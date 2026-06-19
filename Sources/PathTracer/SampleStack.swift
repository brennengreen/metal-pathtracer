import Metal
import simd
import Foundation

/// Export a rigorous "effective samples" measurement set for one *held-out*
/// camera view:
///   * ref.bin    float32 [H*W, 3]      independent high-spp reference
///   * stack.bin  float32 [M, H*W, 3]   M independent single-sample (1 spp) images
///   * feats.bin  float32 [H*W, 21]     model input features (incl. one cast ray)
///   * stack_meta.json                  shapes + scene/clamp/view metadata
///
/// research/eval_amplify.py averages prefixes of the stack to build the true
/// Monte-Carlo convergence curve relMSE(K) and finds the K at which plain path
/// tracing matches the neural amplifier's error — a directly measured
/// (assumption-free) "one ray is worth K samples".
func exportSampleStack(device: MTLDevice, scene baseScene: Scene, sceneName: String,
                       camera: Camera, width: Int, height: Int,
                       stackM: Int, refSpp: Int, bounces: Int, clamp: Float,
                       view: Int, outDir: String) throws {
    let fm = FileManager.default
    try fm.createDirectory(atPath: outDir, withIntermediateDirectories: true)

    var scene = baseScene
    scene.camera = camera
    let norm = SceneNorm(scene)

    let r = try Renderer(device: device, scene: scene, width: width, height: height)
    r.uniforms.radianceClamp = clamp

    print("  rendering reference (\(refSpp) spp)…")
    let ref = r.renderReferenceLinear(spp: refSpp, bounces: bounces)

    print("  rendering \(stackM) independent 1-spp samples…")
    var stack = [Float](); stack.reserveCapacity(stackM * width * height * 3)
    for m in 0..<stackM {
        let img = r.renderSingleSampleLinear(seed: UInt32(1_000_000 + m), bounces: bounces)
        for p in img { stack.append(p.x); stack.append(p.y); stack.append(p.z) }
    }

    let feats = r.captureFeatures(bounces: bounces)
    var fflat = [Float](); fflat.reserveCapacity(width * height * 21)
    for f in feats { fflat.append(contentsOf: packFeature(f, norm: norm)) }

    func write(_ a: [Float], _ name: String) throws {
        let url = URL(fileURLWithPath: outDir).appendingPathComponent(name)
        try a.withUnsafeBytes { try Data($0).write(to: url) }
    }
    var refflat = [Float](); refflat.reserveCapacity(width * height * 3)
    for p in ref { refflat.append(p.x); refflat.append(p.y); refflat.append(p.z) }
    try write(refflat, "ref.bin")
    try write(stack, "stack.bin")
    try write(fflat, "feats.bin")

    let meta: [String: Any] = [
        "width": width, "height": height, "stackM": stackM, "refSpp": refSpp,
        "bounces": bounces, "clamp": clamp, "view": view, "scene": sceneName,
        "center": [norm.center.x, norm.center.y, norm.center.z], "invExtent": norm.invExtent,
    ]
    let url = URL(fileURLWithPath: outDir).appendingPathComponent("stack_meta.json")
    try JSONSerialization.data(withJSONObject: meta, options: [.prettyPrinted, .sortedKeys]).write(to: url)
    print("Sample stack written to \(outDir)/  (stack: \(stackM)x\(width*height)x3, ref: \(width*height)x3)")
}
