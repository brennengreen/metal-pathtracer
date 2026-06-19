import Metal
import simd
import Foundation

/// Generate a neural-amplifier dataset across several camera views of a scene.
///
/// For every pixel of every view we record:
///   * first-hit G-buffer features + the single *physically cast* ray's 1-spp radiance
///   * the high-spp converged ("multi-sample") radiance as the regression target
///
/// Output (consumed by research/train.py):
///   X.bin    float32 [N, IN_DIM]   feature vectors
///   Y.bin    float32 [N, 3]        converged radiance targets
///   meta.json                      shapes + per-view train/val split
func exportDataset(device: MTLDevice, baseScene: Scene, width: Int, height: Int,
                   targetSpp: Int, bounces: Int, views: Int, outDir: String, clamp: Float) throws {
    let IN_DIM = 21
    let fm = FileManager.default
    try fm.createDirectory(atPath: outDir, withIntermediateDirectories: true)

    // Scene normalization (position scaling so the MLP sees [-1,1]-ish coords).
    let norm = SceneNorm(baseScene)
    let center = norm.center
    let extent = 2.0 / norm.invExtent

    var X = [Float](); X.reserveCapacity(views * width * height * IN_DIM)
    var Y = [Float](); Y.reserveCapacity(views * width * height * 3)
    var cameras = [CameraRecord]()

    var rng = SplitMix64(seed: 0xC0FFEE_1234)        // deterministic, reproducible views
    func jitter(_ s: Float) -> Float { Float.random(in: -s...s, using: &rng) }

    let eyeJit = 0.10 * extent
    let tgtJit = 0.05 * extent

    for v in 0..<views {
        // Reproducible per-view camera perturbation around the base view.
        var sc = baseScene
        let base = baseScene.camera
        if v == 0 {
            sc.camera = base                              // canonical view kept verbatim
        } else {
            let eye = base.position + SIMD3(jitter(eyeJit), jitter(eyeJit) * 0.5, jitter(eyeJit))
            let tgt = center + SIMD3(jitter(tgtJit), jitter(tgtJit), jitter(tgtJit))
            sc.camera = Camera.lookAt(eye: eye, target: tgt, fovYDeg: 39.3 + jitter(3),
                                      aspect: Float(width) / Float(height))
        }
        cameras.append(CameraRecord(from: sc.camera))

        let r = try Renderer(device: device, scene: sc, width: width, height: height)
        r.uniforms.radianceClamp = clamp
        let (feats, target) = r.exportTrainingData(targetSpp: targetSpp, bounces: bounces)

        for i in 0..<(width * height) {
            X.append(contentsOf: packFeature(feats[i], norm: norm))
            Y.append(contentsOf: [target[i].x, target[i].y, target[i].z])
        }
        print(String(format: "  view %2d/%d exported (%d px, %d spp target)", v + 1, views, width * height, targetSpp))
    }

    func writeFloats(_ a: [Float], _ name: String) throws {
        let url = URL(fileURLWithPath: outDir).appendingPathComponent(name)
        try a.withUnsafeBytes { try Data($0).write(to: url) }
    }
    try writeFloats(X, "X.bin")
    try writeFloats(Y, "Y.bin")

    // Last 2 views held out for validation (unseen camera positions).
    let valViews = views >= 4 ? [views - 2, views - 1] : [max(0, views - 1)]
    let trainViews = (0..<views).filter { !valViews.contains($0) }
    let meta: [String: Any] = [
        "n": X.count / IN_DIM, "inDim": IN_DIM, "outDim": 3,
        "views": views, "perView": width * height,
        "trainViews": trainViews, "valViews": valViews,
        "width": width, "height": height, "targetSpp": targetSpp, "bounces": bounces,
        "scene": sceneNameHint(baseScene), "clamp": clamp,
        "center": [center.x, center.y, center.z], "invExtent": norm.invExtent,
        "featureLayout": ["hit", "normal.xyz", "wo.xyz", "baseColor.xyz",
                          "metallic", "roughness", "posScaled.xyz", "firstDir.xyz", "oneSample.xyz"],
    ]
    let metaURL = URL(fileURLWithPath: outDir).appendingPathComponent("meta.json")
    try JSONSerialization.data(withJSONObject: meta, options: [.prettyPrinted, .sortedKeys]).write(to: metaURL)

    // Per-view cameras so a held-out view can be reproduced for measurement.
    let camURL = URL(fileURLWithPath: outDir).appendingPathComponent("cameras.json")
    try JSONEncoder().encode(cameras).write(to: camURL)

    print("Dataset written to \(outDir)/  (X: \(X.count / IN_DIM)x\(IN_DIM), Y: \(Y.count / 3)x3)")
    print("Train views: \(trainViews)   Val views: \(valViews)")
}

private func sceneNameHint(_ s: Scene) -> String {
    // Heuristic label for reporting (geometry-based).
    return s.triangleCount <= 64 ? "cornell" : "showcase"
}
