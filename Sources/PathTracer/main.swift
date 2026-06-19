import Metal
import Foundation

// Minimal argument parsing: --key value  and  --flag.
struct Args {
    var map: [String: String] = [:]
    var flags: Set<String> = []
    init(_ argv: [String]) {
        var i = 0
        while i < argv.count {
            let a = argv[i]
            if a.hasPrefix("--") {
                let key = String(a.dropFirst(2))
                if i + 1 < argv.count && !argv[i + 1].hasPrefix("--") {
                    map[key] = argv[i + 1]; i += 2
                } else { flags.insert(key); i += 1 }
            } else { i += 1 }
        }
    }
    func int(_ k: String, _ d: Int) -> Int { map[k].flatMap { Int($0) } ?? d }
    func float(_ k: String, _ d: Float) -> Float { map[k].flatMap { Float($0) } ?? d }
    func str(_ k: String, _ d: String) -> String { map[k] ?? d }
    func has(_ k: String) -> Bool { flags.contains(k) || map[k] != nil }
}

let args = Args(Array(CommandLine.arguments.dropFirst()))

guard let device = MTLCreateSystemDefaultDevice() else {
    FileHandle.standardError.write("No Metal device available.\n".data(using: .utf8)!)
    exit(1)
}
guard device.supportsRaytracing else {
    FileHandle.standardError.write("GPU \(device.name) lacks hardware ray tracing.\n".data(using: .utf8)!)
    exit(1)
}

let width   = args.int("width", 512)
let height  = args.int("height", 512)
let spp     = args.int("spp", 256)
let bounces = args.int("bounces", 8)
let sceneName = args.str("scene", "cornell")
let aspect = Float(width) / Float(height)

// Instanced (BLAS/TLAS) path — Moana-scale scenes, the instancing test, and full OBJ scenes.
if ["insttest", "island", "moana", "sponza", "objscene"].contains(sceneName) {
    do {
        let isc: InstancedScene
        if sceneName == "insttest" {
            isc = InstancedScene.test(aspect: aspect)
        } else if sceneName == "moana", let dir = args.map["moana"] ?? args.map["data"] {
            isc = try MoanaLoader.load(dir: dir, aspect: aspect, maxInstances: args.int("maxInst", 0))
        } else if sceneName == "sponza" || sceneName == "objscene" {
            let objPath = args.map["obj"] ?? args.map["model"] ?? "assets/sponza/sponza.obj"
            isc = try loadInstancedOBJScene(objPath: objPath, aspect: aspect,
                                            targetExtent: args.float("extent", 1200))
        } else {
            isc = IslandScene.build(aspect: aspect, density: args.int("density", 1),
                                    seed: UInt64(args.int("seed", 1)))
        }
        print("Metal PBR Path Tracer — \(device.name), INSTANCED (BLAS/TLAS)")
        let effTris = isc.effectiveTriangles
        print("scene=\(sceneName)  \(width)x\(height)  spp=\(spp)  bounces=\(bounces)  meshes=\(isc.meshes.count)  instances=\(isc.instanceCount)  effective-tris=\(effTris)")
        let r = try InstancedRenderer(device: device, scene: isc, width: width, height: height)
        r.uniforms.radianceClamp = args.float("clamp", 0)
        r.viewMode = args.int("view", 0)
        r.texturesEnabled = !args.has("notex")

        if args.has("window") {
            r.uniforms.radianceClamp = args.float("clamp", 8)
            runWindowApp(r, bounces: bounces, exposure: args.float("exposure", 1.0))
            exit(0)
        }

        let useNormals = args.str("kernel", "pathtrace") == "normals"
        let secs = r.render(spp: spp, bounces: bounces, useNormals: useNormals)
        let rgba = r.resolveRGBA8(exposure: args.float("exposure", 1.0))
        let out = args.str("out", "render.png")
        writePNG(rgba8: rgba, width: width, height: height, to: out)
        let n = useNormals ? 1 : spp
        print(String(format: "rendered %d spp in %.2fs (%.1f Meffective-tris, %.2f ms/frame)",
                     n, secs, Double(effTris) / 1e6, secs / Double(n) * 1000))
        print("wrote \(out)")
        exit(0)
    } catch {
        FileHandle.standardError.write("Instanced render error: \(error)\n".data(using: .utf8)!)
        exit(1)
    }
}

let scene: Scene
if let modelPath = args.map["model"] {
    let t0 = Date()
    let mesh = try! loadOBJ(modelPath)
    let secs = Date().timeIntervalSince(t0)
    print(String(format: "loaded %@  (%d verts, %d tris, %.2fs)",
                 (modelPath as NSString).lastPathComponent, mesh.positions.count, mesh.triangleCount, secs))
    let mat = Scene.modelMaterial(args.str("modelMat", "porcelain"))
    let yaw = args.float("yaw", 0)
    if args.str("env", "cornell") == "studio" {
        scene = Scene.studioModel(mesh, material: mat, aspect: aspect,
                                  fitHeight: args.float("fit", 3.0), yawDeg: yaw)
    } else {
        scene = Scene.cornellModel(mesh, material: mat, aspect: aspect,
                                   fitHeight: args.float("fit", 320), yawDeg: yaw)
    }
} else {
    scene = Scene.named(sceneName, aspect: aspect)
}

print("Metal PBR Path Tracer — \(device.name), HW ray tracing")
print("scene=\(args.map["model"] != nil ? "model" : sceneName)  \(width)x\(height)  spp=\(spp)  bounces=\(bounces)  tris=\(scene.triangleCount)  lights=\(scene.emissiveTris.count)")

do {
    let renderer = try Renderer(device: device, scene: scene, width: width, height: height)

    if args.has("window") {
        renderer.uniforms.radianceClamp = args.float("clamp", 8)
        renderer.viewMode = args.int("view", 0)     // open directly on an ML feature channel
        runWindowApp(renderer, bounces: bounces, exposure: args.float("exposure", 1.0))
        exit(0)
    }

    if args.has("export") {
        // Research data export across multiple camera views (P5).
        let outDir = args.str("out", "research/data")
        try exportDataset(device: device, baseScene: scene, width: width, height: height,
                          targetSpp: args.int("targetSpp", 512), bounces: bounces,
                          views: args.int("views", 12), outDir: outDir,
                          clamp: args.float("clamp", 8))
        exit(0)
    }

    if args.has("samplestack") {
        // Reproduce a held-out view from a dataset's cameras.json and emit a
        // Monte-Carlo convergence stack for the rigorous effective-spp measurement.
        let dataDir = args.str("data", "research/data")
        let viewIdx = args.int("view", 15)
        let camURL = URL(fileURLWithPath: dataDir).appendingPathComponent("cameras.json")
        let cams = try JSONDecoder().decode([CameraRecord].self, from: Data(contentsOf: camURL))
        guard viewIdx < cams.count else { throw Renderer.RErr.msg("view \(viewIdx) out of range (\(cams.count) cameras)") }
        let cam = cams[viewIdx].makeCamera()
        try exportSampleStack(device: device, scene: scene, sceneName: sceneName, camera: cam,
                              width: width, height: height,
                              stackM: args.int("stackM", 128), refSpp: args.int("refSpp", 1024),
                              bounces: bounces, clamp: args.float("clamp", 8),
                              view: viewIdx, outDir: args.str("out", "research/stack"))
        exit(0)
    }

    let useNormals = args.str("kernel", "pathtrace") == "normals"
    let useBDPT = args.str("integrator", "pt") == "bdpt"
    renderer.uniforms.radianceClamp = args.float("clamp", 0)
    renderer.debugSMax = UInt32(args.int("smax", 0))
    var dbg: UInt32 = 0
    if args.has("noS0") { dbg |= (1 << 16) }
    if args.has("noS1") { dbg |= (1 << 17) }
    if args.has("mis1") { dbg |= (1 << 18) }
    renderer.debugFlags = dbg

    // ML-pipeline feature diagnostics, headless: `--view N` (1..9) dumps one
    // channel of the neural-amplifier input vector to PNG (the same channels the
    // interactive viewer shows), so the AOVs double as figures / sanity checks.
    let featureView = args.int("view", 0)
    if featureView > 0 {
        let rgba = renderer.featureVizRGBA8(mode: featureView, bounces: bounces,
                                            exposure: args.float("exposure", 1.0))
        let out = args.str("out", "render.png")
        writePNG(rgba8: rgba, width: width, height: height, to: out)
        let names = Renderer.featureViewModeNames
        let label = featureView < names.count ? names[featureView] : "view\(featureView)"
        print("wrote \(out)  (ML feature channel \(featureView): \(label))")
        exit(0)
    }

    let secs = renderer.renderHeadless(spp: spp, bounces: bounces, useNormals: useNormals, bdpt: useBDPT)
    let rgba = renderer.resolveRGBA8(exposure: args.float("exposure", 1.0))
    let out = args.str("out", "render.png")
    writePNG(rgba8: rgba, width: width, height: height, to: out)

    let totalSamples = useNormals ? 1 : spp
    let msPerFrame = secs / Double(totalSamples) * 1000
    print(String(format: "[%@] rendered %d samples in %.2fs  (%.2f ms/frame @1spp,  %.1f Mpath/s)",
                 useBDPT ? "bdpt" : "pt", totalSamples, secs, msPerFrame,
                 Double(width * height) * Double(totalSamples) / secs / 1e6))
    print("wrote \(out)")
} catch {
    FileHandle.standardError.write("Error: \(error)\n".data(using: .utf8)!)
    exit(1)
}
