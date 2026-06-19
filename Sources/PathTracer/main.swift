import Metal
import Foundation
import simd

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

// Instanced (BLAS/TLAS) path — the instancing stress test and full textured OBJ scenes (Sponza).
if ["insttest", "sponza", "objscene"].contains(sceneName) {
    do {
        let isc: InstancedScene
        if sceneName == "insttest" {
            isc = InstancedScene.test(aspect: aspect)
        } else {
            let objPath = args.map["obj"] ?? args.map["model"] ?? "assets/sponza/sponza.obj"
            isc = try loadInstancedOBJScene(objPath: objPath, aspect: aspect,
                                            targetExtent: args.float("extent", 1200))
        }
        print("Metal PBR Path Tracer — \(device.name), INSTANCED (BLAS/TLAS)")
        let effTris = isc.effectiveTriangles
        print("scene=\(sceneName)  \(width)x\(height)  spp=\(spp)  bounces=\(bounces)  meshes=\(isc.meshes.count)  instances=\(isc.instanceCount)  effective-tris=\(effTris)")
        let r = try InstancedRenderer(device: device, scene: isc, width: width, height: height)
        r.uniforms.radianceClamp = args.float("clamp", 0)
        r.viewMode = args.int("view", 0)
        r.texturesEnabled = !args.has("notex")

        if let nnPath = args.map["neural"] {
            // Realtime in-Metal neural deferred shading: load the trained per-pixel
            // MLP weights and shade entirely on the GPU (G-buffer → MLP → tonemap).
            try r.loadNeuralWeights(path: nnPath)
            r.texturesEnabled = true
            r.uniforms.radianceClamp = args.float("clamp", 8)
            if args.has("window") {
                runWindowApp(r, bounces: bounces, exposure: args.float("exposure", 1.0))
                exit(0)
            }
            r.renderNeuralHeadless()
            let rgba = r.resolveRGBA8(exposure: args.float("exposure", 1.0))
            let out = args.str("out", "render.png")
            writePNG(rgba8: rgba, width: width, height: height, to: out)
            let benchN = 40
            let t0 = Date()
            for _ in 0..<benchN { r.renderNeuralHeadless() }
            let ms = Date().timeIntervalSince(t0) / Double(benchN) * 1000
            print(String(format: "wrote %@  (realtime per-pixel neural shade: %.2f ms/frame, %.0f FPS @ %dx%d)",
                         out, ms, 1000.0 / ms, width, height))
            exit(0)
        }

        if args.has("window") {
            r.uniforms.radianceClamp = args.float("clamp", 8)
            runWindowApp(r, bounces: bounces, exposure: args.float("exposure", 1.0))
            exit(0)
        }

        if args.has("exportseq") {
            // Textured G-buffer sequence export for neural deferred shading on Sponza.
            r.texturesEnabled = !args.has("notex")
            r.uniforms.radianceClamp = args.float("clamp", 8)
            try exportSequenceInstanced(renderer: r, sceneName: sceneName, width: width, height: height,
                                        frames: args.int("frames", 48), targetSpp: args.int("targetSpp", 128),
                                        bounces: bounces, arcDeg: args.float("arcDeg", 60),
                                        outDir: args.str("out", "research/data_seq"))
            exit(0)
        }

        if args.has("gbufserver") {
            // Resident textured G-buffer server for the interactive viewer (research/play.py).
            r.texturesEnabled = !args.has("notex")
            r.uniforms.radianceClamp = args.float("clamp", 8)
            let (center, invExtent) = r.instNorm()
            let cam0 = isc.camera
            let fovY = 2 * atan(cam0.tanHalfFovY) * 180 / Float.pi
            let asp = Float(width) / Float(height)
            let e0 = cam0.position
            let outPath = args.str("out", "/tmp/pt_gbuf.bin")
            func emit(_ s: String) { FileHandle.standardOutput.write(Data((s + "\n").utf8)) }
            emit("READY \(width) \(height) \(center.x) \(center.y) \(center.z) \(e0.x) \(e0.y) \(e0.z) \(fovY)")
            while let line = readLine(strippingNewline: true) {
                if line == "quit" { break }
                let v = line.split(separator: ",").compactMap { Float($0) }
                guard v.count >= 6 else { emit("OK"); continue }
                let fov = v.count >= 7 ? v[6] : fovY
                r.uniforms.camera = Camera.lookAt(eye: SIMD3(v[0], v[1], v[2]),
                                                  target: SIMD3(v[3], v[4], v[5]), fovYDeg: fov, aspect: asp)
                var g = r.captureGBuffer()
                for i in 0..<(width * height) {                    // normalize RAW position -> posScaled
                    let b = i * 10
                    g[b + 7] = (g[b + 7] - center.x) * invExtent
                    g[b + 8] = (g[b + 8] - center.y) * invExtent
                    g[b + 9] = (g[b + 9] - center.z) * invExtent
                }
                try? g.withUnsafeBytes { try Data($0).write(to: URL(fileURLWithPath: outPath)) }
                emit("OK")
            }
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

    if args.has("exportseq") {
        // Temporally-coherent sequence export for neural deferred shading (Stage 2).
        try exportSequence(device: device, baseScene: scene, width: width, height: height,
                           frames: args.int("frames", 48), targetSpp: args.int("targetSpp", 128),
                           bounces: bounces, clamp: args.float("clamp", 8),
                           arcDeg: args.float("arcDeg", 100), outDir: args.str("out", "research/data_seq"))
        exit(0)
    }

    if args.has("gbufserver") {
        // Persistent G-buffer server for the interactive neural viewer (research/play.py):
        // stays resident (one accel build), reads a camera per line on stdin, runs a single
        // primary-ray pass, and writes the minimal 10-channel deferred G-buffer to `--out`.
        let r = try Renderer(device: device, scene: scene, width: width, height: height)
        r.uniforms.radianceClamp = args.float("clamp", 8)
        let norm = SceneNorm(scene)
        let center = norm.center
        let cam0 = scene.camera
        let fovY = 2 * atan(cam0.tanHalfFovY) * 180 / Float.pi
        let aspect = Float(width) / Float(height)
        let outPath = args.str("out", "/tmp/pt_gbuf.bin")
        func emit(_ s: String) { FileHandle.standardOutput.write(Data((s + "\n").utf8)) }
        // Handshake: image size, scene centre, initial eye, vertical FoV.
        let e0 = cam0.position
        emit("READY \(width) \(height) \(center.x) \(center.y) \(center.z) \(e0.x) \(e0.y) \(e0.z) \(fovY)")
        while let line = readLine(strippingNewline: true) {
            if line == "quit" { break }
            let v = line.split(separator: ",").compactMap { Float($0) }
            guard v.count >= 6 else { emit("OK"); continue }
            let fov = v.count >= 7 ? v[6] : fovY
            r.uniforms.camera = Camera.lookAt(eye: SIMD3(v[0], v[1], v[2]),
                                              target: SIMD3(v[3], v[4], v[5]),
                                              fovYDeg: fov, aspect: aspect)
            let feats = r.captureFeatures(bounces: bounces)
            var buf = [Float](); buf.reserveCapacity(width * height * 10)
            for ft in feats {
                let p = (ft.hitPos - center) * norm.invExtent
                buf.append(contentsOf: [ft.hit, ft.normal.x, ft.normal.y, ft.normal.z,
                                        ft.baseColor.x, ft.baseColor.y, ft.baseColor.z, p.x, p.y, p.z])
            }
            try? buf.withUnsafeBytes { try Data($0).write(to: URL(fileURLWithPath: outPath)) }
            emit("OK")
        }
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
