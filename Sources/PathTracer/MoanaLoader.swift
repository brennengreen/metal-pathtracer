import simd
import Foundation

/// Loads the real Disney *Moana Island Scene* data set into the instanced
/// renderer. The format is a set of per-element JSON files that reference a base
/// OBJ (`geomObjFile`), optional whole-element copies (`instancedCopies`), and
/// instanced-primitive files (`instancedPrimitiveJsonFiles`) that map archive OBJ
/// paths to large arrays of 4x4 transforms — the millions of palms/rocks/debris.
///
/// Usage: point `--moana` at the unpacked island root (containing `json/` and
/// `obj/`), or at a single element JSON. `--maxInst N` caps instances to fit RAM
/// on a laptop; `--moanaTranspose` flips the matrix convention if geometry looks
/// sheared.
enum MoanaLoader {
    struct Builder {
        var scene = InstancedScene()
        var meshCache: [String: Int] = [:]          // obj path -> mesh index
        var root: URL
        var transpose: Bool
        var maxInstances: Int
        var instCount = 0
        var defaultMat: UInt32 = 0
    }

    static func load(dir: String, aspect: Float, maxInstances: Int, transpose: Bool = false) throws -> InstancedScene {
        let fm = FileManager.default
        let path = URL(fileURLWithPath: dir)
        var isDir: ObjCBool = false
        guard fm.fileExists(atPath: path.path, isDirectory: &isDir) else {
            throw Err.msg("Moana path not found: \(dir).  Download the island data set (see research/MOANA.md) and point --moana at the unpacked root.")
        }
        // Determine root + the list of element JSON files to load.
        var root = path
        var elementFiles: [URL] = []
        if isDir.boolValue {
            // root/json/<element>/<element>.json
            let jsonDir = path.appendingPathComponent("json")
            let base = fm.fileExists(atPath: jsonDir.path) ? jsonDir : path
            if let groups = try? fm.contentsOfDirectory(at: base, includingPropertiesForKeys: nil) {
                for g in groups where (try? g.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true {
                    let candidate = g.appendingPathComponent(g.lastPathComponent + ".json")
                    if fm.fileExists(atPath: candidate.path) { elementFiles.append(candidate) }
                }
            }
            root = path
        } else {
            elementFiles = [path]
            root = path.deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        }
        guard !elementFiles.isEmpty else { throw Err.msg("no element JSON files found under \(dir)") }

        var b = Builder(root: root, transpose: transpose, maxInstances: maxInstances == 0 ? Int.max : maxInstances)
        // A few neutral PBR materials keyed by element-name heuristics.
        b.scene.materials = [
            Material(baseColor: SIMD3(0.78, 0.71, 0.52), roughness: 0.6),  // 0 sand/ground
            Material(baseColor: SIMD3(0.12, 0.40, 0.13), roughness: 0.6),  // 1 foliage
            Material(baseColor: SIMD3(0.36, 0.24, 0.13), roughness: 0.8),  // 2 wood
            Material(baseColor: SIMD3(0.34, 0.33, 0.31), roughness: 0.85), // 3 rock
            Material(baseColor: SIMD3(0.03, 0.12, 0.18), metallic: 0, roughness: 0.05), // 4 water
        ]

        print("Moana: loading \(elementFiles.count) element(s) from \(root.path)")
        for ef in elementFiles {
            if b.instCount >= b.maxInstances { break }
            do { try loadElement(ef, into: &b) }
            catch { FileHandle.standardError.write("  skipped \(ef.lastPathComponent): \(error)\n".data(using: .utf8)!) }
        }
        guard !b.scene.instances.isEmpty else { throw Err.msg("loaded 0 instances — check the data path / format") }

        // Frame the whole scene.
        var lo = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        var hi = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)
        for inst in b.scene.instances { let t = inst.transform.columns.3; lo = simd_min(lo, SIMD3(t.x,t.y,t.z)); hi = simd_max(hi, SIMD3(t.x,t.y,t.z)) }
        let c = (lo + hi) * 0.5, ext = simd_length(hi - lo)
        b.scene.camera = Camera.lookAt(eye: c + SIMD3(ext*0.35, ext*0.22, -ext*0.45),
                                       target: c, fovYDeg: 42, aspect: aspect)
        b.scene.sunDir = simd_normalize(SIMD3(0.5, 0.5, 0.2))
        b.scene.sunColor = SIMD3(8, 7.2, 6)
        print("Moana: \(b.scene.meshes.count) unique meshes, \(b.scene.instances.count) instances, \(b.scene.effectiveTriangles) effective triangles")
        return b.scene
    }

    // MARK: element parsing

    private static func loadElement(_ file: URL, into b: inout Builder) throws {
        guard let data = FileManager.default.contents(atPath: file.path),
              let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw Err.msg("bad JSON")
        }
        let name = file.deletingPathExtension().lastPathComponent
        let mat = material(forName: name)
        let elemXform = matrix(obj["transformMatrix"], transpose: b.transpose)

        // Base geometry, instanced once (+ instancedCopies).
        if let geom = obj["geomObjFile"] as? String, let mi = try meshIndex(geom, &b) {
            addInstance(meshIndex: mi, transform: elemXform, mat: mat, into: &b)
            if let copies = obj["instancedCopies"] as? [String: Any] {
                for (_, v) in copies {
                    guard let cv = v as? [String: Any] else { continue }
                    let xf = matrix(cv["transformMatrix"], transpose: b.transpose)
                    let cmi = (cv["geomObjFile"] as? String).flatMap { try? meshIndex($0, &b) ?? nil } ?? mi
                    addInstance(meshIndex: cmi, transform: xf, mat: mat, into: &b)
                    if b.instCount >= b.maxInstances { return }
                }
            }
        }

        // Instanced primitives: archives of scattered objects with transform arrays.
        if let prims = obj["instancedPrimitiveJsonFiles"] as? [String: Any] {
            for (_, v) in prims {
                guard let pv = v as? [String: Any], let jf = pv["jsonFile"] as? String else { continue }
                try loadInstanceArchive(jf, elementTransform: elemXform, mat: mat, into: &b)
                if b.instCount >= b.maxInstances { return }
            }
        }
    }

    /// An instance-archive JSON maps archive-OBJ paths to { instanceName: [16 floats] }.
    private static func loadInstanceArchive(_ rel: String, elementTransform: simd_float4x4,
                                            mat: UInt32, into b: inout Builder) throws {
        let url = b.root.appendingPathComponent(rel)
        guard let data = FileManager.default.contents(atPath: url.path),
              let top = try JSONSerialization.jsonObject(with: data) as? [String: Any] else { return }
        for (objPath, val) in top {
            guard let mi = try meshIndex(objPath, &b) else { continue }
            guard let xforms = val as? [String: Any] else { continue }
            for (_, m) in xforms {
                let xf = elementTransform * matrix(m, transpose: b.transpose)
                addInstance(meshIndex: mi, transform: xf, mat: mat, into: &b)
                if b.instCount >= b.maxInstances { return }
            }
        }
    }

    // MARK: helpers

    private static func addInstance(meshIndex: Int, transform: simd_float4x4, mat: UInt32, into b: inout Builder) {
        b.scene.instances.append(InstanceDef(meshIndex: meshIndex, transform: transform, materialId: mat))
        b.instCount += 1
    }

    /// Load (and cache) an OBJ referenced relative to the island root.
    private static func meshIndex(_ relPath: String, _ b: inout Builder) throws -> Int? {
        if let i = b.meshCache[relPath] { return i }
        let url = b.root.appendingPathComponent(relPath)
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        let o = try loadOBJ(url.path)
        if o.triangleCount == 0 { return nil }
        let idx = b.scene.meshes.count
        b.scene.meshes.append(Mesh.fromOBJ(o))
        b.meshCache[relPath] = idx
        return idx
    }

    private static func matrix(_ any: Any?, transpose: Bool) -> simd_float4x4 {
        guard let arr = any as? [Any], arr.count >= 16 else { return matrix_identity_float4x4 }
        let f = arr.prefix(16).map { ($0 as? NSNumber)?.floatValue ?? 0 }
        // Moana stores 16 floats as 4 consecutive columns (column-major).
        var m = simd_float4x4(SIMD4(f[0],f[1],f[2],f[3]), SIMD4(f[4],f[5],f[6],f[7]),
                              SIMD4(f[8],f[9],f[10],f[11]), SIMD4(f[12],f[13],f[14],f[15]))
        if transpose { m = simd_transpose(m) }
        return m
    }

    private static func material(forName n: String) -> UInt32 {
        let s = n.lowercased()
        if s.contains("ocean") || s.contains("water") { return 4 }
        if s.contains("palm") || s.contains("tree") || s.contains("trunk") || s.contains("log") { return 2 }
        if s.contains("rock") || s.contains("stone") || s.contains("coral") || s.contains("mountain") { return 3 }
        if s.contains("beach") || s.contains("sand") || s.contains("dune") || s.contains("dunes") { return 0 }
        return 1   // foliage / default
    }

    enum Err: Error { case msg(String) }
}
