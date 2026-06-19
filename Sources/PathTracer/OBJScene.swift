import simd
import Foundation
#if canImport(Darwin)
import Darwin
#endif

/// A Wavefront material (the subset we use): diffuse color, specular exponent,
/// opacity, and the diffuse texture path (`map_Kd`).
struct MTLMaterialDef {
    var name: String
    var kd: SIMD3<Float> = SIMD3(repeating: 0.7)
    var ks: SIMD3<Float> = .zero
    var ns: Float = 32
    var d: Float = 1                 // opacity (1 = opaque)
    var mapKd: String? = nil         // diffuse texture (relative to the .mtl)
    var mapD: String? = nil          // alpha mask
}

/// A full OBJ *scene*: shared vertex arrays (position / uv / normal, de-indexed so
/// every unique v/vt/vn combination is its own vertex) plus one index group per
/// material — exactly what the instanced renderer needs to draw a textured,
/// multi-material scene (e.g. Crytek Sponza).
struct OBJSceneData {
    var positions: [SIMD3<Float>] = []
    var uvs: [SIMD2<Float>] = []
    var normals: [SIMD3<Float>] = []
    var groups: [(material: String, indices: [UInt32])] = []

    func bounds() -> (lo: SIMD3<Float>, hi: SIMD3<Float>) {
        var lo = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        var hi = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)
        for p in positions { lo = simd_min(lo, p); hi = simd_max(hi, p) }
        return (lo, hi)
    }
    var triangleCount: Int { groups.reduce(0) { $0 + $1.indices.count } / 3 }
}

/// Parse a `.mtl` file into a name→material map.
func parseMTL(_ path: String) -> [String: MTLMaterialDef] {
    guard let text = try? String(contentsOfFile: path, encoding: .utf8) else { return [:] }
    var out: [String: MTLMaterialDef] = [:]
    var cur: MTLMaterialDef? = nil
    func flush() { if let c = cur { out[c.name] = c } }
    for raw in text.split(separator: "\n", omittingEmptySubsequences: false) {
        let line = raw.trimmingCharacters(in: .whitespaces)
        let parts = line.split(separator: " ", omittingEmptySubsequences: true).map(String.init)
        guard let key = parts.first else { continue }
        switch key {
        case "newmtl":
            flush(); cur = MTLMaterialDef(name: parts.count > 1 ? parts[1] : "unnamed")
        case "Kd" where parts.count >= 4:
            cur?.kd = SIMD3(Float(parts[1]) ?? 0.7, Float(parts[2]) ?? 0.7, Float(parts[3]) ?? 0.7)
        case "Ks" where parts.count >= 4:
            cur?.ks = SIMD3(Float(parts[1]) ?? 0, Float(parts[2]) ?? 0, Float(parts[3]) ?? 0)
        case "Ns" where parts.count >= 2:
            cur?.ns = Float(parts[1]) ?? 32
        case "d" where parts.count >= 2:
            cur?.d = Float(parts[1]) ?? 1
        case "Tr" where parts.count >= 2:               // some exporters use Tr = 1 - d
            cur?.d = 1 - (Float(parts[1]) ?? 0)
        case "map_Kd" where parts.count >= 2:
            cur?.mapKd = parts.last
        case "map_d" where parts.count >= 2:
            cur?.mapD = parts.last
        default: break
        }
    }
    flush()
    return out
}

/// Fast byte-level OBJ loader that preserves UVs and per-material face groups.
/// De-indexes faces on the unique (position, uv, normal) tuple, fan-triangulates
/// polygons, and uses file normals where present (recomputing smooth normals only
/// for vertices that lack one).
func loadOBJScene(_ path: String) throws -> OBJSceneData {
    guard var bytes = FileManager.default.contents(atPath: path).map({ [UInt8]($0) }) else {
        throw OBJError.io("cannot read \(path)")
    }
    bytes.append(0)

    var vPos: [SIMD3<Float>] = [], vUV: [SIMD2<Float>] = [], vN: [SIMD3<Float>] = []
    vPos.reserveCapacity(1 << 17); vUV.reserveCapacity(1 << 17); vN.reserveCapacity(1 << 17)

    var out = OBJSceneData()
    var hasNormal: [Bool] = []
    var dedup: [UInt64: UInt32] = [:]
    dedup.reserveCapacity(1 << 19)

    var groupMap: [String: Int] = [:]            // material name -> index in out.groups
    var curGroup = -1
    func ensureGroup(_ name: String) {
        if let g = groupMap[name] { curGroup = g; return }
        groupMap[name] = out.groups.count; curGroup = out.groups.count
        out.groups.append((material: name, indices: []))
    }
    ensureGroup("__default__")

    // face vertex tuple -> output vertex index (dedup on packed key)
    func vertex(pos: Int, uv: Int, nrm: Int) -> UInt32 {
        let p = UInt64(pos & 0x1FFFFF), u = UInt64((uv + 1) & 0x1FFFFF), n = UInt64((nrm + 1) & 0x1FFFFF)
        let key = (p << 42) | (u << 21) | n
        if let i = dedup[key] { return i }
        let idx = UInt32(out.positions.count)
        out.positions.append(pos >= 0 && pos < vPos.count ? vPos[pos] : .zero)
        out.uvs.append(uv >= 0 && uv < vUV.count ? vUV[uv] : .zero)
        if nrm >= 0 && nrm < vN.count { out.normals.append(vN[nrm]); hasNormal.append(true) }
        else { out.normals.append(.zero); hasNormal.append(false) }
        dedup[key] = idx
        return idx
    }

    bytes.withUnsafeBufferPointer { buf in
        let base = buf.baseAddress!
        let end = base + buf.count - 1
        var p = base
        @inline(__always) func isSpace(_ b: UInt8) -> Bool { b == 32 || b == 9 }
        @inline(__always) func isNL(_ b: UInt8) -> Bool { b == 10 || b == 13 }
        @inline(__always) func cchar(_ q: UnsafePointer<UInt8>) -> UnsafePointer<CChar> {
            UnsafeRawPointer(q).assumingMemoryBound(to: CChar.self)
        }
        @inline(__always) func pf(_ q: inout UnsafePointer<UInt8>) -> Float {
            var e: UnsafeMutablePointer<CChar>? = nil
            let v = strtof(cchar(q), &e)
            if let e = e { q = UnsafeRawPointer(e).assumingMemoryBound(to: UInt8.self) }
            return v
        }
        @inline(__always) func pi(_ q: inout UnsafePointer<UInt8>) -> Int {
            var e: UnsafeMutablePointer<CChar>? = nil
            let v = strtol(cchar(q), &e, 10)
            if let e = e { q = UnsafeRawPointer(e).assumingMemoryBound(to: UInt8.self) }
            return v
        }
        var face: [(Int, Int, Int)] = []; face.reserveCapacity(8)

        while p < end {
            while p < end && (isSpace(p.pointee) || isNL(p.pointee)) { p += 1 }
            if p >= end { break }
            let c0 = p.pointee
            let c1 = (p + 1 < end) ? (p + 1).pointee : 0

            if c0 == UInt8(ascii: "v") && isSpace(c1) {
                var q = p + 2
                let x = pf(&q), y = pf(&q), z = pf(&q); vPos.append(SIMD3(x, y, z))
                while q < end && !isNL(q.pointee) { q += 1 }; p = q
            } else if c0 == UInt8(ascii: "v") && c1 == UInt8(ascii: "t") {
                var q = p + 2
                let u = pf(&q), v = pf(&q); vUV.append(SIMD2(u, v))
                while q < end && !isNL(q.pointee) { q += 1 }; p = q
            } else if c0 == UInt8(ascii: "v") && c1 == UInt8(ascii: "n") {
                var q = p + 2
                let x = pf(&q), y = pf(&q), z = pf(&q); vN.append(SIMD3(x, y, z))
                while q < end && !isNL(q.pointee) { q += 1 }; p = q
            } else if c0 == UInt8(ascii: "f") && isSpace(c1) {
                var q = p + 2
                face.removeAll(keepingCapacity: true)
                while q < end && !isNL(q.pointee) {
                    while q < end && isSpace(q.pointee) { q += 1 }
                    if q >= end || isNL(q.pointee) { break }
                    let pa = pi(&q)
                    var ub = 0, nc = 0
                    if q < end && q.pointee == UInt8(ascii: "/") {
                        q += 1
                        if q < end && q.pointee != UInt8(ascii: "/") { ub = pi(&q) }
                        if q < end && q.pointee == UInt8(ascii: "/") { q += 1; nc = pi(&q) }
                    }
                    let posI = pa > 0 ? pa - 1 : vPos.count + pa
                    let uvI  = ub > 0 ? ub - 1 : (ub < 0 ? vUV.count + ub : -1)
                    let nI   = nc > 0 ? nc - 1 : (nc < 0 ? vN.count + nc : -1)
                    face.append((posI, uvI, nI))
                    while q < end && !isSpace(q.pointee) && !isNL(q.pointee) { q += 1 }
                }
                if face.count >= 3 {
                    let v0 = vertex(pos: face[0].0, uv: face[0].1, nrm: face[0].2)
                    for k in 1..<(face.count - 1) {
                        let v1 = vertex(pos: face[k].0, uv: face[k].1, nrm: face[k].2)
                        let v2 = vertex(pos: face[k+1].0, uv: face[k+1].1, nrm: face[k+1].2)
                        out.groups[curGroup].indices.append(contentsOf: [v0, v1, v2])
                    }
                }
                p = q
            } else if c0 == UInt8(ascii: "u") {           // usemtl
                // read the rest of the line as the material name
                var q = p
                while q < end && !isSpace(q.pointee) { q += 1 }   // skip "usemtl"
                while q < end && isSpace(q.pointee) { q += 1 }
                let nameStart = q
                while q < end && !isNL(q.pointee) { q += 1 }
                let name = String(decoding: UnsafeBufferPointer(start: nameStart, count: q - nameStart), as: UTF8.self)
                    .trimmingCharacters(in: .whitespaces)
                ensureGroup(name.isEmpty ? "__default__" : name)
                p = q
            } else {
                while p < end && !isNL(p.pointee) { p += 1 }
            }
        }
    }

    // Fill any missing normals with area-weighted smooth normals.
    if hasNormal.contains(false) {
        var acc = [SIMD3<Float>](repeating: .zero, count: out.positions.count)
        for g in out.groups {
            var i = 0
            while i < g.indices.count {
                let a = Int(g.indices[i]), b = Int(g.indices[i+1]), c = Int(g.indices[i+2])
                let fn = simd_cross(out.positions[b] - out.positions[a], out.positions[c] - out.positions[a])
                acc[a] += fn; acc[b] += fn; acc[c] += fn; i += 3
            }
        }
        for j in out.normals.indices where !hasNormal[j] {
            let l = simd_length(acc[j]); out.normals[j] = l > 1e-12 ? acc[j] / l : SIMD3(0, 1, 0)
        }
    }
    // drop the empty default group if unused
    out.groups.removeAll { $0.indices.isEmpty }
    return out
}

// MARK: - Build a textured, multi-material instanced scene from an OBJ + MTL.

/// Load a full OBJ scene (e.g. Crytek Sponza) into an `InstancedScene`: one
/// compacted mesh+instance per material group, per-material PBR from the sibling
/// `.mtl` (diffuse colour + `map_Kd` albedo texture), framed and lit by sun+sky.
/// Positions are uniformly scaled so the scene fits a sane world size (keeps the
/// ray-offset epsilon well-conditioned).
func loadInstancedOBJScene(objPath: String, aspect: Float,
                           targetExtent: Float = 1200) throws -> InstancedScene {
    let data = try loadOBJScene(objPath)
    guard !data.groups.isEmpty else { throw OBJError.io("no geometry in \(objPath)") }
    let objURL = URL(fileURLWithPath: objPath)
    let dir = objURL.deletingLastPathComponent()
    let mtlURL = objURL.deletingPathExtension().appendingPathExtension("mtl")
    let mtl = parseMTL(mtlURL.path)

    // World-fit scale + recenter on the floor.
    let (lo, hi) = data.bounds()
    let ext = simd_length(hi - lo)
    let s: Float = ext > 0 ? targetExtent / ext : 1
    let center = (lo + hi) * 0.5

    var scene = InstancedScene()

    // Texture table: unique map_Kd paths in first-seen order.
    var texIndex: [String: Int] = [:]
    func textureId(_ rel: String?) -> Float {
        guard let rel = rel else { return -1 }
        if let i = texIndex[rel] { return Float(i) }
        if texIndex.count >= 32 { return -1 }       // array is fixed at 32 slots
        let id = texIndex.count
        texIndex[rel] = id
        scene.texturePaths.append(dir.appendingPathComponent(rel).path)
        return Float(id)
    }

    // Build one compacted mesh + instance per material group.
    var remap = [Int32](repeating: -1, count: data.positions.count)
    let keepRoof = ProcessInfo.processInfo.environment["SPONZA_KEEPROOF"] != nil
    for group in data.groups {
        // The full-footprint roof caps the scene and blocks the sky dome; drop it
        // (and the outer shell) so skylight floods the courtyard — the classic look.
        if !keepRoof {
            let n = group.material.lowercased()
            if n.contains("roof") || n.contains("ceiling") { continue }
        }
        let def = mtl[group.material]
        let tex = ProcessInfo.processInfo.environment["SPONZA_NOTEX"] != nil ? -1 : textureId(def?.mapKd)
        // matte stone/fabric: keep PBR diffuse; Kd is a fallback when no texture.
        let mat = Material(baseColor: def?.kd ?? SIMD3(repeating: 0.6),
                           metallic: 0, roughness: 0.82, texId: tex)
        let matId = UInt32(scene.materials.count)
        scene.materials.append(mat)

        var lp: [SIMD3<Float>] = [], ln: [SIMD3<Float>] = [], lu: [SIMD2<Float>] = [], li: [UInt32] = []
        for gi in group.indices {
            let g = Int(gi)
            if remap[g] < 0 {
                remap[g] = Int32(lp.count)
                lp.append((data.positions[g] - center) * s)
                ln.append(data.normals[g])
                lu.append(data.uvs[g])
            }
            li.append(UInt32(remap[g]))
        }
        for gi in group.indices { remap[Int(gi)] = -1 }      // reset for next group
        let meshIdx = scene.meshes.count
        scene.meshes.append(Mesh(positions: lp, normals: ln, indices: li, uvs: lu))
        scene.instances.append(InstanceDef(meshIndex: meshIdx, transform: matrix_identity_float4x4, materialId: matId))
    }

    // Framing: the roof sits high and wide, which inflates the bounds — so measure
    // the atrium "core" from the lower geometry (below mid-height) and frame off it:
    // stand inside, near one end of the longer horizontal axis, look down it.
    let hCut = lo.y + (hi.y - lo.y) * 0.55
    var cMinX: Float = .greatestFiniteMagnitude, cMaxX = -Float.greatestFiniteMagnitude
    var cMinZ: Float = .greatestFiniteMagnitude, cMaxZ = -Float.greatestFiniteMagnitude
    var cMaxY = -Float.greatestFiniteMagnitude
    for p in data.positions where p.y < hCut {
        cMinX = min(cMinX, p.x); cMaxX = max(cMaxX, p.x)
        cMinZ = min(cMinZ, p.z); cMaxZ = max(cMaxZ, p.z)
        cMaxY = max(cMaxY, p.y)
    }
    let coreCx = (cMinX + cMaxX) * 0.5, coreCz = (cMinZ + cMaxZ) * 0.5
    let colH = max(cMaxY - lo.y, 1)               // atrium height (floor->columns)
    if ProcessInfo.processInfo.environment["SPONZA_DBG"] != nil {
        print("CORE x[\(cMinX),\(cMaxX)] z[\(cMinZ),\(cMaxZ)] y[\(lo.y),\(cMaxY)] colH=\(colH) center=\(center)")
    }
    let longestX = (cMaxX - cMinX) >= (cMaxZ - cMinZ)
    func toWorld(_ p: SIMD3<Float>) -> SIMD3<Float> { (p - center) * s }
    let eyeN: SIMD3<Float>, tgtN: SIMD3<Float>
    if longestX {
        eyeN = SIMD3(cMinX + (cMaxX - cMinX) * 0.245, lo.y + colH * 0.57, coreCz)
        tgtN = SIMD3(cMaxX - (cMaxX - cMinX) * 0.248, lo.y + colH * 0.43, coreCz)
    } else {
        eyeN = SIMD3(coreCx, lo.y + colH * 0.57, cMinZ + (cMaxZ - cMinZ) * 0.245)
        tgtN = SIMD3(coreCx, lo.y + colH * 0.43, cMaxZ - (cMaxZ - cMinZ) * 0.248)
    }
    scene.camera = Camera.lookAt(eye: toWorld(eyeN), target: toWorld(tgtN), fovYDeg: 65, aspect: aspect)
    if let env = ProcessInfo.processInfo.environment["SPONZA_CAM"] {  // "ex,ey,ez,tx,ty,tz" native
        let v = env.split(separator: ",").compactMap { Float($0) }
        if v.count == 6 {
            scene.camera = Camera.lookAt(eye: toWorld(SIMD3(v[0], v[1], v[2])),
                                         target: toWorld(SIMD3(v[3], v[4], v[5])), fovYDeg: 65, aspect: aspect)
        }
    }

    // Sun pours in through the open roof; bright sky fills the shadows via GI.
    scene.sunDir = simd_normalize(SIMD3(0.28, 0.92, 0.27))
    scene.sunColor = SIMD3(7.6, 7.0, 6.0)
    scene.skyZenith = SIMD3(0.22, 0.40, 0.66)
    scene.skyHorizon = SIMD3(0.85, 0.89, 0.95)

    print("OBJ scene: \(objURL.lastPathComponent) — \(data.triangleCount) tris, " +
          "\(scene.meshes.count) material groups, \(scene.texturePaths.count) textures, scale=\(String(format: "%.4f", s))")
    return scene
}
