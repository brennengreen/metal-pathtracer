import simd
import Foundation
#if canImport(Darwin)
import Darwin
#endif

/// Triangle mesh loaded from a Wavefront OBJ file.
struct OBJMesh {
    var positions: [SIMD3<Float>] = []
    var normals: [SIMD3<Float>] = []
    var indices: [UInt32] = []

    var triangleCount: Int { indices.count / 3 }

    func bounds() -> (lo: SIMD3<Float>, hi: SIMD3<Float>) {
        var lo = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        var hi = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)
        for p in positions { lo = simd_min(lo, p); hi = simd_max(hi, p) }
        return (lo, hi)
    }
}

enum OBJError: Error { case io(String) }

/// Fast OBJ loader: parses `v` positions and `f` faces (any of v, v/vt, v/vt/vn,
/// v//vn), triangulates polygons (fan), supports negative (relative) indices, and
/// computes area-weighted smooth vertex normals. Texture coords and file normals
/// are ignored (we synthesize smooth normals), which keeps it robust across the
/// many slightly-different OBJ exporters in the wild. Uses C strtof/strtol so it
/// stays fast on multi-million-triangle research meshes.
func loadOBJ(_ path: String) throws -> OBJMesh {
    guard var bytes = FileManager.default.contents(atPath: path).map({ [UInt8]($0) }) else {
        throw OBJError.io("cannot read \(path)")
    }
    bytes.append(0)  // null terminator so strtof/strtol never overrun

    var positions: [SIMD3<Float>] = []
    var indices: [UInt32] = []
    positions.reserveCapacity(1 << 16)
    indices.reserveCapacity(1 << 17)
    var face: [Int] = []; face.reserveCapacity(8)

    bytes.withUnsafeBufferPointer { buf in
        let base = buf.baseAddress!
        let end = base + buf.count - 1            // exclude the trailing 0
        var p = base

        @inline(__always) func isSpace(_ b: UInt8) -> Bool { b == 32 || b == 9 }
        @inline(__always) func isNewline(_ b: UInt8) -> Bool { b == 10 || b == 13 }
        @inline(__always) func cchar(_ q: UnsafePointer<UInt8>) -> UnsafePointer<CChar> {
            UnsafeRawPointer(q).assumingMemoryBound(to: CChar.self)
        }
        @inline(__always) func parseFloat(_ q: inout UnsafePointer<UInt8>) -> Float {
            var e: UnsafeMutablePointer<CChar>? = nil
            let v = strtof(cchar(q), &e)
            if let e = e { q = UnsafeRawPointer(e).assumingMemoryBound(to: UInt8.self) }
            return v
        }
        @inline(__always) func parseInt(_ q: inout UnsafePointer<UInt8>) -> Int {
            var e: UnsafeMutablePointer<CChar>? = nil
            let v = strtol(cchar(q), &e, 10)
            if let e = e { q = UnsafeRawPointer(e).assumingMemoryBound(to: UInt8.self) }
            return v
        }

        while p < end {
            while p < end && (isSpace(p.pointee) || isNewline(p.pointee)) { p += 1 }
            if p >= end { break }
            let c0 = p.pointee
            let next = (p + 1 < end) ? (p + 1).pointee : 0

            if c0 == UInt8(ascii: "v") && isSpace(next) {
                var q = p + 2
                let x = parseFloat(&q), y = parseFloat(&q), z = parseFloat(&q)
                positions.append(SIMD3(x, y, z))
                while q < end && !isNewline(q.pointee) { q += 1 }
                p = q
            } else if c0 == UInt8(ascii: "f") && isSpace(next) {
                var q = p + 2
                face.removeAll(keepingCapacity: true)
                while q < end && !isNewline(q.pointee) {
                    while q < end && isSpace(q.pointee) { q += 1 }
                    if q >= end || isNewline(q.pointee) { break }
                    let raw = parseInt(&q)
                    let idx = raw > 0 ? raw - 1 : positions.count + raw
                    face.append(idx)
                    while q < end && !isSpace(q.pointee) && !isNewline(q.pointee) { q += 1 }
                }
                if face.count >= 3 {
                    for k in 1..<(face.count - 1) {
                        indices.append(UInt32(face[0]))
                        indices.append(UInt32(face[k]))
                        indices.append(UInt32(face[k + 1]))
                    }
                }
                p = q
            } else {
                while p < end && !isNewline(p.pointee) { p += 1 }
            }
        }
    }

    // Area-weighted smooth vertex normals.
    var normals = [SIMD3<Float>](repeating: .zero, count: positions.count)
    var i = 0
    while i < indices.count {
        let a = Int(indices[i]), b = Int(indices[i + 1]), c = Int(indices[i + 2])
        let fn = simd_cross(positions[b] - positions[a], positions[c] - positions[a])
        normals[a] += fn; normals[b] += fn; normals[c] += fn
        i += 3
    }
    for j in normals.indices {
        let l = simd_length(normals[j])
        normals[j] = l > 1e-12 ? normals[j] / l : SIMD3(0, 1, 0)
    }

    return OBJMesh(positions: positions, normals: normals, indices: indices)
}
