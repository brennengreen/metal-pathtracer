import simd
import Foundation

/// Deterministic, seedable PRNG (SplitMix64) so dataset camera views and the
/// held-out sample-stack measurement are fully reproducible.
struct SplitMix64: RandomNumberGenerator {
    var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }
}

/// Serializable camera so a specific (held-out) view can be reproduced exactly
/// across separate program runs.
struct CameraRecord: Codable {
    var eye: [Float]
    var target: [Float]
    var fovY: Float
    var aspect: Float

    init(from c: Camera) {
        eye = [c.position.x, c.position.y, c.position.z]
        let t = c.position + c.forward
        target = [t.x, t.y, t.z]
        fovY = 2 * atan(c.tanHalfFovY) * 180 / .pi
        aspect = c.aspect
    }
    func makeCamera() -> Camera {
        Camera.lookAt(eye: SIMD3(eye[0], eye[1], eye[2]),
                      target: SIMD3(target[0], target[1], target[2]),
                      fovYDeg: fovY, aspect: aspect)
    }
}

/// Scene position normalization used to feed positions to the network in a
/// roughly [-1,1] range. Depends only on geometry, so it is identical wherever
/// the same scene is rebuilt.
struct SceneNorm {
    let center: SIMD3<Float>
    let invExtent: Float
    init(_ scene: Scene) {
        var lo = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        var hi = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)
        for p in scene.positions { lo = simd_min(lo, p); hi = simd_max(hi, p) }
        center = (lo + hi) * 0.5
        let extent = simd_reduce_max(hi - lo)
        invExtent = extent > 0 ? 2.0 / extent : 1.0
    }
}

/// Pack a per-pixel `Feature` into the 21-dim model input vector. Layout MUST
/// match research/model.py and DataExport's X.bin writer.
@inline(__always)
func packFeature(_ f: Feature, norm: SceneNorm) -> [Float] {
    let pos = (f.hitPos - norm.center) * norm.invExtent
    return [
        f.hit,
        f.normal.x, f.normal.y, f.normal.z,
        f.wo.x, f.wo.y, f.wo.z,
        f.baseColor.x, f.baseColor.y, f.baseColor.z,
        f.metallic, f.roughness,
        pos.x, pos.y, pos.z,
        f.firstDir.x, f.firstDir.y, f.firstDir.z,
        f.oneSample.x, f.oneSample.y, f.oneSample.z,
    ]
}
