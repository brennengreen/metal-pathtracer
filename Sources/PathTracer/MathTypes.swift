import simd
import Foundation

// MARK: - Host mirrors of the MSL structs (layouts must match pathtrace.metal).

struct Camera {
    var position: SIMD3<Float> = .zero
    var right: SIMD3<Float> = SIMD3(1, 0, 0)
    var up: SIMD3<Float> = SIMD3(0, 1, 0)
    var forward: SIMD3<Float> = SIMD3(0, 0, 1)
    var tanHalfFovY: Float = 0.5
    var aspect: Float = 1
    var aperture: Float = 0
    var focusDist: Float = 1
}

struct Uniforms {
    var camera: Camera = Camera()
    var imageSize: SIMD2<UInt32> = .zero
    var frameIndex: UInt32 = 0
    var samplesPerFrame: UInt32 = 1
    var maxBounces: UInt32 = 8
    var numEmissive: UInt32 = 0
    var numTriangles: UInt32 = 0
    var flags: UInt32 = 0
    var background: SIMD3<Float> = .zero
    var radianceClamp: Float = 0
}

struct Material {
    var baseColor: SIMD3<Float> = SIMD3(repeating: 0.8)
    var emission: SIMD3<Float> = .zero
    var metallic: Float = 0
    var roughness: Float = 0.5
    var ior: Float = 1.5
    var texId: Float = -1            // index into the albedo texture array, or -1 for none

    init(baseColor: SIMD3<Float> = SIMD3(repeating: 0.8),
         emission: SIMD3<Float> = .zero,
         metallic: Float = 0, roughness: Float = 0.5, ior: Float = 1.5, texId: Float = -1) {
        self.baseColor = baseColor
        self.emission = emission
        self.metallic = metallic
        self.roughness = roughness
        self.ior = ior
        self.texId = texId
    }
}

// Per-pixel exported feature record (matches `struct Feature` in MSL).
struct Feature {
    var hitPos: SIMD3<Float> = .zero
    var normal: SIMD3<Float> = .zero
    var wo: SIMD3<Float> = .zero
    var baseColor: SIMD3<Float> = .zero
    var metallic: Float = 0
    var roughness: Float = 0
    var hit: Float = 0
    var _pad: Float = 0
    var oneSample: SIMD3<Float> = .zero
    var firstDir: SIMD3<Float> = .zero
}

// MARK: - Math helpers.

@inline(__always) func radians(_ deg: Float) -> Float { deg * .pi / 180 }

extension Camera {
    /// Build a look-at pinhole camera. `fovYDeg` is the vertical field of view.
    static func lookAt(eye: SIMD3<Float>, target: SIMD3<Float>, upHint: SIMD3<Float> = SIMD3(0, 1, 0),
                       fovYDeg: Float, aspect: Float,
                       aperture: Float = 0, focusDist: Float = 1) -> Camera {
        let forward = simd_normalize(target - eye)
        let right = simd_normalize(simd_cross(forward, upHint))
        let up = simd_cross(right, forward)
        var c = Camera()
        c.position = eye
        c.forward = forward
        c.right = right
        c.up = up
        c.tanHalfFovY = tan(radians(fovYDeg) * 0.5)
        c.aspect = aspect
        c.aperture = aperture
        c.focusDist = focusDist
        return c
    }
}

// MARK: - Tonemapping (ACES filmic approximation, Narkowicz 2015) + sRGB.

@inline(__always) func acesFilmic(_ x: SIMD3<Float>) -> SIMD3<Float> {
    let a: Float = 2.51, b: Float = 0.03, c: Float = 2.43, d: Float = 0.59, e: Float = 0.14
    let num = x * (a * x + SIMD3(repeating: b))
    let den = x * (c * x + SIMD3(repeating: d)) + SIMD3(repeating: e)
    return simd_clamp(num / den, SIMD3(repeating: 0), SIMD3(repeating: 1))
}

@inline(__always) func linearToSRGB(_ c: Float) -> Float {
    let x = max(0, min(1, c))
    return x <= 0.0031308 ? x * 12.92 : 1.055 * powf(x, 1 / 2.4) - 0.055
}
