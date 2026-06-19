import Metal
import simd
import Foundation

/// Owns the Metal device, scene buffers, acceleration structure, pipelines, and
/// the progressive accumulation buffer. Drives both headless and realtime render.
final class Renderer {
    let device: MTLDevice
    let queue: MTLCommandQueue

    // Scene GPU buffers.
    private let positionsBuf: MTLBuffer
    private let normalsBuf: MTLBuffer
    private let indicesBuf: MTLBuffer
    private let triMaterialBuf: MTLBuffer
    private let materialsBuf: MTLBuffer
    private let emissiveBuf: MTLBuffer
    private let accel: MTLAccelerationStructure

    // Pipelines.
    private let psPathtrace: MTLComputePipelineState
    private let psNormals: MTLComputePipelineState
    private let psExport: MTLComputePipelineState
    private let psResolve: MTLComputePipelineState
    private let psBDPT: MTLComputePipelineState
    private let psFeatureViz: MTLComputePipelineState

    // Accumulation.
    let width: Int
    let height: Int
    private(set) var accumBuf: MTLBuffer
    var uniforms: Uniforms
    private(set) var accumulatedFrames: UInt32 = 0
    var debugSMax: UInt32 = 0     // 0 = unlimited; else cap BDPT light-subpath length
    var debugFlags: UInt32 = 0    // extra flag bits ORed into uniforms.flags (debug)

    // MARK: ML-pipeline feature diagnostics (interactive viewer)

    /// View modes the viewer can show for this (Cornell/showcase) backend.
    /// Index 0 is Beauty; 1..9 are the individual channels of the 21-dim feature
    /// vector the neural ray-amplifier consumes (see research/model.py /
    /// `packFeature`); 10 is the screen-space motion vector (the temporal cue a
    /// neural deferred renderer reprojects the previous frame with).
    static let featureViewModeNames = [
        "Beauty", "1-spp Radiance", "Shading Normal", "View Dir", "Base Color",
        "Metallic", "Roughness", "Norm. Position", "Bounce Dir", "Hit Mask",
        "Motion Vectors",
    ]
    /// Selected feature channel (0 = Beauty path trace). Driven by the viewer.
    var viewMode = 0 { didSet { if viewMode != oldValue { featVizValid = false } } }
    private var featVizBuf: MTLBuffer?        // per-pixel Feature buffer reused for the viz
    private var featVizValid = false          // are captured features current for the camera?
    private lazy var sceneNorm = SceneNorm(scene)
    /// Camera pose of the *previous* distinct view, set by the viewer on each
    /// camera change; used to build screen-space motion vectors. nil → headless,
    /// where a small synthetic camera nudge stands in so the field is viewable.
    var prevCamera: Camera?
    private let motionVizScale: Float = 24    // pixels mapped to the [0,1] color range
    /// One-line summary of the ML input distribution for the current view,
    /// refreshed whenever features are recaptured; nil in Beauty mode.
    private(set) var featureDiag: String?

    let scene: Scene

    init(device: MTLDevice, scene: Scene, width: Int, height: Int) throws {
        // Validate host/device layout agreement early.
        precondition(MemoryLayout<Uniforms>.stride == 144, "Uniforms stride \(MemoryLayout<Uniforms>.stride) != 144")
        precondition(MemoryLayout<Material>.stride == 48, "Material stride mismatch")
        precondition(MemoryLayout<Feature>.stride == 112, "Feature stride mismatch")

        self.device = device
        self.scene = scene
        self.width = width
        self.height = height
        guard let q = device.makeCommandQueue() else { throw RErr.msg("no command queue") }
        self.queue = q

        func buf<T>(_ arr: [T], _ label: String) throws -> MTLBuffer {
            let len = max(MemoryLayout<T>.stride * arr.count, MemoryLayout<T>.stride)
            guard let b = device.makeBuffer(length: len, options: .storageModeShared) else {
                throw RErr.msg("buffer \(label)")
            }
            if !arr.isEmpty { arr.withUnsafeBytes { b.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) } }
            b.label = label
            return b
        }
        positionsBuf  = try buf(scene.positions, "positions")
        normalsBuf    = try buf(scene.normals, "normals")
        indicesBuf    = try buf(scene.indices, "indices")
        triMaterialBuf = try buf(scene.triMaterial, "triMaterial")
        materialsBuf  = try buf(scene.materials, "materials")
        emissiveBuf   = try buf(scene.emissiveTris.isEmpty ? [UInt32(0)] : scene.emissiveTris, "emissive")

        // Build the primitive acceleration structure (hardware ray tracing).
        let geom = MTLAccelerationStructureTriangleGeometryDescriptor()
        geom.vertexBuffer = positionsBuf
        geom.vertexStride = MemoryLayout<SIMD3<Float>>.stride       // 16, .float3 reads first 12
        geom.vertexFormat = .float3
        geom.triangleCount = scene.triangleCount
        geom.indexBuffer = indicesBuf
        geom.indexType = .uint32
        let primDesc = MTLPrimitiveAccelerationStructureDescriptor()
        primDesc.geometryDescriptors = [geom]

        let sizes = device.accelerationStructureSizes(descriptor: primDesc)
        guard let a = device.makeAccelerationStructure(size: sizes.accelerationStructureSize),
              let scratch = device.makeBuffer(length: max(sizes.buildScratchBufferSize, 32),
                                              options: .storageModePrivate) else {
            throw RErr.msg("acceleration structure alloc")
        }
        a.label = "scene-accel"
        let cmd = q.makeCommandBuffer()!
        let enc = cmd.makeAccelerationStructureCommandEncoder()!
        enc.build(accelerationStructure: a, descriptor: primDesc, scratchBuffer: scratch, scratchBufferOffset: 0)
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        self.accel = a

        // Compile the MSL kernels at runtime from the bundled .metal source.
        guard let url = Bundle.module.url(forResource: "pathtrace", withExtension: "metal") else {
            throw RErr.msg("pathtrace.metal not found in bundle")
        }
        let src = try String(contentsOf: url, encoding: .utf8)
        let opts = MTLCompileOptions()
        opts.languageVersion = .version3_0
        let lib = try device.makeLibrary(source: src, options: opts)
        func pso(_ name: String) throws -> MTLComputePipelineState {
            guard let fn = lib.makeFunction(name: name) else { throw RErr.msg("missing kernel \(name)") }
            return try device.makeComputePipelineState(function: fn)
        }
        psPathtrace = try pso("pathtrace")
        psNormals   = try pso("renderNormals")
        psExport    = try pso("exportFeatures")
        psResolve   = try pso("resolve")
        psBDPT      = try pso("bdpt")
        psFeatureViz = try pso("resolveFeatureViz")

        guard let acc = device.makeBuffer(length: width * height * MemoryLayout<SIMD4<Float>>.stride,
                                          options: .storageModeShared) else {
            throw RErr.msg("accum buffer")
        }
        acc.label = "accum"
        accumBuf = acc

        var u = Uniforms()
        u.camera = scene.camera
        u.imageSize = SIMD2(UInt32(width), UInt32(height))
        u.maxBounces = 8
        u.samplesPerFrame = 1
        u.numEmissive = UInt32(scene.emissiveTris.count)
        u.numTriangles = UInt32(scene.triangleCount)
        u.background = scene.background
        self.uniforms = u
    }

    enum RErr: Error { case msg(String) }

    func resetAccumulation() { accumulatedFrames = 0; featVizValid = false }

    /// Dispatch one accumulation frame of the path tracer (or normals debug view).
    func renderFrame(samplesPerFrame: Int, bounces: Int, kernel: MTLComputePipelineState) {
        uniforms.samplesPerFrame = UInt32(samplesPerFrame)
        uniforms.maxBounces = UInt32(bounces)
        uniforms.frameIndex = accumulatedFrames
        uniforms.flags = (accumulatedFrames == 0) ? 1 : 0     // bit0 = reset accumulation
        uniforms.flags |= (debugSMax & 0xFF) << 8
        uniforms.flags |= debugFlags

        let cmd = queue.makeCommandBuffer()!
        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(kernel)
        enc.setBuffer(accumBuf, offset: 0, index: 0)
        enc.setBytes(&uniforms, length: MemoryLayout<Uniforms>.stride, index: 1)
        enc.setBuffer(positionsBuf, offset: 0, index: 2)
        enc.setBuffer(normalsBuf, offset: 0, index: 3)
        enc.setBuffer(indicesBuf, offset: 0, index: 4)
        enc.setBuffer(triMaterialBuf, offset: 0, index: 5)
        enc.setBuffer(materialsBuf, offset: 0, index: 6)
        enc.setBuffer(emissiveBuf, offset: 0, index: 7)
        enc.setAccelerationStructure(accel, bufferIndex: 8)
        dispatch(enc, kernel)
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        accumulatedFrames += 1
    }

    private func dispatch(_ enc: MTLComputeCommandEncoder, _ pso: MTLComputePipelineState) {        let w = pso.threadExecutionWidth
        let h = max(1, pso.maxTotalThreadsPerThreadgroup / w)
        let tg = MTLSize(width: w, height: h, depth: 1)
        let grid = MTLSize(width: width, height: height, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
    }

    var pathtracePipeline: MTLComputePipelineState { psPathtrace }
    var normalsPipeline: MTLComputePipelineState { psNormals }
    var bdptPipeline: MTLComputePipelineState { psBDPT }

    /// Encode an accumulate-one-frame + tonemap-resolve pass into a drawable
    /// texture (used by the realtime window app). Caller commits/presents.
    func encodeRealtimeFrame(into cmd: MTLCommandBuffer, target: MTLTexture,
                             samplesPerFrame: Int, bounces: Int, exposure: Float) {
        // ML-pipeline diagnostics: when a feature channel is selected, display the
        // neural amplifier's input instead of the path-traced beauty pass.
        if viewMode != 0 {
            encodeFeatureViz(into: cmd, target: target, bounces: bounces, exposure: exposure)
            return
        }

        uniforms.samplesPerFrame = UInt32(samplesPerFrame)
        uniforms.maxBounces = UInt32(bounces)
        uniforms.frameIndex = accumulatedFrames
        uniforms.flags = (accumulatedFrames == 0) ? 1 : 0

        let pt = cmd.makeComputeCommandEncoder()!
        pt.setComputePipelineState(psPathtrace)
        pt.setBuffer(accumBuf, offset: 0, index: 0)
        pt.setBytes(&uniforms, length: MemoryLayout<Uniforms>.stride, index: 1)
        pt.setBuffer(positionsBuf, offset: 0, index: 2)
        pt.setBuffer(normalsBuf, offset: 0, index: 3)
        pt.setBuffer(indicesBuf, offset: 0, index: 4)
        pt.setBuffer(triMaterialBuf, offset: 0, index: 5)
        pt.setBuffer(materialsBuf, offset: 0, index: 6)
        pt.setBuffer(emissiveBuf, offset: 0, index: 7)
        pt.setAccelerationStructure(accel, bufferIndex: 8)
        dispatch(pt, psPathtrace)
        pt.endEncoding()
        accumulatedFrames += 1

        let rs = cmd.makeComputeCommandEncoder()!
        rs.setComputePipelineState(psResolve)
        rs.setTexture(target, index: 0)
        rs.setBuffer(accumBuf, offset: 0, index: 0)
        var size = SIMD2<UInt32>(UInt32(width), UInt32(height))
        rs.setBytes(&size, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 1)
        var e = exposure
        rs.setBytes(&e, length: MemoryLayout<Float>.stride, index: 2)
        dispatch(rs, psResolve)
        rs.endEncoding()
    }

    /// Encode the ML feature-channel diagnostic into the drawable: (re)capture the
    /// per-pixel `Feature` buffer when the view changed, then map the selected
    /// channel to color via `resolveFeatureViz`. The capture runs the same
    /// `exportFeatures` kernel that produces the training data, so what the viewer
    /// shows is exactly what the neural amplifier consumes.
    private func encodeFeatureViz(into cmd: MTLCommandBuffer, target: MTLTexture,
                                  bounces: Int, exposure: Float) {
        if !featVizValid { captureFeatureViz(bounces: bounces) }

        let rs = cmd.makeComputeCommandEncoder()!
        rs.setComputePipelineState(psFeatureViz)
        rs.setTexture(target, index: 0)
        rs.setBuffer(featVizBuf!, offset: 0, index: 0)
        var size = SIMD2<UInt32>(UInt32(width), UInt32(height))
        rs.setBytes(&size, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 1)
        var mode = UInt32(viewMode)
        rs.setBytes(&mode, length: MemoryLayout<UInt32>.stride, index: 2)
        var e = exposure
        rs.setBytes(&e, length: MemoryLayout<Float>.stride, index: 3)
        var center = sceneNorm.center
        rs.setBytes(&center, length: MemoryLayout<SIMD3<Float>>.stride, index: 4)
        var inv = sceneNorm.invExtent
        rs.setBytes(&inv, length: MemoryLayout<Float>.stride, index: 5)
        var prevCam = effectivePrevCamera()
        rs.setBytes(&prevCam, length: MemoryLayout<Camera>.stride, index: 6)
        var mscale = motionVizScale
        rs.setBytes(&mscale, length: MemoryLayout<Float>.stride, index: 7)
        dispatch(rs, psFeatureViz)
        rs.endEncoding()
        accumulatedFrames += 1
    }

    /// Previous-view camera for motion vectors. The viewer supplies the real one;
    /// headless renders synthesize a small rotation about the scene centre so the
    /// "Motion Vectors" channel shows a coherent flow field that can be validated.
    private func effectivePrevCamera() -> Camera {
        if let p = prevCamera { return p }
        let cur = uniforms.camera
        let center = sceneNorm.center
        let off = cur.position - center
        let a: Float = -0.05                                   // ~2.9° nudge
        let ca = cos(a), sa = sin(a)
        let prevEye = center + SIMD3(ca * off.x + sa * off.z, off.y, -sa * off.x + ca * off.z)
        let fovY = 2 * atan(cur.tanHalfFovY) * 180 / .pi
        return Camera.lookAt(eye: prevEye, target: center, fovYDeg: fovY, aspect: cur.aspect)
    }

    /// Project a world point to pixel coordinates under `cam` (CPU mirror of the
    /// shader's `projectToPixel`), used to build motion vectors head-less.
    private func projectToPixel(_ cam: Camera, world P: SIMD3<Float>) -> SIMD2<Float> {
        let d = P - cam.position
        let fz = simd_dot(d, cam.forward)
        let sx = simd_dot(d, cam.right) / fz
        let sy = simd_dot(d, cam.up) / fz
        let ndcx = (sx / (cam.tanHalfFovY * cam.aspect) + 1) * 0.5
        let ndcy = (1 - sy / cam.tanHalfFovY) * 0.5
        return SIMD2(ndcx * Float(width), ndcy * Float(height))
    }

    /// Run `exportFeatures` once into the diagnostics buffer and summarize the ML
    /// input distribution (coverage + the single-sample signal/material stats).
    private func captureFeatureViz(bounces: Int) {
        if featVizBuf == nil {
            featVizBuf = device.makeBuffer(length: width * height * MemoryLayout<Feature>.stride,
                                           options: .storageModeShared)
            featVizBuf?.label = "featureViz"
        }
        uniforms.samplesPerFrame = 1
        uniforms.maxBounces = UInt32(bounces)
        uniforms.frameIndex = 0
        uniforms.flags = 1                                   // reset → also captures features
        let cmd = queue.makeCommandBuffer()!
        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(psExport)
        enc.setBuffer(accumBuf, offset: 0, index: 0)
        bindScene(enc)
        enc.setBuffer(featVizBuf!, offset: 0, index: 9)
        dispatch(enc, psExport)
        enc.endEncoding()
        cmd.commit(); cmd.waitUntilCompleted()
        featureDiag = summarizeFeatures()
        featVizValid = true
    }

    /// Aggregate the captured features into a compact ML-input readout: hit
    /// coverage (fraction of valid training pixels) plus the mean single-sample
    /// luminance and material parameters the amplifier conditions on.
    private func summarizeFeatures() -> String {
        guard let buf = featVizBuf else { return "" }
        let n = width * height
        let fptr = buf.contents().bindMemory(to: Feature.self, capacity: n)
        var hits = 0
        var sumLum: Float = 0, sumRough: Float = 0, sumMetal: Float = 0
        for i in 0..<n {
            let f = fptr[i]
            guard f.hit > 0.5 else { continue }
            hits += 1
            sumLum += dot(f.oneSample, SIMD3<Float>(0.2126, 0.7152, 0.0722))
            sumRough += f.roughness
            sumMetal += f.metallic
        }
        let cov = n > 0 ? 100 * Float(hits) / Float(n) : 0
        guard hits > 0 else { return String(format: "ML in: hit %.0f%%", cov) }
        let inv = 1 / Float(hits)
        return String(format: "ML in: hit %.0f%%  ·  1-spp Lμ %.3f  ·  rough μ %.2f  ·  metal μ %.2f",
                      cov, sumLum * inv, sumRough * inv, sumMetal * inv)
    }

    /// Headless render: accumulate `frames` frames of `spp` samples each.
    /// Returns wall-clock seconds spent in GPU dispatches.
    @discardableResult
    func renderHeadless(spp: Int, bounces: Int, useNormals: Bool = false, bdpt: Bool = false) -> Double {
        resetAccumulation()
        let kernel = useNormals ? psNormals : (bdpt ? psBDPT : psPathtrace)
        let frames = useNormals ? 1 : spp
        let t0 = Date()
        for _ in 0..<frames {
            renderFrame(samplesPerFrame: 1, bounces: bounces, kernel: kernel)
        }
        return Date().timeIntervalSince(t0)
    }

    /// Resolve the accumulation buffer to tonemapped 8-bit sRGB RGBA.
    func resolveRGBA8(exposure: Float = 1.0) -> [UInt8] {
        let ptr = accumBuf.contents().bindMemory(to: SIMD4<Float>.self, capacity: width * height)
        var out = [UInt8](repeating: 0, count: width * height * 4)
        for i in 0..<(width * height) {
            let a = ptr[i]
            let n = max(a.w, 1)
            var c = SIMD3(a.x, a.y, a.z) / n * exposure
            c = acesFilmic(c)
            out[i * 4 + 0] = UInt8(linearToSRGB(c.x) * 255 + 0.5)
            out[i * 4 + 1] = UInt8(linearToSRGB(c.y) * 255 + 0.5)
            out[i * 4 + 2] = UInt8(linearToSRGB(c.z) * 255 + 0.5)
            out[i * 4 + 3] = 255
        }
        return out
    }

    /// Mean linear radiance per pixel (rgb), for metrics / data export.
    func resolveLinear() -> [SIMD3<Float>] {
        let ptr = accumBuf.contents().bindMemory(to: SIMD4<Float>.self, capacity: width * height)
        return (0..<(width * height)).map { i in
            let a = ptr[i]; let n = max(a.w, 1); return SIMD3(a.x, a.y, a.z) / n
        }
    }

    // MARK: Research data export (P5)

    /// Bind all scene buffers (indices 2..8) on an existing compute encoder.
    private func bindScene(_ enc: MTLComputeCommandEncoder) {
        enc.setBytes(&uniforms, length: MemoryLayout<Uniforms>.stride, index: 1)
        enc.setBuffer(positionsBuf, offset: 0, index: 2)
        enc.setBuffer(normalsBuf, offset: 0, index: 3)
        enc.setBuffer(indicesBuf, offset: 0, index: 4)
        enc.setBuffer(triMaterialBuf, offset: 0, index: 5)
        enc.setBuffer(materialsBuf, offset: 0, index: 6)
        enc.setBuffer(emissiveBuf, offset: 0, index: 7)
        enc.setAccelerationStructure(accel, bufferIndex: 8)
    }

    /// Accumulate a high-spp reference image and return mean linear radiance.
    func renderReferenceLinear(spp: Int, bounces: Int) -> [SIMD3<Float>] {
        resetAccumulation()
        for _ in 0..<spp { renderFrame(samplesPerFrame: 1, bounces: bounces, kernel: psPathtrace) }
        return resolveLinear()
    }

    /// Render ONE independent single-sample (1 spp) image with an explicit RNG
    /// seed, isolated in the accumulation buffer. Successive seeds are decorrelated.
    func renderSingleSampleLinear(seed: UInt32, bounces: Int) -> [SIMD3<Float>] {
        uniforms.samplesPerFrame = 1
        uniforms.maxBounces = UInt32(bounces)
        uniforms.frameIndex = seed
        uniforms.flags = 1                                   // reset → accum holds just this frame
        let cmd = queue.makeCommandBuffer()!
        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(psPathtrace)
        enc.setBuffer(accumBuf, offset: 0, index: 0)
        bindScene(enc)
        dispatch(enc, psPathtrace)
        enc.endEncoding()
        cmd.commit(); cmd.waitUntilCompleted()
        return resolveLinear()
    }

    /// Capture per-pixel first-hit features + a single cast ray's 1-spp radiance.
    func captureFeatures(bounces: Int) -> [Feature] {
        let featBuf = device.makeBuffer(length: width * height * MemoryLayout<Feature>.stride,
                                        options: .storageModeShared)!
        uniforms.samplesPerFrame = 1
        uniforms.maxBounces = UInt32(bounces)
        uniforms.frameIndex = 0
        uniforms.flags = 1
        let cmd = queue.makeCommandBuffer()!
        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(psExport)
        enc.setBuffer(accumBuf, offset: 0, index: 0)
        bindScene(enc)
        enc.setBuffer(featBuf, offset: 0, index: 9)
        dispatch(enc, psExport)
        enc.endEncoding()
        cmd.commit(); cmd.waitUntilCompleted()
        let fptr = featBuf.contents().bindMemory(to: Feature.self, capacity: width * height)
        return (0..<(width * height)).map { fptr[$0] }
    }

    /// Headless ML feature-channel visualization → tonemapped 8-bit sRGB RGBA,
    /// mirroring the `resolveFeatureViz` GPU kernel (just as `resolveRGBA8`
    /// mirrors `resolve`). `mode` indexes `Renderer.featureViewModeNames`; modes
    /// 1..9 each render one channel of the vector the neural amplifier consumes.
    func featureVizRGBA8(mode: Int, bounces: Int, exposure: Float = 1.0) -> [UInt8] {
        let feats = captureFeatures(bounces: bounces)
        let norm = sceneNorm
        let prevCam = effectivePrevCamera()
        let zero = SIMD3<Float>(repeating: 0), one = SIMD3<Float>(repeating: 1)
        var out = [UInt8](repeating: 0, count: width * height * 4)
        for i in 0..<(width * height) {
            let f = feats[i]
            let hit = f.hit > 0.5
            var c = zero
            switch mode {
            case 1:
                let r = acesFilmic(f.oneSample * exposure)
                c = SIMD3(linearToSRGB(r.x), linearToSRGB(r.y), linearToSRGB(r.z))
            case 2: c = hit ? 0.5 * (f.normal + 1) : zero
            case 3: c = hit ? 0.5 * (f.wo + 1) : zero
            case 4: c = hit ? f.baseColor : zero
            case 5: c = hit ? SIMD3(repeating: f.metallic) : zero
            case 6: c = hit ? SIMD3(repeating: f.roughness) : zero
            case 7:
                let p = (f.hitPos - norm.center) * norm.invExtent
                c = hit ? simd_clamp(0.5 * (p + 1), zero, one) : zero
            case 8: c = hit ? 0.5 * (f.firstDir + 1) : zero
            case 9: c = SIMD3(repeating: f.hit)
            case 10:
                if hit {
                    let px = i % width, py = i / width
                    let prev = projectToPixel(prevCam, world: f.hitPos)
                    let mvx = (Float(px) + 0.5 - prev.x) / max(motionVizScale, 1e-4)
                    let mvy = (Float(py) + 0.5 - prev.y) / max(motionVizScale, 1e-4)
                    c = SIMD3(simd_clamp(0.5 + 0.5 * mvx, 0, 1),
                              simd_clamp(0.5 - 0.5 * mvy, 0, 1), 0.5)
                }
            default: break
            }
            c = simd_clamp(c, zero, one)
            out[i * 4 + 0] = UInt8(c.x * 255 + 0.5)
            out[i * 4 + 1] = UInt8(c.y * 255 + 0.5)
            out[i * 4 + 2] = UInt8(c.z * 255 + 0.5)
            out[i * 4 + 3] = 255
        }
        return out
    }

    /// Run the export kernel: accumulate a high-spp converged target while also
    /// capturing per-pixel first-hit features + a single-sample (1 spp) estimate.
    /// Returns (features, convergedTargetLinear).
    func exportTrainingData(targetSpp: Int, bounces: Int) -> (feats: [Feature], target: [SIMD3<Float>]) {
        resetAccumulation()
        let featBuf = device.makeBuffer(length: width * height * MemoryLayout<Feature>.stride,
                                        options: .storageModeShared)!
        for f in 0..<targetSpp {
            uniforms.samplesPerFrame = 1
            uniforms.maxBounces = UInt32(bounces)
            uniforms.frameIndex = UInt32(f)
            uniforms.flags = (f == 0) ? 1 : 0
            let cmd = queue.makeCommandBuffer()!
            let enc = cmd.makeComputeCommandEncoder()!
            enc.setComputePipelineState(psExport)
            enc.setBuffer(accumBuf, offset: 0, index: 0)
            enc.setBytes(&uniforms, length: MemoryLayout<Uniforms>.stride, index: 1)
            enc.setBuffer(positionsBuf, offset: 0, index: 2)
            enc.setBuffer(normalsBuf, offset: 0, index: 3)
            enc.setBuffer(indicesBuf, offset: 0, index: 4)
            enc.setBuffer(triMaterialBuf, offset: 0, index: 5)
            enc.setBuffer(materialsBuf, offset: 0, index: 6)
            enc.setBuffer(emissiveBuf, offset: 0, index: 7)
            enc.setAccelerationStructure(accel, bufferIndex: 8)
            enc.setBuffer(featBuf, offset: 0, index: 9)
            dispatch(enc, psExport)
            enc.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
        }
        accumulatedFrames = UInt32(targetSpp)
        let fptr = featBuf.contents().bindMemory(to: Feature.self, capacity: width * height)
        let feats = (0..<(width * height)).map { fptr[$0] }
        return (feats, resolveLinear())
    }
}
