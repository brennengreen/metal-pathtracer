import Metal
import simd
import Foundation

// MARK: - Host mirrors of the instanced MSL structs.

struct IslandUniforms {
    var camera: Camera = Camera()
    var imageSize: SIMD2<UInt32> = .zero
    var frameIndex: UInt32 = 0
    var samplesPerFrame: UInt32 = 1
    var maxBounces: UInt32 = 6
    var flags: UInt32 = 0
    var sunDir: SIMD3<Float> = simd_normalize(SIMD3(0.4, 0.7, 0.3))
    var sunColor: SIMD3<Float> = SIMD3(repeating: 5)
    var skyZenith: SIMD3<Float> = SIMD3(0.20, 0.36, 0.62)
    var skyHorizon: SIMD3<Float> = SIMD3(0.74, 0.83, 0.92)
    var radianceClamp: Float = 0
    var exposure: Float = 1
}

struct InstanceData {
    var nrm0: SIMD4<Float>
    var nrm1: SIMD4<Float>
    var nrm2: SIMD4<Float>
    var indexOffset: UInt32
    var materialId: UInt32
    var _pad0: UInt32 = 0
    var _pad1: UInt32 = 0
}

// MARK: - CPU scene description.

/// A unique object-space triangle mesh (replicated via instances).
struct Mesh {
    var positions: [SIMD3<Float>]
    var normals: [SIMD3<Float>]
    var indices: [UInt32]          // 0-based, local to this mesh
    var uvs: [SIMD2<Float>] = []   // optional; empty => zeros padded at upload
    var triangleCount: Int { indices.count / 3 }
}

struct InstanceDef {
    var meshIndex: Int
    var transform: simd_float4x4
    var materialId: UInt32
}

struct InstancedScene {
    var meshes: [Mesh] = []
    var instances: [InstanceDef] = []
    var materials: [Material] = []
    var texturePaths: [String] = []          // albedo textures; Material.texId indexes this
    var camera = Camera()
    var sunDir = simd_normalize(SIMD3<Float>(0.4, 0.75, 0.35))
    var sunColor = SIMD3<Float>(repeating: 6)
    var skyZenith = SIMD3<Float>(0.20, 0.36, 0.62)
    var skyHorizon = SIMD3<Float>(0.80, 0.86, 0.95)

    var instanceCount: Int { instances.count }
    var effectiveTriangles: Int { instances.reduce(0) { $0 + meshes[$1.meshIndex].triangleCount } }

    /// Approximate world-space bounds (instance transforms × per-mesh AABB
    /// corners). Capped for huge scenes — enough to centre an orbit camera.
    func worldBounds(cap: Int = 200_000) -> (lo: SIMD3<Float>, hi: SIMD3<Float>) {
        var lo = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        var hi = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)
        let mb = meshes.map { $0.bounds() }
        for inst in instances.prefix(cap) {
            let (blo, bhi) = mb[inst.meshIndex]
            for cx in [blo.x, bhi.x] { for cy in [blo.y, bhi.y] { for cz in [blo.z, bhi.z] {
                let p = inst.transform * SIMD4<Float>(cx, cy, cz, 1)
                let w = SIMD3<Float>(p.x, p.y, p.z)
                lo = simd_min(lo, w); hi = simd_max(hi, w)
            }}}
        }
        if lo.x > hi.x { lo = SIMD3(repeating: -1); hi = SIMD3(repeating: 1) }
        return (lo, hi)
    }
}

// MARK: - Mesh primitives.

extension Mesh {
    static func cube(size: Float = 1) -> Mesh {
        let h = size * 0.5
        var p: [SIMD3<Float>] = [], n: [SIMD3<Float>] = [], idx: [UInt32] = []
        let faces: [(SIMD3<Float>, SIMD3<Float>, SIMD3<Float>)] = [
            (SIMD3(0,0,1), SIMD3(1,0,0), SIMD3(0,1,0)), (SIMD3(0,0,-1), SIMD3(-1,0,0), SIMD3(0,1,0)),
            (SIMD3(1,0,0), SIMD3(0,0,-1), SIMD3(0,1,0)), (SIMD3(-1,0,0), SIMD3(0,0,1), SIMD3(0,1,0)),
            (SIMD3(0,1,0), SIMD3(1,0,0), SIMD3(0,0,-1)), (SIMD3(0,-1,0), SIMD3(1,0,0), SIMD3(0,0,1)),
        ]
        for (nrm, u, v) in faces {
            let base = UInt32(p.count)
            let c = nrm * h
            for (du, dv) in [(-h,-h),(h,-h),(h,h),(-h,h)] {
                p.append(c + u*du + v*dv); n.append(nrm)
            }
            idx.append(contentsOf: [base, base+1, base+2, base, base+2, base+3])
        }
        return Mesh(positions: p, normals: n, indices: idx)
    }

    static func fromOBJ(_ o: OBJMesh) -> Mesh {
        Mesh(positions: o.positions, normals: o.normals, indices: o.indices)
    }

    func bounds() -> (lo: SIMD3<Float>, hi: SIMD3<Float>) {
        var lo = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        var hi = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)
        for p in positions { lo = simd_min(lo, p); hi = simd_max(hi, p) }
        return (lo, hi)
    }
}

// MARK: - Transform helpers.

@inline(__always) func translation(_ t: SIMD3<Float>) -> simd_float4x4 {
    var m = matrix_identity_float4x4; m.columns.3 = SIMD4(t, 1); return m
}
@inline(__always) func scaling(_ s: SIMD3<Float>) -> simd_float4x4 {
    simd_float4x4(diagonal: SIMD4(s, 1))
}
@inline(__always) func rotationY(_ a: Float) -> simd_float4x4 {
    let c = cos(a), s = sin(a)
    return simd_float4x4(SIMD4(c,0,-s,0), SIMD4(0,1,0,0), SIMD4(s,0,c,0), SIMD4(0,0,0,1))
}

private func packedTransform(_ m: simd_float4x4) -> MTLPackedFloat4x3 {
    func c(_ v: SIMD4<Float>) -> MTLPackedFloat3 { MTLPackedFloat3Make(v.x, v.y, v.z) }
    var r = MTLPackedFloat4x3()
    r.columns = (c(m.columns.0), c(m.columns.1), c(m.columns.2), c(m.columns.3))
    return r
}

// MARK: - Instanced renderer (BLAS per mesh + top-level instance accel structure).

final class InstancedRenderer {
    let device: MTLDevice
    let queue: MTLCommandQueue
    let width: Int, height: Int
    let scene: InstancedScene

    private let positionsBuf, normalsBuf, uvsBuf, indicesBuf, instanceDataBuf, materialsBuf: MTLBuffer
    private let blasList: [MTLAccelerationStructure]
    private let tlas: MTLAccelerationStructure
    private let psPathtrace, psNormals, psResolve: MTLComputePipelineState
    private let accumBuf: MTLBuffer
    private let albedoTextures: [MTLTexture]      // exactly 32 (padded with a white dummy)
    var uniforms = IslandUniforms()
    private var frames: UInt32 = 0

    // Interactive viewer state (also honoured by headless `--view`).
    var viewMode: Int = 0          // 0 Beauty · 1 Albedo · 2 Normals · 3 GeoNormals · 4 UV · 5 MatID · 6 AO · 7 Direct
    var texturesEnabled: Bool = true
    var accumulatedFrames: UInt32 { frames }
    func resetAccumulation() { frames = 0 }

    private func frameFlags() -> UInt32 {
        var f: UInt32 = (frames == 0) ? 1 : 0     // bit0 = reset accumulation
        if !texturesEnabled { f |= (1 << 3) }     // bit3 = textures off
        f |= UInt32(viewMode & 0xF) << 4          // bits4-7 = view mode
        return f
    }

    init(device: MTLDevice, scene: InstancedScene, width: Int, height: Int) throws {
        precondition(MemoryLayout<IslandUniforms>.stride == 192, "IslandUniforms stride \(MemoryLayout<IslandUniforms>.stride) != 192")
        precondition(MemoryLayout<InstanceData>.stride == 64, "InstanceData stride mismatch")
        self.device = device; self.scene = scene; self.width = width; self.height = height
        guard let q = device.makeCommandQueue() else { throw Err.msg("queue") }
        queue = q

        // Concatenate meshes into global vertex/normal/uv/index buffers (global indices).
        var positions: [SIMD3<Float>] = [], normals: [SIMD3<Float>] = [], indices: [UInt32] = []
        var uvs: [SIMD2<Float>] = []
        var meshIndexStart = [Int](), meshTriCount = [Int]()
        for m in scene.meshes {
            let vbase = UInt32(positions.count)
            meshIndexStart.append(indices.count)
            meshTriCount.append(m.triangleCount)
            positions.append(contentsOf: m.positions)
            normals.append(contentsOf: m.normals)
            if m.uvs.count == m.positions.count { uvs.append(contentsOf: m.uvs) }
            else { uvs.append(contentsOf: repeatElement(SIMD2<Float>(0, 0), count: m.positions.count)) }
            indices.append(contentsOf: m.indices.map { $0 + vbase })
        }

        func buf<T>(_ a: [T], _ label: String) throws -> MTLBuffer {
            let len = max(MemoryLayout<T>.stride * a.count, 16)
            guard let b = device.makeBuffer(length: len, options: .storageModeShared) else { throw Err.msg(label) }
            if !a.isEmpty { a.withUnsafeBytes { b.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) } }
            b.label = label; return b
        }
        positionsBuf = try buf(positions, "positions")
        normalsBuf = try buf(normals, "normals")
        uvsBuf = try buf(uvs, "uvs")
        indicesBuf = try buf(indices, "indices")
        materialsBuf = try buf(scene.materials, "materials")

        // Per-instance shading data.
        var instData: [InstanceData] = []
        instData.reserveCapacity(scene.instances.count)
        for inst in scene.instances {
            let m = inst.transform
            let m3 = simd_float3x3(SIMD3(m.columns.0.x, m.columns.0.y, m.columns.0.z),
                                   SIMD3(m.columns.1.x, m.columns.1.y, m.columns.1.z),
                                   SIMD3(m.columns.2.x, m.columns.2.y, m.columns.2.z))
            let nm = simd_transpose(simd_inverse(m3))
            instData.append(InstanceData(
                nrm0: SIMD4(nm.columns.0, 0), nrm1: SIMD4(nm.columns.1, 0), nrm2: SIMD4(nm.columns.2, 0),
                indexOffset: UInt32(meshIndexStart[inst.meshIndex]), materialId: inst.materialId))
        }
        instanceDataBuf = try buf(instData, "instanceData")

        // Build one BLAS per unique mesh (shared vertex buffer, global indices).
        var blases: [MTLAccelerationStructure] = []
        var built: [(MTLAccelerationStructure, MTLPrimitiveAccelerationStructureDescriptor)] = []
        for mi in scene.meshes.indices {
            let g = MTLAccelerationStructureTriangleGeometryDescriptor()
            g.vertexBuffer = positionsBuf
            g.vertexStride = MemoryLayout<SIMD3<Float>>.stride
            g.vertexFormat = .float3
            g.triangleCount = meshTriCount[mi]
            g.indexBuffer = indicesBuf
            g.indexBufferOffset = meshIndexStart[mi] * MemoryLayout<UInt32>.stride
            g.indexType = .uint32
            let d = MTLPrimitiveAccelerationStructureDescriptor()
            d.geometryDescriptors = [g]
            let sizes = device.accelerationStructureSizes(descriptor: d)
            guard let a = device.makeAccelerationStructure(size: sizes.accelerationStructureSize) else { throw Err.msg("blas") }
            a.label = "blas\(mi)"
            blases.append(a); built.append((a, d))
        }
        // Instance descriptors for the TLAS.
        let instDescBuf = device.makeBuffer(
            length: max(MemoryLayout<MTLAccelerationStructureInstanceDescriptor>.stride * scene.instances.count, 64),
            options: .storageModeShared)!
        let dp = instDescBuf.contents().bindMemory(to: MTLAccelerationStructureInstanceDescriptor.self,
                                                   capacity: max(scene.instances.count, 1))
        for (i, inst) in scene.instances.enumerated() {
            var d = MTLAccelerationStructureInstanceDescriptor()
            d.transformationMatrix = packedTransform(inst.transform)
            d.options = .opaque
            d.mask = 0xFF
            d.intersectionFunctionTableOffset = 0
            d.accelerationStructureIndex = UInt32(inst.meshIndex)
            dp[i] = d
        }
        let tdesc = MTLInstanceAccelerationStructureDescriptor()
        tdesc.instanceCount = scene.instances.count
        tdesc.instanceDescriptorBuffer = instDescBuf
        tdesc.instancedAccelerationStructures = blases

        // Build all acceleration structures (BLAS then TLAS).
        let tsizes = device.accelerationStructureSizes(descriptor: tdesc)
        guard let tl = device.makeAccelerationStructure(size: tsizes.accelerationStructureSize) else { throw Err.msg("tlas") }
        tl.label = "tlas"
        let scratchLen = max(tsizes.buildScratchBufferSize,
                             built.map { device.accelerationStructureSizes(descriptor: $0.1).buildScratchBufferSize }.max() ?? 32)
        let scratch = device.makeBuffer(length: max(scratchLen, 32), options: .storageModePrivate)!
        let cb = q.makeCommandBuffer()!
        let enc = cb.makeAccelerationStructureCommandEncoder()!
        for (a, d) in built { enc.build(accelerationStructure: a, descriptor: d, scratchBuffer: scratch, scratchBufferOffset: 0) }
        enc.build(accelerationStructure: tl, descriptor: tdesc, scratchBuffer: scratch, scratchBufferOffset: 0)
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        blasList = blases; tlas = tl

        // Compile kernels.
        guard let url = Bundle.module.url(forResource: "pathtrace", withExtension: "metal") else { throw Err.msg("shader") }
        let src = try String(contentsOf: url, encoding: .utf8)
        let opts = MTLCompileOptions(); opts.languageVersion = .version3_0
        let lib = try device.makeLibrary(source: src, options: opts)
        func pso(_ n: String) throws -> MTLComputePipelineState {
            guard let f = lib.makeFunction(name: n) else { throw Err.msg("kernel \(n)") }
            return try device.makeComputePipelineState(function: f)
        }
        psPathtrace = try pso("instPathtrace")
        psNormals = try pso("instNormals")
        psResolve = try pso("resolve")

        accumBuf = device.makeBuffer(length: width * height * MemoryLayout<SIMD4<Float>>.stride, options: .storageModeShared)!

        // Load albedo textures (sRGB + mips); pad the fixed 32-slot array with a white dummy.
        let dummy = TextureLoader.white(device: device)
        var texs: [MTLTexture] = []
        for (i, p) in scene.texturePaths.prefix(32).enumerated() {
            if let t = TextureLoader.load(device: device, path: p, queue: q) { texs.append(t) }
            else { FileHandle.standardError.write("  texture \(i) failed: \(p)\n".data(using: .utf8)!); texs.append(dummy) }
        }
        while texs.count < 32 { texs.append(dummy) }
        albedoTextures = texs

        uniforms.camera = scene.camera
        uniforms.imageSize = SIMD2(UInt32(width), UInt32(height))
        uniforms.sunDir = scene.sunDir
        uniforms.sunColor = scene.sunColor
        uniforms.skyZenith = scene.skyZenith
        uniforms.skyHorizon = scene.skyHorizon
    }

    enum Err: Error { case msg(String) }

    private func dispatch(_ pso: MTLComputePipelineState, normals: Bool) {
        uniforms.frameIndex = frames
        uniforms.flags = frameFlags()
        let cb = queue.makeCommandBuffer()!
        let e = cb.makeComputeCommandEncoder()!
        e.setComputePipelineState(pso)
        e.setBuffer(accumBuf, offset: 0, index: 0)
        e.setBytes(&uniforms, length: MemoryLayout<IslandUniforms>.stride, index: 1)
        e.setBuffer(positionsBuf, offset: 0, index: 2)
        e.setBuffer(normalsBuf, offset: 0, index: 3)
        e.setBuffer(indicesBuf, offset: 0, index: 4)
        e.setBuffer(instanceDataBuf, offset: 0, index: 5)
        e.setBuffer(materialsBuf, offset: 0, index: 6)
        e.setAccelerationStructure(tlas, bufferIndex: 7)
        e.setBuffer(uvsBuf, offset: 0, index: 8)
        e.setTextures(albedoTextures, range: 0..<32)
        e.useResources(blasList, usage: .read)
        let w = pso.threadExecutionWidth, h = max(1, pso.maxTotalThreadsPerThreadgroup / w)
        e.dispatchThreads(MTLSize(width: width, height: height, depth: 1),
                          threadsPerThreadgroup: MTLSize(width: w, height: h, depth: 1))
        e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        frames += 1
    }

    @discardableResult
    func render(spp: Int, bounces: Int, useNormals: Bool = false) -> Double {
        frames = 0
        uniforms.maxBounces = UInt32(bounces)
        uniforms.samplesPerFrame = 1
        let n = useNormals ? 1 : spp
        let t0 = Date()
        for _ in 0..<n { dispatch(useNormals ? psNormals : psPathtrace, normals: useNormals) }
        return Date().timeIntervalSince(t0)
    }

    /// Accumulate one progressive frame and tonemap-resolve it into a drawable
    /// texture (used by the realtime window app). Caller commits/presents.
    func encodeRealtimeFrame(into cmd: MTLCommandBuffer, target: MTLTexture,
                             samplesPerFrame: Int, bounces: Int, exposure: Float) {
        uniforms.maxBounces = UInt32(bounces)
        uniforms.samplesPerFrame = UInt32(samplesPerFrame)
        uniforms.frameIndex = frames
        uniforms.flags = frameFlags()

        let pt = cmd.makeComputeCommandEncoder()!
        pt.setComputePipelineState(psPathtrace)
        pt.setBuffer(accumBuf, offset: 0, index: 0)
        pt.setBytes(&uniforms, length: MemoryLayout<IslandUniforms>.stride, index: 1)
        pt.setBuffer(positionsBuf, offset: 0, index: 2)
        pt.setBuffer(normalsBuf, offset: 0, index: 3)
        pt.setBuffer(indicesBuf, offset: 0, index: 4)
        pt.setBuffer(instanceDataBuf, offset: 0, index: 5)
        pt.setBuffer(materialsBuf, offset: 0, index: 6)
        pt.setAccelerationStructure(tlas, bufferIndex: 7)
        pt.setBuffer(uvsBuf, offset: 0, index: 8)
        pt.setTextures(albedoTextures, range: 0..<32)
        pt.useResources(blasList, usage: .read)
        let w = psPathtrace.threadExecutionWidth, h = max(1, psPathtrace.maxTotalThreadsPerThreadgroup / w)
        pt.dispatchThreads(MTLSize(width: width, height: height, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: w, height: h, depth: 1))
        pt.endEncoding()
        frames += 1

        let rs = cmd.makeComputeCommandEncoder()!
        rs.setComputePipelineState(psResolve)
        rs.setTexture(target, index: 0)
        rs.setBuffer(accumBuf, offset: 0, index: 0)
        var size = SIMD2<UInt32>(UInt32(width), UInt32(height))
        rs.setBytes(&size, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 1)
        var e = exposure
        rs.setBytes(&e, length: MemoryLayout<Float>.stride, index: 2)
        let rw = psResolve.threadExecutionWidth, rh = max(1, psResolve.maxTotalThreadsPerThreadgroup / rw)
        rs.dispatchThreads(MTLSize(width: width, height: height, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: rw, height: rh, depth: 1))
        rs.endEncoding()
    }

    func resolveRGBA8(exposure: Float = 1) -> [UInt8] {
        let ptr = accumBuf.contents().bindMemory(to: SIMD4<Float>.self, capacity: width * height)
        var out = [UInt8](repeating: 0, count: width * height * 4)
        for i in 0..<(width * height) {
            let a = ptr[i]; let n = max(a.w, 1)
            var c = SIMD3(a.x, a.y, a.z) / n * exposure
            c = acesFilmic(c)
            out[i*4+0] = UInt8(linearToSRGB(c.x) * 255 + 0.5)
            out[i*4+1] = UInt8(linearToSRGB(c.y) * 255 + 0.5)
            out[i*4+2] = UInt8(linearToSRGB(c.z) * 255 + 0.5)
            out[i*4+3] = 255
        }
        return out
    }
}

// MARK: - Instancing validation scene.

extension InstancedScene {
    /// A grid of instanced cubes on a ground slab — validates BLAS/TLAS transforms.
    static func test(aspect: Float) -> InstancedScene {
        var s = InstancedScene()
        s.meshes = [Mesh.cube(size: 1)]                       // single unique mesh, many instances
        s.materials = [
            Material(baseColor: SIMD3(0.55, 0.55, 0.60), roughness: 0.35),
            Material(baseColor: SIMD3(0.85, 0.28, 0.20), metallic: 0, roughness: 0.30),
            Material(baseColor: SIMD3(0.90, 0.74, 0.32), metallic: 1, roughness: 0.18),
        ]
        s.instances.append(InstanceDef(meshIndex: 0,
            transform: translation(SIMD3(0, -0.5, 0)) * scaling(SIMD3(40, 1, 40)), materialId: 0))
        var rng = SplitMix64(seed: 7)
        let n = 7
        for i in 0..<n { for j in 0..<n {
            let x = (Float(i) - Float(n-1)/2) * 2.4, z = (Float(j) - Float(n-1)/2) * 2.4
            let hgt = Float.random(in: 0.5...2.2, using: &rng)
            let rot = Float.random(in: 0...(2 * .pi), using: &rng)
            let t = translation(SIMD3(x, hgt*0.5, z)) * rotationY(rot) * scaling(SIMD3(0.85, hgt, 0.85))
            s.instances.append(InstanceDef(meshIndex: 0, transform: t, materialId: UInt32(1 + ((i + j) % 2))))
        }}
        s.camera = Camera.lookAt(eye: SIMD3(12, 9, -16), target: SIMD3(0, 1.2, 0), fovYDeg: 40, aspect: aspect)
        s.sunDir = simd_normalize(SIMD3(0.5, 0.85, -0.25))
        s.sunColor = SIMD3(repeating: 5.5)
        return s
    }
}
