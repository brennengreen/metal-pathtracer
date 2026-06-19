import simd
import Foundation

/// Procedural "Moana-scale" tropical island built from a handful of unique meshes
/// replicated into a very large number of instances (palms, ferns, rocks) scattered
/// over a terrain heightfield, plus an ocean plane — demonstrating the same
/// massive-instancing technique the real Disney Moana Island Scene relies on.
///
/// Key structural choices that make it read as an island rather than a thicket:
///  * the terrain is split into three height bands (beach sand / jungle grass /
///    volcanic rock), each its own instance+material, so the shoreline is golden,
///    the slopes are green and the peak is dark rock;
///  * palms are TWO meshes (brown trunk + green fronds) instanced together, so the
///    canopy is actually green instead of a forest of brown sticks;
///  * vegetation is confined to the vegetated band, leaving a clean beach ring and
///    open ocean for a classic establishing composition.
enum IslandScene {
    static func build(aspect: Float, density: Int = 1, seed: UInt64 = 1) -> InstancedScene {
        var s = InstancedScene()
        var rng = SplitMix64(seed: seed)

        // --- materials -----------------------------------------------------
        let mSand  = UInt32(s.materials.count); s.materials.append(Material(baseColor: SIMD3(0.86, 0.76, 0.55), roughness: 0.75))
        let mGrass = UInt32(s.materials.count); s.materials.append(Material(baseColor: SIMD3(0.20, 0.40, 0.13), roughness: 0.8))
        let mRock  = UInt32(s.materials.count); s.materials.append(Material(baseColor: SIMD3(0.12, 0.11, 0.105), roughness: 0.9))
        let mOcean = UInt32(s.materials.count); s.materials.append(Material(baseColor: SIMD3(0.015, 0.09, 0.13), metallic: 0, roughness: 0.035))
        let mTrunk = UInt32(s.materials.count); s.materials.append(Material(baseColor: SIMD3(0.32, 0.21, 0.12), roughness: 0.85))
        let mLeaf  = UInt32(s.materials.count); s.materials.append(Material(baseColor: SIMD3(0.10, 0.34, 0.09), roughness: 0.6))
        let mFern  = UInt32(s.materials.count); s.materials.append(Material(baseColor: SIMD3(0.13, 0.40, 0.12), roughness: 0.6))

        // --- terrain (one heightfield, three banded meshes) ----------------
        let size: Float = 320, segs = 240, hSeed = 99
        let h = terrainSampler(heightSeed: hSeed)
        let bands = makeTerrainBands(size: size, segments: segs, h: h)
        let mTerrSand  = s.meshes.count; s.meshes.append(bands.sand)
        let mTerrGrass = s.meshes.count; s.meshes.append(bands.grass)
        let mTerrRock  = s.meshes.count; s.meshes.append(bands.rock)
        s.instances.append(InstanceDef(meshIndex: mTerrSand,  transform: matrix_identity_float4x4, materialId: mSand))
        s.instances.append(InstanceDef(meshIndex: mTerrGrass, transform: matrix_identity_float4x4, materialId: mGrass))
        s.instances.append(InstanceDef(meshIndex: mTerrRock,  transform: matrix_identity_float4x4, materialId: mRock))

        // --- ocean ---------------------------------------------------------
        let ocean = Mesh.cube(size: 1)
        let mOceanIdx = s.meshes.count; s.meshes.append(ocean)
        s.instances.append(InstanceDef(meshIndex: mOceanIdx,
            transform: translation(SIMD3(0, -0.35, 0)) * scaling(SIMD3(6000, 0.4, 6000)), materialId: mOcean))

        // --- unique vegetation meshes (a few variants each) ----------------
        var palmTrunkMeshes: [Int] = [], palmFrondMeshes: [Int] = []
        for _ in 0..<4 {
            let p = makePalm(rng: &rng)
            palmTrunkMeshes.append(s.meshes.count); s.meshes.append(p.trunk)
            palmFrondMeshes.append(s.meshes.count); s.meshes.append(p.fronds)
        }
        var fernMeshes: [Int] = []
        for sd in [3, 9] { fernMeshes.append(s.meshes.count); s.meshes.append(makeFern(seed: sd)) }
        var rockMeshes: [Int] = []
        for sd in [12, 71] { rockMeshes.append(s.meshes.count); s.meshes.append(makeRock(seed: sd)) }

        // --- scattered vegetation/rocks (the massive instancing) -----------
        let d = Float(max(density, 1))
        func place(count: Int, minH: Float, maxH: Float, maxSlope: Float,
                   body: (_ x: Float, _ z: Float, _ y: Float, _ scl: Float, _ rot: Float, _ tilt: Float) -> Void) {
            var placed = 0, tries = 0
            let cap = count * 12
            while placed < count && tries < cap {
                tries += 1
                let x = Float.random(in: -150...150, using: &rng)
                let z = Float.random(in: -150...150, using: &rng)
                let y = h(x, z)
                if y < minH || y > maxH { continue }
                let dx = h(x + 1.5, z) - h(x - 1.5, z), dz = h(x, z + 1.5) - h(x, z - 1.5)
                if sqrt(dx*dx + dz*dz) / 3.0 > maxSlope { continue }
                let scl = Float.random(in: 0.7...1.0, using: &rng)
                let rot = Float.random(in: 0...(2 * .pi), using: &rng)
                let tilt = Float.random(in: -0.06...0.06, using: &rng)
                body(x, z, y, scl, rot, tilt); placed += 1
            }
        }

        func xform(_ x: Float, _ z: Float, _ y: Float, _ scl: Float, _ rot: Float, _ tilt: Float) -> simd_float4x4 {
            var t = translation(SIMD3(x, y, z)) * rotationY(rot)
            t = t * simd_float4x4(SIMD4(cos(tilt), sin(tilt), 0, 0), SIMD4(-sin(tilt), cos(tilt), 0, 0), SIMD4(0, 0, 1, 0), SIMD4(0, 0, 0, 1))
            t = t * scaling(SIMD3(repeating: scl))
            return t
        }

        // Palms: vegetated slopes, sparse enough to keep the canopy legible.
        place(count: Int(2600 * d), minH: 2.0, maxH: 22, maxSlope: 1.0) { x, z, y, scl, rot, tilt in
            let v = Int(rng.next() % UInt64(palmTrunkMeshes.count))
            let t = xform(x, z, y, scl * 1.25, rot, tilt)
            s.instances.append(InstanceDef(meshIndex: palmTrunkMeshes[v], transform: t, materialId: mTrunk))
            s.instances.append(InstanceDef(meshIndex: palmFrondMeshes[v], transform: t, materialId: mLeaf))
        }
        // Ferns / undergrowth: denser green fill across the jungle floor.
        place(count: Int(9000 * d), minH: 1.6, maxH: 24, maxSlope: 1.4) { x, z, y, scl, rot, tilt in
            let v = Int(rng.next() % UInt64(fernMeshes.count))
            s.instances.append(InstanceDef(meshIndex: fernMeshes[v],
                transform: xform(x, z, y, scl, rot, tilt), materialId: (rng.next() & 1) == 0 ? mFern : mLeaf))
        }
        // Rocks: along the shoreline and up near the rocky peak.
        place(count: Int(3500 * d), minH: -0.3, maxH: 2.4, maxSlope: 2.0) { x, z, y, scl, rot, tilt in
            let v = Int(rng.next() % UInt64(rockMeshes.count))
            s.instances.append(InstanceDef(meshIndex: rockMeshes[v],
                transform: xform(x, z, y, scl * 1.6, rot, tilt), materialId: mRock))
        }
        place(count: Int(2500 * d), minH: 22, maxH: 40, maxSlope: 3.0) { x, z, y, scl, rot, tilt in
            let v = Int(rng.next() % UInt64(rockMeshes.count))
            s.instances.append(InstanceDef(meshIndex: rockMeshes[v],
                transform: xform(x, z, y, scl * 2.2, rot, tilt), materialId: mRock))
        }

        // --- camera / sun / sky (establishing 3/4 aerial) ------------------
        s.camera = Camera.lookAt(eye: SIMD3(168, 92, 196), target: SIMD3(-6, 24, -10),
                                 fovYDeg: 36, aspect: aspect)
        s.sunDir = simd_normalize(SIMD3(-0.32, 0.46, 0.50))   // afternoon, glints off the water toward camera
        s.sunColor = SIMD3(11.0, 9.6, 7.6)
        s.skyZenith = SIMD3(0.13, 0.30, 0.60)
        s.skyHorizon = SIMD3(0.78, 0.86, 0.93)
        return s
    }

    // MARK: terrain

    private static func valueNoise(_ x: Float, _ z: Float, _ seed: Int) -> Float {
        func hsh(_ i: Int, _ j: Int) -> Float {
            var n = UInt64(bitPattern: Int64(i &* 374761393 &+ j &* 668265263 &+ seed &* 362437))
            n = (n ^ (n >> 13)) &* 1274126177; n = n ^ (n >> 16)
            return Float(n & 0xffff) / 65535.0
        }
        let xi = Int(floor(x)), zi = Int(floor(z))
        let fx = x - Float(xi), fz = z - Float(zi)
        let u = fx*fx*(3-2*fx), v = fz*fz*(3-2*fz)
        let a = hsh(xi,zi), b = hsh(xi+1,zi), c = hsh(xi,zi+1), dd = hsh(xi+1,zi+1)
        return (a*(1-u)+b*u)*(1-v) + (c*(1-u)+dd*u)*v
    }

    /// Island terrain height at world (x,z): radial dome falloff + fractal noise +
    /// an offset volcanic peak.
    private static func terrainSampler(heightSeed: Int) -> (Float, Float) -> Float {
        return { (x: Float, z: Float) -> Float in
            let r: Float = sqrt(x*x + z*z)
            let island: Float = max(0, 1 - powf(r / 125, 2.3))     // broad dome
            var n: Float = 0
            var amp: Float = 1
            var freq: Float = 0.011
            for o in 0..<5 {
                n += amp * valueNoise(x*freq, z*freq, heightSeed + o*17)
                amp *= 0.5
                freq *= 2.13
            }
            let beach: Float = smoothstepF(0.0, 0.16, island)
            let base: Float = island * 26 * (0.30 + 0.70*n) * beach
            let pdx: Float = x + 18, pdz: Float = z + 12           // offset volcanic cone
            let pr2: Float = pdx*pdx + pdz*pdz
            let peak: Float = 27 * expf(-pr2 / (2*40*40)) * smoothstepF(0.0, 0.25, island)
            return base + peak - 0.45
        }
    }

    private static func smoothstepF(_ a: Float, _ b: Float, _ x: Float) -> Float {
        let t = max(0, min(1, (x - a) / (b - a))); return t*t*(3-2*t)
    }

    /// Build the terrain grid once, then emit three meshes selecting quads by the
    /// height band of their centre (beach / jungle / rock).
    private static func makeTerrainBands(size: Float, segments n: Int, h: (Float, Float) -> Float)
        -> (sand: Mesh, grass: Mesh, rock: Mesh) {
        var pos = [SIMD3<Float>](); pos.reserveCapacity((n+1)*(n+1))
        for i in 0...n { for j in 0...n {
            let x = (Float(i)/Float(n) - 0.5) * size
            let z = (Float(j)/Float(n) - 0.5) * size
            pos.append(SIMD3(x, h(x, z), z))
        }}
        var nrm = [SIMD3<Float>](repeating: .zero, count: pos.count)
        let stride = n + 1
        func vid(_ i: Int, _ j: Int) -> Int { i*stride + j }
        for i in 0..<n { for j in 0..<n {
            let a = vid(i,j), b = vid(i+1,j), c = vid(i,j+1), dd = vid(i+1,j+1)
            for (p,q,r) in [(a,b,dd),(a,dd,c)] {
                let fn = simd_cross(pos[q]-pos[p], pos[r]-pos[p])
                nrm[p] += fn; nrm[q] += fn; nrm[r] += fn
            }
        }}
        for i in nrm.indices { let l = simd_length(nrm[i]); nrm[i] = l > 1e-9 ? nrm[i]/l : SIMD3(0,1,0) }

        var sIdx = [UInt32](), gIdx = [UInt32](), rIdx = [UInt32]()
        for i in 0..<n { for j in 0..<n {
            let a = UInt32(vid(i,j)), b = UInt32(vid(i+1,j)), c = UInt32(vid(i,j+1)), dd = UInt32(vid(i+1,j+1))
            let hc = (pos[Int(a)].y + pos[Int(b)].y + pos[Int(c)].y + pos[Int(dd)].y) * 0.25
            let slope = max(abs(pos[Int(b)].y - pos[Int(a)].y), abs(pos[Int(c)].y - pos[Int(a)].y))
            let band: Int
            if hc < 1.7 { band = 0 }
            else if hc > 23 || slope > 3.8 { band = 2 }
            else { band = 1 }
            let tri: [UInt32] = [a, b, dd, a, dd, c]
            switch band { case 0: sIdx += tri; case 2: rIdx += tri; default: gIdx += tri }
        }}
        return (Mesh(positions: pos, normals: nrm, indices: sIdx),
                Mesh(positions: pos, normals: nrm, indices: gIdx),
                Mesh(positions: pos, normals: nrm, indices: rIdx))
    }

    // MARK: vegetation meshes

    /// Stylised palm split into a brown trunk mesh and a green frond mesh so the two
    /// can be instanced with different materials (one material per instance).
    private static func makePalm(rng: inout SplitMix64) -> (trunk: Mesh, fronds: Mesh) {
        var tb = MeshBuilder(), fb = MeshBuilder()
        let trunkH: Float = 7.5, segs = 7
        var prev = SIMD3<Float>(0,0,0)
        var bend = SIMD3<Float>(0,0,0)
        let lean = SIMD3(Float.random(in: -0.18...0.18, using: &rng), 0, Float.random(in: -0.05...0.22, using: &rng))
        for i in 1...segs {
            let t = Float(i)/Float(segs)
            bend += lean
            let top = SIMD3(bend.x * t, trunkH*t, bend.z * t + t*t*1.0)
            tb.cylinder(from: prev, to: top, r0: 0.30*(1-t)+0.13, r1: 0.28*(1-t)+0.11, sides: 7)
            prev = top
        }
        let crown = prev
        for _ in 0..<3 {
            let a = Float.random(in: 0...(2 * .pi), using: &rng)
            tb.icosa(center: crown + SIMD3(cos(a)*0.35, -0.35, sin(a)*0.35), radius: 0.22)
        }
        let fronds = 12
        for f in 0..<fronds {
            let a = Float(f)/Float(fronds) * 2 * .pi + Float.random(in: -0.12...0.12, using: &rng)
            let outward = SIMD3(cos(a), 0, sin(a))
            let len: Float = 4.6, fsegs = 6
            var p = crown
            var dir = simd_normalize(outward + SIMD3(0, 0.85, 0))
            for k in 0..<fsegs {
                let tk = Float(k)/Float(fsegs)
                let w = (1 - tk) * 0.55 + 0.06
                let step = len / Float(fsegs)
                let next = p + dir * step
                let side  = simd_normalize(simd_cross(dir, SIMD3<Float>(0,1,0))) * w
                let nside = simd_normalize(simd_cross(dir, SIMD3<Float>(0,1,0))) * ((1 - (tk + 1.0/Float(fsegs))) * 0.55 + 0.04)
                fb.quad(p - side, p + side, next + nside, next - nside)
                p = next
                dir = simd_normalize(dir + SIMD3(0, -0.42, 0))
            }
        }
        return (tb.mesh(), fb.mesh())
    }

    private static func makeFern(seed: Int) -> Mesh {
        var b = MeshBuilder()
        var rng = SplitMix64(seed: UInt64(seed))
        let blades = 9
        for i in 0..<blades {
            let a = Float(i)/Float(blades) * 2 * .pi + Float.random(in: -0.2...0.2, using: &rng)
            let hh = Float.random(in: 1.3...2.0, using: &rng)
            let dir = SIMD3(cos(a)*0.7, hh, sin(a)*0.7)
            let base = SIMD3<Float>(0,0,0)
            let mid = base + dir*0.5
            let tip = base + dir + SIMD3(cos(a)*0.5, -0.2, sin(a)*0.5)
            let side = simd_normalize(simd_cross(dir, SIMD3<Float>(0,1,0))) * 0.14
            b.quad(base - side, base + side, mid + side*0.7, mid - side*0.7)
            b.quad(mid - side*0.7, mid + side*0.7, tip + side*0.2, tip - side*0.2)
        }
        return b.mesh()
    }

    private static func makeRock(seed: Int) -> Mesh {
        var b = MeshBuilder()
        let n = 3
        func disp(_ v: SIMD3<Float>) -> SIMD3<Float> {
            let d = 0.22 * (valueNoise(v.x*3+10, v.z*3+5, seed) - 0.5) + 0.12 * (valueNoise(v.y*4, v.x*4, seed+3) - 0.5)
            return simd_normalize(v) * (1 + d)
        }
        for (nrm, u, v) in cubeFaces() {
            for i in 0..<n { for j in 0..<n {
                func corner(_ a: Int, _ c: Int) -> SIMD3<Float> {
                    let fu = (Float(a)/Float(n) - 0.5) * 2, fv = (Float(c)/Float(n) - 0.5) * 2
                    return disp(nrm + u*fu + v*fv) * 0.5
                }
                b.tri(corner(i,j), corner(i+1,j), corner(i+1,j+1))
                b.tri(corner(i,j), corner(i+1,j+1), corner(i,j+1))
            }}
        }
        for i in b.positions.indices where b.positions[i].y < -0.18 { b.positions[i].y = -0.18 }
        return b.mesh()
    }

    private static func cubeFaces() -> [(SIMD3<Float>, SIMD3<Float>, SIMD3<Float>)] {
        [(SIMD3(0,0,1),SIMD3(1,0,0),SIMD3(0,1,0)), (SIMD3(0,0,-1),SIMD3(-1,0,0),SIMD3(0,1,0)),
         (SIMD3(1,0,0),SIMD3(0,0,-1),SIMD3(0,1,0)), (SIMD3(-1,0,0),SIMD3(0,0,1),SIMD3(0,1,0)),
         (SIMD3(0,1,0),SIMD3(1,0,0),SIMD3(0,0,-1)), (SIMD3(0,-1,0),SIMD3(1,0,0),SIMD3(0,0,1))]
    }
}

/// Tiny mesh accumulator that triangulates simple primitives with flat normals.
struct MeshBuilder {
    var positions: [SIMD3<Float>] = []
    var normals: [SIMD3<Float>] = []
    var indices: [UInt32] = []

    mutating func tri(_ a: SIMD3<Float>, _ b: SIMD3<Float>, _ c: SIMD3<Float>) {
        let n = simd_normalize(simd_cross(b - a, c - a))
        let base = UInt32(positions.count)
        positions.append(contentsOf: [a, b, c]); normals.append(contentsOf: [n, n, n])
        indices.append(contentsOf: [base, base+1, base+2])
    }
    mutating func quad(_ a: SIMD3<Float>, _ b: SIMD3<Float>, _ c: SIMD3<Float>, _ d: SIMD3<Float>) {
        tri(a, b, c); tri(a, c, d)
    }
    mutating func cylinder(from: SIMD3<Float>, to: SIMD3<Float>, r0: Float, r1: Float, sides: Int) {
        let axis = simd_normalize(to - from)
        var up = SIMD3<Float>(0,1,0); if abs(simd_dot(axis, up)) > 0.95 { up = SIMD3(1,0,0) }
        let t = simd_normalize(simd_cross(up, axis)), bvec = simd_cross(axis, t)
        for i in 0..<sides {
            let a0 = Float(i)/Float(sides) * 2 * .pi, a1 = Float(i+1)/Float(sides) * 2 * .pi
            let d0 = t*cos(a0) + bvec*sin(a0), d1 = t*cos(a1) + bvec*sin(a1)
            quad(from + d0*r0, from + d1*r0, to + d1*r1, to + d0*r1)
        }
    }
    mutating func icosa(center: SIMD3<Float>, radius: Float) {
        let v = [SIMD3<Float>(1,0,0), SIMD3(-1,0,0), SIMD3(0,1,0), SIMD3(0,-1,0), SIMD3(0,0,1), SIMD3(0,0,-1)]
            .map { center + $0 * radius }
        let f = [(0,2,4),(2,1,4),(1,3,4),(3,0,4),(2,0,5),(1,2,5),(3,1,5),(0,3,5)]
        for (a,b,c) in f { tri(v[a], v[b], v[c]) }
    }
    func mesh() -> Mesh { Mesh(positions: positions, normals: normals, indices: indices) }
}
