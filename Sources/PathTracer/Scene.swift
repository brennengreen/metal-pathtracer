import simd
import Foundation

/// CPU-side scene: triangle soup + materials + camera, ready to upload to Metal.
struct Scene {
    var positions: [SIMD3<Float>] = []
    var normals:   [SIMD3<Float>] = []
    var indices:   [UInt32] = []
    var triMaterial: [UInt32] = []      // one material index per triangle
    var materials: [Material] = []
    var emissiveTris: [UInt32] = []     // triangle indices whose material emits
    var camera = Camera()
    var background = SIMD3<Float>(repeating: 0)

    var triangleCount: Int { indices.count / 3 }

    // MARK: Builders

    mutating func addMaterial(_ m: Material) -> UInt32 {
        materials.append(m); return UInt32(materials.count - 1)
    }

    /// Add a quad p0->p1->p2->p3 (CCW) as two triangles with a flat normal.
    mutating func addQuad(_ p0: SIMD3<Float>, _ p1: SIMD3<Float>, _ p2: SIMD3<Float>,
                          _ p3: SIMD3<Float>, material: UInt32) {
        let n = simd_normalize(simd_cross(p1 - p0, p2 - p0))
        let base = UInt32(positions.count)
        positions.append(contentsOf: [p0, p1, p2, p3])
        normals.append(contentsOf: [n, n, n, n])
        addTri(base + 0, base + 1, base + 2, material: material)
        addTri(base + 0, base + 2, base + 3, material: material)
    }

    mutating func addTri(_ a: UInt32, _ b: UInt32, _ c: UInt32, material: UInt32) {
        let tri = UInt32(indices.count / 3)
        indices.append(contentsOf: [a, b, c])
        triMaterial.append(material)
        if simd_reduce_max(materials[Int(material)].emission) > 0 { emissiveTris.append(tri) }
    }

    /// Axis-aligned box from `lo` to `hi` with outward-facing quads.
    mutating func addBox(lo: SIMD3<Float>, hi: SIMD3<Float>, material: UInt32) {
        let p = [
            SIMD3(lo.x, lo.y, lo.z), SIMD3(hi.x, lo.y, lo.z),
            SIMD3(hi.x, hi.y, lo.z), SIMD3(lo.x, hi.y, lo.z),
            SIMD3(lo.x, lo.y, hi.z), SIMD3(hi.x, lo.y, hi.z),
            SIMD3(hi.x, hi.y, hi.z), SIMD3(lo.x, hi.y, hi.z),
        ]
        addQuad(p[0], p[3], p[2], p[1], material: material)  // -z
        addQuad(p[4], p[5], p[6], p[7], material: material)  // +z
        addQuad(p[0], p[4], p[7], p[3], material: material)  // -x
        addQuad(p[1], p[2], p[6], p[5], material: material)  // +x
        addQuad(p[0], p[1], p[5], p[4], material: material)  // -y
        addQuad(p[3], p[7], p[6], p[2], material: material)  // +y
    }

    /// UV sphere with smooth per-vertex normals.
    mutating func addSphere(center: SIMD3<Float>, radius: Float, material: UInt32,
                            segments: Int = 48, rings: Int = 24) {
        let base = UInt32(positions.count)
        for i in 0...rings {
            let v = Float(i) / Float(rings)
            let phi = v * Float.pi
            for j in 0...segments {
                let u = Float(j) / Float(segments)
                let theta = u * 2 * Float.pi
                let n = SIMD3<Float>(sin(phi) * cos(theta), cos(phi), sin(phi) * sin(theta))
                positions.append(center + radius * n)
                normals.append(n)
            }
        }
        let stride = UInt32(segments + 1)
        for i in 0..<UInt32(rings) {
            for j in 0..<UInt32(segments) {
                let a = base + i * stride + j
                let b = a + stride
                addTri(a, b, a + 1, material: material)
                addTri(a + 1, b, b + 1, material: material)
            }
        }
    }

    // MARK: Scene presets

    /// Canonical Cornell box (cm units, 0..555) with two interior blocks.
    /// Add the Cornell box shell (walls + ceiling light) and return the white
    /// material id. Shared by the empty box, the two-block box, and model scenes.
    @discardableResult
    private mutating func addCornellShell() -> UInt32 {
        let white = addMaterial(Material(baseColor: SIMD3(0.73, 0.73, 0.73), roughness: 0.85))
        let red   = addMaterial(Material(baseColor: SIMD3(0.65, 0.05, 0.05), roughness: 0.85))
        let green = addMaterial(Material(baseColor: SIMD3(0.12, 0.45, 0.15), roughness: 0.85))
        let light = addMaterial(Material(emission: SIMD3(repeating: 22)))

        addQuad(SIMD3(552.8,0,0), SIMD3(0,0,0), SIMD3(0,0,559.2), SIMD3(549.6,0,559.2), material: white)      // floor
        addQuad(SIMD3(556,548.8,0), SIMD3(556,548.8,559.2), SIMD3(0,548.8,559.2), SIMD3(0,548.8,0), material: white) // ceiling
        addQuad(SIMD3(549.6,0,559.2), SIMD3(0,0,559.2), SIMD3(0,548.8,559.2), SIMD3(556,548.8,559.2), material: white) // back
        addQuad(SIMD3(0,0,559.2), SIMD3(0,0,0), SIMD3(0,548.8,0), SIMD3(0,548.8,559.2), material: green)      // right
        addQuad(SIMD3(552.8,0,0), SIMD3(549.6,0,559.2), SIMD3(556,548.8,559.2), SIMD3(556,548.8,0), material: red) // left
        addQuad(SIMD3(343,548.7,227), SIMD3(343,548.7,332), SIMD3(213,548.7,332), SIMD3(213,548.7,227), material: light) // light (faces -y)
        return white
    }

    static func cornell(aspect: Float) -> Scene {
        var s = Scene()
        let white = s.addCornellShell()

        // Short block.
        s.addQuad(SIMD3(130,165,65), SIMD3(82,165,225), SIMD3(240,165,272), SIMD3(290,165,114), material: white)
        s.addQuad(SIMD3(290,0,114), SIMD3(290,165,114), SIMD3(240,165,272), SIMD3(240,0,272), material: white)
        s.addQuad(SIMD3(130,0,65), SIMD3(130,165,65), SIMD3(290,165,114), SIMD3(290,0,114), material: white)
        s.addQuad(SIMD3(82,0,225), SIMD3(82,165,225), SIMD3(130,165,65), SIMD3(130,0,65), material: white)
        s.addQuad(SIMD3(240,0,272), SIMD3(240,165,272), SIMD3(82,165,225), SIMD3(82,0,225), material: white)

        // Tall block.
        s.addQuad(SIMD3(423,330,247), SIMD3(265,330,296), SIMD3(314,330,456), SIMD3(472,330,406), material: white)
        s.addQuad(SIMD3(423,0,247), SIMD3(423,330,247), SIMD3(472,330,406), SIMD3(472,0,406), material: white)
        s.addQuad(SIMD3(472,0,406), SIMD3(472,330,406), SIMD3(314,330,456), SIMD3(314,0,456), material: white)
        s.addQuad(SIMD3(314,0,456), SIMD3(314,330,456), SIMD3(265,330,296), SIMD3(265,0,296), material: white)
        s.addQuad(SIMD3(265,0,296), SIMD3(265,330,296), SIMD3(423,330,247), SIMD3(423,0,247), material: white)

        s.camera = Camera.lookAt(eye: SIMD3(278, 273, -800), target: SIMD3(278, 273, 0),
                                 fovYDeg: 39.3, aspect: aspect)
        s.background = SIMD3(repeating: 0)
        return s
    }

    /// Load an OBJ mesh, fit it onto the Cornell-box floor, and render it inside
    /// the box so it picks up classic color-bleeding global illumination.
    /// `yawDeg` lets you turn the model to face the camera.
    static func cornellModel(_ mesh: OBJMesh, material: Material, aspect: Float,
                             fitHeight: Float = 320, yawDeg: Float = 0) -> Scene {
        var s = Scene()
        s.addCornellShell()
        let mat = s.addMaterial(material)

        let (lo, hi) = mesh.bounds()
        let center = (lo + hi) * 0.5
        let extent = simd_reduce_max(hi - lo)
        let scale = extent > 0 ? fitHeight / extent : 1
        let cy = cos(radians(yawDeg)), sy = sin(radians(yawDeg))
        // Target: centered in plan (x≈278, z≈300), resting on the floor (y=0).
        let place = SIMD3<Float>(278, 0, 300)

        let base = UInt32(s.positions.count)
        for k in mesh.positions.indices {
            var q = (mesh.positions[k] - center) * scale            // center + scale
            q = SIMD3(cy * q.x + sy * q.z, q.y, -sy * q.x + cy * q.z) // yaw about Y
            var n = mesh.normals[k]
            n = SIMD3(cy * n.x + sy * n.z, n.y, -sy * n.x + cy * n.z)
            s.positions.append(q + SIMD3(place.x, place.y - (lo.y - center.y) * scale, place.z))
            s.normals.append(n)
        }
        var i = 0
        while i < mesh.indices.count {
            s.addTri(base + mesh.indices[i], base + mesh.indices[i + 1], base + mesh.indices[i + 2], material: mat)
            i += 3
        }

        s.camera = Camera.lookAt(eye: SIMD3(278, 273, -800), target: SIMD3(278, 240, 280),
                                 fovYDeg: 42, aspect: aspect)
        s.background = SIMD3(repeating: 0)
        return s
    }

    /// Studio environment for showing off a loaded model: large soft ground,
    /// a big key area light, and a neutral environment fill. More presentable
    /// than the Cornell box for a model gallery.
    static func studioModel(_ mesh: OBJMesh, material: Material, aspect: Float,
                            fitHeight: Float = 3.0, yawDeg: Float = 0) -> Scene {
        var s = Scene()
        let floor = s.addMaterial(Material(baseColor: SIMD3(0.45, 0.45, 0.48), roughness: 0.18))
        s.addQuad(SIMD3(-30,0,-30), SIMD3(30,0,-30), SIMD3(30,0,30), SIMD3(-30,0,30), material: floor)
        let mat = s.addMaterial(material)

        let (lo, hi) = mesh.bounds()
        let center = (lo + hi) * 0.5
        let extent = simd_reduce_max(hi - lo)
        let scale = extent > 0 ? fitHeight / extent : 1
        let cy = cos(radians(yawDeg)), sy = sin(radians(yawDeg))

        let base = UInt32(s.positions.count)
        for k in mesh.positions.indices {
            var q = (mesh.positions[k] - center) * scale
            q = SIMD3(cy * q.x + sy * q.z, q.y, -sy * q.x + cy * q.z)
            var n = mesh.normals[k]
            n = SIMD3(cy * n.x + sy * n.z, n.y, -sy * n.x + cy * n.z)
            s.positions.append(q + SIMD3(0, -(lo.y - center.y) * scale, 0))
            s.normals.append(n)
        }
        var i = 0
        while i < mesh.indices.count {
            s.addTri(base + mesh.indices[i], base + mesh.indices[i + 1], base + mesh.indices[i + 2], material: mat)
            i += 3
        }

        // Key light: a large soft quad above and in front.
        let light = s.addMaterial(Material(emission: SIMD3(11, 10.4, 9.5)))
        let ly = fitHeight * 2.4, ls = fitHeight * 1.1
        s.addQuad(SIMD3(-ls, ly, -ls - fitHeight*0.3), SIMD3(ls, ly, -ls - fitHeight*0.3),
                  SIMD3(ls, ly, ls - fitHeight*0.3), SIMD3(-ls, ly, ls - fitHeight*0.3), material: light)

        s.camera = Camera.lookAt(eye: SIMD3(fitHeight*0.7, fitHeight*0.85, -fitHeight*1.7),
                                 target: SIMD3(0, fitHeight*0.42, 0), fovYDeg: 40, aspect: aspect)
        s.background = SIMD3(0.50, 0.55, 0.62) * 0.7      // neutral environment fill
        return s
    }

    /// Named PBR material presets for loaded models.
    static func modelMaterial(_ name: String) -> Material {
        switch name.lowercased() {
        case "gold":   return Material(baseColor: SIMD3(1.0, 0.78, 0.34), metallic: 1.0, roughness: 0.12)
        case "copper": return Material(baseColor: SIMD3(0.95, 0.64, 0.54), metallic: 1.0, roughness: 0.18)
        case "silver", "mirror": return Material(baseColor: SIMD3(0.97, 0.96, 0.95), metallic: 1.0, roughness: 0.04)
        case "jade":   return Material(baseColor: SIMD3(0.10, 0.55, 0.30), metallic: 0.0, roughness: 0.18)
        case "red":    return Material(baseColor: SIMD3(0.80, 0.12, 0.12), metallic: 0.0, roughness: 0.30)
        case "white":  return Material(baseColor: SIMD3(0.85, 0.85, 0.85), metallic: 0.0, roughness: 0.35)
        default:        return Material(baseColor: SIMD3(0.92, 0.88, 0.80), metallic: 0.0, roughness: 0.25) // porcelain
        }
    }


    /// PBR showcase: rows of spheres sweeping roughness (metal) and metallic.
    static func materialShowcase(aspect: Float) -> Scene {
        var s = Scene()
        let floor = s.addMaterial(Material(baseColor: SIMD3(0.55, 0.55, 0.58), roughness: 0.25))
        s.addQuad(SIMD3(-12,0,-12), SIMD3(12,0,-12), SIMD3(12,0,12), SIMD3(-12,0,12), material: floor)

        let n = 6
        for i in 0..<n {
            let t = Float(i) / Float(n - 1)
            let x = (Float(i) - Float(n - 1) / 2) * 2.4
            // Back row: gold metal, roughness sweep.
            let metal = s.addMaterial(Material(baseColor: SIMD3(1.0, 0.78, 0.34),
                                               metallic: 1.0, roughness: max(0.03, t)))
            s.addSphere(center: SIMD3(x, 1.0, 1.6), radius: 1.0, material: metal)
            // Front row: dielectric->metal sweep at low roughness.
            let mix = s.addMaterial(Material(baseColor: SIMD3(0.85, 0.18, 0.18),
                                             metallic: t, roughness: 0.12))
            s.addSphere(center: SIMD3(x, 1.0, -1.6), radius: 1.0, material: mix)
        }

        // Large soft area light overhead.
        let light = s.addMaterial(Material(emission: SIMD3(repeating: 12)))
        s.addQuad(SIMD3(-5,7,-5), SIMD3(5,7,-5), SIMD3(5,7,5), SIMD3(-5,7,5), material: light)

        s.camera = Camera.lookAt(eye: SIMD3(0, 4.5, -11), target: SIMD3(0, 1.0, 0),
                                 fovYDeg: 38, aspect: aspect)
        s.background = SIMD3(0.45, 0.6, 0.85) * 0.6   // soft sky as uniform environment
        return s
    }

    static func named(_ name: String, aspect: Float) -> Scene {
        switch name.lowercased() {
        case "showcase", "materials", "spheres": return materialShowcase(aspect: aspect)
        default: return cornell(aspect: aspect)
        }
    }
}
