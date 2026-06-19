#if canImport(AppKit)
import AppKit
import MetalKit
import simd
import Foundation

/// A renderer that can drive the realtime viewer: progressive accumulation into
/// a drawable, an orbit camera, and feature (AOV) view-mode toggles. Both the
/// classic `Renderer` (Cornell/showcase) and the `InstancedRenderer`
/// (Sponza/island/Moana) conform, so one viewer serves every scene.
protocol RealtimeBackend: AnyObject {
    var device: MTLDevice { get }
    var queue: MTLCommandQueue { get }
    var width: Int { get }
    var height: Int { get }
    var accumulatedFrames: UInt32 { get }
    func resetAccumulation()
    func encodeRealtimeFrame(into cmd: MTLCommandBuffer, target: MTLTexture,
                             samplesPerFrame: Int, bounces: Int, exposure: Float)

    var rtInitialEye: SIMD3<Float> { get }
    var rtFovYDeg: Float { get }
    func rtSetCamera(eye: SIMD3<Float>, target: SIMD3<Float>, fovYDeg: Float, aspect: Float)
    var rtSceneBounds: (lo: SIMD3<Float>, hi: SIMD3<Float>) { get }

    var rtViewModeNames: [String] { get }       // index 0 is always "Beauty"
    var rtViewMode: Int { get set }
    var rtTexturesEnabled: Bool { get set }
    var rtSupportsTextures: Bool { get }

    /// Optional one-line diagnostics appended to the window title (e.g. the ML
    /// feature-input summary for the neural-amplifier backend). nil = nothing.
    var rtDiagnostics: String? { get }
}

extension RealtimeBackend {
    var rtDiagnostics: String? { nil }
}

extension Renderer: RealtimeBackend {
    var rtInitialEye: SIMD3<Float> { scene.camera.position }
    var rtFovYDeg: Float { 2 * atan(scene.camera.tanHalfFovY) * 180 / .pi }
    func rtSetCamera(eye: SIMD3<Float>, target: SIMD3<Float>, fovYDeg: Float, aspect: Float) {
        prevCamera = uniforms.camera        // remember prior pose for motion vectors
        uniforms.camera = Camera.lookAt(eye: eye, target: target, fovYDeg: fovYDeg, aspect: aspect)
    }
    var rtSceneBounds: (lo: SIMD3<Float>, hi: SIMD3<Float>) {
        var lo = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        var hi = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)
        for p in scene.positions { lo = simd_min(lo, p); hi = simd_max(hi, p) }
        return (lo, hi)
    }
    var rtViewModeNames: [String] { Renderer.featureViewModeNames }
    var rtViewMode: Int {
        get { viewMode }
        set { viewMode = max(0, min(Renderer.featureViewModeNames.count - 1, newValue)) }
    }
    var rtTexturesEnabled: Bool { get { true } set {} }
    var rtSupportsTextures: Bool { false }
    // Surface the ML-pipeline input summary while a feature channel is shown.
    var rtDiagnostics: String? { viewMode == 0 ? nil : featureDiag }
}

extension InstancedRenderer: RealtimeBackend {
    var rtInitialEye: SIMD3<Float> { scene.camera.position }
    var rtFovYDeg: Float { 2 * atan(scene.camera.tanHalfFovY) * 180 / .pi }
    func rtSetCamera(eye: SIMD3<Float>, target: SIMD3<Float>, fovYDeg: Float, aspect: Float) {
        uniforms.camera = Camera.lookAt(eye: eye, target: target, fovYDeg: fovYDeg, aspect: aspect)
    }
    var rtSceneBounds: (lo: SIMD3<Float>, hi: SIMD3<Float>) { scene.worldBounds() }
    var rtViewModeNames: [String] {
        ["Beauty", "Albedo", "Normals", "Geo Normals", "UVs", "Material ID", "Ambient Occlusion", "Direct Light"]
    }
    var rtViewMode: Int {
        get { viewMode }
        set { viewMode = max(0, min(rtViewModeNames.count - 1, newValue)) }
    }
    var rtTexturesEnabled: Bool { get { texturesEnabled } set { texturesEnabled = newValue } }
    var rtSupportsTextures: Bool { true }
}

/// Realtime viewer: progressive path tracing into an MTKView with an orbit
/// camera and feature toggles. Accumulation resets whenever anything that
/// changes the image (camera, view mode, textures, bounces, exposure) changes.
final class PTView: MTKView {
    var backend: RealtimeBackend!
    var bounces = 8
    var exposure: Float = 1.0

    // Orbit camera state.
    var target = SIMD3<Float>(repeating: 0)
    var radius: Float = 1
    var yaw: Float = 0
    var pitch: Float = 0
    private var fovYDeg: Float = 39.3
    private var initialEye = SIMD3<Float>(repeating: 0)

    private var lastFrameLog = Date()
    private var frameCounter = 0

    override var acceptsFirstResponder: Bool { true }

    func setupCamera(center: SIMD3<Float>) {
        fovYDeg = backend.rtFovYDeg
        initialEye = backend.rtInitialEye
        target = center
        let off = initialEye - center
        radius = max(simd_length(off), 0.001)
        pitch = asin(max(-0.999, min(0.999, off.y / radius)))
        yaw = atan2(off.x, off.z)
        updateCamera()
    }

    func updateCamera() {
        pitch = max(-1.5, min(1.5, pitch))
        let eye = target + radius * SIMD3(cos(pitch) * sin(yaw), sin(pitch), cos(pitch) * cos(yaw))
        backend.rtSetCamera(eye: eye, target: target, fovYDeg: fovYDeg,
                            aspect: Float(backend.width) / Float(backend.height))
        backend.resetAccumulation()
    }

    private func setTitle() {
        let modes = backend.rtViewModeNames
        let mode = modes[min(backend.rtViewMode, modes.count - 1)]
        var parts = ["mode: \(mode)"]
        if backend.rtSupportsTextures { parts.append("tex: \(backend.rtTexturesEnabled ? "on" : "off")") }
        parts.append("bounces: \(bounces)")
        parts.append(String(format: "exp: %.2f", exposure))
        parts.append("spp: \(backend.accumulatedFrames)")
        if let diag = backend.rtDiagnostics { parts.append(diag) }
        window?.title = "Metal PBR Path Tracer  —  " + parts.joined(separator: "  |  ")
    }

    // keyCode → digit (top-row number keys).
    private static let digitKeys: [UInt16: Int] =
        [29: 0, 18: 1, 19: 2, 20: 3, 21: 4, 23: 5, 22: 6, 26: 7, 28: 8, 25: 9]

    override func keyDown(with e: NSEvent) {
        let panStep = radius * 0.05
        let rotStep: Float = 0.06
        let right = simd_normalize(SIMD3<Float>(cos(yaw), 0, -sin(yaw)))

        if let digit = PTView.digitKeys[e.keyCode] {     // 0-7 select view mode
            backend.rtViewMode = digit
            backend.resetAccumulation(); setTitle(); return
        }

        var cameraChanged = true     // most keys move the camera; toggles set false
        switch e.keyCode {
        case 123: yaw -= rotStep                       // left arrow
        case 124: yaw += rotStep                       // right arrow
        case 125: pitch -= rotStep                     // down arrow
        case 126: pitch += rotStep                     // up arrow
        case 13:  radius *= 0.92                        // W zoom in
        case 1:   radius *= 1.08                        // S zoom out
        case 0:   target -= right * panStep            // A pan left
        case 2:   target += right * panStep            // D pan right
        case 12:  target.y += panStep                  // Q up
        case 14:  target.y -= panStep                  // E down
        case 15:                                        // R reset camera
            setupCamera(center: target); setTitle(); return
        case 46:                                        // M cycle view mode forward
            backend.rtViewMode = (backend.rtViewMode + 1) % backend.rtViewModeNames.count
            cameraChanged = false
        case 45:                                        // N cycle view mode back
            backend.rtViewMode = (backend.rtViewMode + backend.rtViewModeNames.count - 1) % backend.rtViewModeNames.count
            cameraChanged = false
        case 17:                                        // T toggle textures
            backend.rtTexturesEnabled.toggle(); cameraChanged = false
        case 33:  bounces = max(1, bounces - 1); cameraChanged = false   // [
        case 30:  bounces = min(32, bounces + 1); cameraChanged = false  // ]
        case 27:  exposure = max(0.05, exposure / 1.15); cameraChanged = false   // -
        case 24:  exposure = min(20, exposure * 1.15); cameraChanged = false     // =
        case 53:  NSApp.terminate(nil)                 // Esc
        default:  cameraChanged = false
        }

        if cameraChanged { updateCamera() } else { backend.resetAccumulation() }
        setTitle()
    }

    override func draw(_ dirtyRect: NSRect) {
        guard let drawable = currentDrawable,
              let cmd = backend.queue.makeCommandBuffer() else { return }
        backend.encodeRealtimeFrame(into: cmd, target: drawable.texture,
                                    samplesPerFrame: 1, bounces: bounces, exposure: exposure)
        cmd.present(drawable)
        cmd.commit()

        frameCounter += 1
        if Date().timeIntervalSince(lastFrameLog) > 0.5 { setTitle(); frameCounter = 0; lastFrameLog = Date() }
    }
}

func runWindowApp(_ backend: RealtimeBackend, bounces: Int, exposure: Float = 1.0) {
    let app = NSApplication.shared
    app.setActivationPolicy(.regular)

    let w = backend.width, h = backend.height
    let view = PTView(frame: NSRect(x: 0, y: 0, width: w, height: h), device: backend.device)
    view.backend = backend
    view.bounces = bounces
    view.exposure = exposure
    view.colorPixelFormat = .rgba8Unorm
    view.framebufferOnly = false                 // compute writes into the drawable
    view.drawableSize = CGSize(width: w, height: h)
    view.preferredFramesPerSecond = 60
    view.isPaused = false
    view.enableSetNeedsDisplay = false

    let (lo, hi) = backend.rtSceneBounds
    view.setupCamera(center: (lo + hi) * 0.5)

    let window = NSWindow(contentRect: view.frame,
                          styleMask: [.titled, .closable, .resizable, .miniaturizable],
                          backing: .buffered, defer: false)
    window.title = "Metal PBR Path Tracer"
    window.contentView = view
    window.center()
    window.makeKeyAndOrderFront(nil)
    window.makeFirstResponder(view)
    app.activate(ignoringOtherApps: true)

    let modes = backend.rtViewModeNames.enumerated().map { "\($0.offset)=\($0.element)" }.joined(separator: "  ")
    print("""
    Realtime viewer controls
      camera  : arrows orbit · W/S zoom · A/D/Q/E pan · R reset · Esc quit
      view    : number keys select an AOV · M/N cycle  (\(modes))
      toggles : T textures on/off · [ / ] bounces · - / = exposure
    """)
    app.run()
}
#else
func runWindowApp(_ backend: AnyObject, bounces: Int, exposure: Float = 1.0) {
    print("Window mode requires AppKit (macOS).")
}
#endif
