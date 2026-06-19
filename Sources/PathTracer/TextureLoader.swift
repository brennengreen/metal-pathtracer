import Metal
import Foundation
import CoreGraphics
import ImageIO

/// Loads an image file (TGA/PNG/JPG — anything ImageIO can decode) into an sRGB
/// `MTLTexture` with a full mip chain. Sampling such a texture returns *linear*
/// colour (the GPU undoes sRGB), which is what the path tracer's linear shading
/// expects. Returns nil on failure.
enum TextureLoader {
    static func load(device: MTLDevice, path: String, queue: MTLCommandQueue) -> MTLTexture? {
        guard FileManager.default.fileExists(atPath: path),
              let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil),
              let cg = CGImageSourceCreateImageAtIndex(src, 0, nil) else { return nil }
        let w = cg.width, h = cg.height
        guard w > 0, h > 0 else { return nil }

        // Decode into tightly packed RGBA8 (premultiplied-last, sRGB bytes).
        var rgba = [UInt8](repeating: 0, count: w * h * 4)
        let cs = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(data: &rgba, width: w, height: h, bitsPerComponent: 8,
                                  bytesPerRow: w * 4, space: cs,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))

        let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm_srgb,
                                                            width: w, height: h, mipmapped: true)
        desc.usage = [.shaderRead]
        desc.storageMode = .shared
        guard let tex = device.makeTexture(descriptor: desc) else { return nil }
        rgba.withUnsafeBytes {
            tex.replace(region: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0,
                        withBytes: $0.baseAddress!, bytesPerRow: w * 4)
        }
        // Generate the mip chain (essential: Sponza textures tile and minify hard).
        if tex.mipmapLevelCount > 1, let cb = queue.makeCommandBuffer(), let blit = cb.makeBlitCommandEncoder() {
            blit.generateMipmaps(for: tex)
            blit.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        }
        return tex
    }

    /// A 1×1 white sRGB texture used to pad unused albedo-array slots.
    static func white(device: MTLDevice) -> MTLTexture {
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm_srgb,
                                                         width: 1, height: 1, mipmapped: false)
        d.usage = [.shaderRead]; d.storageMode = .shared
        let t = device.makeTexture(descriptor: d)!
        let px: [UInt8] = [255, 255, 255, 255]
        px.withUnsafeBytes { t.replace(region: MTLRegionMake2D(0, 0, 1, 1), mipmapLevel: 0, withBytes: $0.baseAddress!, bytesPerRow: 4) }
        return t
    }
}
