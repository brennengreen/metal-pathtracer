import Foundation
import CoreGraphics
import ImageIO

/// Write an 8-bit RGBA buffer to a PNG file using ImageIO.
@discardableResult
func writePNG(rgba8: [UInt8], width: Int, height: Int, to path: String) -> Bool {
    let cs = CGColorSpaceCreateDeviceRGB()
    let info = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
    guard let provider = CGDataProvider(data: Data(rgba8) as CFData),
          let img = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
                            bytesPerRow: width * 4, space: cs, bitmapInfo: info,
                            provider: provider, decode: nil, shouldInterpolate: false,
                            intent: .defaultIntent) else { return false }
    let url = URL(fileURLWithPath: path) as CFURL
    guard let dest = CGImageDestinationCreateWithURL(url, "public.png" as CFString, 1, nil) else { return false }
    CGImageDestinationAddImage(dest, img, nil)
    return CGImageDestinationFinalize(dest)
}
