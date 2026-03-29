import AVFoundation
import CoreImage

final class JPEGEncoder {
    private let context = CIContext()
    private let colorSpace = CGColorSpaceCreateDeviceRGB()

    func encode(
        sampleBuffer: CMSampleBuffer,
        compressionQuality: CGFloat = 0.4
    ) -> Data? {
        guard let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return nil
        }

        let ciImage = CIImage(cvPixelBuffer: imageBuffer)
        return context.jpegRepresentation(
            of: ciImage,
            colorSpace: colorSpace,
            options: [kCGImageDestinationLossyCompressionQuality as CIImageRepresentationOption: compressionQuality]
        )
    }
}
