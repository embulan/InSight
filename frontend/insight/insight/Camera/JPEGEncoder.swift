import AVFoundation
import UIKit
import CoreImage

final class JPEGEncoder {
    private let context = CIContext()

    func encode(
        sampleBuffer: CMSampleBuffer,
        compressionQuality: CGFloat = 0.4
    ) -> Data? {
        guard let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return nil
        }

        let ciImage = CIImage(cvPixelBuffer: imageBuffer)

        guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else {
            return nil
        }

        let image = UIImage(cgImage: cgImage)
        return image.jpegData(compressionQuality: compressionQuality)
    }
}
