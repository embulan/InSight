import AVFoundation

final class CameraManager: NSObject {
    let session = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let sessionQueue = DispatchQueue(label: "camera.session.queue")
    private let outputQueue = DispatchQueue(label: "camera.output.queue", qos: .userInitiated)
    private var isConfigured = false

    var onSampleBuffer: ((CMSampleBuffer) -> Void)?

    func start() throws {
        // Configure inputs/outputs only once — they survive start/stop cycles
        if !isConfigured {
            try configure()
        }
        guard !session.isRunning else { return }
        sessionQueue.async { self.session.startRunning() }
    }

    func stop() {
        guard session.isRunning else { return }
        sessionQueue.async { self.session.stopRunning() }
    }

    private func configure() throws {
        session.beginConfiguration()
        session.sessionPreset = .medium

        guard
            let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
            let input = try? AVCaptureDeviceInput(device: device),
            session.canAddInput(input)
        else {
            session.commitConfiguration()
            throw NSError(domain: "CameraManager", code: -1, userInfo: [
                NSLocalizedDescriptionKey: "Failed to create camera input"
            ])
        }
        session.addInput(input)

        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        videoOutput.setSampleBufferDelegate(self, queue: outputQueue)

        guard session.canAddOutput(videoOutput) else {
            session.commitConfiguration()
            throw NSError(domain: "CameraManager", code: -2, userInfo: [
                NSLocalizedDescriptionKey: "Failed to add video output"
            ])
        }
        session.addOutput(videoOutput)
        session.commitConfiguration()
        isConfigured = true
    }
}

extension CameraManager: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        onSampleBuffer?(sampleBuffer)
    }
}
