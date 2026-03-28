import SwiftUI
import AVFoundation

struct CameraPreviewView: View {
    let session: AVCaptureSession

    var body: some View {
        #if targetEnvironment(simulator)
        SimulatorCameraPlaceholder()
        #else
        _CameraPreviewView(session: session)
        #endif
    }
}

// MARK: - Real device

private struct _CameraPreviewView: UIViewRepresentable {
    let session: AVCaptureSession

    func makeUIView(context: Context) -> _PreviewUIView {
        let view = _PreviewUIView()
        view.previewLayer.session = session
        view.previewLayer.videoGravity = .resizeAspectFill
        return view
    }

    func updateUIView(_ uiView: _PreviewUIView, context: Context) {}
}

private final class _PreviewUIView: UIView {
    override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
    var previewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
}

// MARK: - Simulator placeholder

private struct SimulatorCameraPlaceholder: View {
    var body: some View {
        ZStack {
            Color.black
            VStack(spacing: 12) {
                Image(systemName: "camera.slash.fill")
                    .font(.system(size: 48))
                    .foregroundStyle(.white.opacity(0.4))
                Text("Camera not available\nin Simulator")
                    .font(.caption)
                    .multilineTextAlignment(.center)
                    .foregroundStyle(.white.opacity(0.4))
            }
        }
    }
}
