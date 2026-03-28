import Foundation
import Combine

final class BackendSocketClient: NSObject, ObservableObject {
    private var session: URLSession?
    private var socketTask: URLSessionWebSocketTask?

    var onEvent: ((IncomingBackendEvent) -> Void)?

    func connect(url: URL) {
        let configuration = URLSessionConfiguration.default
        session = URLSession(configuration: configuration)
        socketTask = session?.webSocketTask(with: url)
        socketTask?.resume()
        receiveLoop()
    }

    func disconnect() {
        socketTask?.cancel(with: .normalClosure, reason: nil)
        socketTask = nil
        session = nil
    }

    func sendFrame(_ jpegData: Data) {
        sendJSON(OutgoingFrameMessage(
            type: "frame",
            timestampMs: now(),
            jpegBase64: jpegData.base64EncodedString()
        ))
    }

    func sendAudio(_ pcmData: Data, sampleRate: Double) {
        sendJSON(OutgoingAudioMessage(
            timestampMs: now(),
            sampleRate: Int(sampleRate),
            pcmBase64: pcmData.base64EncodedString()
        ))
    }

    func sendSubmit() {
        sendJSON(OutgoingSubmitMessage(timestampMs: now()))
        if Config.verboseLogging { print("WebSocket: submit sent") }
    }

    // MARK: - Helpers

    private func now() -> Int64 {
        Int64(Date().timeIntervalSince1970 * 1000)
    }

    private func sendJSON<T: Encodable>(_ value: T) {
        do {
            let data = try JSONEncoder().encode(value)
            let text = String(decoding: data, as: UTF8.self)
            socketTask?.send(.string(text)) { error in
                if let error { print("WebSocket send error: \(error)") }
            }
        } catch {
            print("WebSocket encode error: \(error)")
        }
    }

    private func receiveLoop() {
        socketTask?.receive { [weak self] result in
            switch result {
            case .failure(let error):
                print("WebSocket receive error: \(error)")

            case .success(let message):
                switch message {
                case .string(let text):
                    self?.handleString(text)
                case .data(let data):
                    self?.handleData(data)
                @unknown default:
                    break
                }

                self?.receiveLoop()
            }
        }
    }

    private func handleString(_ text: String) {
        guard let data = text.data(using: .utf8) else { return }
        handleData(data)
    }

    private func handleData(_ data: Data) {
        do {
            let event = try JSONDecoder().decode(IncomingBackendEvent.self, from: data)
            DispatchQueue.main.async {
                self.onEvent?(event)
            }
        } catch {
            print("Backend decode error: \(error)")
        }
    }
}
