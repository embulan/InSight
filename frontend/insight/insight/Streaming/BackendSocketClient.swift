import Foundation
import Combine

final class BackendSocketClient: NSObject, ObservableObject, URLSessionWebSocketDelegate {
    private enum ConnectionState {
        case disconnected
        case connecting
        case connected
    }

    private var session: URLSession?
    private var socketTask: URLSessionWebSocketTask?
    private var connectionState: ConnectionState = .disconnected
    private var pendingMessages: [URLSessionWebSocketTask.Message] = []
    private let decoder = JSONDecoder()
    private var isDisconnecting = false
    private var pingTimer: Timer?

    var onEvent: ((IncomingBackendEvent) -> Void)?
    /// Called when the connection drops unexpectedly (not from an explicit disconnect() call).
    var onDisconnect: (() -> Void)?

    func connect(url: URL) {
        disconnect()
        isDisconnecting = false

        let configuration = URLSessionConfiguration.default
        session = URLSession(configuration: configuration, delegate: self, delegateQueue: nil)

        var request = URLRequest(url: url)
        if let subprotocol = Config.backendWebSocketSubprotocol {
            request.setValue(subprotocol, forHTTPHeaderField: "Sec-WebSocket-Protocol")
        }

        socketTask = session?.webSocketTask(with: request)
        connectionState = .connecting
        socketTask?.resume()
        receiveLoop()
    }

    func disconnect() {
        isDisconnecting = true
        stopPingTimer()
        pendingMessages.removeAll()
        connectionState = .disconnected
        socketTask?.cancel(with: .normalClosure, reason: nil)
        socketTask = nil
        session?.invalidateAndCancel()
        session = nil
    }

    // MARK: - Keepalive ping

    private func startPingTimer() {
        stopPingTimer()
        // Fire every 30 s on the main run loop — well within typical NAT idle timeouts (60–90 s)
        pingTimer = Timer.scheduledTimer(withTimeInterval: 30, repeats: true) { [weak self] _ in
            self?.sendPing()
        }
    }

    private func stopPingTimer() {
        pingTimer?.invalidate()
        pingTimer = nil
    }

    private func sendPing() {
        guard connectionState == .connected, let socketTask else { return }
        socketTask.sendPing { [weak self] error in
            if let error {
                print("WebSocket ping failed: \(error) — treating as disconnect")
                self?.handleUnexpectedDisconnect()
            }
        }
    }

    private func handleUnexpectedDisconnect() {
        guard !isDisconnecting else { return }
        connectionState = .disconnected
        stopPingTimer()
        DispatchQueue.main.async { [weak self] in
            self?.onDisconnect?()
        }
    }

    func sendFrame(_ jpegData: Data) {
        sendJSONObject([
            "type": "frame",
            "timestampMs": now(),
            "jpegBase64": jpegData.base64EncodedString()
        ])
    }

    func sendAudio(_ pcmData: Data, sampleRate: Double) {
        sendJSONObject([
            "type": "audio",
            "timestampMs": now(),
            "sampleRate": Int(sampleRate),
            "pcmBase64": pcmData.base64EncodedString()
        ])
    }

    func sendListeningStart() {
        sendJSONObject([
            "type": "listening_start",
            "timestampMs": now()
        ])
    }

    func sendAudioDone() {
        sendJSONObject([
            "type": "audio_done",
            "timestampMs": now()
        ])
    }

    func sendSubmit() {
        sendJSONObject([
            "type": "submit",
            "timestampMs": now()
        ])
        if Config.verboseLogging { print("WebSocket: submit sent") }
    }

    // MARK: - Helpers

    private func now() -> Int64 {
        Int64(Date().timeIntervalSince1970 * 1000)
    }

    private func sendJSONObject(_ object: [String: Any]) {
        do {
            let data = try JSONSerialization.data(withJSONObject: object, options: [])
            let text = String(decoding: data, as: UTF8.self)
            if Config.verboseLogging {
                let type_ = object["type"] as? String ?? "?"
                let payloadBytes = data.count
                print("WebSocket → [\(type_)]  \(payloadBytes) bytes")
            }
            enqueueOrSend(.string(text))
        } catch {
            print("WebSocket encode error: \(error)")
        }
    }

    private func enqueueOrSend(_ message: URLSessionWebSocketTask.Message) {
        guard let socketTask else {
            if Config.verboseLogging { print("WebSocket send dropped: no active task") }
            return
        }

        guard connectionState == .connected else {
            pendingMessages.append(message)
            if Config.verboseLogging { print("WebSocket send queued until connection opens") }
            return
        }

        socketTask.send(message) { error in
            if let error { print("WebSocket send error: \(error)") }
        }
    }

    private func flushPendingMessages() {
        guard connectionState == .connected else { return }

        let messages = pendingMessages
        pendingMessages.removeAll()
        for message in messages {
            enqueueOrSend(message)
        }
    }

    private func receiveLoop() {
        socketTask?.receive { [weak self] result in
            guard let self else { return }
            switch result {
            case .failure(let error):
                if !isDisconnecting {
                    print("WebSocket receive error: \(error)")
                    handleUnexpectedDisconnect()
                }

            case .success(let message):
                switch message {
                case .string(let text):
                    handleString(text)
                case .data(let data):
                    handleData(data)
                @unknown default:
                    break
                }
                receiveLoop()
            }
        }
    }

    private func handleString(_ text: String) {
        guard let data = text.data(using: .utf8) else { return }
        handleData(data)
    }

    private func handleData(_ data: Data) {
        Task { @MainActor [weak self] in
            guard let self else { return }

            do {
                let event = try decoder.decode(IncomingBackendEvent.self, from: data)
                if Config.verboseLogging {
                    let detail = event.message ?? (event.data != nil ? "<\(event.data!.count) chars>" : "-")
                    print("WebSocket ← [\(event.type)]  \(detail.prefix(80))")
                }
                onEvent?(event)
            } catch {
                print("Backend decode error: \(error)")
            }
        }
    }

    func urlSession(
        _ session: URLSession,
        webSocketTask: URLSessionWebSocketTask,
        didOpenWithProtocol `protocol`: String?
    ) {
        connectionState = .connected
        isDisconnecting = false
        if Config.verboseLogging {
            print("WebSocket connected. Negotiated protocol:", `protocol` ?? "none")
        }
        DispatchQueue.main.async { self.startPingTimer() }
        flushPendingMessages()
    }

    func urlSession(
        _ session: URLSession,
        webSocketTask: URLSessionWebSocketTask,
        didCloseWith closeCode: URLSessionWebSocketTask.CloseCode,
        reason: Data?
    ) {
        let wasDisconnecting = isDisconnecting
        connectionState = .disconnected
        isDisconnecting = false
        stopPingTimer()
        if Config.verboseLogging {
            let reasonText = reason.flatMap { String(data: $0, encoding: .utf8) } ?? "none"
            let closeKind = wasDisconnecting ? "client disconnect" : "remote close"
            print("WebSocket closed:", closeCode.rawValue, closeKind, "reason:", reasonText)
        }
        if !wasDisconnecting {
            DispatchQueue.main.async { [weak self] in
                self?.onDisconnect?()
            }
        }
    }
}
