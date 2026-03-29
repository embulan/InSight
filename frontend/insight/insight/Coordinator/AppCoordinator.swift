import Foundation
import AVFoundation
import Combine

final class AppCoordinator: ObservableObject {
    static let shared = AppCoordinator()

    @Published var state: AssistState = .idle
    @Published var latestCaption: String = ""
    /// Current walking-nav instruction from the server (`nav_step`); empty when inactive.
    @Published var navStepLine: String = ""

    private let phoneSession: PhoneSessionManager? = Config.watchConnectivityEnabled ? .shared : nil
    private let cameraManager = CameraManager()
    private let locationSender = LocationSender()
    var cameraSession: AVCaptureSession { cameraManager.session }
    private let frameSampler = FrameSampler(secondsBetweenFrames: Config.frameInterval)
    private let jpegEncoder = JPEGEncoder()
    private let socketClient = BackendSocketClient()
    private let audioManager = AudioManager()
    private let audioPlayer = AudioPlayerManager()
    private let frameProcessingQueue = DispatchQueue(label: "app.frame.processing.queue", qos: .userInitiated)
    private let audioBufferQueue = DispatchQueue(label: "app.audio.buffer.queue")
    private var bufferedAudioChunks: [Data] = []
    private var bufferedAudioByteCount = 0

    private init() {
        if let phoneSession {
            phoneSession.onCommandReceived = { [weak self] command in
                self?.handleCommand(command)
            }
        }

        cameraManager.onSampleBuffer = { [weak self] sampleBuffer in
            self?.handleSampleBuffer(sampleBuffer)
        }

        audioManager.onAudioChunk = { [weak self] data in
            guard let self else { return }
            guard case .liveAssist(phase: .listening) = self.state else { return }
            self.appendAudioChunk(data)
        }

        socketClient.onEvent = { [weak self] event in
            self?.handleBackendEvent(event)
        }

        locationSender.attach(client: socketClient)
    }

    // MARK: - Public controls (called from UI and watch)

    func startStreaming()  { startAssist() }
    func stopStreaming()   { stopAssist()  }

    func enterQueryMode() {
        guard case .liveAssist = state else { return }
        resetBufferedAudio()

        // Stop any TTS that is currently playing so it doesn't bleed into the recording
        audioPlayer.stop()

        state = .liveAssist(phase: .listening)
        sendWatchStatus(mode: "liveAssist", phase: AssistPhase.listening.rawValue)

        do {
            try audioManager.start()
            if Config.verboseLogging { print("Audio capture started") }
        } catch {
            print("Audio capture unavailable (non-fatal):", error)
        }
    }

    func submitRequest() {
        guard case .liveAssist = state else { return }

        audioManager.stop()
        if let bufferedAudio = drainBufferedAudio() {
            socketClient.sendAudio(bufferedAudio, sampleRate: audioManager.sampleRate)
        }
        socketClient.sendSubmit()

        state = .liveAssist(phase: .processing)
        sendWatchStatus(mode: "liveAssist", phase: AssistPhase.processing.rawValue)

        // Safety net: if no caption/error arrives within 30 s, recover automatically
        DispatchQueue.main.asyncAfter(deadline: .now() + 30) { [weak self] in
            guard let self else { return }
            if case .liveAssist(phase: .processing) = self.state {
                self.state = .liveAssist(phase: .observing)
                self.sendWatchStatus(mode: "liveAssist", phase: AssistPhase.observing.rawValue)
            }
        }
    }

    // MARK: - Private

    private func handleCommand(_ command: AssistCommand) {
        switch command {
        case .startAssist:    startAssist()
        case .stopAssist:     stopAssist()
        case .enterQueryMode: enterQueryMode()
        case .submitRequest:  submitRequest()
        }
    }

    private func startAssist() {
        guard state == .idle else { return }

        state = .liveAssist()
        sendWatchStatus(mode: "liveAssist", phase: AssistPhase.observing.rawValue)

        do {
            try cameraManager.start()
        } catch {
            print("Camera unavailable (non-fatal):", error)
        }

        socketClient.connect(url: Config.backendWebSocketURL)
        locationSender.start()
    }

    private func stopAssist() {
        audioManager.stop()
        cameraManager.stop()
        locationSender.stop()
        socketClient.disconnect()
        resetBufferedAudio()
        state = .idle
        latestCaption = ""
        navStepLine = ""
        sendWatchStatus(mode: "idle")
    }

    private func handleSampleBuffer(_ sampleBuffer: CMSampleBuffer) {
        guard isStreamingState else { return }
        // Do not send frames while recording a voice query — the backend
        // should stay quiet and save Gemini quota until submit arrives.
        if case .liveAssist(phase: .listening) = state { return }
        guard frameSampler.shouldSendFrame() else { return }

        frameProcessingQueue.async { [weak self] in
            guard let self else { return }
            guard let jpeg = self.jpegEncoder.encode(sampleBuffer: sampleBuffer) else { return }
            self.socketClient.sendFrame(jpeg)
        }
    }

    // MARK: - Backend event handling

    private func handleBackendEvent(_ event: IncomingBackendEvent) {
        if Config.verboseLogging { print("Backend event [\(event.type)]:", event.message ?? event.data?.prefix(40) ?? "-") }

        switch event.type {

        case "caption":
            guard let text = event.message, !text.isEmpty else { return }
            DispatchQueue.main.async {
                self.latestCaption = text
                // Return from processing → observing and push caption to watch
                if case .liveAssist(phase: .processing) = self.state {
                    self.state = .liveAssist(phase: .observing)
                }
                self.sendWatchStatus(
                    mode: "liveAssist",
                    phase: AssistPhase.observing.rawValue,
                    caption: text
                )
            }

        case "audio":
            guard let b64 = event.data, let mp3Data = Data(base64Encoded: b64) else { return }
            audioPlayer.stop()   // cut off any previous clip before starting the new one
            audioPlayer.play(mp3Data: mp3Data)

        case "nav_step":
            let line = event.message ?? ""
            DispatchQueue.main.async {
                self.navStepLine = line
            }

        case "status":
            if Config.verboseLogging { print("Server status:", event.message ?? "") }

        case "error":
            print("Backend error:", event.message ?? "unknown")
            DispatchQueue.main.async {
                // Recover to observing if we were waiting for a response
                if case .liveAssist(phase: .processing) = self.state {
                    self.state = .liveAssist(phase: .observing)
                }
            }

        default:
            break
        }
    }

    // MARK: - Walking mode (called externally when navigation response arrives)

    func setWalkingMode(destination: String, etaText: String) {
        state = .walking(destination: destination, etaText: etaText)
        sendWatchStatus(mode: "walking", destination: destination, etaText: etaText)
    }

    // MARK: - Watch status helper

    private func sendWatchStatus(
        mode: String,
        phase: String? = nil,
        caption: String? = nil,
        destination: String? = nil,
        etaText: String? = nil
    ) {
        guard Config.watchConnectivityEnabled else { return }
        phoneSession?.sendWatchStatus(
            mode: mode,
            phase: phase,
            caption: caption,
            destination: destination,
            etaText: etaText
        )
    }

    private var isStreamingState: Bool {
        switch state {
        case .idle:
            return false
        case .liveAssist(_), .walking(_, _):
            return true
        }
    }

    private func appendAudioChunk(_ data: Data) {
        audioBufferQueue.async {
            self.bufferedAudioChunks.append(data)
            self.bufferedAudioByteCount += data.count
        }
    }

    private func drainBufferedAudio() -> Data? {
        audioBufferQueue.sync {
            guard bufferedAudioByteCount > 0 else {
                bufferedAudioChunks.removeAll()
                return nil
            }

            var combined = Data(capacity: bufferedAudioByteCount)
            for chunk in bufferedAudioChunks {
                combined.append(chunk)
            }

            bufferedAudioChunks.removeAll()
            bufferedAudioByteCount = 0
            return combined
        }
    }

    private func resetBufferedAudio() {
        audioBufferQueue.async {
            self.bufferedAudioChunks.removeAll()
            self.bufferedAudioByteCount = 0
        }
    }
}
