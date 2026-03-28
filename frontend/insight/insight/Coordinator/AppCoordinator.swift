import Foundation
import AVFoundation
import Combine 

final class AppCoordinator: ObservableObject {
    static let shared = AppCoordinator()

    @Published var state: AssistState = .idle

    private let phoneSession: PhoneSessionManager? = Config.watchConnectivityEnabled ? .shared : nil
    private let cameraManager = CameraManager()
    var cameraSession: AVCaptureSession { cameraManager.session }
    private let frameSampler = FrameSampler(secondsBetweenFrames: 1.0 / Config.streamingFPS)
    private let jpegEncoder = JPEGEncoder()
    private let socketClient = BackendSocketClient()
    private let audioManager = AudioManager()

    private init() {
        if let phoneSession {
            phoneSession.onCommandReceived = { [weak self] command in
                self?.handleCommand(command)
            }
        }

        cameraManager.onSampleBuffer = { [weak self] sampleBuffer in
            self?.handleSampleBuffer(sampleBuffer)
        }

        socketClient.onEvent = { event in
            if Config.verboseLogging { print("Backend event:", event) }
        }

        audioManager.onAudioChunk = { [weak self] data in
            guard let self else { return }
            guard case .liveAssist(phase: .listening) = self.state else { return }
            self.socketClient.sendAudio(data, sampleRate: self.audioManager.sampleRate)
        }
    }

    private func handleCommand(_ command: AssistCommand) {
        switch command {
        case .startAssist:
            startAssist()

        case .stopAssist:
            stopAssist()

        case .enterQueryMode:
            enterQueryMode()

        case .submitRequest:
            submitRequest()
        }
    }

    func startStreaming()  { startAssist() }
    func stopStreaming()   { stopAssist()  }
    func enterQueryMode() {
        guard case .liveAssist = state else { return }
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
        socketClient.sendSubmit()

        state = .liveAssist(phase: .processing)
        sendWatchStatus(mode: "liveAssist", phase: AssistPhase.processing.rawValue)
    }

    private func startAssist() {
        guard state == .idle else { return }

        // Transition state immediately so the UI and watch update right away
        state = .liveAssist()
        sendWatchStatus(mode: "liveAssist", phase: AssistPhase.observing.rawValue)

        // Camera — non-fatal: simulator has no camera hardware
        do {
            try cameraManager.start()
        } catch {
            print("Camera unavailable (non-fatal):", error)
        }

        // Socket — connects to URL defined in Config.swift
        socketClient.connect(url: Config.backendWebSocketURL)
    }

    private func stopAssist() {
        audioManager.stop()
        cameraManager.stop()
        socketClient.disconnect()
        state = .idle
        sendWatchStatus(mode: "idle")
    }

    private func handleSampleBuffer(_ sampleBuffer: CMSampleBuffer) {
        guard isStreamingState else { return }
        guard frameSampler.shouldSendFrame() else { return }
        guard let jpeg = jpegEncoder.encode(sampleBuffer: sampleBuffer) else { return }

        socketClient.sendFrame(jpeg)
    }

    func setWalkingMode(destination: String, etaText: String) {
        state = .walking(destination: destination, etaText: etaText)
        sendWatchStatus(
            mode: "walking",
            destination: destination,
            etaText: etaText
        )
    }

    private func sendWatchStatus(
        mode: String,
        phase: String? = nil,
        destination: String? = nil,
        etaText: String? = nil
    ) {
        guard Config.watchConnectivityEnabled else { return }

        phoneSession?.sendWatchStatus(
            mode: mode,
            phase: phase,
            destination: destination,
            etaText: etaText
        )
    }

    private var isStreamingState: Bool {
        switch state {
        case .idle:
            return false
        case .liveAssist(_), .walking:
            return true
        }
    }
}
