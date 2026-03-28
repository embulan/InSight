import AVFoundation

final class AudioManager {

    // A new engine instance is created on every start() — guarantees no stale tap
    private var engine: AVAudioEngine?
    private(set) var sampleRate: Double = 44100

    /// Called on a background thread with each raw Float32 PCM chunk
    var onAudioChunk: ((Data) -> Void)?

    func start() throws {
        // Tear down any previous engine before creating a new one
        teardown()

        // Configure the shared audio session for recording
        let audioSession = AVAudioSession.sharedInstance()
        try audioSession.setCategory(.playAndRecord, mode: .default,
                                     options: [.defaultToSpeaker, .allowBluetooth])
        try audioSession.setActive(true)

        // Fresh engine — inputNode has no tap by definition
        let newEngine = AVAudioEngine()
        let input = newEngine.inputNode
        let format = input.outputFormat(forBus: 0)
        sampleRate = format.sampleRate

        input.installTap(onBus: 0, bufferSize: 4096, format: format) { [weak self] buffer, _ in
            guard let data = Self.extractPCM(buffer) else { return }
            self?.onAudioChunk?(data)
        }

        try newEngine.start()
        engine = newEngine
    }

    func stop() {
        teardown()
        try? AVAudioSession.sharedInstance().setActive(false,
             options: .notifyOthersOnDeactivation)
    }

    private func teardown() {
        guard let e = engine else { return }
        e.inputNode.removeTap(onBus: 0)
        e.stop()
        engine = nil
    }

    // Extracts mono Float32 PCM samples as raw Data
    private static func extractPCM(_ buffer: AVAudioPCMBuffer) -> Data? {
        guard let channelData = buffer.floatChannelData else { return nil }
        let frameCount = Int(buffer.frameLength)
        return Data(bytes: channelData[0], count: frameCount * MemoryLayout<Float>.size)
    }
}
