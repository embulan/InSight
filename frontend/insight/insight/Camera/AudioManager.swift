import AVFoundation

final class AudioManager {

    // A new engine instance is created on every start() — guarantees no stale tap
    private var engine: AVAudioEngine?
    private(set) var sampleRate: Double = 44100

    /// Called on a background thread with each raw mono Float32 PCM chunk
    var onAudioChunk: ((Data) -> Void)?

    func start() throws {
        // Tear down any previous engine before creating a new one
        teardown()

        // Configure the shared audio session for recording
        let audioSession = AVAudioSession.sharedInstance()
        try audioSession.setCategory(.playAndRecord, mode: .default,
                                     options: [.defaultToSpeaker, .allowBluetoothHFP])
        try audioSession.setActive(true)

        // Fresh engine — inputNode has no tap by definition
        let newEngine = AVAudioEngine()
        let input = newEngine.inputNode
        let format = input.outputFormat(forBus: 0)
        sampleRate = format.sampleRate

        input.installTap(onBus: 0, bufferSize: 4096, format: format) { [weak self] buffer, _ in
            guard let data = Self.extractMonoPCM(buffer) else { return }
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

    // Downmixes any available channels to a mono Float32 PCM payload.
    private static func extractMonoPCM(_ buffer: AVAudioPCMBuffer) -> Data? {
        guard let channelData = buffer.floatChannelData else { return nil }
        let frameCount = Int(buffer.frameLength)
        let channelCount = Int(buffer.format.channelCount)

        guard channelCount > 0 else { return nil }
        guard channelCount > 1 else {
            return Data(bytes: channelData[0], count: frameCount * MemoryLayout<Float>.size)
        }

        var monoSamples = Array(repeating: Float.zero, count: frameCount)
        let scale = 1.0 / Float(channelCount)

        for channel in 0..<channelCount {
            let samples = channelData[channel]
            for frame in 0..<frameCount {
                monoSamples[frame] += samples[frame] * scale
            }
        }

        return monoSamples.withUnsafeBytes { Data($0) }
    }
}
