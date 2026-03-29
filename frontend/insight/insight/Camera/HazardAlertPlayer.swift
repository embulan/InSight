import AVFoundation

/// Plays a synthesized audio alert whose urgency matches the hazard priority
/// reported by the backend.  All tones are generated from raw PCM — no
/// bundled audio files are required.
///
/// Priority mapping:
///   critical  → 3 rapid beeps at 880 Hz  (stops immediately for any new call)
///   high      → 2 beeps at 660 Hz
///   medium    → 1 soft chime at 440 Hz
///   ambient   → silent (no tone)
final class HazardAlertPlayer: NSObject {

    private var player: AVAudioPlayer?

    // MARK: - Public

    func play(level: String) {
        let tone = toneData(for: level)
        guard let tone else { return }
        DispatchQueue.main.async { [weak self] in
            self?._play(tone)
        }
    }

    func stop() {
        DispatchQueue.main.async { [weak self] in
            self?.player?.stop()
            self?.player = nil
        }
    }

    // MARK: - Playback

    private func _play(_ wavData: Data) {
        player?.stop()
        do {
            let session = AVAudioSession.sharedInstance()
            if session.category != .playAndRecord {
                try session.setCategory(.playback, mode: .default)
            }
            try session.setActive(true)
            player = try AVAudioPlayer(data: wavData, fileTypeHint: AVFileType.wav.rawValue)
            player?.delegate = self
            player?.prepareToPlay()
            player?.play()
        } catch {
            print("HazardAlertPlayer error: \(error)")
        }
    }

    // MARK: - Tone synthesis

    /// Returns a WAV-encoded Data for the given priority level, or nil for ambient.
    private func toneData(for level: String) -> Data? {
        switch level {
        case "critical":
            // 3 short bursts at 880 Hz — loud, urgent
            return beepSequence(hz: 880, count: 3, onMs: 100, offMs: 60, amplitude: 0.9)
        case "high":
            // 2 medium beeps at 660 Hz
            return beepSequence(hz: 660, count: 2, onMs: 140, offMs: 70, amplitude: 0.7)
        case "medium":
            // 1 soft chime at 440 Hz with a quick fade-out
            return beepSequence(hz: 440, count: 1, onMs: 280, offMs: 0,  amplitude: 0.5)
        default:
            return nil
        }
    }

    /// Builds a WAV Data containing `count` beeps of the given frequency.
    /// Each beep has a linear fade-in/fade-out envelope.
    private func beepSequence(
        hz: Double,
        count: Int,
        onMs: Int,
        offMs: Int,
        amplitude: Float
    ) -> Data {
        let sampleRate: Double = 44100
        let onSamples  = Int(Double(onMs)  / 1000 * sampleRate)
        let offSamples = Int(Double(offMs) / 1000 * sampleRate)
        let totalSamples = count * (onSamples + offSamples)

        var samples = [Int16](repeating: 0, count: totalSamples)
        let ramp = min(onSamples / 4, 441)   // ~10 ms fade-in / fade-out

        for b in 0 ..< count {
            let offset = b * (onSamples + offSamples)
            for i in 0 ..< onSamples {
                let t = Double(offset + i) / sampleRate
                var env: Double = 1.0
                if i < ramp {
                    env = Double(i) / Double(ramp)
                } else if i > onSamples - ramp {
                    env = Double(onSamples - i) / Double(ramp)
                }
                let sample = Double(amplitude) * env * sin(2 * .pi * hz * t)
                samples[offset + i] = Int16(sample * Double(Int16.max))
            }
        }

        return wavData(from: samples, sampleRate: Int32(sampleRate))
    }

    /// Wraps a flat array of 16-bit mono PCM samples in a minimal WAV header.
    private func wavData(from samples: [Int16], sampleRate: Int32) -> Data {
        let numChannels: Int16 = 1
        let bitsPerSample: Int16 = 16
        let byteRate = sampleRate * Int32(numChannels) * Int32(bitsPerSample / 8)
        let blockAlign = numChannels * (bitsPerSample / 8)
        let dataSize = Int32(samples.count) * Int32(bitsPerSample / 8)
        let chunkSize = 36 + dataSize

        var header = Data()
        func append(_ string: String) { header.append(contentsOf: string.utf8) }
        func append<T: FixedWidthInteger>(_ v: T) {
            withUnsafeBytes(of: v.littleEndian) { header.append(contentsOf: $0) }
        }

        append("RIFF");  append(chunkSize);   append("WAVE")
        append("fmt ");  append(Int32(16))
        append(Int16(1));       // PCM
        append(numChannels)
        append(sampleRate)
        append(byteRate)
        append(blockAlign)
        append(bitsPerSample)
        append("data");  append(dataSize)

        var sampleData = Data(capacity: samples.count * 2)
        for s in samples {
            withUnsafeBytes(of: s.littleEndian) { sampleData.append(contentsOf: $0) }
        }

        return header + sampleData
    }
}

extension HazardAlertPlayer: AVAudioPlayerDelegate {
    func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully _: Bool) {
        self.player = nil
    }
}
