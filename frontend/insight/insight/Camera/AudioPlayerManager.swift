import AVFoundation

/// Plays MP3 audio received as raw Data from the backend.
/// Falls back to on-device speech when the backend does not provide audio.
final class AudioPlayerManager: NSObject {

    private var player: AVAudioPlayer?
    private let speechSynthesizer = AVSpeechSynthesizer()
    private var didFinishFromStop = false

    override init() {
        super.init()
        speechSynthesizer.delegate = self
    }

    /// Called on the main queue when the current clip finishes naturally
    /// (not when stopped early via `stop()`).
    var onPlaybackFinished: (() -> Void)?

    /// Play raw MP3 bytes.  Safe to call from any thread.
    func play(mp3Data: Data) {
        DispatchQueue.main.async { [weak self] in
            self?._play(mp3Data)
        }
    }

    /// Stop playback immediately.  Safe to call from any thread.
    /// Does NOT fire onPlaybackFinished — use that only for natural completion.
    func stop() {
        DispatchQueue.main.async { [weak self] in
            self?.didFinishFromStop = true
            self?.player?.stop()
            self?.player = nil
            self?.speechSynthesizer.stopSpeaking(at: .immediate)
            if Config.verboseLogging { print("AudioPlayer: playback stopped") }
        }
    }

    /// Speak plain text on-device when no prerecorded audio is available.
    func speak(text: String) {
        DispatchQueue.main.async { [weak self] in
            self?._speak(text)
        }
    }

    private func _play(_ data: Data) {
        do {
            didFinishFromStop = false
            speechSynthesizer.stopSpeaking(at: .immediate)
            let session = AVAudioSession.sharedInstance()
            // Keep recording category if the mic is open; otherwise switch to playback.
            if session.category != .playAndRecord {
                try session.setCategory(.playback, mode: .default)
            }
            try session.setActive(true)

            player = try AVAudioPlayer(data: data, fileTypeHint: AVFileType.mp3.rawValue)
            player?.delegate = self
            player?.prepareToPlay()
            player?.play()

            if Config.verboseLogging { print("AudioPlayer: playback started (\(data.count) bytes)") }
        } catch {
            print("AudioPlayer error: \(error)")
        }
    }

    private func _speak(_ text: String) {
        let cleanedText = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !cleanedText.isEmpty else { return }

        do {
            didFinishFromStop = false
            player?.stop()
            player = nil

            let session = AVAudioSession.sharedInstance()
            if session.category != .playAndRecord {
                try session.setCategory(.playback, mode: .spokenAudio)
            }
            try session.setActive(true)

            let utterance = AVSpeechUtterance(string: cleanedText)
            utterance.voice = AVSpeechSynthesisVoice(language: "en-US")
            utterance.rate = 0.5
            utterance.prefersAssistiveTechnologySettings = true

            speechSynthesizer.stopSpeaking(at: .immediate)
            speechSynthesizer.speak(utterance)

            if Config.verboseLogging { print("AudioPlayer: local speech started") }
        } catch {
            print("AudioPlayer speech error: \(error)")
        }
    }
}

extension AudioPlayerManager: AVAudioPlayerDelegate {
    func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        if Config.verboseLogging { print("AudioPlayer: playback finished (success: \(flag))") }
        self.player = nil
        onPlaybackFinished?()
    }

    func audioPlayerDecodeErrorDidOccur(_ player: AVAudioPlayer, error: Error?) {
        print("AudioPlayer decode error: \(error?.localizedDescription ?? "unknown")")
    }
}

extension AudioPlayerManager: AVSpeechSynthesizerDelegate {
    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        if Config.verboseLogging { print("AudioPlayer: local speech finished") }
        guard !didFinishFromStop else {
            didFinishFromStop = false
            return
        }
        onPlaybackFinished?()
    }

    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didCancel utterance: AVSpeechUtterance) {
        didFinishFromStop = false
    }
}
