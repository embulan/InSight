import AVFoundation

/// Plays MP3 audio received as raw Data from the backend.
/// Keeps a reference to the player so it is not released mid-playback.
final class AudioPlayerManager: NSObject {

    private var player: AVAudioPlayer?

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
            self?.player?.stop()
            self?.player = nil
            if Config.verboseLogging { print("AudioPlayer: playback stopped") }
        }
    }

    private func _play(_ data: Data) {
        do {
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
