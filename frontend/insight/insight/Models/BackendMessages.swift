import Foundation

// MARK: - Incoming

struct IncomingBackendEvent: Decodable {
    let type: String
    let message: String?
    let data: String?
}

// MARK: - Outgoing

struct OutgoingFrameMessage: Encodable {
    let type: String
    let timestampMs: Int64
    let jpegBase64: String
}

/// Sent once when the user submits a voice prompt (swipe down).
/// Signals the backend to process everything received so far.
struct OutgoingSubmitMessage: Encodable {
    let type = "submit"
    let timestampMs: Int64
}

/// Sent once when the user submits a recorded prompt.
/// `pcmBase64` is mono Float32 PCM, little-endian.
struct OutgoingAudioMessage: Encodable {
    let type = "audio"
    let timestampMs: Int64
    let sampleRate: Int
    let pcmBase64: String
}
