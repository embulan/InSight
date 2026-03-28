import Foundation

enum WatchDisplayState: Equatable {
    case idle
    case liveAssist(phase: WatchAssistPhase = .observing)
    case walking(destination: String, etaText: String)
}

enum WatchAssistPhase: String, Equatable {
    case observing
    case listening
    case processing
}
