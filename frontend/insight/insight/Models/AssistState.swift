import Foundation

enum AssistState: Equatable {
    case idle
    case liveAssist(phase: AssistPhase = .observing)
    case walking(destination: String, etaText: String)
}

enum AssistPhase: String, Equatable {
    case observing
    case listening
    case processing
}
