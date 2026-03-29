import Foundation

final class GestureRouter {

    // Tap: toggle start / stop Live Assist
    func handleTap(currentState: WatchDisplayState) {
        switch currentState {
        case .idle:
            WatchSessionManager.shared.sendCommand(.startAssist)
        default:
            WatchSessionManager.shared.sendCommand(.stopAssist)
        }
    }

    // Swipe up:
    //   observing → enter query (start mic)
    //   all other states → no-op
    func handleSwipeUp(currentState: WatchDisplayState) {
        guard case .liveAssist(let phase) = currentState, phase == .observing else { return }
        WatchSessionManager.shared.sendCommand(.enterQueryMode)
    }

    // Swipe down:
    //   listening → submit (send recording)
    //   all other states → no-op
    func handleSwipeDown(currentState: WatchDisplayState) {
        guard case .liveAssist(let phase) = currentState, phase == .listening else { return }
        WatchSessionManager.shared.sendCommand(.submitRequest)
    }
}
