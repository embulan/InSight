import Foundation

final class GestureRouter {
    func handleTap(currentState: WatchDisplayState) {
        switch currentState {
        case .idle:
            WatchSessionManager.shared.sendCommand(.startAssist)
        default:
            WatchSessionManager.shared.sendCommand(.stopAssist)
        }
    }

    func handleSwipeUp() {
        WatchSessionManager.shared.sendCommand(.enterQueryMode)
    }

    func handleSwipeDown() {
        WatchSessionManager.shared.sendCommand(.submitRequest)
    }
}
