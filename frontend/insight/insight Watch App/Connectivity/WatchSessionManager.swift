import Foundation
import Combine
import WatchConnectivity

final class WatchSessionManager: NSObject, ObservableObject {
    static let shared = WatchSessionManager()

    @Published var displayState: WatchDisplayState = .idle

    private let session = WCSession.default

    override private init() {
        super.init()
    }

    func start() {
        guard WCSession.isSupported() else { return }
        session.delegate = self
        session.activate()
    }

    func sendCommand(_ command: WatchCommand) {
        guard session.isReachable else {
            print("Watch session not reachable")
            return
        }

        session.sendMessage(
            ["command": command.rawValue],
            replyHandler: nil,
            errorHandler: { error in
                print("Watch send error: \(error)")
            }
        )
    }

    private func applyStatus(mode: String, phase: String?, destination: String?, etaText: String?) {
        switch mode {
        case "idle":
            displayState = .idle
        case "liveAssist":
            displayState = .liveAssist(
                phase: WatchAssistPhase(rawValue: phase ?? "") ?? .observing
            )
        case "walking":
            displayState = .walking(
                destination: destination ?? "Destination",
                etaText: etaText ?? "--"
            )
        default:
            displayState = .idle
        }
    }
}

extension WatchSessionManager: WCSessionDelegate {
    func session(
        _ session: WCSession,
        activationDidCompleteWith activationState: WCSessionActivationState,
        error: Error?
    ) {
        if let error = error {
            print("Watch activation error: \(error)")
        }
    }

    func sessionReachabilityDidChange(_ session: WCSession) {}

    func session(_ session: WCSession, didReceiveMessage message: [String: Any]) {
        let mode = message["mode"] as? String ?? "idle"
        let phase = message["phase"] as? String
        let destination = message["destination"] as? String
        let etaText = message["etaText"] as? String

        DispatchQueue.main.async {
            self.applyStatus(mode: mode, phase: phase, destination: destination, etaText: etaText)
        }
    }
}
