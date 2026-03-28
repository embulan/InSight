import Foundation
import Combine
import WatchConnectivity

final class PhoneSessionManager: NSObject, ObservableObject {
    static let shared = PhoneSessionManager()

    var onCommandReceived: ((AssistCommand) -> Void)?

    private let session = WCSession.default

    override private init() {
        super.init()

        if WCSession.isSupported() {
            session.delegate = self
            session.activate()
        }
    }

    func sendWatchStatus(
        mode: String,
        phase: String? = nil,
        destination: String? = nil,
        etaText: String? = nil
    ) {
        guard session.isReachable else {
            print("Phone session not reachable")
            return
        }

        var payload: [String: Any] = ["mode": mode]
        if let phase { payload["phase"] = phase }
        if let destination { payload["destination"] = destination }
        if let etaText { payload["etaText"] = etaText }

        session.sendMessage(
            payload,
            replyHandler: nil,
            errorHandler: { error in
                print("Phone send status error: \(error)")
            }
        )
    }
}

extension PhoneSessionManager: WCSessionDelegate {
    func session(
        _ session: WCSession,
        activationDidCompleteWith activationState: WCSessionActivationState,
        error: Error?
    ) {
        if let error = error {
            print("Phone activation error: \(error)")
        }
    }

    func sessionDidBecomeInactive(_ session: WCSession) {}

    func sessionDidDeactivate(_ session: WCSession) {
        session.activate()
    }

    func session(_ session: WCSession, didReceiveMessage message: [String: Any]) {
        guard
            let raw = message["command"] as? String,
            let command = AssistCommand(rawValue: raw)
        else { return }

        DispatchQueue.main.async {
            self.onCommandReceived?(command)
        }
    }
}
