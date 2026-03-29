import Foundation
import Combine
import WatchConnectivity

final class PhoneSessionManager: NSObject, ObservableObject {
    static let shared = PhoneSessionManager()

    var onCommandReceived: ((AssistCommand) -> Void)?

    private let session = WCSession.default
    private var hasLoggedInactiveSession = false

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
        caption: String? = nil,
        destination: String? = nil,
        etaText: String? = nil
    ) {
        guard session.activationState == .activated else {
            if Config.verboseLogging, !hasLoggedInactiveSession {
                print("[phone] WCSession not activated — status not sent")
                hasLoggedInactiveSession = true
            }
            return
        }

        var payload: [String: Any] = ["mode": mode]
        if let phase       { payload["phase"]       = phase }
        if let caption     { payload["caption"]     = caption }
        if let destination { payload["destination"] = destination }
        if let etaText     { payload["etaText"]     = etaText }

        if session.isReachable {
            // Watch is in foreground — deliver immediately
            session.sendMessage(payload, replyHandler: nil) { error in
                print("[phone] sendMessage error: \(error)")
                // Fall back to context so watch picks it up when it wakes
                try? self.session.updateApplicationContext(payload)
            }
        } else {
            // Watch is backgrounded — queue via application context
            // (only the latest value matters, so context is fine here)
            do {
                try session.updateApplicationContext(payload)
            } catch {
                print("[phone] updateApplicationContext error: \(error)")
            }
        }
    }

    private func dispatch(command raw: String) {
        guard let command = AssistCommand(rawValue: raw) else { return }
        DispatchQueue.main.async { self.onCommandReceived?(command) }
    }
}

extension PhoneSessionManager: WCSessionDelegate {
    func session(
        _ session: WCSession,
        activationDidCompleteWith activationState: WCSessionActivationState,
        error: Error?
    ) {
        if let error { print("[phone] activation error: \(error)"); return }
        hasLoggedInactiveSession = false
    }

    func sessionDidBecomeInactive(_ session: WCSession) {}

    func sessionDidDeactivate(_ session: WCSession) {
        session.activate()
    }

    // Real-time message (watch was in foreground when it sent)
    func session(_ session: WCSession, didReceiveMessage message: [String: Any]) {
        guard let raw = message["command"] as? String else { return }
        dispatch(command: raw)
    }

    // Queued delivery (watch used transferUserInfo fallback)
    func session(_ session: WCSession, didReceiveUserInfo userInfo: [String: Any]) {
        guard let raw = userInfo["command"] as? String else { return }
        dispatch(command: raw)
    }
}
