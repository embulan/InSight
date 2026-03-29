import Foundation
import Combine
import WatchConnectivity
import WatchKit

final class WatchSessionManager: NSObject, ObservableObject {
    static let shared = WatchSessionManager()
    private static let verboseLogging = true

    @Published var displayState: WatchDisplayState = .idle
    @Published var latestCaption: String = ""
    @Published var isPhoneReachable: Bool = false

    private let session = WCSession.default
    private var hasLoggedInactiveSession = false

    override private init() {
        super.init()
    }

    func start() {
        guard WCSession.isSupported() else { return }
        session.delegate = self
        session.activate()
    }

    func sendCommand(_ command: WatchCommand) {
        guard session.activationState == .activated else {
            if Self.verboseLogging, !hasLoggedInactiveSession {
                print("[watch] WCSession not activated — command not sent")
                hasLoggedInactiveSession = true
            }
            return
        }

        WKInterfaceDevice.current().play(.click)
        let payload: [String: Any] = ["command": command.rawValue]

        if session.isReachable {
            // Phone is in foreground — deliver immediately
            session.sendMessage(payload, replyHandler: nil) { error in
                print("[watch] sendMessage failed (\(error)) — queuing via transferUserInfo")
                self.session.transferUserInfo(payload)
            }
        } else {
            // Phone is backgrounded — queue and deliver when it wakes
            session.transferUserInfo(payload)
            if Self.verboseLogging { print("[watch] phone not reachable — queued command: \(command.rawValue)") }
        }
    }

    // MARK: - Private

    private func applyStatus(
        mode: String,
        phase: String?,
        caption: String?,
        destination: String?,
        etaText: String?
    ) {
        let newState: WatchDisplayState
        switch mode {
        case "liveAssist":
            newState = .liveAssist(phase: WatchAssistPhase(rawValue: phase ?? "") ?? .observing)
        case "walking":
            newState = .walking(destination: destination ?? "Destination", etaText: etaText ?? "--")
        default:
            newState = .idle
        }

        if newState != displayState {
            playHaptic(for: newState, previous: displayState)
        }
        displayState = newState

        if let caption, !caption.isEmpty {
            latestCaption = caption
            WKInterfaceDevice.current().play(.notification)
        }
    }

    private func playHaptic(for new: WatchDisplayState, previous: WatchDisplayState) {
        switch new {
        case .idle:
            WKInterfaceDevice.current().play(.stop)
        case .liveAssist(let phase):
            switch phase {
            case .observing:
                if case .idle = previous { WKInterfaceDevice.current().play(.start) }
            case .listening:
                WKInterfaceDevice.current().play(.directionUp)
            case .processing:
                WKInterfaceDevice.current().play(.click)
            }
        case .walking:
            WKInterfaceDevice.current().play(.success)
        }
    }
}

extension WatchSessionManager: WCSessionDelegate {
    func sessionReachabilityDidChange(_ session: WCSession) {
        DispatchQueue.main.async {
            self.isPhoneReachable = session.isReachable
        }
        if Self.verboseLogging {
            print("Watch reachability changed:", session.isReachable)
        }
    }

    func session(
        _ session: WCSession,
        activationDidCompleteWith activationState: WCSessionActivationState,
        error: Error?
    ) {
        DispatchQueue.main.async {
            self.isPhoneReachable = (activationState == .activated) && session.isReachable
            self.hasLoggedInactiveSession = false
        }
        if let error { print("Watch activation error: \(error)") }
    }

    func session(_ session: WCSession, didReceiveApplicationContext applicationContext: [String : Any]) {
        apply(message: applicationContext)
    }

    func session(_ session: WCSession, didReceiveMessage message: [String: Any]) {
        apply(message: message)
    }

    private func apply(message: [String: Any]) {
        let mode        = message["mode"]        as? String
        let phase       = message["phase"]       as? String
        let caption     = message["caption"]     as? String
        let destination = message["destination"] as? String
        let etaText     = message["etaText"]     as? String

        DispatchQueue.main.async {
            self.applyStatus(
                mode: mode ?? "idle",
                phase: phase,
                caption: caption,
                destination: destination,
                etaText: etaText
            )
        }
    }
}
