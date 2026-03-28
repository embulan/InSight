import SwiftUI
import WatchConnectivity

struct TestPanelView: View {
    @ObservedObject var session: WatchSessionManager
    @Environment(\.dismiss) private var dismiss

    @State private var lastSent: String = "—"
    @State private var isReachable: Bool = false

    private let commands: [(label: String, command: WatchCommand, icon: String)] = [
        ("Start Assist",   .startAssist,   "video.fill"),
        ("Stop Assist",    .stopAssist,    "video.slash.fill"),
        ("Query Mode",     .enterQueryMode, "mic.fill"),
        ("Submit Request", .submitRequest,  "paperplane.fill"),
    ]

    var body: some View {
        ScrollView {
            VStack(spacing: 10) {

                Text("Test Panel")
                    .font(.headline)
                    .padding(.top, 4)

                HStack(spacing: 6) {
                    Circle()
                        .fill(isReachable ? Color.green : Color.red)
                        .frame(width: 8, height: 8)
                    Text(isReachable ? "Phone reachable" : "Phone not reachable")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }

                Divider()

                ForEach(commands, id: \.label) { item in
                    Button {
                        sendCommand(item.command, label: item.label)
                    } label: {
                        Label(item.label, systemImage: item.icon)
                            .font(.caption)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    .buttonStyle(.bordered)
                    .tint(lastSent == item.label ? .green : .blue)
                }

                Divider()

                VStack(spacing: 2) {
                    Text("Last sent")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    Text(lastSent)
                        .font(.caption)
                        .foregroundStyle(.primary)
                }

                VStack(spacing: 2) {
                    Text("Display state")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    Text(stateLabel)
                        .font(.caption)
                        .foregroundStyle(.primary)
                }

                Button("Done") { dismiss() }
                    .font(.caption)
                    .padding(.top, 4)
            }
            .padding(.horizontal, 8)
        }
        .onAppear { refreshReachability() }
    }

    private var stateLabel: String {
        switch session.displayState {
        case .idle:                              return "idle"
        case .liveAssist(let phase):            return "liveAssist[\(phase.rawValue)]"
        case .walking(let dest, let eta):        return "walking → \(dest) (\(eta))"
        }
    }

    private func sendCommand(_ command: WatchCommand, label: String) {
        refreshReachability()
        session.sendCommand(command)
        lastSent = label
    }

    private func refreshReachability() {
        isReachable = WCSession.default.isReachable
    }
}
