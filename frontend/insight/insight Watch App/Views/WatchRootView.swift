import SwiftUI

struct WatchRootView: View {
    @StateObject private var session = WatchSessionManager.shared
    private let router = GestureRouter()

    @State private var showTestPanel = false

    var body: some View {
        ZStack(alignment: .bottomTrailing) {
            Color.black.ignoresSafeArea()

            content
                .padding()

            Button {
                showTestPanel = true
            } label: {
                Image(systemName: "wrench.and.screwdriver.fill")
                    .font(.system(size: 12))
                    .foregroundStyle(.gray)
                    .padding(6)
            }
            .buttonStyle(.plain)
        }
        .contentShape(Rectangle())
        .onAppear {
            session.start()
        }
        .onTapGesture {
            router.handleTap(currentState: session.displayState)
        }
        .gesture(
            DragGesture(minimumDistance: 20)
                .onEnded { value in
                    if value.translation.height < -20 {
                        router.handleSwipeUp()
                    } else if value.translation.height > 20 {
                        router.handleSwipeDown()
                    }
                }
        )
        .sheet(isPresented: $showTestPanel) {
            TestPanelView(session: session)
        }
    }

    @ViewBuilder
    private var content: some View {
        switch session.displayState {
        case .idle:
            VStack(spacing: 6) {
                Text("👁️")
                    .font(.system(size: 36))

                Text("insight")
                    .font(.system(size: 20, weight: .semibold, design: .rounded))
                    .foregroundStyle(.white)

                Text("Tap to start")
                    .font(.caption2)
                    .foregroundStyle(.gray)
            }

        case .liveAssist(let phase):
            VStack(spacing: 8) {
                Image(systemName: "video.fill")
                    .font(.system(size: 28))
                Text("Live Assist On")
                    .font(.headline)
                phaseLabel(phase)
            }
            .foregroundStyle(.white)

        case .walking(let destination, let etaText):
            VStack(spacing: 6) {
                Text("Walking Mode")
                    .font(.caption2)
                    .foregroundStyle(.gray)

                Image(systemName: "figure.walk")
                    .font(.system(size: 26))

                Text(destination)
                    .font(.headline)
                    .multilineTextAlignment(.center)

                Text(etaText)
                    .font(.caption2)
                    .foregroundStyle(.gray)
            }
            .foregroundStyle(.white)
        }
    }

    @ViewBuilder
    private func phaseLabel(_ phase: WatchAssistPhase) -> some View {
        switch phase {
        case .observing:
            Text("Connected")
                .font(.caption2)
                .foregroundStyle(.gray)
        case .listening:
            Label("Listening...", systemImage: "mic.fill")
                .font(.caption2)
                .foregroundStyle(.gray)
        case .processing:
            HStack(spacing: 6) {
                ProgressView()
                    .scaleEffect(0.7)
                Text("Processing...")
            }
            .font(.caption2)
            .foregroundStyle(.gray)
        }
    }
}
