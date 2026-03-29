import SwiftUI

struct WatchRootView: View {
    @StateObject private var session = WatchSessionManager.shared
    private let router = GestureRouter()

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            content
                .transition(.opacity.animation(.easeInOut(duration: 0.25)))
                .id(stateKey)
        }
        .contentShape(Rectangle())
        // highPriorityGesture ensures swipes are captured even when a
        // child ScrollView (caption text) is on screen.
        .highPriorityGesture(
            DragGesture(minimumDistance: 30)
                .onEnded { value in
                    let dx = value.translation.width
                    let dy = value.translation.height
                    guard abs(dy) > abs(dx) else { return }
                    if dy < -30 {
                        router.handleSwipeUp(currentState: session.displayState)
                    } else if dy > 30 {
                        router.handleSwipeDown(currentState: session.displayState)
                    }
                }
        )
        .onTapGesture {
            router.handleTap(currentState: session.displayState)
        }
    }

    // Force a view swap (and re-run onAppear animations) on meaningful state changes
    private var stateKey: String {
        switch session.displayState {
        case .idle:                     return "idle"
        case .liveAssist(let phase):    return "liveAssist-\(phase.rawValue)"
        case .walking:                  return "walking"
        }
    }

    @ViewBuilder
    private var content: some View {
        switch session.displayState {
        case .idle:
            IdleView(isPhoneReachable: session.isPhoneReachable)
        case .liveAssist(let phase):
            switch phase {
            case .observing:  ObservingView(caption: session.latestCaption)
            case .listening:  ListeningView()
            case .processing: ProcessingView()
            }
        case .walking(let destination, let etaText):
            WalkingView(destination: destination, etaText: etaText)
        }
    }
}

// MARK: - Idle

private struct IdleView: View {
    let isPhoneReachable: Bool

    var body: some View {
        VStack(spacing: 0) {
            Spacer()

            ZStack {
                Circle()
                    .stroke(Color.white.opacity(0.18), lineWidth: 1.5)
                    .frame(width: 64, height: 64)
                Image(systemName: "eye")
                    .font(.system(size: 28, weight: .ultraLight))
                    .foregroundStyle(.white)
            }

            Spacer().frame(height: 10)

            Text("InSight")
                .font(.system(size: 19, weight: .bold, design: .rounded))
                .foregroundStyle(.white)

            Spacer()

            if isPhoneReachable {
                Text("Tap to begin")
                    .font(.system(size: 11))
                    .foregroundStyle(.gray)
            } else {
                Label("No iPhone", systemImage: "iphone.slash")
                    .font(.system(size: 10))
                    .foregroundStyle(.orange.opacity(0.8))
            }
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 4)
    }
}

// MARK: - Observing

private struct ObservingView: View {
    let caption: String

    @State private var pulseScale: CGFloat = 1.0

    var body: some View {
        VStack(spacing: 0) {
            if caption.isEmpty {
                // No caption yet — show animated eye
                Spacer()
                ZStack {
                    Circle()
                        .stroke(Color.white.opacity(0.12), lineWidth: 1)
                        .frame(width: 60 * pulseScale, height: 60 * pulseScale)
                    Circle()
                        .stroke(Color.white.opacity(0.25), lineWidth: 1.5)
                        .frame(width: 44, height: 44)
                    Image(systemName: "eye.fill")
                        .font(.system(size: 18))
                        .foregroundStyle(.white.opacity(0.75))
                }
                .onAppear {
                    withAnimation(.easeInOut(duration: 1.8).repeatForever(autoreverses: true)) {
                        pulseScale = 1.25
                    }
                }
                Spacer().frame(height: 8)
                Text("Observing")
                    .font(.system(size: 12))
                    .foregroundStyle(.gray)
                Spacer()
            } else {
                // Caption — fixed, no scroll, truncates to 4 lines max
                Text(caption)
                    .font(.system(size: 12, weight: .regular))
                    .foregroundStyle(.white)
                    .multilineTextAlignment(.leading)
                    .lineLimit(4)
                    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
                    .padding(.horizontal, 6)
                    .padding(.top, 6)
            }

            // Hint row
            HStack {
                Label("ask", systemImage: "mic")
                    .font(.system(size: 9))
                    .foregroundStyle(.gray.opacity(0.7))
                Spacer()
                Text("tap to stop")
                    .font(.system(size: 9))
                    .foregroundStyle(.gray.opacity(0.5))
            }
            .padding(.horizontal, 6)
            .padding(.bottom, 2)
        }
        .padding(.top, 6)
    }
}

// MARK: - Listening

private struct ListeningView: View {
    @State private var expanding = false

    var body: some View {
        VStack(spacing: 0) {
            Spacer()

            ZStack {
                // Three expanding rings
                ForEach(0..<3, id: \.self) { i in
                    Circle()
                        .stroke(
                            Color.red.opacity(expanding ? (0.45 - Double(i) * 0.12) : 0.08),
                            lineWidth: 1.5
                        )
                        .frame(
                            width:  CGFloat(40 + i * 18),
                            height: CGFloat(40 + i * 18)
                        )
                        .scaleEffect(expanding ? (1.0 + CGFloat(i) * 0.1) : 1.0)
                        .animation(
                            .easeInOut(duration: 1.0)
                                .repeatForever(autoreverses: true)
                                .delay(Double(i) * 0.18),
                            value: expanding
                        )
                }

                Circle()
                    .fill(Color.red.opacity(0.15))
                    .frame(width: 42, height: 42)

                Image(systemName: "mic.fill")
                    .font(.system(size: 19))
                    .foregroundStyle(.red)
            }
            .onAppear { expanding = true }

            Spacer().frame(height: 10)

            Text("Listening...")
                .font(.system(size: 15, weight: .semibold))
                .foregroundStyle(.white)

            Spacer()

            HStack {
                Label("send", systemImage: "arrow.down")
                    .font(.system(size: 9))
                    .foregroundStyle(.gray.opacity(0.7))
                Spacer()
                Text("tap to stop")
                    .font(.system(size: 9))
                    .foregroundStyle(.gray.opacity(0.5))
            }
            .padding(.horizontal, 6)
            .padding(.bottom, 2)
        }
        .padding(.top, 6)
    }
}

// MARK: - Processing

private struct ProcessingView: View {
    var body: some View {
        VStack(spacing: 12) {
            Spacer()
            ProgressView()
                .progressViewStyle(.circular)
                .tint(.white)
                .scaleEffect(1.3)
            Text("Thinking...")
                .font(.system(size: 13, weight: .medium))
                .foregroundStyle(.white.opacity(0.8))
            Spacer()
        }
    }
}

// MARK: - Walking

private struct WalkingView: View {
    let destination: String
    let etaText: String

    var body: some View {
        VStack(spacing: 6) {
            Spacer()
            Image(systemName: "figure.walk")
                .font(.system(size: 26))
                .foregroundStyle(.white)
            Text(destination)
                .font(.system(size: 13, weight: .semibold))
                .foregroundStyle(.white)
                .multilineTextAlignment(.center)
            Text(etaText)
                .font(.caption2)
                .foregroundStyle(.gray)
            Spacer()
            Text("Tap to stop")
                .font(.system(size: 9))
                .foregroundStyle(.gray.opacity(0.5))
                .padding(.bottom, 2)
        }
        .padding(.horizontal, 8)
    }
}
