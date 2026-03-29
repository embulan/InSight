import SwiftUI

struct PhoneDebugView: View {
    @EnvironmentObject var coordinator: AppCoordinator

    @State private var showGestureHint = false

    private var isStreaming: Bool {
        if case .liveAssist(_) = coordinator.state {
            return true
        }
        return false
    }
    private var isActive: Bool {
        switch coordinator.state {
        case .idle: return false
        default:    return true
        }
    }

    var body: some View {
        ZStack {
            // Camera preview — fills screen when streaming
            if isActive {
                CameraPreviewView(session: coordinator.cameraSession)
                    .ignoresSafeArea()
                    .transition(.opacity)
            } else {
                Color(uiColor: .systemBackground)
                    .ignoresSafeArea()
            }

            // Gesture hint — appears briefly on entering liveAssist
            if showGestureHint {
                VStack(spacing: 12) {
                    HintRow(symbol: "chevron.up",   label: "Swipe up to record")
                    HintRow(symbol: "chevron.down", label: "Swipe down to submit")
                }
                .padding(24)
                .background(.ultraThinMaterial)
                .clipShape(RoundedRectangle(cornerRadius: 18))
                .transition(.opacity.combined(with: .scale))
            }

            if let phase = liveAssistPhaseOverlay {
                LiveAssistPhaseOverlay(phase: phase)
                    .padding(.top, 96)
                    .frame(maxHeight: .infinity, alignment: .top)
                    .transition(.opacity.combined(with: .scale))
            }

            // Walking navigation — current step (from server `nav_step`)
            if isActive && !coordinator.navStepLine.isEmpty {
                VStack {
                    HStack(spacing: 10) {
                        Image(systemName: "location.north.line.fill")
                            .font(.body.weight(.semibold))
                            .foregroundStyle(.cyan.opacity(0.95))
                        Text(coordinator.navStepLine)
                            .font(.subheadline.weight(.semibold))
                            .foregroundStyle(.white)
                            .multilineTextAlignment(.leading)
                        Spacer(minLength: 0)
                    }
                    .padding(.horizontal, 14)
                    .padding(.vertical, 10)
                    .background(.black.opacity(0.65))
                    .clipShape(RoundedRectangle(cornerRadius: 12))
                    .padding(.horizontal, 20)
                    .padding(.top, coordinator.latestCaption.isEmpty ? 120 : 56)
                    Spacer()
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
                .transition(.opacity.combined(with: .move(edge: .top)))
            }

            // Caption overlay — appears when backend sends a description
            if isActive && !coordinator.latestCaption.isEmpty {
                VStack {
                    Spacer()
                    Text(coordinator.latestCaption)
                        .font(.body.weight(.medium))
                        .foregroundStyle(.white)
                        .multilineTextAlignment(.leading)
                        .padding(.horizontal, 16)
                        .padding(.vertical, 12)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(.black.opacity(0.65))
                        .clipShape(RoundedRectangle(cornerRadius: 14))
                        .padding(.horizontal, 20)
                        .padding(.bottom, 148)
                        .transition(.opacity.combined(with: .move(edge: .bottom)))
                }
            }

            // Bottom overlay — status + button
            VStack {
                Spacer()

                VStack(spacing: 20) {
                    HStack(spacing: 8) {
                        Circle()
                            .fill(statusColor)
                            .frame(width: 10, height: 10)
                        Text(stateText)
                            .font(.title3.weight(.semibold))
                            .foregroundStyle(isActive ? .white : .primary)
                    }
                    .padding(.horizontal, 20)
                    .padding(.vertical, 10)
                    .background(.ultraThinMaterial)
                    .clipShape(Capsule())

                    Button {
                        withAnimation(.easeInOut(duration: 0.3)) {
                            if isActive {
                                coordinator.stopStreaming()
                            } else {
                                coordinator.startStreaming()
                            }
                        }
                    } label: {
                        Label(
                            isActive ? "Stop Streaming" : "Start Streaming",
                            systemImage: isActive ? "stop.circle.fill" : "play.circle.fill"
                        )
                        .font(.headline)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 16)
                        .background(isActive ? Color.red.opacity(0.85) : Color.green.opacity(0.85))
                        .foregroundStyle(.white)
                        .clipShape(RoundedRectangle(cornerRadius: 16))
                    }
                    .padding(.horizontal, 32)
                }
                .padding(.bottom, 48)
            }
        }
        .gesture(
            DragGesture(minimumDistance: 40)
                .onEnded { value in
                    guard isActive else { return }
                    if value.translation.height < -40 {
                        withAnimation { coordinator.enterQueryMode() }
                    } else if value.translation.height > 40 {
                        withAnimation { coordinator.submitRequest() }
                    }
                }
        )
        .animation(.easeInOut(duration: 0.3), value: coordinator.state)
        .animation(.easeInOut(duration: 0.4), value: coordinator.latestCaption)
        .animation(.easeInOut(duration: 0.3), value: coordinator.navStepLine)
        .onChange(of: isStreaming) { streaming in
            guard streaming else { return }
            withAnimation { showGestureHint = true }
            DispatchQueue.main.asyncAfter(deadline: .now() + 2.5) {
                withAnimation { showGestureHint = false }
            }
        }
    }

    private var stateText: String {
        switch coordinator.state {
        case .idle:                                    return "Idle"
        case .liveAssist(let phase):
            switch phase {
            case .observing:  return "Live Assist"
            case .listening:  return "Live Assist • Listening"
            case .processing: return "Live Assist • Processing"
            }
        case .walking(let destination, let etaText):   return "Walking to \(destination) • \(etaText)"
        }
    }

    private var statusColor: Color {
        switch coordinator.state {
        case .idle:        return .gray
        case .liveAssist(let phase):
            switch phase {
            case .observing:  return .green
            case .listening:  return .blue
            case .processing: return .orange
            }
        case .walking:     return .cyan
        }
    }

    private var liveAssistPhaseOverlay: AssistPhase? {
        guard case .liveAssist(let phase) = coordinator.state else { return nil }
        switch phase {
        case .observing:
            return nil
        case .listening, .processing:
            return phase
        }
    }
}

private struct HintRow: View {
    let symbol: String
    let label: String

    var body: some View {
        HStack(spacing: 10) {
            Image(systemName: symbol)
                .font(.callout.weight(.semibold))
                .foregroundStyle(.secondary)
                .frame(width: 20)
            Text(label)
                .font(.subheadline)
                .foregroundStyle(.primary)
        }
    }
}

private struct LiveAssistPhaseOverlay: View {
    let phase: AssistPhase

    var body: some View {
        HStack(spacing: 10) {
            if phase == .processing {
                ProgressView()
                    .tint(.white)
            } else {
                Image(systemName: "mic.fill")
                    .font(.headline)
            }

            Text(title)
                .font(.headline)
        }
        .padding(.horizontal, 18)
        .padding(.vertical, 12)
        .background(.ultraThinMaterial)
        .clipShape(Capsule())
        .foregroundStyle(.white)
    }

    private var title: String {
        switch phase {
        case .observing:
            return "Live Assist"
        case .listening:
            return "Listening..."
        case .processing:
            return "Processing..."
        }
    }
}
