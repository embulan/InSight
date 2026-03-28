import Foundation

enum Config {

    // -------------------------------------------------------------------------
    // MARK: Backend WebSocket
    // -------------------------------------------------------------------------

    /// The WebSocket endpoint frames are streamed to.
    /// Swap this out when you move from test → staging → production.
    ///
    /// Public echo server (test): "wss://echo.websocket.org"
    /// Local dev server:          "ws://localhost:8000/ws"
    /// Production:                "wss://api.yourapp.com/ws"
    static let backendWebSocketURL = URL(string: "wss://echo.websocket.org")!

    /// Set this when the backend requires a named WebSocket subprotocol.
    static let backendWebSocketSubprotocol: String? = nil

    // -------------------------------------------------------------------------
    // MARK: Camera
    // -------------------------------------------------------------------------

    /// Frames per second sent to the backend (lower = less bandwidth)
    static let streamingFPS: Double = 2.0

    // -------------------------------------------------------------------------
    // MARK: Feature flags
    // -------------------------------------------------------------------------

    /// Enables Apple Watch messaging. Disable for phone-only MVP work.
    static let watchConnectivityEnabled: Bool = false

    /// When true, extra logs are printed for WebSocket and camera events
    static let verboseLogging: Bool = true
}
