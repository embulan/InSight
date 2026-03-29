import Foundation

enum Config {

    // -------------------------------------------------------------------------
    // MARK: Backend WebSocket
    // -------------------------------------------------------------------------

    /// The WebSocket endpoint frames are streamed to.
    ///
    /// Simulator : "ws://localhost:8000/ws"
    /// Real device: "ws://<your-mac-LAN-ip>:8000/ws"
    ///   → find your Mac's LAN IP with: System Settings → Wi-Fi → Details
    ///   → e.g. "ws://192.168.1.42:8000/ws"
    ///   Both phone and Mac must be on the same Wi-Fi network.
    /// Production: "wss://api.yourapp.com/ws"
    static let backendWebSocketURL = URL(string: "ws://10.74.219.83:8000/ws")!

    /// Set this when the backend requires a named WebSocket subprotocol.
    static let backendWebSocketSubprotocol: String? = nil

    // -------------------------------------------------------------------------
    // MARK: Frame cadence
    // -------------------------------------------------------------------------

    /// Seconds between frames sent to the backend.
    static let frameInterval: TimeInterval = 0.5

    /// How often (seconds) the server saves a frame to vlm_cache and runs sim_check.
    /// Keep this value in sync with CACHE_INTERVAL in backend/server.py.
    /// For best results set it equal to frameInterval so every received frame is evaluated.
    static let cacheInterval: TimeInterval = 0.5

    // -------------------------------------------------------------------------
    // MARK: Feature flags
    // -------------------------------------------------------------------------

    /// Enables Apple Watch messaging. Disable for phone-only MVP work.
    static let watchConnectivityEnabled: Bool = false

    /// When true, extra logs are printed for WebSocket and camera events
    static let verboseLogging: Bool = true
}
