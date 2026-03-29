import SwiftUI

@main
struct InSightWatchApp: App {
    // Start WCSession activation at launch so it's ready before the first gesture.
    @StateObject private var session = WatchSessionManager.shared

    init() {
        WatchSessionManager.shared.start()
    }

    var body: some Scene {
        WindowGroup {
            WatchRootView()
        }
    }
}
