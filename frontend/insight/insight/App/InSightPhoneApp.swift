import SwiftUI

@main
struct InSightPhoneApp: App {
    @StateObject private var coordinator = AppCoordinator.shared

    var body: some Scene {
        WindowGroup {
            PhoneDebugView()
                .environmentObject(coordinator)
        }
    }
}
