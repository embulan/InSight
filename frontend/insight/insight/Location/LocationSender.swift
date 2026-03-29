import CoreLocation
import Foundation

/// Pushes GPS fixes to the backend over the WebSocket for walking navigation.
final class LocationSender: NSObject, CLLocationManagerDelegate {
    private let manager = CLLocationManager()
    private weak var socket: BackendSocketClient?

    func attach(client: BackendSocketClient) {
        socket = client
    }

    func start() {
        manager.delegate = self
        manager.desiredAccuracy = kCLLocationAccuracyNearestTenMeters
        manager.distanceFilter = 8
        manager.requestWhenInUseAuthorization()
        manager.startUpdatingLocation()
        if Config.verboseLogging { print("LocationSender: started") }
    }

    func stop() {
        manager.stopUpdatingLocation()
        if Config.verboseLogging { print("LocationSender: stopped") }
    }

    func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        guard let loc = locations.last else { return }
        guard loc.horizontalAccuracy > 0, loc.horizontalAccuracy < 150 else { return }
        socket?.sendLocation(lat: loc.coordinate.latitude, lon: loc.coordinate.longitude)
    }

    func locationManager(_ manager: CLLocationManager, didFailWithError error: Error) {
        if Config.verboseLogging { print("LocationSender error:", error.localizedDescription) }
    }
}
