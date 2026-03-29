import Foundation

struct WatchStatusPayload: Codable {
    let mode: String
    let phase: String?
    let caption: String?
    let destination: String?
    let etaText: String?
}
