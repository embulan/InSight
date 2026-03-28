import Foundation

enum WatchCommand: String, Codable {
    case startAssist
    case stopAssist
    case enterQueryMode
    case submitRequest
}
