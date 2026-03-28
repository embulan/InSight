import Foundation
import QuartzCore

final class FrameSampler {
    private let minInterval: TimeInterval
    private var lastSentTime: CFTimeInterval = 0

    init(secondsBetweenFrames: TimeInterval = 3.0) {
        self.minInterval = secondsBetweenFrames
    }

    func shouldSendFrame() -> Bool {
        let now = CACurrentMediaTime()

        guard now - lastSentTime >= minInterval else {
            return false
        }

        lastSentTime = now
        return true
    }
}
