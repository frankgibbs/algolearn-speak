import XCTest

/// Throwaway driver for the AlgolearnSpeak simulator verification: attaches to the
/// already-running app (activate, not launch) and taps the Connect/Disconnect button.
final class TapperTests: XCTestCase {
    private func app() -> XCUIApplication {
        let a = XCUIApplication(bundleIdentifier: "ai.algolearn.speak")
        a.activate()
        return a
    }

    func testTapConnect() {
        let a = app()
        let b = a.buttons["Connect"]
        XCTAssertTrue(b.waitForExistence(timeout: 10), "Connect button not found; buttons: \(a.buttons.allElementsBoundByIndex.map(\.label))")
        b.tap()
        // allow the permission alert, if iOS shows one
        let allow = XCUIApplication(bundleIdentifier: "com.apple.springboard").buttons["Allow"]
        if allow.waitForExistence(timeout: 3) { allow.tap() }
        XCTAssertTrue(a.buttons["Disconnect"].waitForExistence(timeout: 10))
        sleep(3)
        NSLog("TAPPER events: %@", a.staticTexts.allElementsBoundByIndex.map(\.label).joined(separator: " | "))
    }

    func testTapDisconnect() {
        let a = app()
        let b = a.buttons["Disconnect"]
        XCTAssertTrue(b.waitForExistence(timeout: 10))
        b.tap()
        XCTAssertTrue(a.buttons["Connect"].waitForExistence(timeout: 10))
    }

    func testDumpLabels() {
        let a = app()
        _ = a.wait(for: .runningForeground, timeout: 5)
        NSLog("TAPPER labels: %@", a.staticTexts.allElementsBoundByIndex.map(\.label).joined(separator: " | "))
    }
}
