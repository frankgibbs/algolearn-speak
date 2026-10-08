// ftcall: read and drive the FaceTime call banner through the Accessibility API.
//
//   ftcall state            {"state": "locked"|"none"|"loading"|"click_to_call"|"ringing"|"connected"|"unknown", "text": "..."}
//                           "locked" is read from IOConsoleLocked before any AX call: while the
//                           screen is locked, AX calls into Notification Center hang.
//   ftcall press <button>   press the banner button labelled exactly <button> (Call, Cancel, End)
//   ftcall devices          {"microphone": "...", "output": "..."} -- the checked items in FaceTime's Video menu
//
// Exit 0 on success, 2 without Accessibility permission, 3 on any other
// failure (reason on stderr). See docs/DESIGN_FACETIME_CALL.md section 4.
//
// The call banner belongs to Notification Center, not FaceTime: an AX group
// whose identifier (or description) is FACETIME_NOTIFICATION.

import AppKit
import ApplicationServices
import Foundation
import IOKit

func fail(_ message: String, code: Int32 = 3) -> Never {
    FileHandle.standardError.write((message + "\n").data(using: .utf8)!)
    exit(code)
}

func emit(_ object: [String: String]) {
    let data = try! JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
    print(String(data: data, encoding: .utf8)!)
}

func attr(_ e: AXUIElement, _ name: String) -> CFTypeRef? {
    var value: CFTypeRef?
    return AXUIElementCopyAttributeValue(e, name as CFString, &value) == .success ? value : nil
}

func text(_ e: AXUIElement, _ name: String) -> String? {
    guard let s = attr(e, name) as? String, !s.isEmpty else { return nil }
    return s
}

func children(_ e: AXUIElement) -> [AXUIElement] {
    (attr(e, kAXChildrenAttribute) as? [AXUIElement]) ?? []
}

func descendants(_ e: AXUIElement) -> [AXUIElement] {
    children(e).flatMap { [$0] + descendants($0) }
}

func appElement(_ bundleID: String) -> AXUIElement? {
    guard let app = NSRunningApplication.runningApplications(withBundleIdentifier: bundleID).first else { return nil }
    return AXUIElementCreateApplication(app.processIdentifier)
}

func banners() -> [AXUIElement] {
    guard let root = appElement("com.apple.notificationcenterui") else {
        fail("Notification Center is not running")
    }
    var found: [AXUIElement] = []
    var stack = children(root)
    while let e = stack.popLast() {
        if text(e, kAXRoleAttribute) == "AXMenuBar" { continue }
        if text(e, kAXIdentifierAttribute) == "FACETIME_NOTIFICATION" || text(e, kAXDescriptionAttribute) == "FACETIME_NOTIFICATION" {
            found.append(e)
            continue
        }
        stack.append(contentsOf: children(e))
    }
    return found
}

/// Static-text labels. The "Click to Call" banner carries them in
/// AXDescription, the in-call banner in AXValue; read both.
func bannerTexts(_ b: AXUIElement) -> [String] {
    descendants(b).filter { text($0, kAXRoleAttribute) == "AXStaticText" }
        .compactMap { text($0, kAXValueAttribute) ?? text($0, kAXDescriptionAttribute) ?? text($0, kAXTitleAttribute) }
}

func classify(_ texts: [String]) -> String {
    if texts.isEmpty { return "loading" }  // banner drawn, labels not filled in yet (seen for ~1 s after dialing)
    if texts.contains("Click to Call") { return "click_to_call" }
    if texts.contains("FaceTime Audio…") { return "ringing" }
    if texts.contains(where: { $0.range(of: #"^FaceTime Audio - \d+:\d{2}(:\d{2})?$"#, options: .regularExpression) != nil }) {
        return "connected"
    }
    return "unknown"
}

func consoleLocked() -> Bool {
    let root = IORegistryGetRootEntry(kIOMainPortDefault)
    defer { IOObjectRelease(root) }
    guard let value = IORegistryEntryCreateCFProperty(root, "IOConsoleLocked" as CFString, kCFAllocatorDefault, 0)?.takeRetainedValue() as? Bool else {
        fail("IOConsoleLocked is missing from the IORegistry root; cannot tell whether the screen is locked")
    }
    return value
}

func cmdState() {
    if consoleLocked() { emit(["state": "locked", "text": ""]); return }
    let all = banners()
    if all.count > 1 {
        emit(["state": "unknown", "text": "\(all.count) FaceTime banners: " + all.map { bannerTexts($0).joined(separator: " | ") }.joined(separator: " // ")])
        return
    }
    guard let b = all.first else { emit(["state": "none", "text": ""]); return }
    let texts = bannerTexts(b)
    emit(["state": classify(texts), "text": texts.joined(separator: " | ")])
}

func cmdPress(_ label: String) {
    let all = banners()
    if all.count > 1 { fail("\(all.count) FaceTime banners are up; refusing to guess which one to press \(label) on") }
    guard let b = all.first else { fail("no FaceTime call banner to press \(label) on") }
    let buttons = descendants(b).filter { text($0, kAXRoleAttribute) == "AXButton" }
    let labels = buttons.map { text($0, kAXDescriptionAttribute) ?? text($0, kAXTitleAttribute) ?? "" }
    guard let i = labels.firstIndex(of: label) else {
        fail("no \(label) button on the FaceTime call banner; buttons: \(labels)")
    }
    let result = AXUIElementPerformAction(buttons[i], kAXPressAction as CFString)
    if result != .success { fail("pressing \(label) failed: AXError \(result.rawValue)") }
}

func cmdDevices() {
    guard let root = appElement("com.apple.FaceTime") else { fail("FaceTime is not running") }
    guard let bar = children(root).first(where: { text($0, kAXRoleAttribute) == "AXMenuBar" }),
          let video = children(bar).first(where: { text($0, kAXTitleAttribute) == "Video" }),
          let menu = children(video).first else {
        fail("FaceTime's Video menu was not found")
    }
    var section = ""
    var checked: [String: String] = [:]
    for item in children(menu) {
        // Section headers ("Microphone", "Output") are disabled items; the
        // device rows follow each header until the next separator/header.
        let title = text(item, kAXTitleAttribute) ?? ""
        if title == "Microphone" || title == "Output" { section = title; continue }
        if title.isEmpty { section = ""; continue }  // separator ends the section
        if let mark = text(item, "AXMenuItemMarkChar"), !mark.isEmpty, !section.isEmpty, checked[section] == nil {
            checked[section] = title
        }
    }
    guard let mic = checked["Microphone"], let out = checked["Output"] else {
        fail("could not read the checked Microphone/Output in FaceTime's Video menu (found \(checked))")
    }
    emit(["microphone": mic, "output": out])
}

guard AXIsProcessTrusted() else {
    fail("ftcall needs Accessibility permission (System Settings > Privacy & Security > Accessibility) for the app that launched it", code: 2)
}
let args = CommandLine.arguments
switch (args.count, args.count > 1 ? args[1] : "") {
case (2, "state"): cmdState()
case (3, "press"): cmdPress(args[2])
case (2, "devices"): cmdDevices()
default: fail("usage: ftcall state | ftcall press <button> | ftcall devices")
}
