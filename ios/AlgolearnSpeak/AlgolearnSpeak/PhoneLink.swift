import AVFoundation
import Foundation
import Observation
import UIKit

/// Thread-safe holder so the audio thread can send mic blocks without hopping actors.
final class SocketSender: @unchecked Sendable {
    private let lock = NSLock()
    private var task: URLSessionWebSocketTask?

    func set(_ task: URLSessionWebSocketTask?) { lock.lock(); self.task = task; lock.unlock() }

    private var onFailure: (@Sendable (String) -> Void)?
    private var failing = false

    func setFailureHandler(_ handler: @escaping @Sendable (String) -> Void) {
        lock.lock(); onFailure = handler; lock.unlock()
    }

    func sendBinary(_ data: Data) { send(.data(data), label: "mic audio") }

    func sendText(_ text: String, label: String) { send(.string(text), label: label) }

    private func send(_ message: URLSessionWebSocketTask.Message, label: String) {
        lock.lock(); let t = task; lock.unlock()
        t?.send(message) { [weak self] error in
            guard let self else { return }
            self.lock.lock()
            let report = error != nil && !self.failing   // report a burst of failures once
            self.failing = error != nil
            let handler = self.onFailure
            self.lock.unlock()
            if report, let error { handler?("send failed (\(label)): \(error.localizedDescription)") }
        }
    }
}

enum LinkConnection: Equatable {
    case disconnected, connecting, connected
}

@MainActor
@Observable
final class PhoneLink {
    static let hostKey = "serverHost"
    static let defaultHost = "192.168.86.188"

    var host: String {
        didSet { UserDefaults.standard.set(host, forKey: Self.hostKey) }
    }
    private(set) var connection: LinkConnection = .disconnected
    private(set) var wantsConnection = false
    private(set) var serverState: ServerState = .idle
    private(set) var micLevel: Float = 0
    private(set) var micOpen = false
    private(set) var events: [String] = []

    @ObservationIgnored private let sender = SocketSender()
    @ObservationIgnored private var audio: AudioEngine?
    @ObservationIgnored private var loop: Task<Void, Never>?
    @ObservationIgnored private var socket: URLSessionWebSocketTask?
    @ObservationIgnored private var foreground = true
    @ObservationIgnored private var outstandingPings = 0
    @ObservationIgnored private let session: URLSession

    private enum SessionEnd { case dropped, busy, fatal(String) }

    init() {
        host = UserDefaults.standard.string(forKey: Self.hostKey) ?? Self.defaultHost
        let cfg = URLSessionConfiguration.default
        cfg.timeoutIntervalForRequest = 60  // idle receive timeout; app-level ping covers liveness
        cfg.waitsForConnectivity = false
        session = URLSession(configuration: cfg)
    }

    // MARK: user actions

    func connect() {
        guard !wantsConnection else { return }
        wantsConnection = true
        log("connecting to \(host)")
        Task { @MainActor in
            await loop?.value  // let a cancelled previous loop finish before starting a new one
            guard wantsConnection else { return }
            guard await Self.microphoneGranted() else {
                log("microphone permission denied (enable it in Settings)")
                wantsConnection = false
                return
            }
            startLoopIfNeeded()
        }
    }

    func disconnect() {
        guard wantsConnection else { return }
        wantsConnection = false
        socket?.cancel(with: .goingAway, reason: nil)
        loop?.cancel()
        log("disconnected by user")
    }

    func setForeground(_ isForeground: Bool) {
        foreground = isForeground
        if isForeground && wantsConnection { startLoopIfNeeded() }
    }

    // MARK: connection loop

    private static func microphoneGranted() async -> Bool {
        await AVAudioApplication.requestRecordPermission()
    }

    private func startLoopIfNeeded() {
        guard loop == nil else { return }
        loop = Task { @MainActor in
            await runLoop()
            audio?.stop()
            connection = .disconnected
            micOpen = false
            micLevel = 0
            serverState = .idle
            loop = nil
        }
    }

    /// Reconnect policy:
    /// - foreground: indefinitely, backoff 1 s doubling to 15 s;
    /// - background: only while the audio session is live (background audio mode), and for at most
    ///   `backgroundGiveUp` after the drop, then stop and show Offline;
    /// - `busy` (a stale slot after a WiFi blip): retry every `busyDelay` up to `maxBusyAttempts`, then fatal;
    /// - `bad-rates`: fatal.
    private func runLoop() async {
        var delay = 1.0
        var busyAttempts = 0
        var dropStart: Date?
        while wantsConnection && !Task.isCancelled {
            if !foreground {
                guard audio?.isRunning == true else { return }  // resumes on foreground
                if let start = dropStart, Date.now.timeIntervalSince(start) > Self.backgroundGiveUp {
                    log("offline: background reconnect gave up after \(Int(Self.backgroundGiveUp)) s")
                    wantsConnection = false
                    return
                }
            }
            let (end, reachedReady) = await runSession()
            guard wantsConnection, !Task.isCancelled else { return }
            if reachedReady { delay = 1; busyAttempts = 0; dropStart = nil }
            if dropStart == nil { dropStart = .now }
            var wait = delay
            switch end {
            case .fatal(let reason):
                log(reason)
                wantsConnection = false
                return
            case .busy:
                busyAttempts += 1
                if busyAttempts > Self.maxBusyAttempts {
                    log("server still busy after \(Self.maxBusyAttempts) attempts; giving up")
                    wantsConnection = false
                    return
                }
                wait = Self.busyDelay
                log("server busy (stale slot?), retry \(busyAttempts)/\(Self.maxBusyAttempts) in \(Int(wait)) s")
            case .dropped:
                log("reconnecting in \(Int(wait)) s")
                delay = min(delay * 2, 15)
            }
            try? await Task.sleep(for: .seconds(wait))
        }
    }

    private static let backgroundGiveUp: TimeInterval = 120
    private static let maxBusyAttempts = 6
    private static let busyDelay: TimeInterval = 10

    private func runSession() async -> (SessionEnd, Bool) {
        guard let url = URL(string: "ws://\(host):\(Wire.port)") else {
            return (.fatal("invalid host \"\(host)\""), false)
        }
        connection = .connecting
        let task = session.webSocketTask(with: url)
        socket = task
        sender.set(task)
        outstandingPings = 0
        var reachedReady = false
        task.resume()

        let pinger = Task { @MainActor [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: Wire.pingInterval)
                guard let self, !Task.isCancelled, self.connection == .connected else { continue }
                if self.outstandingPings >= Wire.maxMissedPongs {
                    self.log("ping timeout")
                    task.cancel(with: .goingAway, reason: nil)
                    return
                }
                self.outstandingPings += 1
                self.send(.ping)
            }
        }

        do {
            try await task.send(.string(try ClientMessage.hello.encoded()))
            while true {
                let message = try await task.receive()
                switch message {
                case .string(let text):
                    if case .ready = (try? ServerMessage.decode(text)) { reachedReady = true }
                    handle(text)
                case .data(let data):
                    audio?.appendPlayback(data)
                @unknown default:
                    break
                }
            }
        } catch {
            if wantsConnection { log("link closed: \(error.localizedDescription)") }
        }

        pinger.cancel()
        let reason = task.closeReason.map { String(decoding: $0, as: UTF8.self) } ?? ""
        sender.set(nil)
        socket = nil
        audio?.resetStreams()  // keep engine + session alive across the backoff
        micOpen = false
        micLevel = 0
        serverState = .idle
        connection = .disconnected
        switch reason {
        case "busy": return (.busy, reachedReady)
        case "bad-rates": return (.fatal("server refused: bad-rates"), reachedReady)
        default: return (.dropped, reachedReady)
        }
    }

    // MARK: protocol handling

    private func handle(_ text: String) {
        let message: ServerMessage
        do { message = try ServerMessage.decode(text) } catch {
            log("bad message: \(error)")
            return
        }
        switch message {
        case .ready(let server, let version):
            connection = .connected
            log("ready: \(server) \(version)")
            startAudio()
        case .state(let s):
            serverState = s
            if s != .listening { micLevel = 0 }
            log("state: \(s.rawValue)")
        case .playStart(let rate):
            if rate != Wire.speakerRate { log("unexpected play rate \(rate)") }
            audio?.beginSegment()
        case .playEnd(let id):
            audio?.endSegment(id: id)
            log("segment \(id) sent")
        case .micStart:
            audio?.setMicEnabled(true)
            micOpen = true
            UIImpactFeedbackGenerator(style: .medium).impactOccurred()
            log("mic open")
        case .micStop:
            audio?.setMicEnabled(false)
            micOpen = false
            micLevel = 0
            log("mic closed")
        case .ping:
            send(.pong)
        case .pong:
            outstandingPings = 0
        case .unknown(let type):
            log("ignored message: \(type)")
        }
    }

    private func startAudio() {
        if audio == nil {
            let sender = self.sender
            sender.setFailureHandler { [weak self] text in Task { @MainActor in self?.log(text) } }
            audio = AudioEngine(
                onMicBlock: { sender.sendBinary($0) },
                onLevel: { [weak self] level in Task { @MainActor in self?.micLevel = level } },
                onPlayed: { [weak self] id in Task { @MainActor in self?.send(.played(id: id)) } },
                onEvent: { [weak self] text in Task { @MainActor in self?.log(text) } })
        }
        do { try audio?.start() } catch {
            log("audio start failed: \(error)")
            wantsConnection = false
            socket?.cancel(with: .goingAway, reason: nil)
        }
    }

    private func send(_ message: ClientMessage) {
        guard connection == .connected, let text = try? message.encoded() else { return }
        let label: String
        if case .played(let id) = message { label = "played \(id)" } else { label = "control" }
        sender.sendText(text, label: label)
    }

    private func log(_ text: String) {
        let stamp = Date.now.formatted(.dateTime.hour().minute().second())
        events.insert("\(stamp)  \(text)", at: 0)
        if events.count > 8 { events.removeLast(events.count - 8) }
    }
}
