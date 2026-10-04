import AVFoundation

/// Wraps a value that is not Sendable but is only touched from one thread at a time.
private struct Unchecked<T>: @unchecked Sendable { let value: T }

private final class OnceBox: @unchecked Sendable { var done = false }

enum AudioEngineError: Error, CustomStringConvertible {
    case noInputFormat
    case converterUnavailable

    var description: String {
        switch self {
        case .noInputFormat: "microphone input format is unavailable (sample rate 0)"
        case .converterUnavailable: "cannot create 16 kHz Int16 converter"
        }
    }
}

/// Converts hardware-format tap buffers to 16 kHz Int16 mono and emits 512-sample blocks.
/// `process` runs on the audio render thread; everything else is lock-protected.
final class MicPipeline: @unchecked Sendable {
    private let converter: AVAudioConverter
    private let outFormat: AVAudioFormat
    private let inputRate: Double
    private let lock = NSLock()
    private var enabled: Bool
    private var accumulator: [Int16] = []
    private let onBlock: @Sendable (Data) -> Void
    private let onLevel: @Sendable (Float) -> Void
    private let onEvent: @Sendable (String) -> Void
    private var errorReported = false

    init(inputFormat: AVAudioFormat, enabled: Bool,
         onBlock: @escaping @Sendable (Data) -> Void,
         onLevel: @escaping @Sendable (Float) -> Void,
         onEvent: @escaping @Sendable (String) -> Void) throws {
        guard let out = AVAudioFormat(commonFormat: .pcmFormatInt16, sampleRate: Double(Wire.micRate),
                                      channels: 1, interleaved: true),
              let conv = AVAudioConverter(from: inputFormat, to: out) else {
            throw AudioEngineError.converterUnavailable
        }
        self.converter = conv
        self.outFormat = out
        self.inputRate = inputFormat.sampleRate
        self.enabled = enabled
        self.onBlock = onBlock
        self.onLevel = onLevel
        self.onEvent = onEvent
        accumulator.reserveCapacity(Wire.micBlockSamples * 4)
    }

    func setEnabled(_ on: Bool) {
        lock.lock(); defer { lock.unlock() }
        if on && !enabled { converter.reset() }
        enabled = on
        accumulator.removeAll(keepingCapacity: true)
    }

    /// Reports a failure once, then stays quiet until a conversion succeeds (avoids log spam at ~40 buffers/s).
    private func reportError(_ text: String) {
        lock.lock()
        let first = !errorReported
        errorReported = true
        lock.unlock()
        if first { onEvent(text) }
    }

    private func clearError() {
        lock.lock(); errorReported = false; lock.unlock()
    }

    func process(_ buffer: AVAudioPCMBuffer) {
        lock.lock()
        guard enabled else { lock.unlock(); return }
        lock.unlock()

        let capacity = AVAudioFrameCount(Double(buffer.frameLength) * Double(Wire.micRate) / inputRate) + 64
        guard let out = AVAudioPCMBuffer(pcmFormat: outFormat, frameCapacity: capacity) else {
            reportError("mic buffer allocation failed")
            return
        }
        let box = OnceBox()
        let input = Unchecked(value: buffer)
        var error: NSError?
        let status = converter.convert(to: out, error: &error) { _, inStatus in
            if box.done { inStatus.pointee = .noDataNow; return nil }
            box.done = true
            inStatus.pointee = .haveData
            return input.value
        }
        if status == .error || error != nil {
            reportError("mic conversion failed: \(error?.localizedDescription ?? "unknown")")
            return
        }
        guard out.frameLength > 0, let ptr = out.int16ChannelData?[0] else { return }
        clearError()
        let samples = Array(UnsafeBufferPointer(start: ptr, count: Int(out.frameLength)))

        var blocks: [[Int16]] = []
        lock.lock()
        if enabled {
            accumulator.append(contentsOf: samples)
            while accumulator.count >= Wire.micBlockSamples {
                blocks.append(Array(accumulator.prefix(Wire.micBlockSamples)))
                accumulator.removeFirst(Wire.micBlockSamples)
            }
        }
        lock.unlock()

        for block in blocks {
            let data = block.withUnsafeBufferPointer { Data(buffer: $0) }  // host is little-endian
            onBlock(data)
            onLevel(PCM.rms(pcm16: block))
        }
    }
}

/// One speaker segment (play_start ... play_end). `played` fires once every scheduled
/// buffer has been rendered and play_end has arrived.
final class PlaybackSegment: @unchecked Sendable {
    private let lock = NSLock()
    private var pending = 0
    private var endID: Int?
    private var fired = false
    private var leftover: UInt8?
    private var abandoned = false

    /// Joins a possibly odd-length chunk with the previous leftover byte; returns whole samples.
    func takeWholeSamples(_ chunk: Data) -> Data {
        lock.lock(); defer { lock.unlock() }
        var data = Data()
        if let b = leftover { data.append(b); leftover = nil }
        data.append(chunk)
        if data.count % 2 == 1 { leftover = data.removeLast() }
        return data
    }

    func bufferScheduled() { lock.lock(); pending += 1; lock.unlock() }

    /// Returns the id to report if this completion finished the segment.
    func bufferCompleted() -> Int? {
        lock.lock(); defer { lock.unlock() }
        pending -= 1
        return finishIfDone()
    }

    func end(id: Int) -> Int? {
        lock.lock(); defer { lock.unlock() }
        endID = id
        return finishIfDone()
    }

    var isAbandoned: Bool { lock.lock(); defer { lock.unlock() }; return abandoned }
    var isFired: Bool { lock.lock(); defer { lock.unlock() }; return fired }

    /// Gives up on the rest of this segment (audio interruption). Later chunks are dropped and
    /// `played` is reported as soon as play_end is known, so the server is never left waiting.
    func abandon() -> Int? {
        lock.lock(); defer { lock.unlock() }
        abandoned = true
        return finishIfDone()
    }

    private func finishIfDone() -> Int? {
        guard !fired, pending == 0 || abandoned, let id = endID else { return nil }
        fired = true
        return id
    }
}

/// AVAudioSession + AVAudioEngine: mic tap -> 16 kHz Int16 blocks; 24 kHz PCM16 -> player node.
final class AudioEngine: @unchecked Sendable {
    private let engine = AVAudioEngine()
    private let player = AVAudioPlayerNode()
    private let playFormat: AVAudioFormat
    private let lock = NSLock()
    private var pipeline: MicPipeline?
    private var micEnabled = false
    private var running = false
    private var attached = false
    private var segment: PlaybackSegment?
    private var active: [PlaybackSegment] = []
    private var observers: [NSObjectProtocol] = []

    private let onMicBlock: @Sendable (Data) -> Void
    private let onLevel: @Sendable (Float) -> Void
    private let onPlayed: @Sendable (Int) -> Void
    private let onEvent: @Sendable (String) -> Void

    init(onMicBlock: @escaping @Sendable (Data) -> Void,
         onLevel: @escaping @Sendable (Float) -> Void,
         onPlayed: @escaping @Sendable (Int) -> Void,
         onEvent: @escaping @Sendable (String) -> Void) {
        self.playFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: Double(Wire.speakerRate),
                                        channels: 1, interleaved: false)!
        self.onMicBlock = onMicBlock
        self.onLevel = onLevel
        self.onPlayed = onPlayed
        self.onEvent = onEvent
    }

    // MARK: lifecycle

    var isRunning: Bool { lock.lock(); defer { lock.unlock() }; return running }

    /// Idempotent. If already running (for example across a reconnect) only the streams are reset.
    func start() throws {
        if isRunning { resetStreams(); return }
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playAndRecord, mode: .voiceChat,
                                options: [.allowBluetoothHFP, .allowBluetoothA2DP, .defaultToSpeaker])
        try session.setActive(true)
        if !attached {
            engine.attach(player)
            engine.connect(player, to: engine.mainMixerNode, format: playFormat)
            attached = true
        }
        try installTap()
        engine.prepare()
        try engine.start()
        player.play()
        lock.lock(); running = true; lock.unlock()
        observeRouteAndInterruptions()
    }

    /// Drops mic gating and any queued playback but keeps the engine and audio session alive
    /// (so the background-audio session survives a reconnect backoff).
    func resetStreams() {
        guard isRunning else { return }
        lock.lock(); micEnabled = false; segment = nil; active.removeAll(); let p = pipeline; lock.unlock()
        p?.setEnabled(false)
        player.stop()
        player.play()
    }

    func stop() {
        guard isRunning else { return }
        lock.lock(); running = false; micEnabled = false; segment = nil; active.removeAll(); lock.unlock()
        observers.forEach { NotificationCenter.default.removeObserver($0) }
        observers.removeAll()
        pipeline?.setEnabled(false)
        engine.inputNode.removeTap(onBus: 0)
        player.stop()
        engine.stop()
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
    }

    private func installTap() throws {
        let input = engine.inputNode
        input.removeTap(onBus: 0)
        let format = input.outputFormat(forBus: 0)
        guard format.sampleRate > 0, format.channelCount > 0 else { throw AudioEngineError.noInputFormat }
        lock.lock(); let enabled = micEnabled; lock.unlock()
        let p = try MicPipeline(inputFormat: format, enabled: enabled, onBlock: onMicBlock, onLevel: onLevel, onEvent: onEvent)
        lock.lock(); pipeline = p; lock.unlock()
        input.installTap(onBus: 0, bufferSize: 1024, format: format) { buffer, _ in
            p.process(buffer)
        }
    }

    private func observeRouteAndInterruptions() {
        let nc = NotificationCenter.default
        observers.append(nc.addObserver(forName: .AVAudioEngineConfigurationChange, object: engine, queue: .main) { [weak self] _ in
            self?.recover(reason: "audio route changed")
        })
        observers.append(nc.addObserver(forName: AVAudioSession.interruptionNotification, object: nil, queue: .main) { [weak self] note in
            guard let raw = note.userInfo?[AVAudioSessionInterruptionTypeKey] as? UInt,
                  let type = AVAudioSession.InterruptionType(rawValue: raw) else { return }
            switch type {
            case .began: self?.abandonPlayback()
            case .ended: self?.recover(reason: "interruption ended")
            @unknown default: break
            }
        })
    }

    /// Interruption began: stop the player and explicitly abandon every unfinished segment.
    private func abandonPlayback() {
        player.stop()
        lock.lock()
        active.removeAll { $0.isFired }
        let segs = active
        lock.unlock()
        var ids: [Int] = []
        for seg in segs { if let id = seg.abandon() { ids.append(id); onPlayed(id) } }
        onEvent("interruption: playback abandoned" + (ids.isEmpty ? "" : " (reported played \(ids))"))
    }

    private func recover(reason: String) {
        lock.lock(); let isRunning = running; lock.unlock()
        guard isRunning else { return }
        do {
            try AVAudioSession.sharedInstance().setActive(true)
            try installTap()
            if !engine.isRunning { try engine.start() }
            player.play()
            onEvent("audio restarted: \(reason)")
        } catch {
            onEvent("audio restart failed: \(error.localizedDescription)")
        }
    }

    // MARK: microphone gate

    /// The mic is streamed only between mic_start and mic_stop.
    func setMicEnabled(_ on: Bool) {
        lock.lock(); micEnabled = on; let p = pipeline; lock.unlock()
        p?.setEnabled(on)
    }

    // MARK: playback

    func beginSegment() {
        lock.lock()
        active.removeAll { $0.isFired }
        let seg = PlaybackSegment()
        segment = seg
        active.append(seg)
        lock.unlock()
    }

    func appendPlayback(_ chunk: Data) {
        lock.lock(); let seg = segment; lock.unlock()
        guard let seg else { onEvent("audio chunk outside a segment ignored"); return }
        if seg.isAbandoned { return }  // interruption already reported once
        let whole = seg.takeWholeSamples(chunk)
        let samples = PCM.float32(fromPCM16: whole)
        if samples.isEmpty { return }
        guard let buffer = AVAudioPCMBuffer(pcmFormat: playFormat, frameCapacity: AVAudioFrameCount(samples.count)),
              let dest = buffer.floatChannelData?[0] else {
            onEvent("playback chunk dropped: buffer allocation failed (\(samples.count) samples)")
            return
        }
        buffer.frameLength = AVAudioFrameCount(samples.count)
        samples.withUnsafeBufferPointer { dest.update(from: $0.baseAddress!, count: samples.count) }
        seg.bufferScheduled()
        player.scheduleBuffer(buffer, completionCallbackType: .dataPlayedBack) { [weak self, seg] _ in
            if let id = seg.bufferCompleted() { self?.onPlayed(id) }
        }
    }

    func endSegment(id: Int) {
        lock.lock(); let seg = segment; segment = nil; lock.unlock()
        guard let seg else { onEvent("play_end \(id) without play_start"); return }
        if let done = seg.end(id: id) { onPlayed(done) }
    }
}
