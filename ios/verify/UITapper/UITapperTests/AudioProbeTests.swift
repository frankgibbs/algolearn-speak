import XCTest
import AVFoundation

/// Probes run INSIDE the simulator (test-runner process) to find which
/// AVAudioSession/AVAudioEngine configuration deadlocks CoreAudio here.
final class AudioProbeTests: XCTestCase {
    private func run(category: AVAudioSession.Category, mode: AVAudioSession.Mode, options: AVAudioSession.CategoryOptions, input: Bool, output: Bool) throws {
        let s = AVAudioSession.sharedInstance()
        try s.setCategory(category, mode: mode, options: options)
        try s.setActive(true)
        NSLog("PROBE session active; route in=%@ out=%@ rate=%f", s.currentRoute.inputs.map(\.portName).description, s.currentRoute.outputs.map(\.portName).description, s.sampleRate)
        let e = AVAudioEngine()
        if input {
            let f = e.inputNode.outputFormat(forBus: 0)
            NSLog("PROBE input format %@", f.description)
            e.inputNode.installTap(onBus: 0, bufferSize: 1024, format: f) { _, _ in }
        }
        if output {
            let p = AVAudioPlayerNode()
            e.attach(p)
            e.connect(p, to: e.mainMixerNode, format: AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 24000, channels: 1, interleaved: false)!)
            NSLog("PROBE mixer connected")
        }
        e.prepare()
        try e.start()
        NSLog("PROBE engine started OK")
        sleep(2)
        e.stop()
    }
    func testPlaybackOnly() throws { try run(category: .playback, mode: .default, options: [], input: false, output: true) }
    func testRecordOnly() throws { try run(category: .record, mode: .default, options: [], input: true, output: false) }
    func testPlayAndRecordDefault() throws { try run(category: .playAndRecord, mode: .default, options: [], input: true, output: true) }
    func testPlayAndRecordDefaultOutputOnly() throws { try run(category: .playAndRecord, mode: .default, options: [], input: false, output: true) }
    func testPlayAndRecordVoiceChatApp() throws { try run(category: .playAndRecord, mode: .voiceChat, options: [.allowBluetoothHFP, .allowBluetoothA2DP, .defaultToSpeaker], input: true, output: true) }
}
