import AVFoundation
import Foundation
// Mac-side AVAudioEngine input probe with a watchdog (CoreAudio HAL stall detector).
let watchdog = Thread { sleep(10); print("MACPROBE: TIMED OUT after 10s (CoreAudio input stalled)"); fflush(stdout); exit(2) }
watchdog.start()
let t = Date()
let e = AVAudioEngine()
let f = e.inputNode.outputFormat(forBus: 0)
print("MACPROBE: input format \(f.sampleRate) Hz \(f.channelCount) ch (\(String(format: "%.2f", Date().timeIntervalSince(t)))s)"); fflush(stdout)
var frames: Int = 0
e.inputNode.installTap(onBus: 0, bufferSize: 1024, format: f) { buf, _ in frames += Int(buf.frameLength) }
e.prepare()
try e.start()
print("MACPROBE: engine started (\(String(format: "%.2f", Date().timeIntervalSince(t)))s)"); fflush(stdout)
sleep(1)
print("MACPROBE: \(frames) frames in 1 s"); fflush(stdout)
exit(0)
