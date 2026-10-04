// Verification helper: list CoreAudio devices and get/set the macOS default
// input device. Usage:
//   swift audiodev.swift list
//   swift audiodev.swift get-input
//   swift audiodev.swift set-input "<device name>"
import CoreAudio
import Foundation

func prop(_ sel: AudioObjectPropertySelector, _ scope: AudioObjectPropertyScope = kAudioObjectPropertyScopeGlobal) -> AudioObjectPropertyAddress {
    AudioObjectPropertyAddress(mSelector: sel, mScope: scope, mElement: kAudioObjectPropertyElementMain)
}

func devices() -> [AudioDeviceID] {
    var addr = prop(kAudioHardwarePropertyDevices)
    var size: UInt32 = 0
    AudioObjectGetPropertyDataSize(AudioObjectID(kAudioObjectSystemObject), &addr, 0, nil, &size)
    var ids = [AudioDeviceID](repeating: 0, count: Int(size) / MemoryLayout<AudioDeviceID>.size)
    AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &addr, 0, nil, &size, &ids)
    return ids
}

func name(_ id: AudioDeviceID) -> String {
    var addr = prop(kAudioObjectPropertyName)
    var cf: Unmanaged<CFString>? = nil
    var size = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
    AudioObjectGetPropertyData(id, &addr, 0, nil, &size, &cf)
    return cf?.takeRetainedValue() as String? ?? "?"
}

func inputChannels(_ id: AudioDeviceID) -> Int {
    var addr = prop(kAudioDevicePropertyStreamConfiguration, kAudioObjectPropertyScopeInput)
    var size: UInt32 = 0
    AudioObjectGetPropertyDataSize(id, &addr, 0, nil, &size)
    let buf = UnsafeMutablePointer<AudioBufferList>.allocate(capacity: Int(size))
    defer { buf.deallocate() }
    AudioObjectGetPropertyData(id, &addr, 0, nil, &size, buf)
    return UnsafeMutableAudioBufferListPointer(buf).reduce(0) { $0 + Int($1.mNumberChannels) }
}

func defaultInput() -> AudioDeviceID {
    var addr = prop(kAudioHardwarePropertyDefaultInputDevice)
    var id: AudioDeviceID = 0
    var size = UInt32(MemoryLayout<AudioDeviceID>.size)
    AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &addr, 0, nil, &size, &id)
    return id
}

let args = CommandLine.arguments
switch args.count > 1 ? args[1] : "list" {
case "list":
    for id in devices() { print("\(id)\t\(name(id))\tin=\(inputChannels(id))") }
case "get-input":
    print(name(defaultInput()))
case "set-input":
    guard let id = devices().first(where: { name($0) == args[2] }) else { print("no device named \(args[2])"); exit(1) }
    var addr = prop(kAudioHardwarePropertyDefaultInputDevice)
    var dev = id
    let st = AudioObjectSetPropertyData(AudioObjectID(kAudioObjectSystemObject), &addr, 0, nil, UInt32(MemoryLayout<AudioDeviceID>.size), &dev)
    print(st == noErr ? "default input -> \(name(id))" : "failed: \(st)")
    exit(st == noErr ? 0 : 1)
default:
    print("usage: list | get-input | set-input <name>"); exit(2)
}
