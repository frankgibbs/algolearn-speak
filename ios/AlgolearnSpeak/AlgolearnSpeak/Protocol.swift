import Foundation

/// Wire contract from docs/DESIGN_PHONE_AUDIO.md section 3.
enum Wire {
    static let port = 8772
    static let micRate = 16_000
    static let speakerRate = 24_000
    static let micBlockSamples = 512
    static let pingInterval: Duration = .seconds(5)
    static let maxMissedPongs = 2
    static let appName = "algolearn-speak-ios"
    static let version = "1"
}

/// PCM16 little-endian <-> Float32 conversion.
enum PCM {
    static func float32(fromPCM16 data: Data) -> [Float] {
        let count = data.count / 2
        var out = [Float](repeating: 0, count: count)
        data.withUnsafeBytes { raw in
            for i in 0..<count {
                let v = Int16(littleEndian: raw.loadUnaligned(fromByteOffset: i * 2, as: Int16.self))
                out[i] = Float(v) / 32768
            }
        }
        return out
    }

    static func pcm16(fromFloat32 samples: [Float]) -> Data {
        var data = Data(capacity: samples.count * 2)
        for s in samples {
            let scaled = (max(-1, min(1, s)) * 32768).rounded()
            let v = Int16(clamping: Int(scaled))
            withUnsafeBytes(of: v.littleEndian) { data.append(contentsOf: $0) }
        }
        return data
    }

    /// Root-mean-square level of PCM16 samples, 0...1.
    static func rms(pcm16 samples: [Int16]) -> Float {
        guard !samples.isEmpty else { return 0 }
        var sum: Float = 0
        for s in samples { let f = Float(s) / 32768; sum += f * f }
        return (sum / Float(samples.count)).squareRoot()
    }
}

enum ServerState: String, Sendable, Equatable {
    case idle, speaking, listening, processing
}

enum ProtocolError: Error, Equatable {
    case notJSONObject
    case missingField(String)
    case badValue(field: String, value: String)
}

/// Control messages the server sends to the phone.
enum ServerMessage: Equatable, Sendable {
    case ready(server: String, version: String)
    case state(ServerState)
    case playStart(rate: Int)
    case playEnd(id: Int)
    case micStart
    case micStop
    case ping
    case pong
    case unknown(type: String)

    static func decode(_ text: String) throws -> ServerMessage {
        guard let obj = try JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any],
              let type = obj["type"] as? String else {
            throw ProtocolError.notJSONObject
        }
        func int(_ key: String) throws -> Int {
            guard let v = obj[key] else { throw ProtocolError.missingField(key) }
            guard let n = v as? Int else { throw ProtocolError.badValue(field: key, value: "\(v)") }
            return n
        }
        func string(_ key: String) throws -> String {
            guard let v = obj[key] else { throw ProtocolError.missingField(key) }
            guard let s = v as? String else { throw ProtocolError.badValue(field: key, value: "\(v)") }
            return s
        }
        switch type {
        case "ready":
            return .ready(server: try string("server"), version: try string("version"))
        case "state":
            let raw = try string("value")
            guard let s = ServerState(rawValue: raw) else {
                throw ProtocolError.badValue(field: "value", value: raw)
            }
            return .state(s)
        case "play_start": return .playStart(rate: try int("rate"))
        case "play_end": return .playEnd(id: try int("id"))
        case "mic_start": return .micStart
        case "mic_stop": return .micStop
        case "ping": return .ping
        case "pong": return .pong
        default: return .unknown(type: type)
        }
    }
}

/// Control messages the phone sends to the server.
enum ClientMessage: Equatable, Sendable {
    case hello
    case played(id: Int)
    case ping
    case pong

    func encoded() throws -> String {
        let obj: [String: Any]
        switch self {
        case .hello:
            obj = ["type": "hello", "app": Wire.appName, "version": Wire.version,
                   "mic_rate": Wire.micRate, "spk_rate": Wire.speakerRate]
        case .played(let id): obj = ["type": "played", "id": id]
        case .ping: obj = ["type": "ping"]
        case .pong: obj = ["type": "pong"]
        }
        let data = try JSONSerialization.data(withJSONObject: obj, options: [.sortedKeys])
        return String(decoding: data, as: UTF8.self)
    }
}
