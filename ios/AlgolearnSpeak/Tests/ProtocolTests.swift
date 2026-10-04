import XCTest
@testable import AlgolearnSpeak

final class ProtocolTests: XCTestCase {
    func testPCM16RoundTripIsExactForEveryInt16() {
        var data = Data()
        var expected: [Int16] = []
        for v in stride(from: Int(Int16.min), through: Int(Int16.max), by: 257) {
            expected.append(Int16(v))
            withUnsafeBytes(of: Int16(v).littleEndian) { data.append(contentsOf: $0) }
        }
        let floats = PCM.float32(fromPCM16: data)
        XCTAssertEqual(floats.count, expected.count)
        XCTAssertEqual(PCM.pcm16(fromFloat32: floats), data)
    }

    func testFloat32ToPCM16KnownValuesAndClipping() {
        let out = PCM.pcm16(fromFloat32: [0, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0])
        let v = out.withUnsafeBytes { raw in
            (0..<7).map { Int16(littleEndian: raw.loadUnaligned(fromByteOffset: $0 * 2, as: Int16.self)) }
        }
        XCTAssertEqual(v, [0, 16384, -16384, 32767, -32768, 32767, -32768])
    }

    func testPCM16LittleEndianByteOrder() {
        let floats = PCM.float32(fromPCM16: Data([0x00, 0x40]))  // 0x4000 = 16384
        XCTAssertEqual(floats, [0.5])
    }

    func testHelloEncoding() throws {
        let text = try ClientMessage.hello.encoded()
        let obj = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any])
        XCTAssertEqual(obj["type"] as? String, "hello")
        XCTAssertEqual(obj["app"] as? String, "algolearn-speak-ios")
        XCTAssertEqual(obj["version"] as? String, "1")
        XCTAssertEqual(obj["mic_rate"] as? Int, 16000)
        XCTAssertEqual(obj["spk_rate"] as? Int, 24000)
        XCTAssertEqual(obj.count, 5)
    }

    func testPlayedPingPongEncoding() throws {
        XCTAssertEqual(try ClientMessage.played(id: 7).encoded(), #"{"id":7,"type":"played"}"#)
        XCTAssertEqual(try ClientMessage.ping.encoded(), #"{"type":"ping"}"#)
        XCTAssertEqual(try ClientMessage.pong.encoded(), #"{"type":"pong"}"#)
    }

    func testServerMessageDecoding() throws {
        XCTAssertEqual(try ServerMessage.decode(#"{"type":"ready","server":"algolearn-speak","version":"0.1.0"}"#),
                       .ready(server: "algolearn-speak", version: "0.1.0"))
        XCTAssertEqual(try ServerMessage.decode(#"{"type":"state","value":"listening"}"#), .state(.listening))
        XCTAssertEqual(try ServerMessage.decode(#"{"type":"play_start","rate":24000}"#), .playStart(rate: 24000))
        XCTAssertEqual(try ServerMessage.decode(#"{"type":"play_end","id":3}"#), .playEnd(id: 3))
        XCTAssertEqual(try ServerMessage.decode(#"{"type":"mic_start"}"#), .micStart)
        XCTAssertEqual(try ServerMessage.decode(#"{"type":"mic_stop"}"#), .micStop)
        XCTAssertEqual(try ServerMessage.decode(#"{"type":"ping"}"#), .ping)
        XCTAssertEqual(try ServerMessage.decode(#"{"type":"pong"}"#), .pong)
        XCTAssertEqual(try ServerMessage.decode(#"{"type":"future"}"#), .unknown(type: "future"))
    }

    func testServerMessageDecodingErrors() {
        XCTAssertThrowsError(try ServerMessage.decode("[]"))
        XCTAssertThrowsError(try ServerMessage.decode(#"{"type":"state","value":"dancing"}"#))
        XCTAssertThrowsError(try ServerMessage.decode(#"{"type":"play_end"}"#)) {
            XCTAssertEqual($0 as? ProtocolError, .missingField("id"))
        }
    }

    func testRMS() {
        XCTAssertEqual(PCM.rms(pcm16: []), 0)
        XCTAssertEqual(PCM.rms(pcm16: [16384, -16384]), 0.5, accuracy: 1e-6)
    }
}
