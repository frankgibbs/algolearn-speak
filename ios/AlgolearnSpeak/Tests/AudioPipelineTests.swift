import AVFoundation
import XCTest
@testable import AlgolearnSpeak

final class PlaybackSegmentTests: XCTestCase {
    func testFiresOnEndWhenNothingPending() {
        let seg = PlaybackSegment()
        XCTAssertEqual(seg.end(id: 4), 4)
        XCTAssertNil(seg.end(id: 4), "must fire once")
    }

    func testFiresOnLastCompletionWhenEndCameFirst() {
        let seg = PlaybackSegment()
        seg.bufferScheduled(); seg.bufferScheduled()
        XCTAssertNil(seg.end(id: 9))
        XCTAssertNil(seg.bufferCompleted())
        XCTAssertEqual(seg.bufferCompleted(), 9)
    }

    func testFiresOnEndWhenCompletionsCameFirst() {
        let seg = PlaybackSegment()
        seg.bufferScheduled()
        XCTAssertNil(seg.bufferCompleted())  // no play_end yet
        XCTAssertEqual(seg.end(id: 2), 2)
    }

    func testFiresExactlyOnce() {
        let seg = PlaybackSegment()
        seg.bufferScheduled()
        XCTAssertNil(seg.end(id: 1))
        XCTAssertEqual(seg.bufferCompleted(), 1)
        seg.bufferScheduled()
        XCTAssertNil(seg.bufferCompleted())
        XCTAssertNil(seg.end(id: 1))
    }

    func testOverlappingSegmentsAreIndependent() {
        let a = PlaybackSegment(), b = PlaybackSegment()
        a.bufferScheduled(); b.bufferScheduled()
        XCTAssertNil(a.end(id: 1))
        XCTAssertNil(b.end(id: 2))
        XCTAssertEqual(b.bufferCompleted(), 2)
        XCTAssertEqual(a.bufferCompleted(), 1)
    }

    func testAbandonReportsOnceEndIsKnown() {
        let seg = PlaybackSegment()
        seg.bufferScheduled()
        XCTAssertNil(seg.abandon())        // play_end not seen yet
        XCTAssertEqual(seg.end(id: 5), 5)  // reported despite a pending buffer
        XCTAssertNil(seg.bufferCompleted())
        XCTAssertTrue(seg.isAbandoned)
    }

    func testOddChunkBoundaries() {
        let seg = PlaybackSegment()
        XCTAssertEqual(seg.takeWholeSamples(Data([1, 2, 3])), Data([1, 2]))
        XCTAssertEqual(seg.takeWholeSamples(Data([4])), Data([3, 4]))
    }
}

private final class Collector: @unchecked Sendable {
    private let lock = NSLock()
    private var _blocks: [Data] = []
    private var _events: [String] = []
    var blocks: [Data] { lock.lock(); defer { lock.unlock() }; return _blocks }
    var events: [String] { lock.lock(); defer { lock.unlock() }; return _events }
    func add(_ d: Data) { lock.lock(); _blocks.append(d); lock.unlock() }
    func event(_ s: String) { lock.lock(); _events.append(s); lock.unlock() }
}

final class MicPipelineTests: XCTestCase {
    /// 16 kHz Float32 input: the converter only changes sample type, so output length equals input length.
    private func makePipeline(_ c: Collector, enabled: Bool) throws -> MicPipeline {
        let fmt = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16_000, channels: 1, interleaved: false)!
        return try MicPipeline(inputFormat: fmt, enabled: enabled,
                               onBlock: { c.add($0) }, onLevel: { _ in }, onEvent: { c.event($0) })
    }

    private func buffer(from start: Int, count: Int) -> AVAudioPCMBuffer {
        let fmt = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16_000, channels: 1, interleaved: false)!
        let b = AVAudioPCMBuffer(pcmFormat: fmt, frameCapacity: AVAudioFrameCount(count))!
        b.frameLength = AVAudioFrameCount(count)
        for i in 0..<count { b.floatChannelData![0][i] = Float(start + i) / 32768 }
        return b
    }

    func testFramingAcrossUnevenInputSizes() throws {
        let c = Collector()
        let p = try makePipeline(c, enabled: true)
        var offset = 0
        for n in [100, 300, 700, 13, 1, 511, 512, 862] {  // total 3000
            p.process(buffer(from: offset, count: n))
            offset += n
        }
        XCTAssertEqual(c.events, [])
        XCTAssertEqual(c.blocks.count, 3000 / 512)
        var expected = 0
        for block in c.blocks {
            XCTAssertEqual(block.count, 512 * 2)
            let samples = block.withUnsafeBytes { raw in
                (0..<512).map { Int16(littleEndian: raw.loadUnaligned(fromByteOffset: $0 * 2, as: Int16.self)) }
            }
            for s in samples {
                XCTAssertEqual(Int(s), expected, accuracy: 1)
                expected += 1
            }
        }
    }

    func testDisabledEmitsNothing() throws {
        let c = Collector()
        let p = try makePipeline(c, enabled: false)
        p.process(buffer(from: 0, count: 2048))
        XCTAssertEqual(c.blocks.count, 0)
    }

    func testAccumulatorResetsOnDisable() throws {
        let c = Collector()
        let p = try makePipeline(c, enabled: true)
        p.process(buffer(from: 0, count: 300))
        p.setEnabled(false)
        p.setEnabled(true)
        p.process(buffer(from: 0, count: 300))  // would be 600 (one block) if 300 had survived
        XCTAssertEqual(c.blocks.count, 0)
        p.process(buffer(from: 300, count: 212))
        XCTAssertEqual(c.blocks.count, 1)
    }
}
