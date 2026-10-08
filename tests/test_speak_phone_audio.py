"""Tests for speak_phone_audio.PhoneAudioServer and the phone routing in
speak_server (docs/DESIGN_PHONE_AUDIO.md section 4).

A real `websockets` client talks to a real PhoneAudioServer over localhost
(port 0 = OS-assigned). No audio device is touched. VAD plumbing is tested
with a stub VAD injected through the constructor (Silero does not reliably
fire on synthetic signals); one test uses the real Silero on digital silence.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import unittest
from unittest import mock

os.environ["SPEAK_AUDIO_DRY_RUN"] = "1"
os.environ["SPEAK_FTCALL_BIN"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fake_ftcall")  # never the real FaceTime banner
os.environ["SPEAK_PHONE_SOCKET"] = "/nonexistent-speak-phone-test/phone.sock"  # never the live daemon
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect

import speak_phone_audio as pa
import speak_server as s

HELLO = {"type": "hello", "app": "algolearn-speak-ios", "version": "1", "mic_rate": 16000, "spk_rate": 24000}
BLOCK = 512


class StubVAD:
    """Energy VAD with Silero's event contract: {"start":..} on the first
    loud block, {"end":..} after `quiet_needed` quiet blocks following speech."""

    def __init__(self, silence_seconds: float) -> None:
        self.quiet_needed = int(silence_seconds * 1000 / 32)
        self.speaking = False
        self.quiet = 0

    def __call__(self, frame: np.ndarray):
        loud = float(np.abs(frame).mean()) > 0.01
        if loud:
            self.quiet = 0
            if not self.speaking:
                self.speaking = True
                return {"start": 0}
            return None
        if self.speaking:
            self.quiet += 1
            if self.quiet >= self.quiet_needed:
                return {"end": 0}
        return None


def speech_blocks(n_silence: int, n_speech: int, n_tail: int) -> list[bytes]:
    rng = np.random.default_rng(1)
    out = [pa.pcm16_from_float(np.zeros(BLOCK, dtype=np.float32)) for _ in range(n_silence)]
    out += [pa.pcm16_from_float((0.5 * rng.standard_normal(BLOCK)).astype(np.float32)) for _ in range(n_speech)]
    out += [pa.pcm16_from_float(np.zeros(BLOCK, dtype=np.float32)) for _ in range(n_tail)]
    return out


class FakePhone:
    """A scripted phone: real websocket client with a reader thread."""

    def __init__(self, port: int, hello: dict | None = HELLO, ack: bool = True, mic_blocks: list[bytes] | None = None, answer_pings: bool = True, read_delay: float = 0.0) -> None:
        self.answer_pings = answer_pings
        self.read_delay = read_delay
        self.ws = connect(f"ws://127.0.0.1:{port}")
        self.received: list = []
        self.ack = ack
        self.mic_blocks = mic_blocks or []
        self.closed = threading.Event()
        self.close_reason: str | None = None
        self.close_code: int | None = None
        if hello is not None:
            try:
                self.ws.send(json.dumps(hello))
            except ConnectionClosed:
                pass  # refused before our hello went out (busy); the reader records the close
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self) -> None:
        try:
            for msg in self.ws:
                if self.read_delay:
                    time.sleep(self.read_delay)
                self.received.append(msg)
                if isinstance(msg, bytes):
                    continue
                m = json.loads(msg)
                if m["type"] == "play_end" and self.ack:
                    self.ws.send(json.dumps({"type": "played", "id": m["id"]}))
                elif m["type"] == "mic_start":
                    for b in self.mic_blocks:
                        self.ws.send(b)
                elif m["type"] == "ping" and self.answer_pings:
                    self.ws.send(json.dumps({"type": "pong"}))
        except ConnectionClosed as e:
            if e.rcvd is not None:
                self.close_code, self.close_reason = e.rcvd.code, e.rcvd.reason
        finally:
            self.closed.set()

    def types(self) -> list[str]:
        return [json.loads(m)["type"] for m in self.received if isinstance(m, str)]

    def close(self) -> None:
        self.ws.close()
        self.closed.wait(5)


def wait_for(cond, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


class PhoneServerBase(unittest.TestCase):
    server_kwargs: dict = {}

    def setUp(self) -> None:
        kwargs = {"vad_factory": StubVAD, **self.server_kwargs}
        self.server = pa.PhoneAudioServer(host="127.0.0.1", port=0, **kwargs)
        self.server.start()
        self.phones: list[FakePhone] = []
        self.addCleanup(self.server.stop)
        self.addCleanup(lambda: [p.ws.close() for p in self.phones])

    def phone(self, **kw) -> FakePhone:
        p = FakePhone(self.server.port, **kw)
        self.phones.append(p)
        return p

    def connect_ready(self, **kw) -> FakePhone:
        p = self.phone(**kw)
        self.assertTrue(wait_for(self.server.connected))
        return p


class TestHandshake(PhoneServerBase):
    def test_hello_gets_ready_and_state(self):
        p = self.connect_ready()
        self.assertTrue(wait_for(lambda: len(p.received) >= 2))
        ready = json.loads(p.received[0])
        self.assertEqual(ready["type"], "ready")
        self.assertEqual(ready["server"], "algolearn-speak")
        self.assertEqual(json.loads(p.received[1]), {"type": "state", "value": "idle"})

    def test_not_connected_before_hello(self):
        p = self.phone(hello=None)
        time.sleep(0.2)
        self.assertFalse(self.server.connected())
        p.ws.send(json.dumps(HELLO))
        self.assertTrue(wait_for(self.server.connected))

    def test_second_phone_refused_busy(self):
        first = self.connect_ready()
        second = self.phone()
        self.assertTrue(second.closed.wait(5))
        self.assertEqual((second.close_code, second.close_reason), (pa.CLOSE_BUSY, "busy"))
        self.assertTrue(self.server.connected())  # the first phone is undisturbed
        first.close()
        self.assertTrue(wait_for(lambda: not self.server.connected()))

    def test_bad_rates_refused(self):
        p = self.phone(hello={**HELLO, "mic_rate": 48000})
        self.assertTrue(p.closed.wait(5))
        self.assertEqual((p.close_code, p.close_reason), (pa.CLOSE_POLICY, "bad-rates"))
        self.assertEqual(pa.CLOSE_POLICY, 1008)
        self.assertFalse(self.server.connected())

    def test_slot_freed_after_disconnect(self):
        self.connect_ready().close()
        self.assertTrue(wait_for(lambda: not self.server.connected()))
        self.connect_ready()

    def test_state_message(self):
        p = self.connect_ready()
        self.server.set_state("processing")
        self.assertTrue(wait_for(lambda: {"type": "state", "value": "processing"} in [json.loads(m) for m in p.received if isinstance(m, str)]))
        with self.assertRaises(ValueError):
            self.server.set_state("dancing")

    def test_ping_answered(self):
        p = self.connect_ready()
        p.ws.send(json.dumps({"type": "ping"}))
        self.assertTrue(wait_for(lambda: "pong" in p.types()))


class TestPlay(PhoneServerBase):
    def test_play_sends_pcm16_and_waits_for_played(self):
        p = self.connect_ready(ack=False)
        audio = (0.5 * np.sin(np.linspace(0, 100, 2400))).astype(np.float32)
        done = threading.Event()
        threading.Thread(target=lambda: (self.server.play(audio, 24000), done.set()), daemon=True).start()
        self.assertTrue(wait_for(lambda: "play_end" in p.types()))
        self.assertFalse(done.wait(0.3), "play() must block until the phone reports played")
        play_end = [json.loads(m) for m in p.received if isinstance(m, str) and json.loads(m)["type"] == "play_end"][0]
        p.ws.send(json.dumps({"type": "played", "id": play_end["id"]}))
        self.assertTrue(done.wait(5))
        start = [json.loads(m) for m in p.received if isinstance(m, str) and json.loads(m)["type"] == "play_start"][0]
        self.assertEqual(start["rate"], 24000)
        pcm = b"".join(m for m in p.received if isinstance(m, bytes))
        self.assertEqual(len(pcm), 2400 * 2)
        np.testing.assert_allclose(pa.float_from_pcm16(pcm), audio, atol=1 / 16000)
        self.assertIn({"type": "state", "value": "speaking"}, [json.loads(m) for m in p.received if isinstance(m, str)])

    def test_play_order_is_start_chunks_end(self):
        p = self.connect_ready()
        self.server.play(np.zeros(30000, dtype=np.float32), 24000)  # 60000 bytes -> 2 chunks
        kinds = ["bin" if isinstance(m, bytes) else json.loads(m)["type"] for m in p.received]
        kinds = [k for k in kinds if k != "state"]
        self.assertEqual(kinds[kinds.index("play_start"):], ["play_start", "bin", "bin", "play_end"])

    def test_play_segments_ids_increase(self):
        p = self.connect_ready()
        self.server.play_segments([np.zeros(240, dtype=np.float32)] * 3, 24000)
        ids = [json.loads(m)["id"] for m in p.received if isinstance(m, str) and json.loads(m)["type"] == "play_end"]
        self.assertEqual(ids, sorted(set(ids)))
        self.assertEqual(len(ids), 3)

    def test_wrong_rate_raises(self):
        self.connect_ready()
        with self.assertRaises(ValueError):
            self.server.play(np.zeros(10, dtype=np.float32), 16000)

    def test_play_without_phone_raises(self):
        with self.assertRaises(RuntimeError):
            self.server.play(np.zeros(10, dtype=np.float32), 24000)

    def test_played_timeout_raises(self):
        self.server._played_grace = 0.3
        self.connect_ready(ack=False)
        with self.assertRaisesRegex(RuntimeError, "did not report played"):
            self.server.play(np.zeros(240, dtype=np.float32), 24000)

    def test_disconnect_mid_play_raises(self):
        p = self.connect_ready(ack=False)
        threading.Timer(0.3, p.ws.close).start()
        with self.assertRaisesRegex(RuntimeError, "phone disconnected during speak"):
            self.server.play(np.zeros(240, dtype=np.float32), 24000)

    def test_unknown_played_id_is_protocol_error(self):
        p = self.connect_ready()
        p.ws.send(json.dumps({"type": "played", "id": 999}))
        self.assertTrue(p.closed.wait(5))
        self.assertEqual(p.close_code, pa.CLOSE_PROTOCOL)


class TestRecord(PhoneServerBase):
    def test_record_control_sequence_and_slice(self):
        # 20 silent blocks, 30 speech, 20 silent tail; silence_seconds 0.2 -> 6 quiet blocks end it.
        p = self.connect_ready(mic_blocks=speech_blocks(20, 30, 20))
        cue = np.zeros(2400, dtype=np.float32)
        audio = self.server.record(max_seconds=10, silence_seconds=0.2, start_timeout_seconds=5, cue=cue)
        self.assertIsInstance(audio, np.ndarray)
        self.assertEqual(audio.dtype, np.float32)
        # 15 blocks pre-roll (0.5 s, includes the start block) + 29 more speech + 6 quiet = 50 blocks.
        self.assertEqual(len(audio), 50 * BLOCK)
        # pre-roll is silent (14 blocks), then speech
        self.assertLess(float(np.abs(audio[: 14 * BLOCK]).max()), 1e-3)
        self.assertGreater(float(np.abs(audio[14 * BLOCK: 15 * BLOCK]).mean()), 0.05)
        self.assertTrue(wait_for(lambda: "mic_stop" in p.types()))
        seq = [t for t in p.types() if t in ("play_start", "play_end", "mic_start", "mic_stop")]
        self.assertEqual(seq, ["play_start", "play_end", "mic_start", "mic_stop"])
        states = [json.loads(m)["value"] for m in p.received if isinstance(m, str) and json.loads(m)["type"] == "state"]
        self.assertIn("listening", states)

    def test_cue_played_before_mic_start(self):
        p = self.connect_ready(ack=False, mic_blocks=speech_blocks(0, 10, 10))
        result: list = []
        t = threading.Thread(
            target=lambda: result.append(self.server.record(5, 0.2, 5, cue=np.zeros(240, dtype=np.float32))), daemon=True,
        )
        t.start()
        self.assertTrue(wait_for(lambda: "play_end" in p.types()))
        time.sleep(0.3)
        self.assertNotIn("mic_start", p.types(), "mic must not open before the cue is reported played")
        end = [json.loads(m) for m in p.received if isinstance(m, str) and json.loads(m)["type"] == "play_end"][0]
        p.ws.send(json.dumps({"type": "played", "id": end["id"]}))
        t.join(5)
        self.assertEqual(len(result), 1)

    def test_timeout_when_nobody_speaks(self):
        p = self.connect_ready(mic_blocks=speech_blocks(5, 0, 0))
        result = self.server.record(max_seconds=5, silence_seconds=0.2, start_timeout_seconds=0.4)
        self.assertIs(result, pa.TIMEOUT)
        self.assertTrue(wait_for(lambda: "mic_stop" in p.types()))

    def test_hard_cap_returns_capture(self):
        # Speech that never ends: capture stops at start_timeout + max_seconds.
        p = self.connect_ready(mic_blocks=speech_blocks(0, 10, 0))
        t0 = time.monotonic()
        audio = self.server.record(max_seconds=0.3, silence_seconds=2.0, start_timeout_seconds=0.2)
        self.assertGreaterEqual(time.monotonic() - t0, 0.5)
        self.assertEqual(len(audio), 10 * BLOCK)

    def test_mic_data_outside_mic_window_is_not_captured(self):
        p = self.connect_ready()
        p.ws.send(speech_blocks(0, 5, 0)[0])  # unsolicited; must be dropped
        time.sleep(0.2)
        self.assertIs(self.server.record(5, 0.2, 0.3), pa.TIMEOUT)

    def test_disconnect_mid_record_raises(self):
        p = self.connect_ready()
        threading.Timer(0.3, p.ws.close).start()
        with self.assertRaisesRegex(RuntimeError, "phone disconnected during listen"):
            self.server.record(max_seconds=5, silence_seconds=0.2, start_timeout_seconds=5)

    def test_silent_mic_raises(self):
        self.server._mic_first_block_timeout = 0.3
        self.connect_ready()
        with self.assertRaisesRegex(RuntimeError, "no microphone audio"):
            self.server.record(max_seconds=5, silence_seconds=0.2, start_timeout_seconds=5)

    def test_record_without_phone_raises(self):
        with self.assertRaises(RuntimeError):
            self.server.record(5, 0.2, 1)


class TestRealSilero(unittest.TestCase):
    def test_silero_on_digital_silence_times_out(self):
        server = pa.PhoneAudioServer(host="127.0.0.1", port=0)  # default factory = real Silero
        server.start()
        self.addCleanup(server.stop)
        phone = FakePhone(server.port, mic_blocks=speech_blocks(40, 0, 0))
        self.addCleanup(phone.ws.close)
        self.assertTrue(wait_for(server.connected))
        self.assertIs(server.record(5, 0.5, 0.5), pa.TIMEOUT)


class TestRobustness(unittest.TestCase):
    def make(self, **kw):
        server = pa.PhoneAudioServer(host="127.0.0.1", port=0, vad_factory=StubVAD, **kw)
        server.start()
        self.addCleanup(server.stop)
        return server

    def phone(self, server, **kw):
        p = FakePhone(server.port, **kw)
        self.addCleanup(p.ws.close)
        self.assertTrue(wait_for(server.connected))
        return p

    def test_bind_conflict_names_the_env_var(self):
        import socket

        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        sock.listen()
        self.addCleanup(sock.close)
        server = pa.PhoneAudioServer(host="127.0.0.1", port=sock.getsockname()[1], vad_factory=StubVAD)
        with self.assertRaisesRegex(RuntimeError, "SPEAK_PHONE_PORT"):
            server.start()

    def test_unanswered_pings_close_with_ping_timeout_and_blocked_play_raises(self):
        server = self.make(ping_interval_seconds=0.1)
        p = self.phone(server, ack=False, answer_pings=False)
        with self.assertRaisesRegex(RuntimeError, "phone disconnected during speak"):
            server.play(np.zeros(240, dtype=np.float32), 24000)
        self.assertTrue(p.closed.wait(5))
        self.assertEqual(p.close_reason, "ping timeout")
        self.assertFalse(server.connected())

    def test_send_that_never_completes_aborts_and_raises_unresponsive(self):
        import asyncio

        server = self.make()
        p = self.phone(server)
        ws = server._ws
        with self.assertRaisesRegex(RuntimeError, "phone unresponsive during listen"):
            server._run(asyncio.sleep(30), "listen", ws, 0.3)
        self.assertTrue(p.closed.wait(5))
        self.assertTrue(wait_for(lambda: not server.connected()))

    def test_stale_slot_is_replaced_when_holder_does_not_answer(self):
        server = self.make()
        first = self.phone(server)
        with mock.patch.object(server, "_alive", mock.AsyncMock(return_value=False)):
            second = FakePhone(server.port)
            self.addCleanup(second.ws.close)
            self.assertTrue(first.closed.wait(5), "the stale connection must be dropped")
            self.assertTrue(wait_for(lambda: "ready" in second.types()))
        self.assertTrue(server.connected())

    def test_live_holder_still_gets_busy(self):
        server = self.make()
        self.phone(server)
        second = FakePhone(server.port)
        self.addCleanup(second.ws.close)
        self.assertTrue(second.closed.wait(5))
        self.assertEqual(second.close_reason, "busy")

    def test_multi_megabyte_play_through_slow_reader(self):
        server = self.make()
        p = self.phone(server, read_delay=0.001)
        rng = np.random.default_rng(3)
        segs = [(0.5 * rng.standard_normal(24000 * 30)).astype(np.float32) for _ in range(3)]  # 3 x 1.44 MB
        server.play_segments(segs, 24000)
        pcm = b"".join(m for m in p.received if isinstance(m, bytes))
        self.assertEqual(len(pcm), 3 * 24000 * 30 * 2)
        self.assertEqual(p.types().count("play_end"), 3)

    def test_played_timeout_pops_pending(self):
        server = self.make(played_grace_seconds=0.2)
        self.phone(server, ack=False)
        with self.assertRaises(RuntimeError):
            server.play(np.zeros(240, dtype=np.float32), 24000)
        self.assertEqual(server._pending, {})

    def test_cue_state_is_listening_not_speaking(self):
        server = self.make()
        p = self.phone(server, mic_blocks=speech_blocks(0, 5, 10))
        server.record(5, 0.2, 5, cue=np.zeros(240, dtype=np.float32))
        states = [json.loads(m)["value"] for m in p.received if isinstance(m, str) and json.loads(m)["type"] == "state"]
        self.assertNotIn("speaking", states)
        self.assertIn("listening", states)

    def test_set_state_after_connection_gone_does_not_raise(self):
        server = self.make()
        p = self.phone(server)
        p.ws.close()
        p.closed.wait(5)
        server.set_state("idle")  # connection gone or going: logged, not raised


class TestSyncRouting(unittest.TestCase):
    """The real _speak_sync/_listen_sync/_converse_sync through _begin_call/_end_call with a real PhoneAudioServer."""

    def setUp(self):
        import shutil
        import tempfile

        self.server = pa.PhoneAudioServer(host="127.0.0.1", port=0, vad_factory=StubVAD)
        self.server.start()
        self.addCleanup(self.server.stop)
        self.tmp = tempfile.mkdtemp(prefix="speak-phone-lock-")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.lock = s._SerializingLock(os.path.join(self.tmp, "a.lock"), os.path.join(self.tmp, "a.json"))
        for name, val in (("phone", self.server), ("audio_lock", self.lock)):
            orig = getattr(s, name)
            setattr(s, name, val)
            self.addCleanup(setattr, s, name, orig)
        orig_ready, orig_err, orig_kokoro = s.engines.ready.is_set(), s.engines.error, s.engines.kokoro
        s.engines.ready.set()
        s.engines.error = None

        class FakeKokoro:
            def generate(self, **kwargs):
                class R:
                    audio = np.zeros(2400, dtype=np.float32)

                yield R()
                yield R()

        s.engines.kokoro = FakeKokoro()

        def restore():
            s.engines.kokoro, s.engines.error = orig_kokoro, orig_err
            if not orig_ready:
                s.engines.ready.clear()

        self.addCleanup(restore)

    def connect(self, **kw):
        p = FakePhone(self.server.port, **kw)
        self.addCleanup(p.ws.close)
        self.assertTrue(wait_for(self.server.connected))
        return p

    def test_speak_sync_routes_to_phone_and_returns_idle(self):
        p = self.connect()
        seconds, route = s._speak_sync("hello there")
        self.assertEqual(route, "phone")
        self.assertAlmostEqual(seconds, 0.2)
        self.assertEqual(p.types().count("play_end"), 2)  # one segment per sentence-chunk
        self.assertEqual(s._route, "mac")
        self.assertTrue(wait_for(lambda: [json.loads(m)["value"] for m in p.received if isinstance(m, str) and json.loads(m)["type"] == "state"][-1] == "idle"))
        self.assertEqual(self.lock.snapshot(), (False, None, 0))

    def test_speak_sync_uses_mac_when_no_phone(self):
        with mock.patch.object(s, "_play_pcm_stream") as stream:
            seconds, route = s._speak_sync("hello there")
        self.assertEqual(route, "mac")
        stream.assert_called_once()

    def test_listen_sync_routes_to_phone_end_to_end(self):
        p = self.connect(mic_blocks=speech_blocks(5, 20, 20))
        import mlx_whisper

        with mock.patch.object(mlx_whisper, "transcribe", return_value={"text": " hello "}):
            text, route = s._listen_sync(10, 0.2, 5)
        self.assertEqual((text, route), ("hello", "phone"))
        types = p.types()
        self.assertLess(types.index("mic_start"), types.index("mic_stop"))
        states = [json.loads(m)["value"] for m in p.received if isinstance(m, str) and json.loads(m)["type"] == "state"]
        self.assertIn("processing", states)
        self.assertEqual(self.lock.snapshot(), (False, None, 0))

    def test_disconnect_mid_call_raises_and_lock_is_released(self):
        p = self.connect()
        threading.Thread(target=lambda: (wait_for(lambda: "mic_start" in p.types(), 30), p.ws.close()), daemon=True).start()
        with self.assertRaisesRegex(RuntimeError, "phone disconnected during listen"):
            s._listen_sync(5, 0.2, 5)
        self.assertEqual(self.lock.snapshot(), (False, None, 0))
        self.assertEqual(s._route, "mac")

    def test_lock_released_when_end_call_raises(self):
        self.connect()
        for fn, args in ((s._speak_sync, ("hi",)), (s._listen_sync, (1, 0.2, 1)), (s._converse_sync, ("hi", 1, 0.2, 1))):
            with mock.patch.object(s, "_end_call", side_effect=RuntimeError("boom")), \
                    mock.patch.object(s, "_speak_impl", return_value=0.1), \
                    mock.patch.object(s, "_listen_impl", return_value="x"):
                with self.assertRaisesRegex(RuntimeError, "boom"):
                    fn(*args)
            self.assertEqual(self.lock.snapshot(), (False, None, 0), fn.__name__)


class TestRouting(unittest.TestCase):
    """speak_server picks phone vs mac once per call, from phone.connected()."""

    def setUp(self) -> None:
        self.orig_phone = s.phone
        self.addCleanup(setattr, s, "phone", self.orig_phone)

    def fake_phone(self, connected: bool):
        ph = mock.Mock()
        ph.connected.return_value = connected
        s.phone = ph
        return ph

    def test_begin_call_follows_connected(self):
        self.fake_phone(False)
        self.assertEqual(s._begin_call("speak"), "mac")
        self.fake_phone(True)
        self.assertEqual(s._begin_call("speak"), "phone")
        s._end_call("phone")
        self.assertEqual(s._route, "mac")

    def test_play_pcm_goes_to_phone_only_when_routed(self):
        ph = self.fake_phone(True)
        audio = np.zeros(100, dtype=np.float32)
        with mock.patch.object(s, "_run_worker") as worker:
            s._begin_call("speak")
            s._play_pcm(audio, 24000)
            s._end_call("phone")
            ph.play.assert_called_once()
            worker.assert_not_called()
        ph2 = self.fake_phone(False)
        with mock.patch.object(s, "_run_worker") as worker:
            s._begin_call("speak")
            s._play_pcm(audio, 24000)
            s._end_call("mac")
            worker.assert_called_once()
            ph2.play.assert_not_called()

    def test_play_stream_sends_one_segment_per_chunk(self):
        ph = self.fake_phone(True)
        chunks = [np.zeros((10, 1), dtype=np.float32), np.zeros((20, 1), dtype=np.float32)]
        s._begin_call("speak")
        try:
            s._play_pcm_stream(chunks, 24000, timeout=1.0)
        finally:
            s._end_call("phone")
        segments = ph.play_segments.call_args.args[0]
        self.assertEqual([len(x) for x in segments], [10, 20])
        self.assertEqual(ph.play_segments.call_args.args[1], 24000)

    def test_route_is_fixed_for_the_whole_call(self):
        # The phone drops after the call started: the call still targets the phone (and would raise there).
        ph = self.fake_phone(True)
        s._begin_call("speak")
        ph.connected.return_value = False
        try:
            s._play_pcm(np.zeros(10, dtype=np.float32), 24000)
        finally:
            s._end_call("phone")
        ph.play.assert_called_once()

    def test_record_routed_to_phone_timeout_raises_timeouterror(self):
        ph = self.fake_phone(True)
        ph.record.return_value = pa.TIMEOUT
        s._begin_call("speak")
        try:
            with self.assertRaises(TimeoutError):
                s._record_for_route(5, 1.0, 3)
        finally:
            s._end_call("phone")
        args = ph.record.call_args
        self.assertEqual(args.args, (5, 1.0, 3))
        self.assertEqual(args.kwargs["cue"].dtype, np.float32)

    def test_status_reports_audio(self):
        import asyncio

        self.fake_phone(True)
        self.assertEqual(asyncio.run(s.status())["audio"], "phone")
        self.fake_phone(False)
        self.assertEqual(asyncio.run(s.status())["audio"], "mac")

    def test_tool_results_carry_audio_note(self):
        import asyncio

        with mock.patch.object(s, "_speak_sync", return_value=(1.5, "phone")):
            self.assertEqual(asyncio.run(s.speak("hi")), "spoke for 1.5s\n(audio: phone)")
        with mock.patch.object(s, "_listen_sync", return_value=("hello there", "mac")):
            self.assertEqual(asyncio.run(s.listen()), "hello there\n(audio: mac)")
        with mock.patch.object(s, "_converse_sync", return_value=("yes", "phone")):
            self.assertEqual(asyncio.run(s.converse("q")), "yes\n(audio: phone)")


if __name__ == "__main__":
    unittest.main()
