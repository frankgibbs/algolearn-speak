"""Tests for speak_telegram.TelegramLink (docs/DESIGN_TELEGRAM_CALL.md section 12).

Telegram is never contacted: Telethon and py-tgcalls are replaced by fakes that
script the call (answer, decline, busy, ring-out, remote hang-up) and the
owner's incoming audio. py-tgcalls' real update/frame/exception types are used
so the link's dispatch is exercised as it runs against the library.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
import wave
from types import SimpleNamespace
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np
from pytgcalls.exceptions import CallBusy, CallDeclined, CallDiscarded, TimedOutAnswer
from pytgcalls.types import ChatUpdate, Device, Direction, Frame, StreamFrames

import speak_telegram as tg

USER_ID = 4242
FAKE_CAFFEINATE = os.path.join(HERE, "fake_caffeinate")


class FakeTelethon:
    authorized = True

    async def connect(self):
        pass

    async def is_user_authorized(self):
        return self.authorized

    async def get_entity(self, target):
        return SimpleNamespace(id=USER_ID)

    async def disconnect(self):
        pass


class FakeCalls:
    """Scripted stand-in for PyTgCalls. `ring` is "answer", or an exception to raise from play()."""

    def __init__(self, client, ring="answer"):
        self.ring = ring
        self.ring_forever = False
        self.fail_record: BaseException | None = None
        self.fail_send: BaseException | None = None
        self.handler = None
        self.plays: list = []
        self.records: list = []
        self.frames: list[bytes] = []
        self.left = 0

    def on_update(self, *_filters):
        def deco(fn):
            self.handler = fn
            return fn
        return deco

    async def start(self):
        pass

    async def play(self, chat_id, stream, config=None):
        self.plays.append((chat_id, stream, config))
        if config is not None:   # the ringing play()
            await asyncio.sleep(0.01)
            while self.ring_forever:
                await asyncio.sleep(0.01)
            if isinstance(self.ring, BaseException):
                raise self.ring

    async def record(self, chat_id, stream):
        self.records.append((chat_id, stream))
        if self.fail_record:
            raise self.fail_record

    async def send_frame(self, chat_id, device, data):
        assert device == Device.MICROPHONE and len(data) == tg.FRAME_BYTES
        if self.fail_send:
            raise self.fail_send
        self.frames.append(data)

    async def leave_call(self, chat_id):
        self.left += 1


class StubVAD:
    """Energy VAD with Silero's event shape: start on the first loud block, end after `silence` quiet seconds."""

    def __init__(self, silence_seconds: float):
        self.need = max(1, int(silence_seconds * tg.VAD_RATE / tg.VAD_BLOCK))
        self.speaking = False
        self.quiet = 0

    def __call__(self, block):
        loud = float(np.sqrt(np.mean(block ** 2))) > 0.02
        if loud and not self.speaking:
            self.speaking = True
            self.quiet = 0
            return {"start": 0}
        if self.speaking:
            self.quiet = 0 if loud else self.quiet + 1
            if self.quiet >= self.need:
                self.speaking = False
                return {"end": 0}
        return None


def speech_frames(seconds: float, loud: bool = True) -> list[bytes]:
    """10 ms 48 kHz stereo s16 frames: a 220 Hz tone, or silence."""
    out = []
    t = np.arange(tg.CALL_RATE // 100) / tg.CALL_RATE
    for i in range(int(seconds * 100)):
        mono = (0.3 * np.sin(2 * np.pi * 220 * (t + i * 0.01))) if loud else np.zeros_like(t)
        out.append(np.repeat((mono * 32767).astype(np.int16), 2).tobytes())
    return out


class LinkCase(unittest.TestCase):
    ring = "answer"

    def setUp(self):
        self.log_dir = tempfile.mkdtemp(prefix="fake-caffeinate-")
        self.addCleanup(shutil.rmtree, self.log_dir, True)
        self.caffeinate_log = os.path.join(self.log_dir, "log")
        p = mock.patch.dict(os.environ, {"FAKE_CAFFEINATE_LOG": self.caffeinate_log})
        p.start()
        self.addCleanup(p.stop)
        for name, value in (("OPENER_MIN_SECONDS", 0.05), ("OPENER_TAIL_SECONDS", 0.02)):
            p = mock.patch.object(tg, name, value)
            p.start()
            self.addCleanup(p.stop)
        self.calls = None

        def calls_factory(client):
            self.calls = FakeCalls(client, self.ring)
            return self.calls

        self.link = tg.TelegramLink("owner", FakeTelethon, calls_factory, vad_factory=StubVAD, caffeinate=FAKE_CAFFEINATE)
        self.link.start()
        self.addCleanup(self.link.stop)

    def emit(self, update):
        asyncio.run_coroutine_threadsafe(self.calls.handler(self.calls, update), self.link._loop).result(5)

    def remote_hang_up(self):
        self.emit(ChatUpdate(USER_ID, ChatUpdate.Status.DISCARDED_CALL))

    def push_incoming(self, frames: list[bytes], pace: float = 0.0):
        for f in frames:
            self.emit(StreamFrames(USER_ID, Direction.INCOMING, Device.MICROPHONE, [Frame(1, f, Frame.Info())]))
            if pace:
                time.sleep(pace)

    def caffeinate_lines(self) -> list[str]:
        if not os.path.exists(self.caffeinate_log):
            return []
        with open(self.caffeinate_log) as f:
            return f.read().splitlines()

    def wait_until(self, predicate, timeout=5.0):
        deadline = time.monotonic() + timeout
        while not predicate():
            if time.monotonic() > deadline:
                self.fail("condition not met in time")
            time.sleep(0.01)


class TestCallAnswered(LinkCase):
    def test_answer_flow_matches_the_spike(self):
        outcome = self.link.call(np.zeros(2400, dtype=np.float32), 24000)
        self.assertEqual(outcome, tg.ANSWERED)
        self.assertEqual(self.link.state(), tg.CONNECTED)
        (uid1, opener, config), (uid2, raw, no_config) = self.calls.plays
        self.assertEqual((uid1, uid2), (USER_ID, USER_ID))
        self.assertIsNotNone(config)                      # ringing play() carries the CallConfig
        self.assertIsNone(no_config)                      # the switch to raw frames does not
        self.assertEqual(len(self.calls.records), 1)      # recording starts before the switch
        self.wait_until(lambda: len(self.calls.frames) >= 5)   # the pump is feeding silence
        self.assertEqual(self.calls.frames[0], bytes(tg.FRAME_BYTES))
        self.wait_until(lambda: self.caffeinate_lines() == [f"-i -w {os.getpid()}"])

    def test_already_connected(self):
        self.link.call(np.zeros(100, dtype=np.float32), 24000)
        self.assertEqual(self.link.call(np.zeros(100, dtype=np.float32), 24000), tg.ALREADY_CONNECTED)
        self.assertEqual(len(self.calls.plays), 2)   # no second dial

    def test_play_sends_the_audio_paced_and_returns(self):
        self.link.call(np.zeros(100, dtype=np.float32), 24000)
        n_before = len(self.calls.frames)
        tone = (0.3 * np.sin(2 * np.pi * 440 * np.arange(12000) / 24000)).astype(np.float32)  # 0.5 s
        t0 = time.monotonic()
        self.link.play(tone, 24000)
        elapsed = time.monotonic() - t0
        self.assertGreater(elapsed, 0.4)   # paced at real time, not dumped
        loud = [f for f in self.calls.frames[n_before:] if f != bytes(tg.FRAME_BYTES)]
        self.assertEqual(len(loud), 50)    # 0.5 s = 50 frames of 10 ms

    def test_remote_hang_up_ends_everything(self):
        self.link.call(np.zeros(100, dtype=np.float32), 24000)
        self.wait_until(lambda: self.caffeinate_lines() != [])
        self.remote_hang_up()
        self.assertEqual(self.link.state(), tg.NONE)
        self.assertIsNone(self.link._caffeinate)
        n = len(self.calls.frames)
        time.sleep(0.1)
        self.assertLessEqual(len(self.calls.frames) - n, 1)   # pump stopped
        with self.assertRaisesRegex(tg.CallEnded, "no connected"):
            self.link.play(np.zeros(100, dtype=np.float32), 24000)

    def test_hang_up_during_play_raises_call_ended(self):
        self.link.call(np.zeros(100, dtype=np.float32), 24000)
        threading.Timer(0.1, self.remote_hang_up).start()
        with self.assertRaisesRegex(tg.CallEnded, "during playback"):
            self.link.play(np.zeros(48000, dtype=np.float32), 24000)   # 2 s

    def test_other_chats_are_ignored(self):
        self.link.call(np.zeros(100, dtype=np.float32), 24000)
        self.emit(ChatUpdate(USER_ID + 1, ChatUpdate.Status.DISCARDED_CALL))
        self.assertEqual(self.link.state(), tg.CONNECTED)

    def test_hang_up(self):
        self.link.call(np.zeros(100, dtype=np.float32), 24000)
        self.assertEqual(self.link.hang_up(), tg.HUNG_UP)
        self.assertEqual(self.calls.left, 1)
        self.assertEqual(self.link.state(), tg.NONE)
        self.assertEqual(self.link.hang_up(), tg.NO_ACTIVE_CALL)


class TestReviewFindings(LinkCase):
    def test_a_dead_pump_ends_the_call(self):
        self.link.call(np.zeros(100, dtype=np.float32), 24000)
        self.calls.fail_send = RuntimeError("ntgcalls exploded")
        self.wait_until(lambda: self.link.state() == tg.NONE)
        self.assertEqual(self.link.info()["end_reason"], tg.END_FAILED)
        self.wait_until(lambda: self.calls.left >= 1)
        with self.assertRaisesRegex(tg.CallEnded, "no connected"):
            self.link.play(np.zeros(100, dtype=np.float32), 24000)

    def test_library_dropping_the_call_is_a_remote_end(self):
        from pytgcalls.exceptions import NotInCallError
        self.link.call(np.zeros(100, dtype=np.float32), 24000)
        self.calls.fail_send = NotInCallError()   # the owner hung up; the pump notices first
        self.wait_until(lambda: self.link.state() == tg.NONE)
        self.assertEqual(self.link.info()["end_reason"], tg.END_REMOTE)

    def test_our_hang_up_is_recorded_even_when_the_pump_notices_first(self):
        from pytgcalls.exceptions import NotInCallError
        self.link.call(np.zeros(100, dtype=np.float32), 24000)
        calls = self.calls

        async def leave(chat_id):   # like the library: leaving makes send_frame fail
            calls.fail_send = NotInCallError()
            await asyncio.sleep(0.05)
            calls.left += 1
        calls.leave_call = leave
        self.assertEqual(self.link.hang_up(), tg.HUNG_UP)
        time.sleep(0.1)
        self.assertEqual(self.link.info()["end_reason"], tg.END_HUNG_UP)

    def test_failure_after_answer_tears_the_call_down(self):
        self.calls.fail_record = RuntimeError("TelegramServerError")
        with self.assertRaisesRegex(RuntimeError, "TelegramServerError"):
            self.link.call(np.zeros(100, dtype=np.float32), 24000)
        self.assertEqual(self.link.state(), tg.NONE)
        self.assertEqual(self.calls.left, 1)
        self.assertIsNone(self.link._caffeinate)

    def test_hang_up_during_the_greeting_counts_as_not_answered(self):
        with mock.patch.object(tg, "OPENER_TAIL_SECONDS", 0.5):
            threading.Timer(0.2, self.remote_hang_up).start()
            self.assertEqual(self.link.call(np.zeros(100, dtype=np.float32), 24000), tg.NOT_ANSWERED)
        self.assertEqual(len(self.calls.plays), 1)   # never switched to raw frames

    def test_hang_up_when_the_library_already_dropped_it(self):
        from pytgcalls.exceptions import NotInCallError
        self.link.call(np.zeros(100, dtype=np.float32), 24000)

        async def gone(chat_id):
            raise NotInCallError()
        self.calls.leave_call = gone
        self.assertEqual(self.link.hang_up(), tg.HUNG_UP)
        self.assertEqual(self.link.info()["end_reason"], tg.END_HUNG_UP)

    def test_hang_up_while_ringing_cancels_the_ring(self):
        self.calls.ring_forever = True
        result = {}
        t = threading.Thread(target=lambda: result.setdefault("r", self._call_catching()))
        t.start()
        self.wait_until(lambda: self.link.state() == tg.RINGING)
        self.assertEqual(self.link.hang_up(), tg.HUNG_UP)
        t.join(5)
        self.assertEqual(self.link.state(), tg.NONE)
        self.assertEqual(self.link.info()["end_reason"], tg.END_HUNG_UP)
        self.assertIsInstance(result["r"], BaseException)   # the dial was cancelled, not answered

    def _call_catching(self):
        try:
            return self.link.call(np.zeros(100, dtype=np.float32), 24000)
        except BaseException as e:
            return e

    def test_late_end_update_does_not_end_the_next_call(self):
        self.link.call(np.zeros(100, dtype=np.float32), 24000)
        self.link.hang_up()
        self.link.call(np.zeros(100, dtype=np.float32), 24000)
        gen = self.link.info()["generation"]
        self.link._state = tg.NONE   # simulate the window where a stale update would arrive
        self.remote_hang_up()        # ignored: nothing is in progress
        self.link._state = tg.CONNECTED
        self.assertEqual(self.link.info(), {"state": tg.CONNECTED, "generation": gen, "end_reason": None})

    def test_a_timed_out_play_does_not_keep_playing(self):
        self.link.call(np.zeros(100, dtype=np.float32), 24000)
        with mock.patch.object(self.link, "_run", wraps=self.link._run) as run:
            def short(coro, timeout, what):
                return tg.TelegramLink._run(self.link, coro, 0.1, what)
            run.side_effect = short
            with self.assertRaisesRegex(TimeoutError, "playing audio into the call did not finish within 0s"):
                self.link.play(np.zeros(24000 * 2, dtype=np.float32), 24000)
        time.sleep(0.1)
        self.assertEqual(len(self.link._outgoing), 0)

    def test_generation_and_end_reason(self):
        self.assertEqual(self.link.info()["generation"], 0)
        self.link.call(np.zeros(100, dtype=np.float32), 24000)
        self.assertEqual(self.link.info(), {"state": tg.CONNECTED, "generation": 1, "end_reason": None})
        self.remote_hang_up()
        self.assertEqual(self.link.info(), {"state": tg.NONE, "generation": 1, "end_reason": tg.END_REMOTE})


class TestRevokedSession(unittest.TestCase):
    def test_refuses_to_start_instead_of_prompting(self):
        class Revoked(FakeTelethon):
            authorized = False
        link = tg.TelegramLink("owner", Revoked, lambda c: FakeCalls(c), vad_factory=StubVAD)
        with self.assertRaisesRegex(RuntimeError, "not authorized"):
            link.start()
        link._loop.call_soon_threadsafe(link._loop.stop)


class TestRecord(LinkCase):
    def test_captures_speech_until_silence_after_the_cue(self):
        self.link.call(np.zeros(100, dtype=np.float32), 24000)
        cue = (0.5 * np.sin(2 * np.pi * 880 * np.arange(2400) / 24000)).astype(np.float32)   # 0.1 s
        result = {}
        t = threading.Thread(target=lambda: result.setdefault("audio", self.link.record(10, 0.3, 5, cue, 24000)))
        t.start()
        self.wait_until(lambda: self.link._listening)
        loud = [f for f in self.calls.frames if f != bytes(tg.FRAME_BYTES)]
        self.assertEqual(len(loud), 10)   # the 0.1 s cue went out before listening began
        self.push_incoming(speech_frames(1.0) + speech_frames(0.6, loud=False))
        t.join(5)
        audio = result["audio"]
        self.assertEqual(audio.dtype, np.float32)
        self.assertGreater(len(audio) / tg.VAD_RATE, 0.9)   # the speech (plus pre-roll and tail)
        self.assertLess(len(audio) / tg.VAD_RATE, 2.0)

    def test_nobody_speaks(self):
        self.link.call(np.zeros(100, dtype=np.float32), 24000)
        self.assertIs(self.link.record(10, 0.3, 0.3, None, 24000), tg.TIMEOUT)

    def test_hang_up_during_listen_raises(self):
        self.link.call(np.zeros(100, dtype=np.float32), 24000)
        threading.Timer(0.2, self.remote_hang_up).start()
        with self.assertRaisesRegex(tg.CallEnded, "during listen"):
            self.link.record(10, 0.3, 5, None, 24000)

    def test_record_without_a_call_raises(self):
        with self.assertRaises(tg.CallEnded):
            self.link.record(10, 0.3, 5, None, 24000)


def _not_answered_case(exc):
    class Case(LinkCase):
        ring = exc

        def test_maps_to_not_answered(self):
            self.assertEqual(self.link.call(np.zeros(100, dtype=np.float32), 24000), tg.NOT_ANSWERED)
            self.assertEqual(self.link.state(), tg.NONE)
            self.assertEqual(self.caffeinate_lines(), [])
            self.assertEqual(len(self.calls.plays), 1)   # never switched to raw frames
    Case.__name__ = f"TestNotAnswered_{type(exc).__name__}"
    return Case


TestDeclined = _not_answered_case(CallDeclined(USER_ID))
TestBusy = _not_answered_case(CallBusy(USER_ID))
TestRingOut = _not_answered_case(TimedOutAnswer())
TestDiscarded = _not_answered_case(CallDiscarded(USER_ID))


class TestUnexpectedFailure(LinkCase):
    ring = ConnectionError("network down")

    def test_raises_and_resets(self):
        with self.assertRaisesRegex(ConnectionError, "network down"):
            self.link.call(np.zeros(100, dtype=np.float32), 24000)
        self.assertEqual(self.link.state(), tg.NONE)


class TestPcmHelpers(unittest.TestCase):
    def test_to_call_pcm_is_48k_stereo_s16(self):
        pcm = tg.to_call_pcm(np.zeros(24000, dtype=np.float32), 24000)   # 1 s at 24 kHz
        self.assertEqual(len(pcm), 48000 * 2 * 2)

    def test_opener_is_padded_to_the_minimum(self):
        path = os.path.join(tempfile.mkdtemp(prefix="opener-"), "o.wav")
        self.addCleanup(shutil.rmtree, os.path.dirname(path), True)
        seconds = tg.write_opener(path, np.zeros(12000, dtype=np.float32), 24000)   # 0.5 s of speech
        self.assertEqual(seconds, tg.OPENER_MIN_SECONDS)
        with wave.open(path) as w:
            self.assertEqual((w.getframerate(), w.getnchannels()), (48000, 1))
            self.assertAlmostEqual(w.getnframes() / 48000, 6.0, places=3)

    def test_long_greeting_is_not_cut(self):
        path = os.path.join(tempfile.mkdtemp(prefix="opener-"), "o.wav")
        self.addCleanup(shutil.rmtree, os.path.dirname(path), True)
        self.assertAlmostEqual(tg.write_opener(path, np.zeros(24000 * 8, dtype=np.float32), 24000), 8.0, places=3)


if __name__ == "__main__":
    unittest.main()
