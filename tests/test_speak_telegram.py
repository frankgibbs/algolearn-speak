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

import ntgcalls
import numpy as np
from pytgcalls.exceptions import CallBusy, CallDeclined, CallDiscarded, TimedOutAnswer
from pytgcalls.types import ChatUpdate, Device, Direction, Frame, RawCallUpdate, StreamFrames

import speak_telegram as tg

USER_ID = 4242


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
        self.accept_before_ring_error = False   # emit the callee's ACCEPTED update before play() raises
        self.mtproto_handler = None
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

    async def _handle_mtproto_updates(self, update):
        pass

    async def start(self):
        self.mtproto_handler = self._handle_mtproto_updates   # as py-tgcalls' start() registers it

    async def play(self, chat_id, stream, config=None):
        self.plays.append((chat_id, stream, config))
        if config is not None:   # the ringing play()
            await asyncio.sleep(0.01)
            while self.ring_forever:
                await asyncio.sleep(0.01)
            if isinstance(self.ring, BaseException):
                if self.accept_before_ring_error:
                    await self.mtproto_handler(RawCallUpdate(chat_id, RawCallUpdate.Type.ACCEPTED))
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
        for name, value in (("OPENER_MIN_SECONDS", 0.004), ("OPENER_TAIL_SECONDS", 0.02),
                            ("REDIAL_GAP_SECONDS", 0.0)):
            p = mock.patch.object(tg, name, value)
            p.start()
            self.addCleanup(p.stop)
        self.calls = None

        def calls_factory(client):
            self.calls = FakeCalls(client, self.ring)
            return self.calls

        self.link = tg.TelegramLink("owner", FakeTelethon, calls_factory, vad_factory=StubVAD, vad_preload=lambda: None)
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
        self.remote_hang_up()
        self.assertEqual(self.link.state(), tg.NONE)
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

    def test_hang_up_during_the_greeting_is_its_own_outcome(self):
        with mock.patch.object(tg, "OPENER_TAIL_SECONDS", 0.5):
            threading.Timer(0.2, self.remote_hang_up).start()
            self.assertEqual(self.link.call(np.zeros(100, dtype=np.float32), 24000), tg.HUNG_UP_DURING_GREETING)
        self.assertEqual(len(self.calls.plays), 1)   # never switched to raw frames
        self.assertEqual(self.link.state(), tg.NONE)
        self.assertIn("hung_up_during_greeting", [e["event"] for e in self.link.events.recent(50)])

    def test_hang_up_from_another_session_during_the_greeting_is_hung_up(self):
        with mock.patch.object(tg, "OPENER_TAIL_SECONDS", 0.5):
            threading.Timer(0.2, self.link.hang_up).start()
            self.assertEqual(self.link.call(np.zeros(100, dtype=np.float32), 24000), tg.HUNG_UP)
        self.assertEqual(self.link.state(), tg.NONE)
        self.assertEqual(self.link.info()["end_reason"], tg.END_HUNG_UP)

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


class TestVadPreload(unittest.TestCase):
    def link(self, preload):
        return tg.TelegramLink("owner", FakeTelethon, lambda c: FakeCalls(c), vad_factory=StubVAD,
                               vad_preload=preload)

    def test_start_loads_the_model_before_connecting(self):
        order = []

        class Client(FakeTelethon):
            async def connect(self):
                order.append("connect")
                await super().connect()

        link = tg.TelegramLink("owner", Client, lambda c: FakeCalls(c), vad_factory=StubVAD,
                               vad_preload=lambda: order.append("preload"))
        link.start()
        self.addCleanup(link.stop)
        self.assertEqual(order[0], "preload")

    def test_load_failure_fails_start(self):
        def boom():
            raise RuntimeError("silero load failed")
        link = self.link(boom)
        with self.assertRaisesRegex(RuntimeError, "silero load failed"):
            link.start()
        self.assertFalse(link._thread.is_alive())

    def test_factory_refuses_when_model_not_preloaded(self):
        with mock.patch.object(tg, "_silero_model", None):
            with self.assertRaisesRegex(RuntimeError, "not loaded"):
                tg.silero_vad_factory(1.0)


class TestRevokedSession(unittest.TestCase):
    def test_refuses_to_start_instead_of_prompting(self):
        class Revoked(FakeTelethon):
            authorized = False
        link = tg.TelegramLink("owner", Revoked, lambda c: FakeCalls(c), vad_factory=StubVAD,
                               vad_preload=lambda: None)
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
            self.assertEqual(len(self.calls.plays), 1)   # never switched to raw frames
    Case.__name__ = f"TestNotAnswered_{type(exc).__name__}"
    return Case


TestDeclined = _not_answered_case(CallDeclined(USER_ID))
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

    def test_short_opener_raises_instead_of_padding(self):
        path = os.path.join(tempfile.mkdtemp(prefix="opener-"), "o.wav")
        self.addCleanup(shutil.rmtree, os.path.dirname(path), True)
        with self.assertRaisesRegex(ValueError, "at least 6s"):
            tg.write_opener(path, np.zeros(12000, dtype=np.float32), 24000)   # 0.5 s of speech
        self.assertFalse(os.path.exists(path))

    def test_opener_at_the_minimum_is_written_as_48k_mono(self):
        path = os.path.join(tempfile.mkdtemp(prefix="opener-"), "o.wav")
        self.addCleanup(shutil.rmtree, os.path.dirname(path), True)
        seconds = tg.write_opener(path, np.zeros(int(tg.OPENER_MIN_SECONDS * 24000), dtype=np.float32), 24000)
        self.assertEqual(seconds, tg.OPENER_MIN_SECONDS)
        with wave.open(path) as w:
            self.assertEqual((w.getframerate(), w.getnchannels()), (48000, 1))

    def test_long_greeting_is_not_cut(self):
        path = os.path.join(tempfile.mkdtemp(prefix="opener-"), "o.wav")
        self.addCleanup(shutil.rmtree, os.path.dirname(path), True)
        self.assertAlmostEqual(tg.write_opener(path, np.zeros(24000 * 8, dtype=np.float32), 24000), 8.0, places=3)


if __name__ == "__main__":
    unittest.main()


class AcceptedCase(LinkCase):
    """The fake callee accepts (a RawCallUpdate reaches the link), then play() raises `ring`."""

    def setUp(self):
        super().setUp()
        self.calls.accept_before_ring_error = True


def _call(link):
    return link.call(np.zeros(100, dtype=np.float32), 24000)


class TestAudioFailure(AcceptedCase):
    ring = ntgcalls.TelegramServerError()   # empty message, as seen live

    def test_distinct_result_and_clean_teardown(self):
        out = _call(self.link)
        self.assertEqual(out, tg.AUDIO_FAILED_PREFIX + "TelegramServerError")
        self.assertTrue(tg.is_audio_failure(out))
        self.assertEqual(self.link.state(), tg.NONE)
        self.assertEqual(self.link.info()["end_reason"], tg.END_FAILED)
        self.assertEqual(self.calls.left, 1)

    def test_events_tell_the_story(self):
        _call(self.link)
        events = self.link.events.recent(50)
        self.assertEqual([e["event"] for e in events],
                         ["requested", "ringing", "accepted", "audio_failed", "ended"])
        self.assertEqual(len({e["call_id"] for e in events}), 1)
        self.assertEqual(events[3]["error"], "TelegramServerError")
        self.assertEqual(events[4]["reason"], tg.END_FAILED)
        self.assertIn("duration_s", events[4])


class TestServerErrorWithoutAcceptance(LinkCase):
    ring = ntgcalls.TelegramServerError()   # clear_call during ring: the callee never accepted

    def test_is_a_decline_not_an_audio_failure(self):
        out = _call(self.link)
        self.assertEqual(out, tg.NOT_ANSWERED)
        self.assertFalse(tg.is_audio_failure(out))
        self.assertEqual(self.link.state(), tg.NONE)
        self.assertEqual(self.link.info()["end_reason"], tg.END_NOT_ANSWERED)
        self.assertIn("discarded_before_accept", [e["event"] for e in self.link.events.recent(50)])


class TestOtherNtgcallsErrorsRaise(AcceptedCase):
    ring = ntgcalls.FFmpegError("ffprobe not installed")

    def test_raises_loudly_even_after_acceptance(self):
        with self.assertRaises(ntgcalls.FFmpegError):
            _call(self.link)
        self.assertEqual(self.link.state(), tg.NONE)
        self.assertEqual(self.link.info()["end_reason"], tg.END_FAILED)
        self.assertEqual(self.calls.left, 1)
        self.assertNotIn("audio_failed", [e["event"] for e in self.link.events.recent(50)])


class TestOtherNtgcallsErrorWithoutAcceptance(LinkCase):
    ring = ntgcalls.FileError("no such file")

    def test_raises(self):
        with self.assertRaises(ntgcalls.FileError):
            _call(self.link)
        self.assertEqual(self.link.state(), tg.NONE)


class TestAudioFailureOnRawSwitch(LinkCase):
    def _failing_switch(self, error):
        async def failing_play(chat_id, stream, config=None):
            if config is None:
                raise error
        self.calls.play = failing_play

    def test_server_error_after_answer_is_reported(self):
        self._failing_switch(ntgcalls.TelegramServerError())
        out = _call(self.link)
        self.assertEqual(out, tg.AUDIO_FAILED_PREFIX + "TelegramServerError")
        self.assertEqual(self.link.state(), tg.NONE)
        self.assertEqual(self.calls.left, 1)

    def test_other_ntgcalls_error_after_answer_raises(self):
        self._failing_switch(ntgcalls.ConnectionError("rtc down"))
        with self.assertRaises(ntgcalls.ConnectionError):
            _call(self.link)
        self.assertEqual(self.link.state(), tg.NONE)
        self.assertEqual(self.calls.left, 1)


class TestBusy(LinkCase):
    ring = CallBusy(USER_ID)

    def test_busy_is_its_own_outcome(self):
        self.assertEqual(_call(self.link), tg.BUSY)
        self.assertNotEqual(tg.BUSY, tg.NOT_ANSWERED)
        self.assertEqual(self.link.state(), tg.NONE)
        self.assertEqual(self.link.info()["end_reason"], tg.END_BUSY)
        self.assertEqual(len(self.calls.plays), 1)   # never switched to raw frames
        self.assertIn("busy", [e["event"] for e in self.link.events.recent(50)])


class TestRedialGap(LinkCase):
    ring = CallBusy(USER_ID)

    def setUp(self):
        super().setUp()
        p = mock.patch.object(tg, "REDIAL_GAP_SECONDS", 0.5)
        p.start()
        self.addCleanup(p.stop)

    def test_first_call_dials_at_once(self):
        t0 = time.monotonic()
        _call(self.link)
        self.assertLess(time.monotonic() - t0, 0.4)
        self.assertNotIn("redial_gap_wait", [e["event"] for e in self.link.events.recent(50)])

    def test_redial_waits_out_the_gap_after_the_last_call_ended(self):
        _call(self.link)
        t0 = time.monotonic()
        self.assertEqual(_call(self.link), tg.BUSY)
        self.assertGreaterEqual(time.monotonic() - t0, 0.45)
        self.assertEqual(len(self.calls.plays), 2)
        events = [e["event"] for e in self.link.events.recent(50)]
        self.assertIn("redial_gap_wait", events)
        self.assertLess(events.index("redial_gap_wait"), len(events) - 1 - events[::-1].index("requested"))

    def test_no_wait_once_the_gap_has_passed(self):
        _call(self.link)
        time.sleep(0.55)
        t0 = time.monotonic()
        _call(self.link)
        self.assertLess(time.monotonic() - t0, 0.4)
        self.assertNotIn("redial_gap_wait", [e["event"] for e in self.link.events.recent(50)])


class TestEventLog(unittest.TestCase):
    def test_ring_buffer_keeps_the_newest(self):
        log = tg.CallEventLog(size=3)
        for i in range(5):
            log.record("c", "e", i=i)
        self.assertEqual([e["i"] for e in log.recent(10)], [2, 3, 4])
        self.assertEqual([e["i"] for e in log.recent(2)], [3, 4])

    def test_describe(self):
        self.assertEqual(tg.describe(ntgcalls.TelegramServerError()), "TelegramServerError")
        self.assertEqual(tg.describe(ValueError("x")), "ValueError: x")


class TestAnsweredEvents(LinkCase):
    def test_success_path_events(self):
        self.link.call(np.zeros(100, dtype=np.float32), 24000)
        names = [e["event"] for e in self.link.events.recent(50)]
        names = [n for n in names if not n.startswith("audio_") and n != "opener_written"]
        self.assertEqual(names[:5], ["requested", "ringing", "answered_media_connected", "recording", "raw_frames"])
        self.assertEqual(names[5], "connected")

    def test_playback_timeline_events(self):
        self.link.call(np.zeros(100, dtype=np.float32), 24000)
        self.link.play(np.zeros(2400, dtype=np.float32), 24000)
        names = [e["event"] for e in self.link.events.recent(100)]
        for n in ("first_frame_sent", "playback_queued", "audio_out_start", "playback_done"):
            self.assertIn(n, names)
        self.assertLess(names.index("connected"), names.index("playback_queued"))
        self.assertLess(names.index("audio_out_start"), names.index("playback_done"))


def events_named(link, name):
    return [e for e in link.events.recent(500) if e["event"] == name]


class TestAudioStats(unittest.TestCase):
    def test_peak_rms_and_zero_frames(self):
        st = tg.AudioStats()
        st.add(bytes(tg.FRAME_BYTES))                     # all zero
        st.add(speech_frames(0.01)[0])                    # 0.3 FS sine
        f = st.fields()
        self.assertEqual((f["frames"], f["zero_frames"]), (2, 1))
        self.assertAlmostEqual(f["peak"], 0.3 * 32767, delta=5)
        # half the frames are silent, so RMS is 0.3/sqrt(2)/sqrt(2) of full scale
        self.assertAlmostEqual(f["rms"], 0.3 * 32767 / 2, delta=60)
        st.reset()
        self.assertEqual(st.fields()["frames"], 0)
        self.assertIsNone(st.fields()["peak_dbfs"])

    def test_wav_stats(self):
        path = os.path.join(tempfile.mkdtemp(prefix="wavstats-"), "o.wav")
        self.addCleanup(shutil.rmtree, os.path.dirname(path), True)
        tone = (0.5 * np.sin(2 * np.pi * 440 * np.arange(24000) / 24000)).astype(np.float32)
        with mock.patch.object(tg, "OPENER_MIN_SECONDS", 0.5):
            tg.write_opener(path, tone, 24000)
        st = tg.wav_stats(path)
        self.assertAlmostEqual(st["peak"], 0.5 * 32767, delta=100)
        self.assertAlmostEqual(st["rms"], 0.5 * 32767 / 2 ** 0.5, delta=80)


class TestAmplitudeEvents(LinkCase):
    def setUp(self):
        super().setUp()
        for name, value in (("STATS_FAST_INTERVAL", 0.05), ("STATS_FAST_SECONDS", 0.2), ("STATS_SLOW_INTERVAL", 0.1)):
            p = mock.patch.object(tg, name, value)
            p.start()
            self.addCleanup(p.stop)
        self.link.audio_stats = True

    def test_no_audio_events_while_debug_is_off(self):
        self.link.audio_stats = False
        self.link.call(np.zeros(100, dtype=np.float32), 24000)
        self.push_incoming(speech_frames(0.05, loud=True))
        time.sleep(0.3)
        self.link.hang_up()
        names = {e["event"] for e in self.link.events.recent(500)}
        self.assertFalse(names & {"opener_written", "audio_in", "audio_out", "audio_opener_out"})

    def test_turning_it_on_mid_call_starts_the_events(self):
        self.link.audio_stats = False
        self.link.call(np.zeros(100, dtype=np.float32), 24000)
        self.link.audio_stats = True
        self.wait_until(lambda: len(events_named(self.link, "audio_in")) >= 1)

    def test_opener_written_event(self):
        tone = (0.5 * np.sin(2 * np.pi * 440 * np.arange(2400) / 24000)).astype(np.float32)
        self.link.call(tone, 24000)
        (e,) = events_named(self.link, "opener_written")
        self.assertAlmostEqual(e["duration_s"], 0.1, places=3)
        self.assertAlmostEqual(e["peak"], 0.5 * 32767, delta=100)
        self.assertGreater(e["rms"], 1000)

    def test_incoming_summary_counts_all_frames_even_when_not_listening(self):
        self.link.call(np.zeros(100, dtype=np.float32), 24000)
        self.push_incoming(speech_frames(0.05, loud=True) + speech_frames(0.03, loud=False))   # not listening
        self.assertEqual(self.link._incoming.qsize(), 0)
        self.wait_until(lambda: sum(e["frames"] for e in events_named(self.link, "audio_in")) >= 8)
        evs = events_named(self.link, "audio_in")
        self.assertEqual(sum(e["frames"] for e in evs), 8)
        self.assertEqual(sum(e["zero_frames"] for e in evs), 3)
        self.assertAlmostEqual(max(e["peak"] for e in evs), 0.3 * 32767, delta=5)
        for key in ("phase", "window_s", "rms", "peak_dbfs", "rms_dbfs"):
            self.assertIn(key, evs[0])
        self.assertEqual(evs[-1]["phase"], "raw")

    def test_summary_is_emitted_even_with_no_frames(self):
        self.link.call(np.zeros(100, dtype=np.float32), 24000)
        self.wait_until(lambda: len(events_named(self.link, "audio_in")) >= 3)
        self.assertTrue(all(e["frames"] == 0 for e in events_named(self.link, "audio_in")))

    def test_outgoing_summary_separates_audio_and_silence(self):
        self.link.call(np.zeros(100, dtype=np.float32), 24000)
        tone = (0.3 * np.sin(2 * np.pi * 440 * np.arange(12000) / 24000)).astype(np.float32)   # 0.5 s
        self.link.play(tone, 24000)
        self.wait_until(lambda: sum(e["audio_frames"] for e in events_named(self.link, "audio_out")) >= 50)
        evs = events_named(self.link, "audio_out")
        self.assertEqual(sum(e["audio_frames"] for e in evs), 50)
        self.assertEqual(sum(e["frames"] for e in evs), sum(e["audio_frames"] + e["silence_frames"] for e in evs))
        self.assertGreater(sum(e["silence_frames"] for e in evs), 0)
        self.assertAlmostEqual(max(e["audio_peak"] for e in evs), 0.3 * 32767, delta=5)
        self.assertGreater(max(e["audio_rms"] for e in evs), 3000)

    def test_stats_stop_with_the_call_after_a_final_summary(self):
        self.link.call(np.zeros(100, dtype=np.float32), 24000)
        self.link.hang_up()
        n = len(events_named(self.link, "audio_in"))
        self.assertGreaterEqual(n, 1)
        time.sleep(0.3)
        self.assertEqual(len(events_named(self.link, "audio_in")), n)
        self.assertIsNone(self.link._stats_task)

    def test_opener_phase_probe_reports_not_exposed_when_no_outgoing_frames(self):
        self.link.call(np.zeros(100, dtype=np.float32), 24000)
        (e,) = events_named(self.link, "audio_opener_out")
        self.assertFalse(e["exposed"])
        self.assertEqual(e["frames"], 0)

    def test_opener_phase_probe_counts_outgoing_frames_if_reported(self):
        def during_opener():
            time.sleep(0.005)
            for f in speech_frames(0.03):
                self.emit(StreamFrames(USER_ID, Direction.OUTGOING, Device.MICROPHONE, [Frame(1, f, Frame.Info())]))
        with mock.patch.object(tg, "OPENER_TAIL_SECONDS", 0.3):
            th = threading.Thread(target=lambda: self.wait_until(lambda: self.link.state() == tg.CONNECTED) or during_opener())
            th.start()
            self.link.call(np.zeros(100, dtype=np.float32), 24000)
            th.join()
        (e,) = events_named(self.link, "audio_opener_out")
        self.assertTrue(e["exposed"])
        self.assertEqual(e["frames"], 3)
        self.assertAlmostEqual(e["peak"], 0.3 * 32767, delta=5)
