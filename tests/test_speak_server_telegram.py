"""speak_server's Telegram route and call tools (docs/DESIGN_TELEGRAM_CALL.md sections 5-7).

`speak_server.telegram` (the daemon client) is replaced by a scripted fake, so
no daemon or Telegram is involved; the Mac route still runs the real dry-run
audio worker.
"""

from __future__ import annotations

import asyncio
import datetime
import os
import subprocess
import sys
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ["SPEAK_AUDIO_DRY_RUN"] = "1"
os.environ["SPEAK_TELEGRAM_SOCKET"] = "/nonexistent-speak-telegram-test/telegram.sock"
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np

import speak_server as s
import speak_telegram as tg
from speak_telegram_daemon import NOT_RUNNING


class FakeTelegram:
    """Scripted daemon client. `states` is consumed one per state() call; the last repeats."""

    def __init__(self, states, outcome=tg.ANSWERED, record_result=None, generation=1, end_reason=None):
        self.states = list(states)
        self.generation = generation
        self.end_reason = end_reason
        self.outcome = outcome
        self.record_result = np.zeros(int(s.MIC_RATE * 0.5), dtype=np.float32) if record_result is None else record_result
        self.ops: list = []

    def state(self):
        value = self.states[0]
        if len(self.states) > 1:
            self.states.pop(0)
        if isinstance(value, Exception):
            raise value
        return value

    def info(self):
        return {"state": self.state(), "generation": self.generation, "end_reason": self.end_reason}

    def call(self, greeting, rate):
        self.ops.append(("call", len(greeting), rate))
        return self.outcome

    def play(self, audio, rate):
        self.ops.append(("play", len(audio), rate))

    def record(self, max_seconds, silence_seconds, start_timeout_seconds, cue, cue_rate):
        self.ops.append(("record", max_seconds, silence_seconds, start_timeout_seconds, cue is not None, cue_rate))
        return self.record_result

    def hang_up(self):
        self.ops.append(("hang_up",))
        return tg.HUNG_UP


class ServerCase(unittest.TestCase):
    def setUp(self):
        self._orig = (s.engines.kokoro, s.engines.ready.is_set(), s.engines.error)
        s.engines.ready.set()
        s.engines.error = None

        class Result:
            audio = np.zeros(2400, dtype=np.float32)

        class FakeKokoro:
            def generate(self, **kwargs):
                yield Result()

        s.engines.kokoro = FakeKokoro()
        p = mock.patch.object(s, "_call_expected", False)
        p.start()
        self.addCleanup(p.stop)

    def tearDown(self):
        s.engines.kokoro, ready, s.engines.error = self._orig
        if not ready:
            s.engines.ready.clear()

    def use(self, fake: FakeTelegram) -> FakeTelegram:
        p = mock.patch.object(s, "telegram", fake)
        p.start()
        self.addCleanup(p.stop)
        return fake

    def at(self, hour, minute):
        fake_dt = mock.Mock(wraps=datetime.datetime)
        fake_dt.now.return_value = datetime.datetime(2026, 10, 9, hour, minute)
        return mock.patch.object(s.datetime, "datetime", fake_dt)


class TestRouting(ServerCase):
    def test_connected_call_carries_speech_cues_and_listening(self):
        fake = self.use(FakeTelegram([tg.CONNECTED]))
        whisper = mock.Mock()
        whisper.transcribe.return_value = {"text": "hello over telegram"}
        workers = []
        real_popen, real_run = subprocess.Popen, subprocess.run
        with mock.patch.object(subprocess, "Popen", side_effect=lambda *a, **k: workers.append(a) or real_popen(*a, **k)), \
                mock.patch.object(subprocess, "run", side_effect=lambda *a, **k: workers.append(a) or real_run(*a, **k)), \
                mock.patch.dict(sys.modules, {"mlx_whisper": whisper}), \
                mock.patch.object(s, "SAVE_DIR", "/should/never/be/written"):
            reply, route = s._converse_sync("hi", 5, 1.2, 3)
        self.assertEqual((reply, route), ("hello over telegram", "telegram"))
        self.assertEqual(workers, [])   # the Mac's audio worker never ran
        kinds = [op[0] for op in fake.ops]
        self.assertEqual(kinds[0], "play")                     # the spoken text
        record = next(op for op in fake.ops if op[0] == "record")
        self.assertEqual(record, ("record", 5, 1.2, 3, True, s.TTS_RATE))   # with the ear-open cue
        self.assertGreaterEqual(kinds.count("play"), 3)        # + ear-closed cue, chime, "Processing"
        self.assertEqual(s._route, "mac")                      # reset afterwards

    def test_no_call_uses_the_mac(self):
        for state in (tg.NONE, NOT_RUNNING):
            fake = self.use(FakeTelegram([state]))
            _, route = s._speak_sync("hi")
            self.assertEqual(route, "mac", state)
            self.assertEqual(fake.ops, [])

    def test_ringing_refuses_to_route(self):
        self.use(FakeTelegram([tg.RINGING]))
        with self.assertRaisesRegex(RuntimeError, "ringing but not connected"):
            s._speak_sync("hi")

    def test_drop_during_speak_raises(self):
        self.use(FakeTelegram([tg.CONNECTED, tg.NONE]))
        with self.assertRaisesRegex(RuntimeError, "Telegram call ended during speak; call back"):
            s._speak_sync("hi")

    def test_drop_during_listen_keeps_the_transcript(self):
        self.use(FakeTelegram([tg.CONNECTED, tg.NONE]))
        whisper = mock.Mock()
        whisper.transcribe.return_value = {"text": "I was saying"}
        with mock.patch.object(s, "_ack"), mock.patch.dict(sys.modules, {"mlx_whisper": whisper}):
            with self.assertRaisesRegex(RuntimeError, "ended during listen.*'I was saying'"):
                s._listen_sync(5, 1.2, 3)

    def test_failure_after_a_drop_reports_the_drop(self):
        fake = self.use(FakeTelegram([tg.CONNECTED, tg.NONE]))
        fake.play = mock.Mock(side_effect=RuntimeError("the Telegram call ended during playback"))
        with self.assertRaisesRegex(RuntimeError, "Telegram call ended during speak.*during playback"):
            s._speak_sync("hi")

    def test_timeout_on_a_live_call_stays_a_timeout(self):
        self.use(FakeTelegram([tg.CONNECTED], record_result=tg.TIMEOUT))
        with mock.patch.object(s, "_cue"):
            with self.assertRaises(TimeoutError):
                s._listen_sync(5, 1.2, 3)

    def test_unreadable_state_after_speech_keeps_both_errors(self):
        fake = self.use(FakeTelegram([tg.CONNECTED, RuntimeError("daemon stalled")]))
        fake.play = mock.Mock(side_effect=RuntimeError("Kokoro synthesis failed: boom"))
        with self.assertRaisesRegex(RuntimeError, "boom.*could not be read.*daemon stalled"):
            s._speak_sync("hi")


class TestDropAfterCapture(ServerCase):
    """Review finding 1: the owner speaks, then hangs up; their words must survive."""

    def test_words_survive_a_hang_up_right_after_speaking(self):
        fake = self.use(FakeTelegram([tg.CONNECTED, tg.NONE]))
        fake.play = mock.Mock(side_effect=RuntimeError("no connected Telegram call"))   # ear-closed cue + ack fail
        whisper = mock.Mock()
        whisper.transcribe.return_value = {"text": "OK, deploy it, bye"}
        with mock.patch.dict(sys.modules, {"mlx_whisper": whisper}):
            with self.assertRaisesRegex(RuntimeError, "ended during listen.*'OK, deploy it, bye'"):
                s._listen_sync(5, 1.2, 3)
        whisper.transcribe.assert_called_once()

    def test_cue_failure_on_a_live_call_still_raises(self):
        fake = self.use(FakeTelegram([tg.CONNECTED]))
        fake.play = mock.Mock(side_effect=RuntimeError("daemon exploded"))
        with self.assertRaisesRegex(RuntimeError, "daemon exploded"):
            s._listen_sync(5, 1.2, 3)


class TestSharedCall(ServerCase):
    """Review finding 7: the call is shared; joiners watch it, and a hang-up is not a drop."""

    def test_a_session_that_talks_on_the_call_expects_it(self):
        self.use(FakeTelegram([tg.CONNECTED], generation=7))
        s._speak_sync("hi")
        self.assertTrue(s._call_expected)
        self.assertEqual(s._call_generation, 7)

    def test_drop_after_joining_raises_instead_of_using_the_mac(self):
        fake = self.use(FakeTelegram([tg.CONNECTED, tg.CONNECTED, tg.NONE], generation=7))
        s._speak_sync("hi")
        fake.end_reason = tg.END_REMOTE
        with self.assertRaisesRegex(RuntimeError, "Telegram call ended during speak; call back"):
            s._speak_sync("hi")

    def test_hang_up_by_another_session_is_not_a_drop(self):
        fake = self.use(FakeTelegram([tg.CONNECTED, tg.CONNECTED, tg.NONE], generation=7))
        s._speak_sync("hi")
        fake.end_reason = tg.END_HUNG_UP
        _, route = s._speak_sync("hi")
        self.assertEqual(route, "mac")
        self.assertFalse(s._call_expected)

    def test_a_hang_up_of_an_older_call_does_not_excuse_a_newer_drop(self):
        fake = self.use(FakeTelegram([tg.NONE], generation=8, end_reason=tg.END_HUNG_UP))
        s._call_expected, s._call_generation = True, 7
        with self.assertRaisesRegex(RuntimeError, "call back"):
            s._speak_sync("hi")
        self.assertEqual(fake.ops, [])

    def test_hang_up_clears_the_expectation_even_when_it_fails(self):
        fake = self.use(FakeTelegram([tg.CONNECTED]))
        fake.hang_up = mock.Mock(side_effect=RuntimeError("NotInCallError"))
        s._call_expected = True
        with self.assertRaises(RuntimeError):
            s._hang_up_sync()
        self.assertFalse(s._call_expected)


class TestCallExpected(ServerCase):
    def test_drop_between_tools_raises_once_then_mac(self):
        self.use(FakeTelegram([tg.NONE, tg.NONE, tg.NONE]))
        s._call_expected = True
        with self.assertRaisesRegex(RuntimeError, "Telegram call ended during speak; call back"):
            s._speak_sync("hi")
        self.assertFalse(s._call_expected)
        _, route = s._speak_sync("hi")
        self.assertEqual(route, "mac")

    def test_daemon_gone_while_expected_raises(self):
        self.use(FakeTelegram([NOT_RUNNING]))
        s._call_expected = True
        with self.assertRaisesRegex(RuntimeError, "daemon not running"):
            s._speak_sync("hi")


class TestCallTool(ServerCase):
    def test_quiet_hours_refuse_without_dialing(self):
        fake = self.use(FakeTelegram([tg.NONE]))
        with self.at(23, 30):
            self.assertEqual(s._call_sync("Hi Frank", False), "not called: quiet hours (22:00-07:00)")
        self.assertEqual(fake.ops, [])

    def test_daytime_call_speaks_the_greeting_and_expects_the_call(self):
        fake = self.use(FakeTelegram([tg.NONE]))
        with self.at(12, 0):
            self.assertEqual(s._call_sync("Hi Frank, the backtest finished.", False), tg.ANSWERED)
        self.assertEqual(fake.ops, [("call", 2400, s.TTS_RATE)])
        self.assertTrue(s._call_expected)

    def test_override_dials_inside_quiet_hours(self):
        fake = self.use(FakeTelegram([tg.NONE]))
        with self.at(5, 0):
            self.assertEqual(s._call_sync("Morning, Frank", True), tg.ANSWERED)
        self.assertEqual(fake.ops[0][0], "call")

    def test_not_answered_does_not_expect_a_call(self):
        self.use(FakeTelegram([tg.NONE], outcome=tg.NOT_ANSWERED))
        with self.at(12, 0):
            self.assertEqual(s._call_sync("Hi", False), tg.NOT_ANSWERED)
        self.assertFalse(s._call_expected)

    def test_already_connected_joins_without_dialing(self):
        fake = self.use(FakeTelegram([tg.CONNECTED]))
        with self.at(12, 0):
            self.assertEqual(s._call_sync("Hi", False), tg.ALREADY_CONNECTED)
        self.assertEqual(fake.ops, [])
        self.assertTrue(s._call_expected)

    def test_daemon_not_running_raises(self):
        self.use(FakeTelegram([NOT_RUNNING]))
        with self.at(12, 0):
            with self.assertRaisesRegex(RuntimeError, "speak-telegram daemon is not running"):
                s._call_sync("Hi", False)

    def test_empty_greeting_raises(self):
        self.use(FakeTelegram([tg.NONE]))
        with self.at(12, 0):
            with self.assertRaisesRegex(ValueError, "non-empty greeting"):
                s._call_sync("   ", False)

    def test_hang_up_clears_the_expectation(self):
        fake = self.use(FakeTelegram([tg.CONNECTED]))
        s._call_expected = True
        self.assertEqual(s._hang_up_sync(), tg.HUNG_UP)
        self.assertEqual(fake.ops, [("hang_up",)])
        self.assertFalse(s._call_expected)

    def test_hang_up_without_a_daemon(self):
        self.use(FakeTelegram([NOT_RUNNING]))
        self.assertEqual(s._hang_up_sync(), tg.NO_ACTIVE_CALL)

    def test_call_and_hang_up_hold_the_audio_lock(self):
        fake = self.use(FakeTelegram([tg.NONE]))
        seen = []
        fake.call = lambda g, r: seen.append(s.audio_lock.snapshot()[1]) or tg.ANSWERED
        fake.hang_up = lambda: seen.append(s.audio_lock.snapshot()[1]) or tg.HUNG_UP
        with self.at(12, 0):
            s._call_sync("Hi", False)
        s._hang_up_sync()
        self.assertEqual(seen, ["call", "hang_up"])

    def test_quiet_hours_are_judged_after_waiting_for_the_lock(self):
        self.use(FakeTelegram([tg.NONE]))
        order = []
        real_acquire = s.audio_lock.acquire
        fake_dt = mock.Mock(wraps=datetime.datetime)
        fake_dt.now.side_effect = lambda: order.append("clock") or datetime.datetime(2026, 10, 9, 23, 0)
        with mock.patch.object(s.datetime, "datetime", fake_dt), \
                mock.patch.object(s.audio_lock, "acquire", side_effect=lambda t: order.append("lock") or real_acquire(t)):
            s._call_sync("Hi", False)
        self.assertEqual(order, ["lock", "clock"])


class TestStatus(ServerCase):
    def test_connected(self):
        self.use(FakeTelegram([tg.CONNECTED]))
        r = asyncio.run(s.status())
        self.assertEqual((r["call"], r["audio"]), (tg.CONNECTED, "telegram"))

    def test_no_daemon(self):
        self.use(FakeTelegram([NOT_RUNNING]))
        r = asyncio.run(s.status())
        self.assertEqual((r["call"], r["audio"]), (NOT_RUNNING, "mac"))

    def test_expected_call_gone_is_an_error(self):
        self.use(FakeTelegram([tg.NONE]))
        s._call_expected = True
        self.assertEqual(asyncio.run(s.status())["audio"], "error")

    def test_unreadable_state_is_reported_not_raised(self):
        self.use(FakeTelegram([RuntimeError("socket timed out")]))
        r = asyncio.run(s.status())
        self.assertEqual((r["call"], r["audio"]), ("error: socket timed out", "error"))


class TestQuietHours(unittest.TestCase):
    def test_window_wrapping_midnight(self):
        w = s._parse_quiet_hours("22:00-07:00")
        for t in (datetime.time(22, 0), datetime.time(23, 59), datetime.time(0, 0), datetime.time(6, 59)):
            self.assertTrue(s._in_quiet_hours(w, t), t)
        for t in (datetime.time(7, 0), datetime.time(12, 0), datetime.time(21, 59)):
            self.assertFalse(s._in_quiet_hours(w, t), t)

    def test_window_inside_a_day(self):
        w = s._parse_quiet_hours("13:00-14:30")
        self.assertTrue(s._in_quiet_hours(w, datetime.time(13, 0)))
        self.assertFalse(s._in_quiet_hours(w, datetime.time(14, 30)))

    def test_bad_windows_raise(self):
        for bad in ("22-07", "25:00-07:00", "22:00-07:61", "08:00-08:00", ""):
            with self.assertRaises(RuntimeError, msg=bad):
                s._parse_quiet_hours(bad)

    def test_startup_check(self):
        with mock.patch.object(s, "QUIET_HOURS", "late"):
            with self.assertRaisesRegex(RuntimeError, "HH:MM"):
                s._check_call_config()


if __name__ == "__main__":
    unittest.main()
