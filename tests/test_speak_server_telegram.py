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

    def debug(self, lines):
        self.ops.append(("debug", lines))
        return {"daemon": "running", "events": [], "log": [], "debug": False}

    def set_debug(self, enabled):
        self.ops.append(("set_debug", enabled))
        return {"daemon": "running", "debug": enabled}


GREETING_SAMPLES = int(tg.OPENER_MIN_SECONDS * s.TTS_RATE)


class ServerCase(unittest.TestCase):
    def setUp(self):
        self._orig = (s.engines.kokoro, s.engines.ready.is_set(), s.engines.error)
        s.engines.ready.set()
        s.engines.error = None

        class Result:
            audio = np.zeros(GREETING_SAMPLES, dtype=np.float32)   # exactly the opener minimum

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


LISTEN = (5, 1.2, 3)   # max_seconds, silence_seconds, start_timeout_seconds


def answered_fake(**kw) -> FakeTelegram:
    """Daemon that is idle at dial time and connected once the owner answers."""
    return FakeTelegram([tg.NONE, tg.CONNECTED], **kw)


def whisper_says(text):
    whisper = mock.Mock()
    whisper.transcribe.return_value = {"text": text}
    return mock.patch.dict(sys.modules, {"mlx_whisper": whisper})


class TestCallTool(ServerCase):
    def test_answered_speaks_the_greeting_then_listens_and_returns_the_words(self):
        fake = self.use(answered_fake())
        with self.at(12, 0), whisper_says("yes, go ahead"):
            result = s._call_sync("Hey Frank, it's Angie, what's up?", False, *LISTEN)
        self.assertEqual(result, "answered\nyes, go ahead\n(audio: telegram)")
        self.assertEqual(fake.ops[0], ("call", GREETING_SAMPLES, s.TTS_RATE))
        self.assertEqual(fake.ops[1], ("record", 5, 1.2, 3, True, s.TTS_RATE))   # with the ear-open beep
        # exact sequence: dial, record, then ear-closed cue, rising chime, spoken "Processing"
        self.assertEqual([op[0] for op in fake.ops], ["call", "record", "play", "play", "play"])
        self.assertTrue(s._call_expected)
        self.assertEqual(s._route, "mac")

    def test_greeting_and_listen_hold_the_lock_as_one_unit(self):
        fake = self.use(answered_fake())
        seen = []
        real_call, real_record = fake.call, fake.record
        fake.call = lambda g, r: seen.append(s.audio_lock.snapshot()[1]) or real_call(g, r)
        fake.record = lambda *a: seen.append(s.audio_lock.snapshot()[1]) or real_record(*a)
        with self.at(12, 0), whisper_says("hi"):
            s._call_sync("Hi, what's up?", False, *LISTEN)
        self.assertEqual(seen, ["call", "call"])
        self.assertEqual(s.audio_lock.snapshot()[0], False)

    def test_listen_timeout_raises_and_the_call_stays_connected(self):
        fake = self.use(answered_fake())
        fake.record = mock.Mock(side_effect=TimeoutError("nobody spoke"))
        with self.at(12, 0), whisper_says("x"):
            with self.assertRaisesRegex(TimeoutError, "nobody spoke"):
                s._call_sync("Hi, what's up?", False, *LISTEN)
        # the low timeout cue went into the call as a play op (after the dial), not to the Mac
        self.assertEqual([op[0] for op in fake.ops], ["call", "play"])
        self.assertEqual(fake.ops[1][2], s.TTS_RATE)
        self.assertTrue(s._call_expected)
        self.assertEqual(s._route, "mac")

    def test_drop_during_the_listen_raises_call_ended(self):
        fake = FakeTelegram([tg.NONE, tg.CONNECTED, tg.CONNECTED, tg.NONE])
        self.use(fake)
        fake.record = mock.Mock(side_effect=RuntimeError("the Telegram call ended during recording"))
        with self.at(12, 0), whisper_says("x"):
            with self.assertRaisesRegex(RuntimeError, "Telegram call ended during call; call back"):
                s._call_sync("Hi, what's up?", False, *LISTEN)
        self.assertFalse(s._call_expected)

    def test_drop_between_answered_and_the_listen_raises_call_ended(self):
        fake = self.use(FakeTelegram([tg.NONE, tg.CONNECTED, tg.NONE], end_reason=tg.END_REMOTE))
        with self.at(12, 0):
            with self.assertRaisesRegex(RuntimeError, "Telegram call ended during call"):
                s._call_sync("Hi, what's up?", False, *LISTEN)
        self.assertEqual([op[0] for op in fake.ops], ["call"])   # never reached the listen
        self.assertFalse(s._call_expected)

    def test_short_greeting_raises_before_any_dial_op(self):
        fake = self.use(FakeTelegram([tg.NONE]))
        short = np.zeros(GREETING_SAMPLES - 1, dtype=np.float32)   # one sample under the minimum
        with mock.patch.object(s, "_synthesize", return_value=short), self.at(12, 0):
            with self.assertRaisesRegex(ValueError, r"only 6\.0s.*at least 6s.*Nothing was dialed.*context.*joke"):
                s._call_sync("Hi.", False, *LISTEN)
        self.assertEqual(fake.ops, [])
        self.assertFalse(s._call_expected)

    def test_greeting_at_the_minimum_dials(self):
        fake = self.use(answered_fake())
        exact = np.zeros(GREETING_SAMPLES, dtype=np.float32)
        with mock.patch.object(s, "_synthesize", return_value=exact), self.at(12, 0), whisper_says("hi"):
            s._call_sync("Hi.", False, *LISTEN)
        self.assertEqual(fake.ops[0], ("call", GREETING_SAMPLES, s.TTS_RATE))

    def test_quiet_hours_and_already_connected_win_over_a_short_greeting(self):
        short = np.zeros(10, dtype=np.float32)
        with mock.patch.object(s, "_synthesize", return_value=short):
            self.use(FakeTelegram([tg.NONE]))
            with self.at(23, 30):
                self.assertEqual(s._call_sync("Hi.", False, *LISTEN), "not called: quiet hours (22:00-07:00)")
            self.use(FakeTelegram([tg.CONNECTED]))
            with self.at(12, 0):
                self.assertEqual(s._call_sync("Hi.", False, *LISTEN), tg.ALREADY_CONNECTED)

    def test_tool_passes_converse_defaults_to_the_listen(self):
        with mock.patch.object(s, "_call_sync", return_value="answered\nhi\n(audio: telegram)") as m:
            self.assertEqual(asyncio.run(s.call("Hi, what's up?")), "answered\nhi\n(audio: telegram)")
        m.assert_called_once_with("Hi, what's up?", False, 120.0, 2.0, 45.0)

    def test_quiet_hours_refuse_without_dialing(self):
        fake = self.use(FakeTelegram([tg.NONE]))
        with self.at(23, 30):
            self.assertEqual(s._call_sync("Hi Frank", False, *LISTEN), "not called: quiet hours (22:00-07:00)")
        self.assertEqual(fake.ops, [])

    def test_override_dials_inside_quiet_hours(self):
        fake = self.use(answered_fake())
        with self.at(5, 0), whisper_says("morning"):
            self.assertEqual(s._call_sync("Morning, Frank, what's up?", True, *LISTEN), "answered\nmorning\n(audio: telegram)")
        self.assertEqual(fake.ops[0][0], "call")

    def test_not_answered_does_not_expect_a_call(self):
        fake = self.use(FakeTelegram([tg.NONE], outcome=tg.NOT_ANSWERED))
        with self.at(12, 0):
            self.assertEqual(s._call_sync("Hi", False, *LISTEN), tg.NOT_ANSWERED)
        self.assertFalse(s._call_expected)
        self.assertNotIn("record", [op[0] for op in fake.ops])   # no listen

    def test_hung_up_during_greeting_does_not_expect_a_call(self):
        fake = self.use(FakeTelegram([tg.NONE], outcome=tg.HUNG_UP_DURING_GREETING))
        with self.at(12, 0):
            self.assertEqual(s._call_sync("Hi", False, *LISTEN), tg.HUNG_UP_DURING_GREETING)
        self.assertFalse(s._call_expected)
        self.assertNotIn("record", [op[0] for op in fake.ops])   # no listen

    def test_already_connected_joins_without_dialing(self):
        fake = self.use(FakeTelegram([tg.CONNECTED]))
        with self.at(12, 0):
            self.assertEqual(s._call_sync("Hi", False, *LISTEN), tg.ALREADY_CONNECTED)
        self.assertEqual(fake.ops, [])
        self.assertTrue(s._call_expected)

    def test_daemon_not_running_raises(self):
        self.use(FakeTelegram([NOT_RUNNING]))
        with self.at(12, 0):
            with self.assertRaisesRegex(RuntimeError, "speak-telegram daemon is not running"):
                s._call_sync("Hi", False, *LISTEN)

    def test_empty_greeting_raises(self):
        self.use(FakeTelegram([tg.NONE]))
        with self.at(12, 0):
            with self.assertRaisesRegex(ValueError, "non-empty greeting"):
                s._call_sync("   ", False, *LISTEN)

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
        fake = self.use(answered_fake())
        seen = []
        fake.call = lambda g, r: seen.append(s.audio_lock.snapshot()[1]) or tg.ANSWERED
        fake.hang_up = lambda: seen.append(s.audio_lock.snapshot()[1]) or tg.HUNG_UP
        with self.at(12, 0), whisper_says("hi"):
            s._call_sync("Hi", False, *LISTEN)
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
            s._call_sync("Hi", False, *LISTEN)
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


class TestCallDebugTools(ServerCase):
    def test_call_debug_passes_lines_and_never_takes_the_audio_lock(self):
        fake = self.use(FakeTelegram([tg.CONNECTED]))
        s.audio_lock.acquire("converse")   # a call in progress holds the lock
        try:
            r = asyncio.run(s.call_debug(7))
            r2 = asyncio.run(s.set_call_debug(True))
        finally:
            s.audio_lock.release()
        self.assertEqual(r["daemon"], "running")
        self.assertEqual(r2, {"daemon": "running", "debug": True})
        self.assertEqual(fake.ops, [("debug", 7), ("set_debug", True)])

    def test_daemon_down_is_reported_not_faked(self):
        class Down(FakeTelegram):
            def debug(self, lines):
                return {"daemon": NOT_RUNNING}

            def set_debug(self, enabled):
                return {"daemon": NOT_RUNNING}
        self.use(Down([NOT_RUNNING]))
        self.assertEqual(asyncio.run(s.call_debug()), {"daemon": NOT_RUNNING})
        self.assertEqual(asyncio.run(s.set_call_debug(False)), {"daemon": NOT_RUNNING})

    def test_bad_arguments_raise(self):
        self.use(FakeTelegram([tg.NONE]))
        for bad in (0, 1001, True, "5"):
            with self.assertRaises(s.ToolError):
                asyncio.run(s.call_debug(bad))
        with self.assertRaises(s.ToolError):
            asyncio.run(s.set_call_debug("yes"))

    def test_default_is_50_lines(self):
        fake = self.use(FakeTelegram([tg.NONE]))
        asyncio.run(s.call_debug())
        self.assertEqual(fake.ops, [("debug", 50)])


class TestAudioFailedResult(ServerCase):
    def test_passes_through_and_does_not_expect_a_call(self):
        out = tg.AUDIO_FAILED_PREFIX + "TelegramServerError"
        fake = self.use(FakeTelegram([tg.NONE], outcome=out))
        with self.at(12, 0):
            self.assertEqual(s._call_sync("Hi", False, *LISTEN), out)
        self.assertFalse(s._call_expected)
        self.assertEqual(fake.ops[0][0], "call")

    def test_docstring_documents_it(self):
        self.assertIn("answered but audio failed to connect", s.call.__doc__)
        self.assertIn("call_debug", s.call.__doc__)


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
