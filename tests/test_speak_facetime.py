"""Tests for FaceTime calls (docs/DESIGN_FACETIME_CALL.md section 12).

The real ftcall helper and the real `open facetime-audio://` are never used:
SPEAK_FTCALL_BIN points at tests/fake_ftcall, which plays back a scripted
sequence of banner states from FAKE_FTCALL_DIR, and FaceTime.dial is
replaced in every test. No call is ever placed.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ["SPEAK_AUDIO_DRY_RUN"] = "1"
os.environ["SPEAK_FTCALL_BIN"] = os.path.join(HERE, "fake_ftcall")
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np

import speak_facetime as ft
import speak_server as s

GOOD_DEVICES = {"microphone": "BlackHole 2ch", "output": "BlackHole 16ch"}


class FakeBanner:
    """A FAKE_FTCALL_DIR for one test: scripted states, devices, recorded presses."""

    def __init__(self, case: unittest.TestCase, states: list[str], devices: dict | None = GOOD_DEVICES) -> None:
        self.dir = tempfile.mkdtemp(prefix="fake-ftcall-")
        case.addCleanup(shutil.rmtree, self.dir, True)
        self.set_states(states)
        if devices is not None:
            with open(os.path.join(self.dir, "devices.json"), "w") as f:
                json.dump(devices, f)
        patcher = mock.patch.dict(os.environ, {"FAKE_FTCALL_DIR": self.dir})
        patcher.start()
        case.addCleanup(patcher.stop)

    def set_states(self, states: list[str]) -> None:
        with open(os.path.join(self.dir, "states"), "w") as f:
            f.write("\n".join(states) + "\n")

    def presses(self) -> list[str]:
        path = os.path.join(self.dir, "presses.log")
        if not os.path.exists(path):
            return []
        with open(path) as f:
            return f.read().split()


def facetime(case: unittest.TestCase) -> tuple[ft.FaceTime, list[str]]:
    """A FaceTime on the fake binary whose dial() only records the number."""
    dialed: list[str] = []
    f = ft.FaceTime(poll_seconds=0.01)
    f.dial = dialed.append
    return f, dialed


class TestQuietHours(unittest.TestCase):
    def test_window_inside_a_day(self):
        w = ft.parse_quiet_hours("13:00-14:30")
        self.assertTrue(ft.in_quiet_hours(w, datetime.time(13, 0)))
        self.assertTrue(ft.in_quiet_hours(w, datetime.time(14, 29)))
        self.assertFalse(ft.in_quiet_hours(w, datetime.time(14, 30)))
        self.assertFalse(ft.in_quiet_hours(w, datetime.time(12, 59)))

    def test_window_wrapping_midnight(self):
        w = ft.parse_quiet_hours("22:00-07:00")
        for t in (datetime.time(22, 0), datetime.time(23, 59), datetime.time(0, 0), datetime.time(6, 59)):
            self.assertTrue(ft.in_quiet_hours(w, t), t)
        for t in (datetime.time(7, 0), datetime.time(12, 0), datetime.time(21, 59)):
            self.assertFalse(ft.in_quiet_hours(w, t), t)

    def test_bad_windows_raise(self):
        for bad in ("22-07", "25:00-07:00", "22:00-07:61", "08:00-08:00", ""):
            with self.assertRaises(RuntimeError, msg=bad):
                ft.parse_quiet_hours(bad)

    def test_number_validation(self):
        ft.validate_number("+15555550100")
        for bad in ("6262304927", "+1626", "+1 626 230 4927", "frank@example.com"):
            with self.assertRaises(RuntimeError, msg=bad):
                ft.validate_number(bad)


class TestPlaceCall(unittest.TestCase):
    def test_answered(self):
        banner = FakeBanner(self, ["none", "none", "click_to_call|Click to Call", "ringing|FaceTime Audio…",
                                   "ringing|FaceTime Audio…", "connected|FaceTime Audio - 0:00"])
        f, dialed = facetime(self)
        self.assertEqual(f.place_call("+15555550100", "BlackHole 2ch", "BlackHole 16ch"), ft.ANSWERED)
        self.assertEqual(dialed, ["+15555550100"])
        self.assertEqual(banner.presses(), ["Call"])

    def test_declined_banner_disappears(self):
        banner = FakeBanner(self, ["none", "click_to_call", "ringing", "ringing", "none"])
        f, _ = facetime(self)
        self.assertEqual(f.place_call("+15555550100", "BlackHole 2ch", "BlackHole 16ch"), ft.NOT_ANSWERED)
        self.assertEqual(banner.presses(), ["Call"])

    def test_gap_after_pressing_call_is_not_a_decline(self):
        # A banner-less moment between Click to Call and ringing must not read as "declined".
        FakeBanner(self, ["none", "click_to_call", "none", "none", "ringing", "connected"])
        f, _ = facetime(self)
        self.assertEqual(f.place_call("+15555550100", "BlackHole 2ch", "BlackHole 16ch"), ft.ANSWERED)

    def test_ringing_past_the_timeout_hangs_up(self):
        banner = FakeBanner(self, ["none", "click_to_call", "ringing"])
        f, _ = facetime(self)
        original_press = f.press

        def press(button):
            original_press(button)
            if button == "End":
                banner.set_states(["none"])

        f.press = press
        with mock.patch.object(ft, "ANSWER_TIMEOUT", 0.2):
            self.assertEqual(f.place_call("+15555550100", "BlackHole 2ch", "BlackHole 16ch"), ft.NOT_ANSWERED)
        self.assertEqual(banner.presses(), ["Call", "End"])

    def test_never_starts_ringing_raises(self):
        FakeBanner(self, ["none", "click_to_call", "none"])
        f, _ = facetime(self)
        with mock.patch.object(ft, "RING_START_TIMEOUT", 0.1):
            with self.assertRaisesRegex(RuntimeError, "did not start ringing"):
                f.place_call("+15555550100", "BlackHole 2ch", "BlackHole 16ch")

    def test_wrong_devices_cancel_and_raise(self):
        banner = FakeBanner(self, ["none", "click_to_call"], devices={"microphone": "BlackHole 16ch", "output": "BlackHole 16ch"})
        f, _ = facetime(self)
        with self.assertRaisesRegex(RuntimeError, "Microphone is 'BlackHole 16ch'.*must be 'BlackHole 2ch'"):
            f.place_call("+15555550100", "BlackHole 2ch", "BlackHole 16ch")
        self.assertEqual(banner.presses(), ["Cancel"])

    def test_already_connected_does_not_dial(self):
        banner = FakeBanner(self, ["connected|FaceTime Audio - 3:12"])
        f, dialed = facetime(self)
        self.assertEqual(f.place_call("+15555550100", "BlackHole 2ch", "BlackHole 16ch"), ft.ALREADY_CONNECTED)
        self.assertEqual(dialed, [])
        self.assertEqual(banner.presses(), [])

    def test_half_set_up_banner_refuses_to_dial(self):
        FakeBanner(self, ["ringing"])
        f, dialed = facetime(self)
        with self.assertRaisesRegex(RuntimeError, "already up"):
            f.place_call("+15555550100", "BlackHole 2ch", "BlackHole 16ch")
        self.assertEqual(dialed, [])

    def test_banner_never_appears_raises(self):
        FakeBanner(self, ["none"])
        f, _ = facetime(self)
        with mock.patch.object(ft, "BANNER_TIMEOUT", 0.1):
            with self.assertRaisesRegex(RuntimeError, "Click to Call"):
                f.place_call("+15555550100", "BlackHole 2ch", "BlackHole 16ch")

    def test_unknown_banner_while_ringing_raises(self):
        FakeBanner(self, ["none", "click_to_call", "ringing", "unknown|Incoming call"])
        f, _ = facetime(self)
        with self.assertRaisesRegex(RuntimeError, "Incoming call"):
            f.place_call("+15555550100", "BlackHole 2ch", "BlackHole 16ch")

    def test_helper_failure_raises_with_its_stderr(self):
        banner = FakeBanner(self, ["none"])
        with open(os.path.join(banner.dir, "fail"), "w") as fh:
            fh.write("ftcall needs Accessibility permission")
        f, _ = facetime(self)
        with self.assertRaisesRegex(RuntimeError, "Accessibility"):
            f.state()


class TestHangUp(unittest.TestCase):
    def test_hangs_up_a_connected_call(self):
        banner = FakeBanner(self, ["connected", "connected", "none"])
        f, _ = facetime(self)
        self.assertEqual(f.hang_up(), ft.HUNG_UP)
        self.assertEqual(banner.presses(), ["End"])

    def test_no_active_call(self):
        banner = FakeBanner(self, ["none"])
        f, _ = facetime(self)
        self.assertEqual(f.hang_up(), ft.NO_ACTIVE_CALL)
        self.assertEqual(banner.presses(), [])

    def test_banner_that_stays_raises(self):
        FakeBanner(self, ["connected"])
        f, _ = facetime(self)
        with mock.patch.object(ft, "HANGUP_TIMEOUT", 0.1):
            with self.assertRaisesRegex(RuntimeError, "did not go away"):
                f.hang_up()


FAKE_CAFFEINATE = os.path.join(HERE, "fake_caffeinate")


def caffeinate_log(case: unittest.TestCase) -> str:
    path = os.path.join(tempfile.mkdtemp(prefix="fake-caffeinate-"), "log")
    case.addCleanup(shutil.rmtree, os.path.dirname(path), True)
    patcher = mock.patch.dict(os.environ, {"FAKE_CAFFEINATE_LOG": path})
    patcher.start()
    case.addCleanup(patcher.stop)
    return path


def read_lines(path: str) -> list[str]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return f.read().splitlines()


class ServerCase(unittest.TestCase):
    """speak_server with a FaceTime on the fake banner, a fake caffeinate, and stubbed engines."""

    def setUp(self):
        f, self.dialed = facetime(self)
        patcher = mock.patch.object(s, "facetime", f)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.caffeinate_log = caffeinate_log(self)
        self.keep_awake = ft.KeepAwake(f, poll=0.05, pulse=0.1, caffeinate=FAKE_CAFFEINATE)
        patcher = mock.patch.object(s, "keep_awake", self.keep_awake)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.keep_awake.stop)
        self._orig = (s.engines.kokoro, s.engines.ready.is_set(), s.engines.error)
        s.engines.ready.set()
        s.engines.error = None

        class FakeResult:
            audio = np.zeros(2400, dtype=np.float32)

        class FakeKokoro:
            def generate(self, **kwargs):
                yield FakeResult()

        s.engines.kokoro = FakeKokoro()

    def tearDown(self):
        self.keep_awake.stop()  # before cleanups remove the fake banner dir it polls
        s.engines.kokoro, ready, s.engines.error = self._orig
        if not ready:
            s.engines.ready.clear()


class TestRouting(ServerCase):
    def test_connected_call_routes_every_path_to_the_call_devices(self):
        FakeBanner(self, ["connected"])
        cmds = []
        real_popen, real_run = subprocess.Popen, subprocess.run

        def popen(cmd, *a, **k):
            cmds.append(cmd)
            return real_popen(cmd, *a, **k)

        def run(cmd, *a, **k):
            cmds.append(cmd)
            return real_run(cmd, *a, **k)

        whisper = mock.Mock()
        whisper.transcribe.return_value = {"text": "hello over facetime"}
        with mock.patch.object(s, "OUTPUT_DEVICE", "Mac Speakers"), mock.patch.object(s, "INPUT_DEVICE", "Mac Mic"):
            with mock.patch.object(subprocess, "Popen", side_effect=popen), mock.patch.object(subprocess, "run", side_effect=run):
                with mock.patch.dict(sys.modules, {"mlx_whisper": whisper}):
                    with mock.patch.dict(os.environ, {"SPEAK_AUDIO_DRY_RUN_SECONDS": "0.2"}):
                        reply, route = s._converse_sync("hi", 5, 1.2, 3)

        self.assertEqual((reply, route), ("hello over facetime", "facetime"))
        worker_cmds = [c for c in cmds if "speak_audio_worker" in c]
        kinds = {c[3] for c in worker_cmds}
        self.assertEqual(kinds, {"play-stream", "record", "play"})
        for c in worker_cmds:
            if c[3] in ("play-stream", "play"):
                self.assertEqual(c[-1], "BlackHole 2ch", c)
            else:  # record: ..., device_name, archive_path, output_device_name
                self.assertEqual(c[-3], "BlackHole 16ch", c)
                self.assertEqual(c[-2], "", c)  # never archived on the facetime route
                self.assertEqual(c[-1], "BlackHole 2ch", c)
        self.assertEqual(s._route, "mac")  # reset after the call

    def test_no_call_uses_the_mac_devices(self):
        FakeBanner(self, ["none"])
        cmds = []
        real_popen = subprocess.Popen

        def popen(cmd, *a, **k):
            cmds.append(cmd)
            return real_popen(cmd, *a, **k)

        with mock.patch.object(s, "OUTPUT_DEVICE", "Mac Speakers"):
            with mock.patch.object(subprocess, "Popen", side_effect=popen):
                _, route = s._speak_sync("hi")
        self.assertEqual(route, "mac")
        self.assertEqual([c[-1] for c in cmds if "play-stream" in c], ["Mac Speakers"])

    def test_call_that_drops_during_speak_raises(self):
        FakeBanner(self, ["connected", "none"])
        with self.assertRaisesRegex(RuntimeError, "FaceTime call ended during speak; call back"):
            s._speak_sync("hi")

    def test_call_that_drops_during_listen_reports_the_partial_transcript(self):
        FakeBanner(self, ["connected", "none"])
        whisper = mock.Mock()
        whisper.transcribe.return_value = {"text": "I was saying"}
        with mock.patch.object(s, "_ack"), mock.patch.dict(sys.modules, {"mlx_whisper": whisper}):
            with mock.patch.dict(os.environ, {"SPEAK_AUDIO_DRY_RUN_SECONDS": "0.2"}):
                with self.assertRaisesRegex(RuntimeError, "ended during listen.*'I was saying'"):
                    s._listen_sync(5, 1.2, 3)

    def test_silence_after_a_drop_raises_call_ended_not_timeout(self):
        FakeBanner(self, ["connected", "none"])
        with mock.patch.dict(os.environ, {"SPEAK_AUDIO_DRY_RUN_TIMEOUT": "1"}):
            with self.assertRaisesRegex(RuntimeError, "FaceTime call ended during listen"):
                s._listen_sync(5, 1.2, 3)

    def test_timeout_on_a_live_call_is_still_a_timeout(self):
        FakeBanner(self, ["connected"])
        with mock.patch.dict(os.environ, {"SPEAK_AUDIO_DRY_RUN_TIMEOUT": "1"}):
            with self.assertRaises(TimeoutError):
                s._listen_sync(5, 1.2, 3)


class TestCallTool(ServerCase):
    def _at(self, hour, minute):
        fake_dt = mock.Mock(wraps=datetime.datetime)
        fake_dt.now.return_value = datetime.datetime(2026, 10, 8, hour, minute)
        return mock.patch.object(s.datetime, "datetime", fake_dt)

    def test_quiet_hours_refuse_without_dialing(self):
        FakeBanner(self, ["none"])
        with mock.patch.object(s, "CALL_NUMBER", "+15555550100"), self._at(23, 30):
            self.assertEqual(s._call_sync(False), "not called: quiet hours (22:00-07:00)")
        self.assertEqual(self.dialed, [])

    def test_daytime_dials(self):
        FakeBanner(self, ["none", "click_to_call", "ringing", "connected"])
        with mock.patch.object(s, "CALL_NUMBER", "+15555550100"), self._at(12, 0):
            self.assertEqual(s._call_sync(False), ft.ANSWERED)
        self.assertEqual(self.dialed, ["+15555550100"])

    def test_override_dials_inside_quiet_hours(self):
        FakeBanner(self, ["none", "click_to_call", "ringing", "connected"])
        with mock.patch.object(s, "CALL_NUMBER", "+15555550100"), self._at(5, 0):
            self.assertEqual(s._call_sync(True), ft.ANSWERED)
        self.assertEqual(self.dialed, ["+15555550100"])

    def test_quiet_hours_are_judged_after_waiting_for_the_lock(self):
        FakeBanner(self, ["none"])
        order = []
        real_acquire = s.audio_lock.acquire

        def acquire(tool):
            order.append("lock")
            real_acquire(tool)

        fake_dt = mock.Mock(wraps=datetime.datetime)
        fake_dt.now.side_effect = lambda: order.append("clock") or datetime.datetime(2026, 10, 8, 23, 0)
        with mock.patch.object(s, "CALL_NUMBER", "+15555550100"), mock.patch.object(s.datetime, "datetime", fake_dt):
            with mock.patch.object(s.audio_lock, "acquire", side_effect=acquire):
                s._call_sync(False)
        self.assertEqual(order, ["lock", "clock"])

    def test_unset_number_raises(self):
        with mock.patch.object(s, "CALL_NUMBER", ""):
            with self.assertRaisesRegex(RuntimeError, "SPEAK_CALL_NUMBER is not set"):
                s._call_sync(True)

    def test_hang_up_tool(self):
        FakeBanner(self, ["connected", "none"])
        self.assertEqual(s._hang_up_sync(), ft.HUNG_UP)

    def test_status_reports_call_and_facetime_audio(self):
        FakeBanner(self, ["connected"])
        result = asyncio.run(s.status())
        self.assertEqual((result["call"], result["audio"]), ("connected", "facetime"))

    def test_status_reports_a_helper_error_instead_of_raising(self):
        banner = FakeBanner(self, ["none"])
        with open(os.path.join(banner.dir, "fail"), "w") as fh:
            fh.write("no permission")
        result = asyncio.run(s.status())
        self.assertTrue(result["call"].startswith("error: "), result)
        self.assertEqual(result["audio"], "error")

    def test_startup_check_rejects_bad_config(self):
        with mock.patch.object(s, "CALL_NUMBER", "6262304927"):
            with self.assertRaisesRegex(RuntimeError, "E.164"):
                s._check_call_config()
        with mock.patch.object(s, "QUIET_HOURS", "late"):
            with self.assertRaisesRegex(RuntimeError, "HH:MM"):
                s._check_call_config()
        with mock.patch.object(s.facetime, "binary", "/nonexistent/ftcall"):
            with self.assertRaisesRegex(RuntimeError, "ftcall/build.sh"):
                s._check_call_config()


if __name__ == "__main__":
    unittest.main()


class TestRobustness(unittest.TestCase):
    """Review findings: transient banners, cleanup after failures, press races."""

    def setUp(self):
        p = mock.patch.object(ft, "UNKNOWN_GRACE", 0.1)
        p.start()
        self.addCleanup(p.stop)
        p = mock.patch.object(ft, "GONE_CONFIRM", 0.1)
        p.start()
        self.addCleanup(p.stop)

    def test_brief_unreadable_banner_is_tolerated(self):
        FakeBanner(self, ["none", "click_to_call", "ringing", "unknown|", "ringing", "connected"])
        f, _ = facetime(self)
        self.assertEqual(f.place_call("+15555550100", "BlackHole 2ch", "BlackHole 16ch"), ft.ANSWERED)

    def test_long_loading_banner_after_dialing_is_waited_out(self):
        # Seen live: the banner exists ~1.1 s with no labels before "Click to Call" fills in.
        FakeBanner(self, ["none"] + ["loading|"] * 30 + ["click_to_call", "ringing", "connected"])
        f, _ = facetime(self)
        self.assertEqual(f.place_call("+15555550100", "BlackHole 2ch", "BlackHole 16ch"), ft.ANSWERED)

    def test_cleanup_waits_for_a_loading_banner_before_cancelling(self):
        banner = FakeBanner(self, ["none", "click_to_call", "loading|", "loading|", "click_to_call"], devices=None)
        f, _ = facetime(self)
        with self.assertRaises(RuntimeError):
            f.place_call("+15555550100", "BlackHole 2ch", "BlackHole 16ch")
        self.assertEqual(banner.presses(), ["Cancel"])

    def test_persistent_unknown_banner_raises(self):
        FakeBanner(self, ["none", "click_to_call", "ringing", "unknown|Call Failed"])
        f, _ = facetime(self)
        with self.assertRaisesRegex(RuntimeError, "Call Failed"):
            f.place_call("+15555550100", "BlackHole 2ch", "BlackHole 16ch")

    def test_momentary_gap_between_ringing_and_connected_is_answered(self):
        FakeBanner(self, ["none", "click_to_call", "ringing", "none", "connected"])
        f, _ = facetime(self)
        self.assertEqual(f.place_call("+15555550100", "BlackHole 2ch", "BlackHole 16ch"), ft.ANSWERED)

    def test_failure_after_dialing_cancels_click_to_call(self):
        banner = FakeBanner(self, ["none", "click_to_call"], devices=None)  # devices read fails
        f, _ = facetime(self)
        with self.assertRaisesRegex(RuntimeError, "FaceTime is not running"):
            f.place_call("+15555550100", "BlackHole 2ch", "BlackHole 16ch")
        self.assertEqual(banner.presses(), ["Cancel"])

    def test_wrong_devices_message_survives_cleanup(self):
        banner = FakeBanner(self, ["none", "click_to_call"], devices={"microphone": "MacBook Pro Microphone", "output": "BlackHole 16ch"})
        f, _ = facetime(self)
        with self.assertRaisesRegex(RuntimeError, "MacBook Pro Microphone"):
            f.place_call("+15555550100", "BlackHole 2ch", "BlackHole 16ch")
        self.assertEqual(banner.presses(), ["Cancel"])

    def test_hang_up_cancels_a_pending_click_to_call(self):
        banner = FakeBanner(self, ["click_to_call", "click_to_call", "none"])
        f, _ = facetime(self)
        self.assertEqual(f.hang_up(), ft.HUNG_UP)
        self.assertEqual(banner.presses(), ["Cancel"])

    def test_owner_hangs_up_first_press_race(self):
        banner = FakeBanner(self, ["connected", "none"])
        open(os.path.join(banner.dir, "press_fail"), "w").close()
        f, _ = facetime(self)
        self.assertEqual(f.hang_up(), ft.HUNG_UP)

    def test_press_failure_on_a_live_banner_raises(self):
        banner = FakeBanner(self, ["connected"])
        open(os.path.join(banner.dir, "press_fail"), "w").close()
        f, _ = facetime(self)
        with self.assertRaisesRegex(RuntimeError, "no FaceTime call banner to press End"):
            f.hang_up()

    def test_unknown_after_end_keeps_waiting_for_none(self):
        banner = FakeBanner(self, ["connected", "unknown|", "none"])
        f, _ = facetime(self)
        self.assertEqual(f.hang_up(), ft.HUNG_UP)
        self.assertEqual(banner.presses(), ["End"])


class TestCallExpected(ServerCase):
    """H1: once a call was answered, its absence raises instead of routing to the Mac."""

    def setUp(self):
        super().setUp()
        p = mock.patch.object(ft, "UNKNOWN_GRACE", 0.1)
        p.start()
        self.addCleanup(p.stop)
        p = mock.patch.object(s, "_call_expected", False)
        p.start()
        self.addCleanup(p.stop)

    def test_call_dropped_between_tools_raises_once_then_mac(self):
        FakeBanner(self, ["none", "click_to_call", "ringing", "connected", "none"])
        with mock.patch.object(s, "CALL_NUMBER", "+15555550100"), mock.patch.object(s, "QUIET_HOURS", "03:00-03:01"):
            self.assertEqual(s._call_sync(True), ft.ANSWERED)
        self.assertTrue(s._call_expected)
        with self.assertRaisesRegex(RuntimeError, "FaceTime call ended during speak; call back"):
            s._speak_sync("hi")
        self.assertFalse(s._call_expected)
        _, route = s._speak_sync("hi")  # reported once; afterwards the Mac is the device again
        self.assertEqual(route, "mac")

    def test_hang_up_clears_the_expectation(self):
        FakeBanner(self, ["connected", "none"])
        s._call_expected = True
        self.assertEqual(s._hang_up_sync(), ft.HUNG_UP)
        self.assertFalse(s._call_expected)

    def test_not_answered_does_not_expect_a_call(self):
        FakeBanner(self, ["none", "click_to_call", "ringing", "none"])
        with mock.patch.object(ft, "GONE_CONFIRM", 0.05), mock.patch.object(s, "CALL_NUMBER", "+15555550100"):
            self.assertEqual(s._call_sync(True), ft.NOT_ANSWERED)
        self.assertFalse(s._call_expected)

    def test_half_set_up_banner_refuses_to_route(self):
        FakeBanner(self, ["ringing"])
        with self.assertRaisesRegex(RuntimeError, "not connected"):
            s._speak_sync("hi")

    def test_state_read_failure_after_speech_keeps_both_errors(self):
        FakeBanner(self, ["connected", "FAIL"])
        with mock.patch.object(s, "_speak_impl", side_effect=RuntimeError("Kokoro synthesis failed: boom")):
            with self.assertRaisesRegex(RuntimeError, "Kokoro synthesis failed: boom.*could not be read.*AX read stalled"):
                s._speak_sync("hi")

    def test_state_read_failure_after_listen_keeps_the_transcript(self):
        FakeBanner(self, ["connected", "FAIL"])
        with mock.patch.object(s, "_listen_impl", return_value="my answer"):
            with self.assertRaisesRegex(RuntimeError, "'my answer'.*could not be read"):
                s._listen_sync(5, 1.2, 3)

    def test_call_and_hang_up_hold_the_audio_lock(self):
        FakeBanner(self, ["connected", "none"])
        seen = []
        real_hang = s.facetime.hang_up
        with mock.patch.object(s.facetime, "hang_up", side_effect=lambda: seen.append(s.audio_lock.snapshot()[1]) or real_hang()):
            s._hang_up_sync()
        with mock.patch.object(s.facetime, "place_call", side_effect=lambda *a: seen.append(s.audio_lock.snapshot()[1]) or ft.ALREADY_CONNECTED):
            with mock.patch.object(s, "CALL_NUMBER", "+15555550100"):
                s._call_sync(True)
        self.assertEqual(seen, ["hang_up", "call"])


class TestLocked(ServerCase):
    def test_call_refuses_while_locked_without_dialing(self):
        FakeBanner(self, ["locked"])
        with mock.patch.object(s, "CALL_NUMBER", "+15555550100"):
            self.assertEqual(s._call_sync(True), ft.NOT_CALLED_LOCKED)
        self.assertEqual(self.dialed, [])
        self.assertFalse(self.keep_awake.active())

    def test_voice_uses_the_mac_while_locked(self):
        FakeBanner(self, ["locked"])
        _, route = s._speak_sync("hi")
        self.assertEqual(route, "mac")

    def test_lock_during_a_call_reports_the_drop(self):
        FakeBanner(self, ["locked"])
        with mock.patch.object(s, "_call_expected", True):
            with self.assertRaisesRegex(RuntimeError, "call ended during speak.*locked"):
                s._speak_sync("hi")

    def test_hang_up_while_locked_is_no_active_call(self):
        FakeBanner(self, ["locked"])
        self.assertEqual(s._hang_up_sync(), ft.NO_ACTIVE_CALL)

    def test_status_while_locked(self):
        FakeBanner(self, ["locked"])
        result = asyncio.run(s.status())
        self.assertEqual((result["call"], result["audio"]), ("locked", "mac"))


class TestKeepAwake(ServerCase):
    def setUp(self):
        super().setUp()
        p = mock.patch.object(s, "_call_expected", False)
        p.start()
        self.addCleanup(p.stop)

    def _wait_until(self, predicate, timeout=5.0):
        deadline = time.monotonic() + timeout
        while not predicate():
            if time.monotonic() > deadline:
                self.fail("condition not met in time")
            time.sleep(0.02)

    def test_answered_call_holds_caffeinate_and_pulses_activity(self):
        FakeBanner(self, ["none", "click_to_call", "ringing", "connected"])
        with mock.patch.object(s, "CALL_NUMBER", "+15555550100"):
            self.assertEqual(s._call_sync(True), ft.ANSWERED)
        self.assertTrue(self.keep_awake.active())
        self._wait_until(lambda: sum(l.startswith("-u") for l in read_lines(self.caffeinate_log)) >= 2)
        held = [l for l in read_lines(self.caffeinate_log) if not l.startswith("-u")]
        self.assertEqual(held, [f"-d -i -w {os.getpid()}"])

    def test_watcher_releases_when_the_call_ends(self):
        banner = FakeBanner(self, ["connected"])
        self.keep_awake.start()
        banner.set_states(["none"])
        self._wait_until(lambda: not self.keep_awake.active())

    def test_watcher_releases_when_the_screen_locks(self):
        banner = FakeBanner(self, ["connected"])
        self.keep_awake.start()
        banner.set_states(["locked"])
        self._wait_until(lambda: not self.keep_awake.active())

    def test_watcher_stays_awake_when_the_state_is_unreadable(self):
        banner = FakeBanner(self, ["connected"])
        self.keep_awake.start()
        open(os.path.join(banner.dir, "fail"), "w").close()
        time.sleep(0.3)
        self.assertTrue(self.keep_awake.active())

    def test_hang_up_releases(self):
        FakeBanner(self, ["connected", "connected", "none"])
        self.keep_awake.start()
        self.assertEqual(s._hang_up_sync(), ft.HUNG_UP)
        self.assertFalse(self.keep_awake.active())

    def test_reported_drop_releases(self):
        FakeBanner(self, ["none"])
        self.keep_awake.start()
        with mock.patch.object(s, "_call_expected", True):
            with self.assertRaises(RuntimeError):
                s._speak_sync("hi")
        self.assertFalse(self.keep_awake.active())

    def test_not_answered_does_not_hold(self):
        FakeBanner(self, ["none", "click_to_call", "ringing", "none"])
        with mock.patch.object(ft, "GONE_CONFIRM", 0.05), mock.patch.object(s, "CALL_NUMBER", "+15555550100"):
            self.assertEqual(s._call_sync(True), ft.NOT_ANSWERED)
        self.assertFalse(self.keep_awake.active())
