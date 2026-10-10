"""Tests for the speak-telegram daemon and its client (docs/DESIGN_TELEGRAM_CALL.md section 3).

A real TelegramDaemon serves a fake link over a temp Unix socket; the real
TelegramClient talks to it. The live socket and Telegram are never touched.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from unittest import mock

import numpy as np

import speak_telegram as tg
import speak_telegram_daemon as td


class FakeLink:
    def __init__(self):
        self.started = self.stopped = False
        self.calls: list = []
        self.next_state = tg.NONE
        self.record_result = np.arange(1600, dtype=np.float32) / 1600
        self.fail_with: Exception | None = None
        self.events = tg.CallEventLog()
        self.call_outcome = tg.ANSWERED

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def state(self):
        return self.next_state

    def info(self):
        return {"state": self.next_state, "generation": 3, "end_reason": tg.END_REMOTE}

    def call(self, greeting, rate):
        self.calls.append(("call", greeting.copy(), rate))
        return self.call_outcome

    def play(self, audio, rate):
        if self.fail_with:
            raise self.fail_with
        self.calls.append(("play", audio.copy(), rate))

    def record(self, max_seconds, silence_seconds, start_timeout_seconds, cue, cue_rate):
        self.calls.append(("record", max_seconds, silence_seconds, start_timeout_seconds,
                           None if cue is None else cue.copy(), cue_rate))
        return self.record_result

    def hang_up(self):
        self.calls.append(("hang_up",))
        return tg.HUNG_UP


class DaemonCase(unittest.TestCase):
    def setUp(self):
        self.restarts: list[str] = []
        d = tempfile.mkdtemp(prefix="spk-tg-")
        self.addCleanup(shutil.rmtree, d, True)
        self.sock = os.path.join(d, "telegram.sock")
        self.link = FakeLink()
        self.daemon = td.TelegramDaemon(self.link, self.sock, on_restart=self.restarts.append)
        self.daemon.start()
        self.addCleanup(self.daemon.stop)
        self.client = td.TelegramClient(self.sock)


class TestRoundTrips(DaemonCase):
    def test_start_owns_a_private_socket(self):
        self.assertTrue(self.link.started)
        self.assertEqual(os.stat(self.sock).st_mode & 0o777, 0o600)

    def test_state(self):
        self.assertEqual(self.client.state(), tg.NONE)
        self.link.next_state = tg.CONNECTED
        self.assertEqual(self.client.state(), tg.CONNECTED)

    def test_call_carries_the_greeting(self):
        greeting = np.linspace(-0.5, 0.5, 2400, dtype=np.float32)
        self.assertEqual(self.client.call(greeting, 24000), tg.ANSWERED)
        op, got, rate = self.link.calls[0]
        self.assertEqual((op, rate), ("call", 24000))
        np.testing.assert_array_equal(got, greeting)

    def test_play(self):
        audio = np.linspace(-1, 1, 480, dtype=np.float32)
        self.client.play(audio, 24000)
        op, got, rate = self.link.calls[0]
        self.assertEqual((op, rate), ("play", 24000))
        np.testing.assert_array_equal(got, audio)

    def test_record_returns_the_capture_and_passes_the_cue(self):
        cue = np.ones(240, dtype=np.float32) * 0.5
        audio = self.client.record(30.0, 2.0, 45.0, cue, 24000)
        np.testing.assert_array_equal(audio, self.link.record_result)
        op, max_s, silence, start_timeout, got_cue, cue_rate = self.link.calls[0]
        self.assertEqual((op, max_s, silence, start_timeout, cue_rate), ("record", 30.0, 2.0, 45.0, 24000))
        np.testing.assert_array_equal(got_cue, cue)

    def test_record_timeout(self):
        self.link.record_result = tg.TIMEOUT
        self.assertIs(self.client.record(30.0, 2.0, 1.0, None, 24000), tg.TIMEOUT)

    def test_hang_up(self):
        self.assertEqual(self.client.hang_up(), tg.HUNG_UP)

    def test_info_carries_generation_and_end_reason(self):
        self.assertEqual(self.client.info(), {"state": tg.NONE, "generation": 3, "end_reason": tg.END_REMOTE})

    def test_os_errors_from_the_link_are_reported_not_dropped(self):
        self.link.fail_with = FileNotFoundError(2, "No such file", "/tmp/gone/opener.wav")
        with self.assertRaisesRegex(RuntimeError, "FileNotFoundError.*opener.wav"):
            self.client.play(np.zeros(10, dtype=np.float32), 24000)

    def test_a_stalled_daemon_is_a_runtime_error_not_a_timeout(self):
        import threading
        release = threading.Event()
        self.link.record = lambda *a: release.wait(5) and None
        self.addCleanup(release.set)
        with mock.patch.object(td, "CONNECT_TIMEOUT_SECONDS", 5.0):
            orig = self.client._request
            def quick(header, payload=b"", reply_timeout=None):
                return orig(header, payload, reply_timeout=0.3)
            with mock.patch.object(self.client, "_request", side_effect=quick):
                with self.assertRaisesRegex(RuntimeError, "did not reply to 'record'"):
                    self.client.record(1.0, 1.0, 1.0, None, 24000)

    def test_link_errors_reach_the_client_with_their_message(self):
        self.link.fail_with = tg.CallEnded("the Telegram call ended during playback")
        with self.assertRaisesRegex(RuntimeError, "ended during playback"):
            self.client.play(np.zeros(10, dtype=np.float32), 24000)

    def test_unexpected_errors_keep_their_type_name(self):
        self.link.fail_with = KeyError("boom")
        with self.assertRaisesRegex(RuntimeError, "KeyError"):
            self.client.play(np.zeros(10, dtype=np.float32), 24000)


class TestAudioFailureRestart(DaemonCase):
    def test_audio_failure_is_returned_then_restart_requested(self):
        self.link.call_outcome = tg.AUDIO_FAILED_PREFIX + "TelegramServerError"
        out = self.client.call(np.zeros(100, dtype=np.float32), 24000)
        self.assertEqual(out, self.link.call_outcome)
        deadline = time.monotonic() + 3
        while not self.restarts and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(self.restarts, [self.link.call_outcome])

    def test_other_outcomes_do_not_restart(self):
        self.link.call_outcome = tg.NOT_ANSWERED
        self.client.call(np.zeros(100, dtype=np.float32), 24000)
        self.client.state()
        self.assertEqual(self.restarts, [])


class TestDebugOps(DaemonCase):
    def test_debug_snapshot(self):
        self.link.events.record("abc", "requested")
        self.link.events.record("abc", "ringing")
        self.daemon.debug.ring._lines.extend(["l1", "l2", "l3"])
        r = self.client.debug(2)
        self.assertEqual(r["daemon"], "running")
        self.assertEqual(r["pid"], os.getpid())
        self.assertEqual(r["debug"], False)
        self.assertEqual([e["event"] for e in r["events"]], ["requested", "ringing"])
        self.assertEqual(r["log"], ["l2", "l3"])
        self.assertEqual(r["call"]["state"], tg.NONE)
        self.assertGreaterEqual(r["uptime_seconds"], 0)
        self.assertNotIn("ok", r)

    def test_debug_while_a_call_op_is_in_flight(self):
        gate, started = threading.Event(), threading.Event()
        real = self.link.call
        self.link.call = lambda g, r: (started.set(), gate.wait(5), real(g, r))[2]
        t = threading.Thread(target=self.client.call, args=(np.zeros(10, dtype=np.float32), 24000))
        t.start()
        self.addCleanup(t.join)
        self.addCleanup(gate.set)
        self.assertTrue(started.wait(3))
        self.assertEqual(self.client.debug(5)["daemon"], "running")   # does not wait for the call
        self.assertEqual(self.client.set_debug(True)["debug"], True)
        self.client.set_debug(False)

    def test_set_debug_switches_library_loggers(self):
        names = td.DEBUG_LOGGERS
        before = {n: logging.getLogger(n).level for n in names}
        self.addCleanup(self.daemon.debug.set, False)
        self.assertEqual(self.client.set_debug(True), {"daemon": "running", "debug": True})
        for n in names:
            self.assertEqual(logging.getLogger(n).level, logging.DEBUG)
        self.assertTrue(self.client.debug(1)["debug"])
        self.assertEqual(self.client.set_debug(False), {"daemon": "running", "debug": False})
        for n in names:
            self.assertEqual(logging.getLogger(n).level, before[n])

    def test_set_debug_false_restores_previous_levels(self):
        logging.getLogger("telethon").setLevel(logging.WARNING)
        self.addCleanup(logging.getLogger("telethon").setLevel, logging.NOTSET)
        self.client.set_debug(True)
        self.client.set_debug(True)   # a repeat must not overwrite the saved level
        self.client.set_debug(False)
        self.assertEqual(logging.getLogger("telethon").level, logging.WARNING)

    def test_bad_arguments_are_errors(self):
        with self.assertRaisesRegex(RuntimeError, "lines must be"):
            self.client.debug(0)
        with self.assertRaisesRegex(RuntimeError, "enabled must be"):
            self.client._request({"op": "set_debug", "enabled": "yes"})

    def test_log_ring_captures_records(self):
        ring = td.LogRing(size=2)
        lg = logging.getLogger("speak.test-ring")
        lg.addHandler(ring)
        self.addCleanup(lg.removeHandler, ring)
        lg.setLevel(logging.INFO)
        for i in range(3):
            lg.info("msg %d", i)
        tail = ring.tail(10)
        self.assertEqual(len(tail), 2)
        self.assertIn("msg 2", tail[-1])


class TestSingleDaemon(DaemonCase):
    def test_a_second_daemon_on_the_same_socket_refuses(self):
        with self.assertRaisesRegex(RuntimeError, "only one may use the Telegram login"):
            td.TelegramDaemon(FakeLink(), self.sock).start()
        self.assertEqual(self.client.state(), tg.NONE)   # the first one is unharmed

    def test_stop_removes_the_socket_and_stops_the_link(self):
        self.daemon.stop()
        self.assertFalse(os.path.exists(self.sock))
        self.assertTrue(self.link.stopped)
        self.assertEqual(td.TelegramClient(self.sock).state(), td.NOT_RUNNING)


class TestNoDaemon(unittest.TestCase):
    def test_not_running_is_reported_not_raised_for_state(self):
        self.assertEqual(td.TelegramClient("/nonexistent-speak-telegram/telegram.sock").state(), td.NOT_RUNNING)

    def test_other_operations_raise(self):
        with self.assertRaises(FileNotFoundError):
            td.TelegramClient("/nonexistent-speak-telegram/telegram.sock").hang_up()


class TestDebugNoDaemon(unittest.TestCase):
    def test_clear_not_running_result(self):
        client = td.TelegramClient("/nonexistent-speak-telegram-test/telegram.sock")
        self.assertEqual(client.debug(10), {"daemon": td.NOT_RUNNING})
        self.assertEqual(client.set_debug(True), {"daemon": td.NOT_RUNNING})


class TestEnv(unittest.TestCase):
    def write(self, text):
        d = tempfile.mkdtemp(prefix="spk-env-")
        self.addCleanup(shutil.rmtree, d, True)
        path = os.path.join(d, ".env")
        with open(path, "w") as f:
            f.write(text)
        return path

    def test_quotes_and_crlf_are_stripped(self):
        env = td.load_env(self.write('TELEGRAM_API_ID="1"\r\nTELEGRAM_API_HASH=\'h\'\r\nTELEGRAM_SESSION=s\r\nTELEGRAM_CALL_TARGET=t\r\n'))
        self.assertEqual(env["TELEGRAM_API_ID"], "1")
        self.assertEqual(env["TELEGRAM_API_HASH"], "h")
        self.assertEqual(env["TELEGRAM_CALL_TARGET"], "t")

    def test_loads_the_telegram_values(self):
        env = td.load_env(self.write("# c\nTELEGRAM_API_ID=1\nTELEGRAM_API_HASH=h\nTELEGRAM_SESSION=s\nTELEGRAM_CALL_TARGET=t\nOTHER=x\n"))
        self.assertEqual(env, {"TELEGRAM_API_ID": "1", "TELEGRAM_API_HASH": "h", "TELEGRAM_SESSION": "s", "TELEGRAM_CALL_TARGET": "t"})

    def test_a_missing_value_names_it(self):
        with self.assertRaisesRegex(RuntimeError, "TELEGRAM_SESSION, TELEGRAM_CALL_TARGET"):
            td.load_env(self.write("TELEGRAM_API_ID=1\nTELEGRAM_API_HASH=h\nTELEGRAM_SESSION=\n"))


if __name__ == "__main__":
    unittest.main()
