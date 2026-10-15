"""Tests for speak_server.py: speech helpers over a faked Telegram call, the
tool results, and the cross-process lock.

`speak_server.telegram` (the daemon client) is replaced by a fake wherever a
test reaches it, so no daemon and no Telegram are involved. There is no local
audio device path to test: the server is Telegram only.
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading
import unittest
from unittest import mock

os.environ["SPEAK_TELEGRAM_SOCKET"] = "/nonexistent-speak-telegram-test/telegram.sock"  # never the live daemon

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np

import speak_server as s
import speak_telegram as tg


def _connected_telegram(record_result=None):
    """A daemon client whose call is connected; play/record are recorded mocks."""
    fake = mock.Mock()
    fake.state.return_value = tg.CONNECTED
    fake.info.return_value = {"state": tg.CONNECTED, "generation": 1, "end_reason": None}
    fake.record.return_value = np.zeros(int(s.MIC_RATE * 0.2), dtype=np.float32) if record_result is None else record_result
    return fake


class TestNoLocalAudio(unittest.TestCase):
    def test_no_local_audio_code_remains(self):
        with open(os.path.join(REPO_ROOT, "speak_server.py")) as f:
            source = f.read()
        for gone in ("sounddevice", "speak_audio_worker", "SPEAK_INPUT_DEVICE", "SPEAK_OUTPUT_DEVICE", "SPEAK_SAVE_DIR", '"mac"'):
            self.assertNotIn(gone, source)
        self.assertFalse(os.path.exists(os.path.join(REPO_ROOT, "speak_audio_worker.py")))

    def test_without_a_call_the_tools_raise_instead_of_using_a_local_device(self):
        from speak_telegram_daemon import NOT_RUNNING

        orig_lock = s.audio_lock
        test_lock, tmpdir = _make_test_lock()
        s.audio_lock = test_lock
        try:
            for state in (tg.NONE, NOT_RUNNING, tg.RINGING):
                fake = mock.Mock()
                fake.state.return_value = state
                fake.info.return_value = {"state": state, "generation": 1, "end_reason": None}
                with mock.patch.object(s, "telegram", fake), mock.patch.object(s, "_call_expected", False):
                    for fn, args in ((s._speak_sync, ("hi",)), (s._listen_sync, (5, 1.2, 3)), (s._converse_sync, ("hi", 5, 1.2, 3))):
                        with self.assertRaisesRegex(RuntimeError, "needs a connected Telegram call"):
                            fn(*args)
                fake.play.assert_not_called()
                fake.record.assert_not_called()
        finally:
            s.audio_lock = orig_lock
            _rmtree(tmpdir)


class TestSpeakImpl(unittest.TestCase):
    """_speak_impl drives Kokoro (stubbed here) then plays into the call.
    Stubbing engines.kokoro keeps this test fast and independent of the real
    MLX model."""

    def setUp(self):
        p = mock.patch.object(s, "telegram", _connected_telegram())
        self.telegram = p.start()
        self.addCleanup(p.stop)
        self._orig_kokoro = s.engines.kokoro
        self._orig_ready = s.engines.ready.is_set()
        self._orig_error = s.engines.error
        s.engines.ready.set()
        s.engines.error = None

    def tearDown(self):
        s.engines.kokoro = self._orig_kokoro
        s.engines.error = self._orig_error
        if not self._orig_ready:
            s.engines.ready.clear()

    def test_speak_impl_plays_synthesized_chunks_and_returns_duration(self):
        class FakeResult:
            def __init__(self, audio):
                self.audio = audio

        class FakeKokoro:
            def generate(self, **kwargs):
                yield FakeResult(np.zeros(2400, dtype=np.float32))
                yield FakeResult(np.zeros(1200, dtype=np.float32))

        s.engines.kokoro = FakeKokoro()
        seconds = s._speak_impl("hello there")
        self.assertAlmostEqual(seconds, 3600 / s.TTS_RATE, places=5)
        self.telegram.play.assert_called_once()
        audio, rate = self.telegram.play.call_args.args
        self.assertEqual((len(audio), rate), (3600, s.TTS_RATE))

    def test_speak_impl_rejects_empty_text(self):
        s.engines.kokoro = mock.Mock()
        with self.assertRaises(ValueError):
            s._speak_impl("   ")

    def test_speak_impl_propagates_kokoro_failure(self):
        class FailingKokoro:
            def generate(self, **kwargs):
                raise RuntimeError("mlx blew up")
                yield  # pragma: no cover

        s.engines.kokoro = FailingKokoro()
        with self.assertRaises(RuntimeError) as ctx:
            s._speak_impl("hello")
        self.assertIn("Kokoro synthesis failed", str(ctx.exception))


class TestListenImpl(unittest.TestCase):
    def setUp(self):
        p = mock.patch.object(s, "telegram", _connected_telegram())
        self.telegram = p.start()
        self.addCleanup(p.stop)
        self._orig_ready = s.engines.ready.is_set()
        self._orig_error = s.engines.error
        s.engines.ready.set()
        s.engines.error = None

    def tearDown(self):
        s.engines.error = self._orig_error
        if not self._orig_ready:
            s.engines.ready.clear()

    def test_listen_impl_raises_timeout_error_when_the_call_hears_nothing(self):
        self.telegram.record.return_value = tg.TIMEOUT
        with self.assertRaises(TimeoutError):
            s._listen_impl(max_seconds=5, silence_seconds=1.2, start_timeout_seconds=3)

    def test_listen_impl_transcribes_recorded_audio(self):
        fake_mlx_whisper = mock.Mock()
        fake_mlx_whisper.transcribe.return_value = {"text": "hello world"}
        with mock.patch.dict(sys.modules, {"mlx_whisper": fake_mlx_whisper}):
            with mock.patch.object(s, "_ack") as fake_ack:
                text = s._listen_impl(max_seconds=5, silence_seconds=1.2, start_timeout_seconds=3)
        self.assertEqual(text, "hello world")
        fake_ack.assert_called_once()
        self.assertEqual(self.telegram.record.call_args.args[:3], (5, 1.2, 3))

    def test_listen_impl_raises_when_transcription_is_empty(self):
        fake_mlx_whisper = mock.Mock()
        fake_mlx_whisper.transcribe.return_value = {"text": "   "}
        with mock.patch.dict(sys.modules, {"mlx_whisper": fake_mlx_whisper}):
            with self.assertRaises(RuntimeError) as ctx:
                s._listen_impl(max_seconds=5, silence_seconds=1.2, start_timeout_seconds=3)
        self.assertIn("no text", str(ctx.exception))



def _make_test_lock():
    """A _SerializingLock pointed at a throwaway temp lockfile/sidecar --
    NEVER the real ~/.algolearn-speak/audio.lock, which a live speak_server.py
    session on this Mac may actually be holding. Returns (lock, tmpdir);
    caller is responsible for cleaning up tmpdir."""
    import tempfile

    tmpdir = tempfile.mkdtemp(prefix="speak-lock-test-")
    lock = s._SerializingLock(
        lock_path=os.path.join(tmpdir, "audio.lock"),
        sidecar_path=os.path.join(tmpdir, "audio.lock.json"),
    )
    return lock, tmpdir


class TestToolResultsCarryAudioNote(unittest.TestCase):
    """Every speak/listen/converse result ends with `(audio: telegram)` on its own line."""

    def test_speak(self):
        with mock.patch.object(s, "_speak_sync", return_value=1.5):
            self.assertEqual(asyncio.run(s.speak("hi")), "spoke for 1.5s\n(audio: telegram)")

    def test_listen(self):
        with mock.patch.object(s, "_listen_sync", return_value="hello there"):
            self.assertEqual(asyncio.run(s.listen()), "hello there\n(audio: telegram)")

    def test_converse(self):
        with mock.patch.object(s, "_converse_sync", return_value="yes"):
            self.assertEqual(asyncio.run(s.converse("ok?")), "yes\n(audio: telegram)")


class TestSerializingLock(unittest.TestCase):
    """The lock behind speak/listen/converse must queue a second caller
    (wait then proceed) rather than reject it, and `status()` must report
    busy/current_tool/waiting without itself blocking on the lock. Every
    test here uses its own throwaway lockfile (see _make_test_lock) so it
    never touches the real ~/.algolearn-speak lock a live session might hold."""

    def test_snapshot_reports_idle_when_unheld(self):
        lock, tmpdir = _make_test_lock()
        try:
            self.assertEqual(lock.snapshot(), (False, None, 0))
        finally:
            _rmtree(tmpdir)

    def test_snapshot_reports_current_tool_while_held(self):
        lock, tmpdir = _make_test_lock()
        try:
            lock.acquire("speak")
            try:
                self.assertEqual(lock.snapshot(), (True, "speak", 0))
            finally:
                lock.release()
            self.assertEqual(lock.snapshot(), (False, None, 0))
        finally:
            _rmtree(tmpdir)

    def test_second_caller_queues_and_waits_rather_than_being_rejected(self):
        import time

        lock, tmpdir = _make_test_lock()
        try:
            order: list[str] = []
            second_acquired = threading.Event()

            lock.acquire("listen")
            order.append("first-acquired")

            def second_caller():
                lock.acquire("converse")
                try:
                    order.append("second-acquired")
                    second_acquired.set()
                finally:
                    lock.release()

            t = threading.Thread(target=second_caller)
            t.start()
            # Give the second caller a moment to reach the blocking acquire() so
            # `waiting` reflects it before the first caller releases.
            deadline = time.time() + 2.0
            while lock.snapshot()[2] < 1 and time.time() < deadline:
                time.sleep(0.01)
            busy, current_tool, waiting = lock.snapshot()
            self.assertTrue(busy)
            self.assertEqual(current_tool, "listen")
            self.assertEqual(waiting, 1)
            self.assertFalse(second_acquired.is_set(), "second caller must wait, not run concurrently")

            order.append("first-releasing")
            lock.release()
            t.join(timeout=5.0)

            self.assertEqual(order, ["first-acquired", "first-releasing", "second-acquired"])
            self.assertEqual(lock.snapshot(), (False, None, 0))
        finally:
            _rmtree(tmpdir)

    def test_speak_sync_and_listen_sync_never_overlap(self):
        # End-to-end through the real _speak_sync/_listen_sync wrappers
        # (engines faked out), proving the lock is actually held for the
        # whole duration of each call, not just acquired and dropped. Uses a
        # throwaway lock swapped in for the module-global s.audio_lock so
        # this never touches the real ~/.algolearn-speak lockfile.
        orig_ready = s.engines.ready.is_set()
        orig_error = s.engines.error
        orig_kokoro = s.engines.kokoro
        orig_audio_lock = s.audio_lock
        s.engines.ready.set()
        s.engines.error = None
        test_lock, tmpdir = _make_test_lock()
        s.audio_lock = test_lock

        active = []
        overlapped = []

        class FakeKokoro:
            def generate(self, **kwargs):
                active.append("speak")
                if len(active) > 1:
                    overlapped.append(True)
                import time
                time.sleep(0.1)
                active.remove("speak")

                class R:
                    audio = np.zeros(10, dtype=np.float32)

                yield R()

        s.engines.kokoro = FakeKokoro()

        def fake_listen_impl(*args, **kwargs):
            active.append("listen")
            if len(active) > 1:
                overlapped.append(True)
            import time
            time.sleep(0.1)
            active.remove("listen")
            return "heard"

        try:
            with mock.patch.object(s, "telegram", _connected_telegram()):
                with mock.patch.object(s, "_listen_impl", side_effect=fake_listen_impl):
                    t1 = threading.Thread(target=s._speak_sync, args=("hello",))
                    t2 = threading.Thread(target=s._listen_sync, args=(5, 1.2, 3))
                    t1.start()
                    t2.start()
                    t1.join(timeout=5.0)
                    t2.join(timeout=5.0)
        finally:
            s.engines.kokoro = orig_kokoro
            s.engines.error = orig_error
            if not orig_ready:
                s.engines.ready.clear()
            s.audio_lock = orig_audio_lock
            _rmtree(tmpdir)

        self.assertEqual(overlapped, [], "speak and listen must never run concurrently")
        self.assertEqual(test_lock.snapshot(), (False, None, 0))

    def test_status_tool_reports_idle_snapshot(self):
        import asyncio

        orig_audio_lock = s.audio_lock
        test_lock, tmpdir = _make_test_lock()
        s.audio_lock = test_lock
        try:
            result = asyncio.run(s.status())
        finally:
            s.audio_lock = orig_audio_lock
            _rmtree(tmpdir)
        self.assertEqual(result, {"busy": False, "current_tool": None, "waiting": 0, "call": "daemon not running"})

    def test_status_tool_reports_busy_snapshot_without_blocking(self):
        import asyncio

        orig_audio_lock = s.audio_lock
        test_lock, tmpdir = _make_test_lock()
        s.audio_lock = test_lock
        test_lock.acquire("speak")
        try:
            result = asyncio.run(s.status())
        finally:
            test_lock.release()
            s.audio_lock = orig_audio_lock
            _rmtree(tmpdir)
        self.assertEqual(result, {"busy": True, "current_tool": "speak", "waiting": 0, "call": "daemon not running"})


def _rmtree(path: str) -> None:
    import shutil

    shutil.rmtree(path, ignore_errors=True)


def _cross_process_worker(lock_path: str, sidecar_path: str, tool_name: str, hold_seconds: float, order_path: str) -> None:
    """Module-level (picklable, importable under `spawn`) target for
    TestCrossProcessLock: acquires the real _CrossProcessLock at the given
    path, appends "<tool_name>-acquired" to the shared order file, holds it
    for hold_seconds, appends "<tool_name>-released", then releases. Uses a
    plain append-only text file rather than a multiprocessing.Queue/Manager
    to keep the child's imports minimal (just this module) and avoid any
    dependency on how the two processes were spawned."""
    import time as _time

    lock = s._CrossProcessLock(lock_path, sidecar_path)
    lock.acquire(tool_name)
    try:
        with open(order_path, "a", encoding="utf-8") as f:
            f.write(f"{tool_name}-acquired\n")
        _time.sleep(hold_seconds)
        with open(order_path, "a", encoding="utf-8") as f:
            f.write(f"{tool_name}-released\n")
    finally:
        lock.release()


class TestCrossProcessLock(unittest.TestCase):
    """Two SEPARATE OS processes (multiprocessing, spawn context -- neither
    forks nor shares this test process's memory, so this genuinely exercises
    flock() across processes, not just across threads in one interpreter)
    contending for the same lockfile must run strictly one after the other,
    never concurrently -- this is the actual fix for "two Claude sessions,
    two speak_server.py processes, talking over each other". Every test uses
    its own throwaway lockfile, never the real ~/.algolearn-speak path."""

    def test_two_processes_contending_for_the_file_lock_run_strictly_in_turn(self):
        import multiprocessing
        import tempfile

        tmpdir = tempfile.mkdtemp(prefix="speak-crossproc-test-")
        try:
            lock_path = os.path.join(tmpdir, "audio.lock")
            sidecar_path = os.path.join(tmpdir, "audio.lock.json")
            order_path = os.path.join(tmpdir, "order.log")

            ctx = multiprocessing.get_context("spawn")
            p1 = ctx.Process(target=_cross_process_worker, args=(lock_path, sidecar_path, "speak", 0.4, order_path))
            p2 = ctx.Process(target=_cross_process_worker, args=(lock_path, sidecar_path, "listen", 0.4, order_path))
            p1.start()
            import time
            time.sleep(0.05)  # give p1 a head start so it acquires first, deterministically
            p2.start()
            p1.join(timeout=15.0)
            p2.join(timeout=15.0)

            self.assertEqual(p1.exitcode, 0)
            self.assertEqual(p2.exitcode, 0)

            with open(order_path, encoding="utf-8") as f:
                order = [line.strip() for line in f if line.strip()]

            # Whichever process won the race to acquire first, it must fully
            # release before the second one ever acquires -- never interleaved.
            self.assertEqual(len(order), 4)
            first_tool = order[0].split("-")[0]
            second_tool = "listen" if first_tool == "speak" else "speak"
            self.assertEqual(order, [
                f"{first_tool}-acquired", f"{first_tool}-released",
                f"{second_tool}-acquired", f"{second_tool}-released",
            ])
        finally:
            _rmtree(tmpdir)

    def test_status_reflects_lock_held_by_another_process(self):
        import multiprocessing
        import tempfile
        import time

        tmpdir = tempfile.mkdtemp(prefix="speak-crossproc-test-")
        try:
            lock_path = os.path.join(tmpdir, "audio.lock")
            sidecar_path = os.path.join(tmpdir, "audio.lock.json")
            order_path = os.path.join(tmpdir, "order.log")

            ctx = multiprocessing.get_context("spawn")
            holder = ctx.Process(target=_cross_process_worker, args=(lock_path, sidecar_path, "converse", 1.5, order_path))
            holder.start()
            try:
                # Wait for the holder to actually acquire before probing --
                # avoids a flaky race where we probe before it opens the lock.
                deadline = time.time() + 10.0
                while not os.path.exists(order_path) and time.time() < deadline:
                    time.sleep(0.02)
                time.sleep(0.1)  # small margin past the acquire so the sidecar write has landed

                # This process itself does not hold the lock, so
                # _SerializingLock.snapshot() must consult _CrossProcessLock
                # and report the other process's tool/pid, not idle.
                probe_lock = s._SerializingLock(lock_path=lock_path, sidecar_path=sidecar_path)
                busy, current_tool, waiting = probe_lock.snapshot()
                self.assertTrue(busy)
                self.assertIn("converse", current_tool)
                self.assertIn(f"pid {holder.pid}", current_tool)
                self.assertEqual(waiting, 0)
            finally:
                holder.join(timeout=15.0)
                self.assertEqual(holder.exitcode, 0)

            # After the holder releases, a fresh probe must see idle again.
            probe_lock = s._SerializingLock(lock_path=lock_path, sidecar_path=sidecar_path)
            self.assertEqual(probe_lock.snapshot(), (False, None, 0))
        finally:
            _rmtree(tmpdir)


if __name__ == "__main__":
    unittest.main()
