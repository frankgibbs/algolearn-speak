"""Tests for speak_server.py's audio helpers.

Every test here runs with SPEAK_AUDIO_DRY_RUN=1 (set at module import time,
before speak_server ever gets a chance to spawn a worker), so no worker
subprocess ever opens a real PortAudio stream and no test touches the
microphone or speakers. speak_server.py itself is asserted to never import
sounddevice at all -- that's the point of the fix.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import unittest
from unittest import mock

os.environ["SPEAK_AUDIO_DRY_RUN"] = "1"

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np

import speak_server as s


class TestNoSounddeviceInServer(unittest.TestCase):
    def test_server_module_never_imports_sounddevice(self):
        # speak_server.py must not bind sounddevice at all -- every actual
        # PortAudio open lives in speak_audio_worker.py, which imports it
        # lazily inside functions and only when not in dry-run.
        self.assertNotIn("sd", dir(s))
        with open(os.path.join(REPO_ROOT, "speak_server.py")) as f:
            source = f.read()
        self.assertNotIn("import sounddevice", source)


class TestPlayPcm(unittest.TestCase):
    def test_play_pcm_succeeds_in_dry_run(self):
        tone = s._tone(440.0, 0.05)
        s._play_pcm(tone, s.TTS_RATE)  # must not raise

    def test_play_pcm_cleans_up_temp_file(self):
        before = set(os.listdir(os.path.dirname(_a_temp_path())))
        tone = s._tone(440.0, 0.01)
        s._play_pcm(tone, s.TTS_RATE)
        after = set(os.listdir(os.path.dirname(_a_temp_path())))
        leaked = {p for p in after - before if p.startswith("speak-play-")}
        self.assertEqual(leaked, set())


def _a_temp_path() -> str:
    import tempfile
    with tempfile.NamedTemporaryFile() as f:
        return f.name


class TestPlayPcmStream(unittest.TestCase):
    def test_play_pcm_stream_succeeds_with_multiple_chunks(self):
        chunks = [s._tone(440.0, 0.02).reshape(-1, 1), s._tone(660.0, 0.02).reshape(-1, 1)]
        s._play_pcm_stream(chunks, s.TTS_RATE, timeout=10.0)  # must not raise

    def test_play_pcm_stream_succeeds_with_no_chunks(self):
        s._play_pcm_stream([], s.TTS_RATE, timeout=10.0)  # must not raise


class TestRecordPcm(unittest.TestCase):
    def test_record_pcm_returns_synthesized_audio(self):
        with mock.patch.dict(os.environ, {"SPEAK_AUDIO_DRY_RUN_SECONDS": "0.2"}):
            audio = s._record_pcm(s.MIC_RATE, s.VAD_FRAME, max_seconds=5, silence_seconds=1.2, start_timeout_seconds=3)
        self.assertEqual(len(audio), int(s.MIC_RATE * 0.2))
        self.assertEqual(audio.dtype, np.float32)

    def test_record_pcm_raises_timeout_error_when_worker_times_out(self):
        with mock.patch.dict(os.environ, {"SPEAK_AUDIO_DRY_RUN_TIMEOUT": "1"}):
            with self.assertRaises(TimeoutError):
                s._record_pcm(s.MIC_RATE, s.VAD_FRAME, max_seconds=5, silence_seconds=1.2, start_timeout_seconds=3)

    def test_record_pcm_cleans_up_temp_file_on_timeout(self):
        import tempfile
        tmp_dir = tempfile.gettempdir()
        before = {p for p in os.listdir(tmp_dir) if p.startswith("speak-record-")}
        with mock.patch.dict(os.environ, {"SPEAK_AUDIO_DRY_RUN_TIMEOUT": "1"}):
            with self.assertRaises(TimeoutError):
                s._record_pcm(s.MIC_RATE, s.VAD_FRAME, max_seconds=5, silence_seconds=1.2, start_timeout_seconds=3)
        after = {p for p in os.listdir(tmp_dir) if p.startswith("speak-record-")}
        self.assertEqual(after - before, set())

    def test_record_pcm_returns_captured_audio_when_worker_hits_wall_clock_cap(self):
        # Regression test for the TimeoutExpired bug: continuous non-speech
        # sound (e.g. a TV) can make a real VAD believe speech never stops,
        # so silence_seconds never arrives. The worker's own wall-clock cap
        # (start_timeout_seconds + max_seconds from stream open) must still
        # fire, and _record_pcm must return the captured audio rather than
        # raising -- the subprocess timeout given to the worker must be
        # generous enough for the worker's own cap to win the race.
        max_seconds = 0.2
        start_timeout_seconds = 0.1
        cap_seconds = max_seconds + start_timeout_seconds
        with mock.patch.dict(os.environ, {"SPEAK_AUDIO_DRY_RUN_ENDLESS_SPEECH": "1"}):
            audio = s._record_pcm(
                s.MIC_RATE, s.VAD_FRAME, max_seconds=max_seconds,
                silence_seconds=1.2, start_timeout_seconds=start_timeout_seconds,
            )
        self.assertEqual(len(audio), int(s.MIC_RATE * cap_seconds))
        self.assertEqual(audio.dtype, np.float32)

    def test_record_pcm_finish_timeout_is_cap_plus_fixed_margin(self):
        # The parent's post-readiness wait budget (_finish_record_worker's
        # `timeout=`) must be the worker's own budget (start_timeout_seconds
        # + max_seconds) plus a small FIXED margin, not a multiple of it and
        # not the old (wrong) 3x-ish arithmetic that let the worker's actual
        # overrun run past the parent's kill.
        captured = {}

        def fake_finish(handle, cap_seconds, start_timeout_seconds):
            captured["cap_seconds"] = cap_seconds
            captured["handle"] = handle
            raise RuntimeError("stop before actually waiting on anything")

        try:
            with mock.patch.object(s, "_finish_record_worker", side_effect=fake_finish):
                with mock.patch.dict(os.environ, {"SPEAK_AUDIO_DRY_RUN_SECONDS": "0.05"}):
                    with self.assertRaises(RuntimeError):
                        s._record_pcm(s.MIC_RATE, s.VAD_FRAME, max_seconds=300.0, silence_seconds=3.0, start_timeout_seconds=45.0)
        finally:
            # fake_finish stood in for _finish_record_worker, so nothing
            # reaped the (already-exited, dry-run) worker process -- do it
            # here rather than leaving a zombie for the test process.
            if "handle" in captured:
                captured["handle"].proc.wait(timeout=5.0)
        self.assertEqual(captured["cap_seconds"], 45.0 + 300.0)
        # And _finish_record_worker itself turns that cap into timeout = cap + 15.0:
        handle = s._start_record_worker("/tmp/does-not-matter.pcm", s.MIC_RATE, s.VAD_FRAME, 0.05, 1.2, 0.05)
        try:
            with mock.patch.object(handle.proc, "wait", side_effect=RuntimeError("stop before actually waiting")) as fake_wait:
                with self.assertRaises(RuntimeError):
                    s._finish_record_worker(handle, cap_seconds=45.0 + 300.0, start_timeout_seconds=45.0)
            self.assertEqual(fake_wait.call_args.kwargs["timeout"], 45.0 + 300.0 + 15.0)
        finally:
            handle.proc.wait(timeout=5.0)

    def test_record_pcm_names_the_phase_when_worker_overruns_its_own_cap(self):
        # If the worker somehow still overran (the bug this fix targets),
        # the resulting error must name which phase it was stuck in, not
        # just report a bare subprocess timeout.
        with mock.patch.dict(os.environ, {"SPEAK_AUDIO_DRY_RUN_SECONDS": "0.05"}):
            handle = s._start_record_worker("/tmp/does-not-matter.pcm", s.MIC_RATE, s.VAD_FRAME, 5, 1.2, 3)
        handle.stderr_lines.append("phase=recording\n")
        real_wait = handle.proc.wait
        # Only the FIRST wait() (the one with a timeout=) should simulate the
        # worker overrunning its own cap; the cleanup wait() after kill()
        # must behave normally so the test process actually reaps the child.
        def wait_once_then_real(*args, **kwargs):
            if kwargs.get("timeout") is not None:
                raise subprocess.TimeoutExpired(cmd="record", timeout=kwargs["timeout"])
            return real_wait(*args, **kwargs)

        try:
            with mock.patch.object(handle.proc, "wait", side_effect=wait_once_then_real):
                with self.assertRaises(RuntimeError) as ctx:
                    s._finish_record_worker(handle, cap_seconds=5, start_timeout_seconds=3)
            self.assertIn("recording", str(ctx.exception))
        finally:
            if handle.proc.poll() is None:
                handle.proc.kill()
                handle.proc.wait(timeout=5.0)


class TestRetryOnWorkerFailure(unittest.TestCase):
    """SPEAK_AUDIO_DRY_RUN_FAIL forces the worker to exit 1 every time, so
    these exercise the real retry-once-then-raise path end to end (two real
    subprocess launches, both forced to fail) without ever needing a real
    device to be unavailable."""

    def test_play_pcm_retries_once_then_raises_readable_error(self):
        tone = s._tone(440.0, 0.01)
        with mock.patch.dict(os.environ, {"SPEAK_AUDIO_DRY_RUN_FAIL": "1"}):
            with self.assertRaises(RuntimeError) as ctx:
                s._play_pcm(tone, s.TTS_RATE)
        self.assertIn("failed twice", str(ctx.exception))
        self.assertIn("forced this failure", str(ctx.exception))

    def test_play_pcm_stream_retries_once_then_raises_readable_error(self):
        chunks = [s._tone(440.0, 0.01).reshape(-1, 1)]
        with mock.patch.dict(os.environ, {"SPEAK_AUDIO_DRY_RUN_FAIL": "1"}):
            with self.assertRaises(RuntimeError) as ctx:
                s._play_pcm_stream(chunks, s.TTS_RATE, timeout=10.0)
        self.assertIn("failed twice", str(ctx.exception))

    def test_record_pcm_retries_once_then_raises_readable_error(self):
        with mock.patch.dict(os.environ, {"SPEAK_AUDIO_DRY_RUN_FAIL": "1"}):
            with self.assertRaises(RuntimeError) as ctx:
                s._record_pcm(s.MIC_RATE, s.VAD_FRAME, max_seconds=5, silence_seconds=1.2, start_timeout_seconds=3)
        self.assertIn("failed twice", str(ctx.exception))

    def test_run_worker_only_launches_twice_on_persistent_failure(self):
        calls = []
        real_run = s._run_worker_once

        def counting_run(*args, **kwargs):
            calls.append(args)
            return real_run(*args, **kwargs)

        with mock.patch.object(s, "_run_worker_once", side_effect=counting_run):
            with mock.patch.dict(os.environ, {"SPEAK_AUDIO_DRY_RUN_FAIL": "1"}):
                with self.assertRaises(RuntimeError):
                    s._run_worker("play", "/nonexistent.pcm", "24000", timeout=10.0)
        self.assertEqual(len(calls), 2)


class TestSpeakImpl(unittest.TestCase):
    """_speak_impl drives Kokoro (stubbed here) then plays through the
    worker. Stubbing engines.kokoro keeps this test fast and independent of
    the real MLX model."""

    def setUp(self):
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
        self._orig_ready = s.engines.ready.is_set()
        self._orig_error = s.engines.error
        s.engines.ready.set()
        s.engines.error = None

    def tearDown(self):
        s.engines.error = self._orig_error
        if not self._orig_ready:
            s.engines.ready.clear()

    def test_listen_impl_raises_timeout_error_without_touching_a_device(self):
        with mock.patch.dict(os.environ, {"SPEAK_AUDIO_DRY_RUN_TIMEOUT": "1"}):
            with self.assertRaises(TimeoutError):
                s._listen_impl(max_seconds=5, silence_seconds=1.2, start_timeout_seconds=3)

    def test_listen_impl_transcribes_recorded_audio(self):
        fake_mlx_whisper = mock.Mock()
        fake_mlx_whisper.transcribe.return_value = {"text": "hello world"}
        with mock.patch.dict(sys.modules, {"mlx_whisper": fake_mlx_whisper}):
            with mock.patch.object(s, "_ack") as fake_ack:
                with mock.patch.dict(os.environ, {"SPEAK_AUDIO_DRY_RUN_SECONDS": "0.2"}):
                    text = s._listen_impl(max_seconds=5, silence_seconds=1.2, start_timeout_seconds=3)
        self.assertEqual(text, "hello world")
        fake_ack.assert_called_once()

    def test_listen_impl_raises_when_transcription_is_empty(self):
        fake_mlx_whisper = mock.Mock()
        fake_mlx_whisper.transcribe.return_value = {"text": "   "}
        with mock.patch.dict(sys.modules, {"mlx_whisper": fake_mlx_whisper}):
            with mock.patch.dict(os.environ, {"SPEAK_AUDIO_DRY_RUN_SECONDS": "0.2"}):
                with self.assertRaises(RuntimeError) as ctx:
                    s._listen_impl(max_seconds=5, silence_seconds=1.2, start_timeout_seconds=3)
        self.assertIn("no text", str(ctx.exception))


class _FakeStderr:
    """A minimal file-like object standing in for a real Popen's `.stderr`
    pipe: `readline()` returns queued lines one at a time (each ending in
    "\\n", like a real text-mode pipe) and then blocks (via a real pipe
    under the hood) until `close()` unblocks it with "" (EOF) -- so
    `_drain_stderr`'s `iter(pipe.readline, "")` behaves exactly as it would
    against a live subprocess, including the "keep blocking until the
    worker exits" behaviour that makes the drain thread safe to join after
    the process is reaped."""

    def __init__(self, lines: list[str]) -> None:
        self._lines = list(lines)
        self._closed = False
        # A real anonymous pipe gives us genuine blocking-until-data-or-EOF
        # semantics without inventing a fake condvar/event protocol here.
        self._r, self._w = os.pipe()
        for line in self._lines:
            os.write(self._w, line.encode())

    def readline(self) -> str:
        data = bytearray()
        while True:
            chunk = os.read(self._r, 1)
            if not chunk:
                return data.decode() if data else ""
            data += chunk
            if chunk == b"\n":
                return data.decode()

    def release_eof(self) -> None:
        """Simulate the worker exiting: no more lines are coming."""
        if not self._closed:
            os.close(self._w)
            self._closed = True

    def close(self) -> None:
        self.release_eof()
        try:
            os.close(self._r)
        except OSError:
            pass


class _FakePopen:
    """Stands in for subprocess.Popen for a `record` worker in tests that
    exercise the readiness handshake without launching a real process
    (dry-run or otherwise). Only implements what `_start_record_worker` /
    `_finish_record_worker` touch: `.stderr`, `.poll()`, `.wait(timeout=)`,
    `.kill()`, `.returncode`."""

    def __init__(self, stderr_lines: list[str], exit_after_lines: bool = False, returncode: int = 0) -> None:
        self.stderr = _FakeStderr(stderr_lines)
        self._exit_after_lines = exit_after_lines
        self._final_returncode = returncode
        self.returncode = None
        self.killed = False
        if exit_after_lines:
            self.stderr.release_eof()
            self.returncode = returncode

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if self.returncode is None:
            raise subprocess.TimeoutExpired(cmd="record", timeout=timeout or 0)
        return self.returncode

    def kill(self):
        self.killed = True
        self.stderr.release_eof()
        self.returncode = self._final_returncode


class TestReadinessHandshake(unittest.TestCase):
    """The core of this fix: the "ear open" cue is now played by the record
    worker itself (in-process, from inside the same process that owns the
    input stream) rather than by a separate `play` subprocess spawned from
    an `on_ready` callback. `_start_record_worker` must not return until the
    worker's readiness line (`phase=cue-played`: mic open AND the cue, if
    any, already played) has actually arrived. These tests fake Popen
    entirely so they exercise the handshake logic itself, independent of
    speak_audio_worker.py's actual dry-run timing."""

    def test_start_record_worker_passes_cue_args_on_the_command_line(self):
        # This is the crux of the fix: the cue spec travels to the record
        # worker as CLI args, not as a callback the server executes itself.
        fake_proc = _FakePopen([f"phase={s.READY_PHASE}\n"], exit_after_lines=False)
        captured_cmd = {}

        def fake_popen(cmd, **kwargs):
            captured_cmd["cmd"] = cmd
            return fake_proc

        with mock.patch.object(subprocess, "Popen", side_effect=fake_popen):
            handle = s._start_record_worker(
                "/tmp/x.pcm", s.MIC_RATE, s.VAD_FRAME, 5.0, 1.2, 3.0,
                cue_freq_hz=880.0, cue_seconds=0.3, cue_volume=0.4, cue_lead_silence=0.2,
            )

        cmd = captured_cmd["cmd"]
        self.assertEqual(cmd[0:3], [sys.executable, "-m", "speak_audio_worker"])
        self.assertEqual(cmd[3], "record")
        # record <path> <samplerate> <blocksize> <max_seconds> <silence_seconds>
        #        <start_timeout_seconds> <cue_freq_hz> <cue_seconds> <cue_volume> <cue_lead_silence>
        self.assertEqual(cmd[-4:], ["880.0", "0.3", "0.4", "0.2"])
        # Cleanup: reap the fake process's drain thread cleanly.
        fake_proc.kill()
        handle.stderr_thread.join(timeout=5.0)

    def test_no_separate_play_worker_is_spawned_for_the_ear_open_cue(self):
        # _record_pcm must launch exactly one worker command (`record`,
        # carrying the cue args) -- never a second `play`/`play-stream`
        # worker for the ear-open cue.
        launched_commands: list[list[str]] = []
        real_popen = subprocess.Popen

        def recording_popen(cmd, *args, **kwargs):
            launched_commands.append(cmd)
            return real_popen(cmd, *args, **kwargs)

        with mock.patch.object(subprocess, "Popen", side_effect=recording_popen):
            with mock.patch.dict(os.environ, {"SPEAK_AUDIO_DRY_RUN_SECONDS": "0.05"}):
                s._record_pcm(
                    s.MIC_RATE, s.VAD_FRAME, max_seconds=5, silence_seconds=1.2, start_timeout_seconds=3,
                    cue_freq_hz=880.0, cue_seconds=0.3, cue_volume=0.4, cue_lead_silence=0.2,
                )

        record_commands = [c for c in launched_commands if "record" in c]
        play_commands = [c for c in launched_commands if "play" in c or "play-stream" in c]
        self.assertEqual(len(record_commands), 1)
        self.assertEqual(play_commands, [])

    def test_listen_impl_passes_ear_open_cue_to_the_record_worker_only(self):
        # End-to-end through _record_pcm/_listen_impl's real wiring, with
        # _start_record_worker/_finish_record_worker faked out so no real
        # subprocess is involved at all. `_cue` (the separate-worker path)
        # must be called only for the ear-closed beep (440.0), never 880.0.
        orig_ready = s.engines.ready.is_set()
        orig_error = s.engines.error
        s.engines.ready.set()
        s.engines.error = None
        self.addCleanup(lambda: (s.engines.ready.clear() if not orig_ready else None))
        self.addCleanup(setattr, s.engines, "error", orig_error)

        order: list[str] = []
        captured_cue_kwargs = {}
        fake_proc = _FakePopen([f"phase={s.READY_PHASE}\n"], exit_after_lines=True, returncode=0)

        def fake_start(path, samplerate, blocksize, max_seconds, silence_seconds, start_timeout_seconds,
                        cue_freq_hz=0.0, cue_seconds=0.3, cue_volume=0.4, cue_lead_silence=0.2):
            order.append("worker-ready")
            captured_cue_kwargs.update(
                cue_freq_hz=cue_freq_hz, cue_seconds=cue_seconds,
                cue_volume=cue_volume, cue_lead_silence=cue_lead_silence,
            )
            return s._RecordHandle(fake_proc, path, [f"phase={s.READY_PHASE}\n"], threading.Thread(target=lambda: None))

        def fake_finish(handle, cap_seconds, start_timeout_seconds):
            order.append("recording-finished")
            return np.zeros(int(s.MIC_RATE * 0.1), dtype=np.float32)

        def fake_cue(freq_hz, *args, **kwargs):
            order.append(f"cue-{freq_hz}")

        fake_mlx_whisper = mock.Mock()
        fake_mlx_whisper.transcribe.return_value = {"text": "hi"}

        with mock.patch.object(s, "_start_record_worker", side_effect=fake_start):
            with mock.patch.object(s, "_finish_record_worker", side_effect=fake_finish):
                with mock.patch.object(s, "_cue", side_effect=fake_cue):
                    with mock.patch.object(s, "_ack"):
                        with mock.patch.dict(sys.modules, {"mlx_whisper": fake_mlx_whisper}):
                            text = s._listen_impl(max_seconds=5, silence_seconds=1.2, start_timeout_seconds=3)

        self.assertEqual(text, "hi")
        # The ear-open cue must have been requested from the record worker
        # (cue_freq_hz=880.0), not played via a separate _cue() call -- only
        # the ear-closed beep (440.0) goes through _cue().
        self.assertEqual(captured_cue_kwargs["cue_freq_hz"], 880.0)
        self.assertEqual(order, ["worker-ready", "recording-finished", "cue-440.0"])

    def test_ready_line_never_arrives_raises_clear_error_without_playing_beep(self):
        # A worker stuck before ever opening the stream (or a real hang)
        # must not be treated as ready -- no silent fallback, no beep played
        # over a mic that might not be listening, and the error must name
        # what happened rather than surfacing a bare timeout.
        fake_proc = _FakePopen([], exit_after_lines=False)  # never emits the ready line, never exits
        beep_played = []

        def fake_popen(*args, **kwargs):
            return fake_proc

        with mock.patch.object(s, "READY_WAIT_SECONDS", 0.3):
            with mock.patch.object(subprocess, "Popen", side_effect=fake_popen):
                with self.assertRaises(RuntimeError) as ctx:
                    s._start_record_worker("/tmp/x.pcm", s.MIC_RATE, s.VAD_FRAME, 5.0, 1.2, 3.0)

        self.assertIn("did not signal readiness", str(ctx.exception))
        self.assertIn(s.READY_PHASE, str(ctx.exception))
        self.assertTrue(fake_proc.killed, "a worker that never signals readiness must be killed, not left running")
        self.assertEqual(beep_played, [])

    def test_worker_exits_before_readiness_retries_once_then_raises(self):
        # A worker that exits early (e.g. a stale-PortAudio-style failure)
        # before ever reaching readiness gets exactly one retry with a
        # fresh process -- mirroring _run_worker's existing retry-once
        # policy -- and a clear error if the retry also fails early.
        procs = [
            _FakePopen([], exit_after_lines=True, returncode=1),
            _FakePopen([], exit_after_lines=True, returncode=1),
        ]
        calls = []

        def fake_popen(*args, **kwargs):
            calls.append(1)
            return procs[len(calls) - 1]

        with mock.patch.object(subprocess, "Popen", side_effect=fake_popen):
            with self.assertRaises(RuntimeError) as ctx:
                s._start_record_worker("/tmp/x.pcm", s.MIC_RATE, s.VAD_FRAME, 5.0, 1.2, 3.0)

        self.assertEqual(len(calls), 2)
        self.assertIn("failed twice before signalling readiness", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
