"""Tests for speak_server.py's audio helpers.

Every test here runs with SPEAK_AUDIO_DRY_RUN=1 (set at module import time,
before speak_server ever gets a chance to spawn a worker), so no worker
subprocess ever opens a real PortAudio stream and no test touches the
microphone or speakers. speak_server.py itself is asserted to never import
sounddevice at all -- that's the point of the fix.
"""

from __future__ import annotations

import os
import sys
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


if __name__ == "__main__":
    unittest.main()
