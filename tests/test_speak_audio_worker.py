"""Tests for speak_audio_worker.py — run only in dry-run mode
(SPEAK_AUDIO_DRY_RUN=1), so no test in this file ever opens a real audio
device or touches the microphone/speakers.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_worker(args: list[str], env_extra: dict[str, str] | None = None, input_bytes: bytes | None = None, timeout: float = 10.0) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["SPEAK_AUDIO_DRY_RUN"] = "1"
    env.update(env_extra or {})
    return subprocess.run(
        [sys.executable, "-m", "speak_audio_worker", *args],
        cwd=REPO_ROOT,
        env=env,
        input=input_bytes,
        capture_output=True,
        timeout=timeout,
    )


class TestPlay(unittest.TestCase):
    def test_play_dry_run_succeeds_without_a_device(self):
        with tempfile.NamedTemporaryFile(suffix=".pcm") as f:
            np.array([0.1, 0.2, -0.1], dtype=np.float32).tofile(f.name)
            proc = run_worker(["play", f.name, "24000"])
        self.assertEqual(proc.returncode, 0, proc.stderr)


class TestPlayStream(unittest.TestCase):
    def test_play_stream_drains_stdin_and_exits_zero(self):
        payload = np.linspace(-1, 1, 4800, dtype=np.float32).tobytes()
        proc = run_worker(["play-stream", "24000"], input_bytes=payload)
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_play_stream_handles_empty_input(self):
        proc = run_worker(["play-stream", "24000"], input_bytes=b"")
        self.assertEqual(proc.returncode, 0, proc.stderr)


class TestRecord(unittest.TestCase):
    def test_record_dry_run_writes_synthesized_audio(self):
        with tempfile.NamedTemporaryFile(suffix=".pcm") as f:
            out_path = f.name
        try:
            proc = run_worker(
                ["record", out_path, "16000", "512", "5", "1.2", "3"],
                env_extra={"SPEAK_AUDIO_DRY_RUN_SECONDS": "0.25"},
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            audio = np.fromfile(out_path, dtype=np.float32)
            self.assertEqual(len(audio), int(16000 * 0.25))
        finally:
            if os.path.exists(out_path):
                os.unlink(out_path)

    def test_record_dry_run_endless_speech_stops_at_cap_and_writes_audio(self):
        # Simulates a VAD permanently fooled by continuous non-speech sound
        # (e.g. a TV in the room): "speech" never stops. The worker must
        # still exit 0 at the wall-clock cap (start_timeout_seconds +
        # max_seconds) and write whatever it captured, rather than running
        # forever or exiting with a timeout code.
        max_seconds = 0.2
        start_timeout_seconds = 0.1
        cap_seconds = max_seconds + start_timeout_seconds
        with tempfile.NamedTemporaryFile(suffix=".pcm") as f:
            out_path = f.name
        try:
            proc = run_worker(
                ["record", out_path, "16000", "512", str(max_seconds), "1.2", str(start_timeout_seconds)],
                env_extra={"SPEAK_AUDIO_DRY_RUN_ENDLESS_SPEECH": "1"},
                timeout=10.0,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            audio = np.fromfile(out_path, dtype=np.float32)
            self.assertEqual(len(audio), int(16000 * cap_seconds))
        finally:
            if os.path.exists(out_path):
                os.unlink(out_path)

    def test_record_dry_run_endless_speech_logs_phase_transitions(self):
        with tempfile.NamedTemporaryFile(suffix=".pcm") as f:
            out_path = f.name
        try:
            proc = run_worker(
                ["record", out_path, "16000", "512", "0.2", "1.2", "0.1"],
                env_extra={"SPEAK_AUDIO_DRY_RUN_ENDLESS_SPEECH": "1"},
                timeout=10.0,
            )
            stderr = proc.stderr.decode("utf-8", "replace") if isinstance(proc.stderr, bytes) else proc.stderr
            self.assertIn("phase=waiting-for-speech", stderr)
            self.assertIn("phase=recording", stderr)
        finally:
            if os.path.exists(out_path):
                os.unlink(out_path)

    def test_record_dry_run_timeout_returns_timeout_exit_code(self):
        with tempfile.NamedTemporaryFile(suffix=".pcm") as f:
            out_path = f.name
        # the `with` block already deleted the file; the worker must not
        # write it either when it exits with the timeout code.
        try:
            proc = run_worker(
                ["record", out_path, "16000", "512", "5", "1.2", "3"],
                env_extra={"SPEAK_AUDIO_DRY_RUN_TIMEOUT": "1"},
            )
            import speak_audio_worker
            self.assertEqual(proc.returncode, speak_audio_worker.TIMEOUT_EXIT_CODE, proc.stderr)
            self.assertFalse(os.path.exists(out_path))
        finally:
            if os.path.exists(out_path):
                os.unlink(out_path)


class TestForcedFailure(unittest.TestCase):
    def test_forced_failure_exits_nonzero_with_message(self):
        with tempfile.NamedTemporaryFile(suffix=".pcm") as f:
            np.array([0.1], dtype=np.float32).tofile(f.name)
            proc = run_worker(["play", f.name, "24000"], env_extra={"SPEAK_AUDIO_DRY_RUN_FAIL": "1"})
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn(b"forced this failure", proc.stderr)

    def test_forced_failure_on_play_stream_still_drains_stdin(self):
        payload = b"\x00" * (65536 * 2 + 17)  # bigger than one drain chunk
        proc = run_worker(["play-stream", "24000"], env_extra={"SPEAK_AUDIO_DRY_RUN_FAIL": "1"}, input_bytes=payload)
        self.assertNotEqual(proc.returncode, 0)


if __name__ == "__main__":
    unittest.main()
