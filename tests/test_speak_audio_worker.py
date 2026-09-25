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
from unittest import mock

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


# record's CLI now takes 4 trailing cue args: cue_freq_hz, cue_seconds,
# cue_volume, cue_lead_silence. "0" for cue_freq_hz means "no cue" -- used
# by every test below that isn't specifically about the cue, since dry-run
# never touches a real device either way and these tests only care about
# the recording behaviour.
NO_CUE_ARGS = ["0", "0", "0", "0"]


class TestRecord(unittest.TestCase):
    def test_record_dry_run_writes_synthesized_audio(self):
        with tempfile.NamedTemporaryFile(suffix=".pcm") as f:
            out_path = f.name
        try:
            proc = run_worker(
                ["record", out_path, "16000", "512", "5", "1.2", "3", *NO_CUE_ARGS],
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
                ["record", out_path, "16000", "512", str(max_seconds), "1.2", str(start_timeout_seconds), *NO_CUE_ARGS],
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
                ["record", out_path, "16000", "512", "0.2", "1.2", "0.1", *NO_CUE_ARGS],
                env_extra={"SPEAK_AUDIO_DRY_RUN_ENDLESS_SPEECH": "1"},
                timeout=10.0,
            )
            stderr = proc.stderr.decode("utf-8", "replace") if isinstance(proc.stderr, bytes) else proc.stderr
            self.assertIn("phase=stream-open", stderr)
            self.assertIn("phase=cue-played", stderr)
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
                ["record", out_path, "16000", "512", "5", "1.2", "3", *NO_CUE_ARGS],
                env_extra={"SPEAK_AUDIO_DRY_RUN_TIMEOUT": "1"},
            )
            import speak_audio_worker
            self.assertEqual(proc.returncode, speak_audio_worker.TIMEOUT_EXIT_CODE, proc.stderr)
            self.assertFalse(os.path.exists(out_path))
        finally:
            if os.path.exists(out_path):
                os.unlink(out_path)


class TestRecordCueArgs(unittest.TestCase):
    """Worker-side test for the cue: cmd_record must accept the cue CLI args
    and, in dry-run mode, log `phase=cue-played` without touching any real
    hardware (sounddevice is never imported in dry-run at all)."""

    def test_dry_run_logs_cue_played_regardless_of_cue_args(self):
        # Dry-run never opens a real device (see cmd_record: the dry-run
        # branch returns before `import sounddevice`), so it logs
        # phase=cue-played unconditionally -- this just proves the CLI
        # parses cue args without error and the phase line still appears.
        with tempfile.NamedTemporaryFile(suffix=".pcm") as f:
            out_path = f.name
        try:
            proc = run_worker(
                ["record", out_path, "16000", "512", "5", "1.2", "3", "880.0", "0.3", "0.4", "0.2"],
                env_extra={"SPEAK_AUDIO_DRY_RUN_SECONDS": "0.05"},
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            stderr = proc.stderr.decode("utf-8", "replace") if isinstance(proc.stderr, bytes) else proc.stderr
            phases = [line.split("=", 1)[1] for line in stderr.splitlines() if line.startswith("phase=")]
            self.assertEqual(phases, ["stream-open", "cue-played", "recording"])
        finally:
            if os.path.exists(out_path):
                os.unlink(out_path)

    def test_cmd_record_accepts_cue_kwargs_directly(self):
        # Call cmd_record() directly (still dry-run, so no device is
        # touched) to pin the parameter names/order cmd_record's CLI parsing
        # in main() relies on.
        import speak_audio_worker as w

        with mock.patch.dict(os.environ, {"SPEAK_AUDIO_DRY_RUN": "1", "SPEAK_AUDIO_DRY_RUN_SECONDS": "0.05"}):
            with tempfile.NamedTemporaryFile(suffix=".pcm") as f:
                out_path = f.name
            try:
                rc = w.cmd_record(
                    out_path, 16000, 512, max_seconds=5, silence_seconds=1.2, start_timeout_seconds=3,
                    cue_freq_hz=880.0, cue_seconds=0.3, cue_volume=0.4, cue_lead_silence=0.2,
                )
                self.assertEqual(rc, 0)
            finally:
                if os.path.exists(out_path):
                    os.unlink(out_path)


class TestResolveInputDevice(unittest.TestCase):
    """_resolve_input_device is the substring -> (index, native_rate)
    lookup used when SPEAK_INPUT_DEVICE is set. It's a thin wrapper around
    sd.query_devices(name, kind="input"), so these tests stub sounddevice
    itself rather than touching a real device."""

    def test_resolves_unique_substring_match(self):
        import speak_audio_worker as w

        fake_sd = mock.Mock()
        fake_sd.query_devices.return_value = {"index": 4, "name": "Logi USB Headset", "default_samplerate": 48000.0}
        with mock.patch.dict(sys.modules, {"sounddevice": fake_sd}):
            index, rate = w._resolve_input_device("Logi USB Headset")
        self.assertEqual(index, 4)
        self.assertEqual(rate, 48000.0)
        fake_sd.query_devices.assert_called_once_with("Logi USB Headset", kind="input")

    def test_kind_input_avoids_duplicate_name_error_for_input_output_devices(self):
        # A headset that exposes both an input and an output entry under the
        # same name would make a kind-less query_devices() raise "multiple
        # devices found" -- proving kind="input" is passed is the whole
        # point of this wrapper.
        import speak_audio_worker as w

        fake_sd = mock.Mock()
        fake_sd.query_devices.return_value = {"index": 6, "name": "AirPods", "default_samplerate": 24000.0}
        with mock.patch.dict(sys.modules, {"sounddevice": fake_sd}):
            w._resolve_input_device("AirPods")
        self.assertEqual(fake_sd.query_devices.call_args.kwargs.get("kind"), "input")

    def test_no_match_raises_without_fallback(self):
        import speak_audio_worker as w

        fake_sd = mock.Mock()
        fake_sd.query_devices.side_effect = ValueError("No input device matching 'nonexistent'")
        with mock.patch.dict(sys.modules, {"sounddevice": fake_sd}):
            with self.assertRaises(ValueError):
                w._resolve_input_device("nonexistent")

    def test_ambiguous_match_raises_without_fallback(self):
        import speak_audio_worker as w

        fake_sd = mock.Mock()
        fake_sd.query_devices.side_effect = ValueError("Multiple input devices found for 'usb'")
        with mock.patch.dict(sys.modules, {"sounddevice": fake_sd}):
            with self.assertRaises(ValueError):
                w._resolve_input_device("usb")


class TestRecordDeviceAndArchiveDryRun(unittest.TestCase):
    """cmd_record's device_name/archive_pcm_path args, exercised entirely in
    dry-run (SPEAK_AUDIO_DRY_RUN=1) via SPEAK_AUDIO_DRY_RUN_NATIVE_RATE,
    which stands in for a resolved device's native rate without ever
    calling sd.query_devices."""

    def test_device_name_without_native_rate_env_behaves_like_default_device(self):
        # A device_name is given but SPEAK_AUDIO_DRY_RUN_NATIVE_RATE is not
        # set (dry-run's stand-in for "native rate not above samplerate") --
        # capture_rate falls back to samplerate, exactly like the unset
        # device_name path, and no archive is written even if an archive
        # path was passed.
        with tempfile.NamedTemporaryFile(suffix=".pcm") as f:
            out_path = f.name
        with tempfile.NamedTemporaryFile(suffix=".pcm") as f:
            archive_path = f.name
        try:
            proc = run_worker(
                ["record", out_path, "16000", "512", "5", "1.2", "3", *NO_CUE_ARGS, "Some Device", archive_path],
                env_extra={"SPEAK_AUDIO_DRY_RUN_SECONDS": "0.2"},
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            audio = np.fromfile(out_path, dtype=np.float32)
            self.assertEqual(len(audio), int(16000 * 0.2))
            self.assertFalse(os.path.exists(archive_path))
        finally:
            if os.path.exists(out_path):
                os.unlink(out_path)
            if os.path.exists(archive_path):
                os.unlink(archive_path)

    def test_native_rate_above_samplerate_writes_archive_at_native_rate(self):
        with tempfile.NamedTemporaryFile(suffix=".pcm") as f:
            out_path = f.name
        with tempfile.NamedTemporaryFile(suffix=".pcm") as f:
            archive_path = f.name
        try:
            proc = run_worker(
                ["record", out_path, "16000", "512", "5", "1.2", "3", *NO_CUE_ARGS, "Logi USB Headset", archive_path],
                env_extra={"SPEAK_AUDIO_DRY_RUN_SECONDS": "0.2", "SPEAK_AUDIO_DRY_RUN_NATIVE_RATE": "48000"},
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            audio = np.fromfile(out_path, dtype=np.float32)
            archive = np.fromfile(archive_path, dtype=np.float32)
            self.assertEqual(len(audio), int(16000 * 0.2))
            self.assertEqual(len(archive), int(48000 * 0.2))
            stderr = proc.stderr.decode("utf-8", "replace") if isinstance(proc.stderr, bytes) else proc.stderr
            self.assertIn("archive-rate=48000", stderr)
        finally:
            if os.path.exists(out_path):
                os.unlink(out_path)
            if os.path.exists(archive_path):
                os.unlink(archive_path)

    def test_no_archive_path_means_no_archive_file_even_with_native_rate(self):
        with tempfile.NamedTemporaryFile(suffix=".pcm") as f:
            out_path = f.name
        try:
            proc = run_worker(
                ["record", out_path, "16000", "512", "5", "1.2", "3", *NO_CUE_ARGS, "Logi USB Headset", ""],
                env_extra={"SPEAK_AUDIO_DRY_RUN_SECONDS": "0.1", "SPEAK_AUDIO_DRY_RUN_NATIVE_RATE": "48000"},
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            audio = np.fromfile(out_path, dtype=np.float32)
            self.assertEqual(len(audio), int(16000 * 0.1))
        finally:
            if os.path.exists(out_path):
                os.unlink(out_path)

    def test_native_rate_at_or_below_samplerate_does_not_trigger_native_capture(self):
        # A device whose native rate is <= samplerate must behave exactly
        # like the unset-device path: no resampling, no archive.
        with tempfile.NamedTemporaryFile(suffix=".pcm") as f:
            out_path = f.name
        with tempfile.NamedTemporaryFile(suffix=".pcm") as f:
            archive_path = f.name
        try:
            proc = run_worker(
                ["record", out_path, "16000", "512", "5", "1.2", "3", *NO_CUE_ARGS, "Built-in Mic", archive_path],
                env_extra={"SPEAK_AUDIO_DRY_RUN_SECONDS": "0.1", "SPEAK_AUDIO_DRY_RUN_NATIVE_RATE": "16000"},
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertFalse(os.path.exists(archive_path))
        finally:
            if os.path.exists(out_path):
                os.unlink(out_path)
            if os.path.exists(archive_path):
                os.unlink(archive_path)

    def test_old_style_ten_arg_invocation_still_works(self):
        # Backward compatibility: device_name/archive_pcm_path are optional
        # trailing CLI args -- an invocation with only the original 10 args
        # (as every pre-existing test in this file uses) must still work.
        with tempfile.NamedTemporaryFile(suffix=".pcm") as f:
            out_path = f.name
        try:
            proc = run_worker(
                ["record", out_path, "16000", "512", "5", "1.2", "3", *NO_CUE_ARGS],
                env_extra={"SPEAK_AUDIO_DRY_RUN_SECONDS": "0.15"},
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            audio = np.fromfile(out_path, dtype=np.float32)
            self.assertEqual(len(audio), int(16000 * 0.15))
        finally:
            if os.path.exists(out_path):
                os.unlink(out_path)


class TestResolveOutputDevice(unittest.TestCase):
    """_resolve_output_device is the substring -> index lookup used when
    SPEAK_OUTPUT_DEVICE is set. Thin wrapper around
    sd.query_devices(name, kind="output"), so stub sounddevice rather than
    touch a real device."""

    def test_resolves_unique_substring_match(self):
        import speak_audio_worker as w

        fake_sd = mock.Mock()
        fake_sd.query_devices.return_value = {"index": 4, "name": "Logi USB Headset", "default_samplerate": 48000.0}
        with mock.patch.dict(sys.modules, {"sounddevice": fake_sd}):
            index = w._resolve_output_device("Logi USB Headset")
        self.assertEqual(index, 4)
        fake_sd.query_devices.assert_called_once_with("Logi USB Headset", kind="output")

    def test_kind_output_avoids_duplicate_name_error_for_input_output_devices(self):
        # Same headset-with-both-directions concern as the input resolver,
        # mirrored for the output side: kind="output" must actually be
        # passed, not just kind=None/omitted.
        import speak_audio_worker as w

        fake_sd = mock.Mock()
        fake_sd.query_devices.return_value = {"index": 6, "name": "AirPods", "default_samplerate": 24000.0}
        with mock.patch.dict(sys.modules, {"sounddevice": fake_sd}):
            w._resolve_output_device("AirPods")
        self.assertEqual(fake_sd.query_devices.call_args.kwargs.get("kind"), "output")

    def test_no_match_raises_without_fallback(self):
        import speak_audio_worker as w

        fake_sd = mock.Mock()
        fake_sd.query_devices.side_effect = ValueError("No output device matching 'nonexistent'")
        with mock.patch.dict(sys.modules, {"sounddevice": fake_sd}):
            with self.assertRaises(ValueError):
                w._resolve_output_device("nonexistent")

    def test_ambiguous_match_raises_without_fallback(self):
        import speak_audio_worker as w

        fake_sd = mock.Mock()
        fake_sd.query_devices.side_effect = ValueError("Multiple output devices found for 'usb'")
        with mock.patch.dict(sys.modules, {"sounddevice": fake_sd}):
            with self.assertRaises(ValueError):
                w._resolve_output_device("usb")


class TestOutputDeviceArgsDryRun(unittest.TestCase):
    """cmd_play/cmd_play_stream/cmd_record accept an output_device_name arg
    on the CLI. Exercised in dry-run, which never resolves or touches a
    real device, so these only pin that the arg is accepted (and, for old-
    style invocations without it, that omitting it still works)."""

    def test_play_accepts_trailing_output_device_arg(self):
        with tempfile.NamedTemporaryFile(suffix=".pcm") as f:
            np.array([0.1, 0.2, -0.1], dtype=np.float32).tofile(f.name)
            proc = run_worker(["play", f.name, "24000", "Logi USB Headset"])
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_play_still_works_without_output_device_arg(self):
        with tempfile.NamedTemporaryFile(suffix=".pcm") as f:
            np.array([0.1], dtype=np.float32).tofile(f.name)
            proc = run_worker(["play", f.name, "24000"])
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_play_stream_accepts_trailing_output_device_arg(self):
        payload = np.linspace(-1, 1, 4800, dtype=np.float32).tobytes()
        proc = run_worker(["play-stream", "24000", "Logi USB Headset"], input_bytes=payload)
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_record_accepts_trailing_output_device_arg(self):
        with tempfile.NamedTemporaryFile(suffix=".pcm") as f:
            out_path = f.name
        try:
            proc = run_worker(
                ["record", out_path, "16000", "512", "5", "1.2", "3", *NO_CUE_ARGS, "", "", "Logi USB Headset"],
                env_extra={"SPEAK_AUDIO_DRY_RUN_SECONDS": "0.1"},
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            audio = np.fromfile(out_path, dtype=np.float32)
            self.assertEqual(len(audio), int(16000 * 0.1))
        finally:
            if os.path.exists(out_path):
                os.unlink(out_path)

    def test_cmd_record_accepts_output_device_name_kwarg_directly(self):
        # Pins the parameter name cmd_record's CLI parsing in main() relies
        # on, same style as the existing cue-kwargs test above.
        import speak_audio_worker as w

        with mock.patch.dict(os.environ, {"SPEAK_AUDIO_DRY_RUN": "1", "SPEAK_AUDIO_DRY_RUN_SECONDS": "0.05"}):
            with tempfile.NamedTemporaryFile(suffix=".pcm") as f:
                out_path = f.name
            try:
                rc = w.cmd_record(
                    out_path, 16000, 512, max_seconds=5, silence_seconds=1.2, start_timeout_seconds=3,
                    output_device_name="Logi USB Headset",
                )
                self.assertEqual(rc, 0)
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
