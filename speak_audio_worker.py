"""Short-lived helper process for algolearn-speak: the only code in this
repo that touches sounddevice/PortAudio.

Why this exists: the long-lived MCP server process must never restart (it
owns the stdio pipe to Claude Code), but PortAudio's device snapshot goes
stale after the machine has been idle for a couple of hours (see
docs/AUDIO_LIFECYCLE_AUDIT.md) and the only reliable fix is a fresh process.
So every actual `sd.OutputStream`/`sd.InputStream` open happens here, in a
process launched fresh for exactly one call and then thrown away. There is
no long-lived PortAudio state in this file to go stale between calls.

Invoked as `sys.executable -m speak_audio_worker <command> [args...]`.
Audio payloads cross the process boundary as raw float32 PCM files (no WAV
header — both ends are this same codebase, so a fixed dtype/layout is
enough). All commands print a single line to stdout on success; on failure
they print a message to stderr and exit non-zero.

Commands:
  play <pcm_path> <samplerate>
      Play a mono float32 PCM file through the default output device.

  play-stream <samplerate>
      Play mono float32 PCM read from stdin as it arrives, so a caller can
      pipe synthesis chunks through as they're produced instead of waiting
      for the whole utterance. Stdin EOF means "no more audio"; the process
      plays out whatever is buffered and exits.

  record <out_pcm_path> <samplerate> <blocksize> <max_seconds>
         <silence_seconds> <start_timeout_seconds>
      Record from the default input device, gated by Silero VAD: stop
      after `silence_seconds` of quiet following detected speech, or after
      `max_seconds` of speech, whichever comes first. If nobody speaks
      within `start_timeout_seconds`, exit with code TIMEOUT_EXIT_CODE and
      write nothing.

      Hard wall-clock cap: `start_timeout_seconds + max_seconds`, measured
      from stream open (not from speech start). This is the absolute upper
      bound on how long `record` can run, checked on every loop iteration
      regardless of VAD state. It exists because a VAD can be fooled by
      continuous non-speech sound (e.g. a TV in the room) into believing
      speech never stops, so "silence_seconds after speech" may never
      arrive — no heuristic is applied to detect that case; the wall-clock
      cap is the only guard. When the cap is hit, whatever audio has been
      captured so far is written to `out_pcm_path` and the process exits 0
      (even if no VAD "end" event ever fired) — the caller always gets
      something rather than nothing.

      On success (including a cap-triggered stop), recorded audio (mono
      float32 PCM) is written to `out_pcm_path`. The process prints a
      single status line to stderr on each phase change
      (`phase=waiting-for-speech` at stream open, `phase=recording` on the
      first detected speech) so a caller that still sees the subprocess
      overrun its own timeout can report which phase it was in.

Env:
  SPEAK_AUDIO_DRY_RUN=1
      Never touch a real device. `play`/`play-stream` no-op (still drain
      stdin for play-stream, so the caller's write doesn't block on a full
      pipe). `record` synthesizes `SPEAK_AUDIO_DRY_RUN_SECONDS` (default 1.0)
      of low-amplitude noise as "speech" instead of opening the mic, so the
      file round-trip can be exercised without hardware. Used by tests only.
  SPEAK_AUDIO_DRY_RUN_TIMEOUT=1
      With dry-run also set: `record` returns TIMEOUT_EXIT_CODE immediately
      instead of synthesizing speech, so the caller's timeout path can be
      exercised deterministically. Used by tests only.
  SPEAK_AUDIO_DRY_RUN_ENDLESS_SPEECH=1
      With dry-run also set: `record` behaves as if speech starts
      immediately and never stops (as continuous non-speech noise, e.g. a
      TV, can fool the real VAD into doing) instead of synthesizing a fixed
      `SPEAK_AUDIO_DRY_RUN_SECONDS` clip. Used to exercise the wall-clock
      cap path deterministically: `record` must still stop at
      `start_timeout_seconds + max_seconds` and write whatever it
      captured, exiting 0. Used by tests only.
  SPEAK_AUDIO_DRY_RUN_FAIL=1
      With dry-run also set: every command exits 1 with a fixed message on
      stderr, so the caller's retry-then-raise path can be exercised
      deterministically without a real device failure. Used by tests only.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

TIMEOUT_EXIT_CODE = 2

VAD_THRESHOLD = 0.5
VAD_SPEECH_PAD_MS = 300


def _dry_run() -> bool:
    return os.environ.get("SPEAK_AUDIO_DRY_RUN") == "1"


def _dry_run_force_fail() -> bool:
    return _dry_run() and os.environ.get("SPEAK_AUDIO_DRY_RUN_FAIL") == "1"


def _dry_run_force_timeout() -> bool:
    return _dry_run() and os.environ.get("SPEAK_AUDIO_DRY_RUN_TIMEOUT") == "1"


def _dry_run_endless_speech() -> bool:
    return _dry_run() and os.environ.get("SPEAK_AUDIO_DRY_RUN_ENDLESS_SPEECH") == "1"


def _log_phase(phase: str) -> None:
    print(f"phase={phase}", file=sys.stderr, flush=True)


def _load_pcm(path: str) -> np.ndarray:
    return np.fromfile(path, dtype=np.float32)


def _save_pcm(path: str, audio: np.ndarray) -> None:
    audio.astype(np.float32).tofile(path)


def cmd_play(pcm_path: str, samplerate: int) -> None:
    audio = _load_pcm(pcm_path).reshape(-1, 1)
    if _dry_run():
        return
    import sounddevice as sd

    sd.play(audio, samplerate=samplerate)
    sd.wait()


# Chunk size (frames) read from stdin at a time for play-stream. Small enough
# to keep latency to first sound low, large enough not to spin the read loop.
STREAM_READ_FRAMES = 2400  # 0.1 s at 24 kHz
BYTES_PER_FRAME = 4  # float32


def cmd_play_stream(samplerate: int) -> None:
    stdin = sys.stdin.buffer
    if _dry_run():
        while stdin.read(STREAM_READ_FRAMES * BYTES_PER_FRAME):
            pass
        return

    import sounddevice as sd

    with sd.OutputStream(samplerate=samplerate, channels=1, dtype="float32") as out:
        while True:
            raw = stdin.read(STREAM_READ_FRAMES * BYTES_PER_FRAME)
            if not raw:
                break
            # A short read at EOF can leave a partial frame; drop it rather
            # than crash the reshape — a few samples of silence lost.
            usable = len(raw) - (len(raw) % BYTES_PER_FRAME)
            if usable == 0:
                continue
            chunk = np.frombuffer(raw[:usable], dtype=np.float32).reshape(-1, 1)
            out.write(chunk)


def cmd_record(
    out_pcm_path: str,
    samplerate: int,
    blocksize: int,
    max_seconds: float,
    silence_seconds: float,
    start_timeout_seconds: float,
) -> int:
    """Returns an exit code: 0 on success, TIMEOUT_EXIT_CODE if nobody spoke.

    Hard wall-clock cap = start_timeout_seconds + max_seconds, measured from
    stream open. Checked every loop iteration regardless of VAD state, so a
    VAD that never emits "end" (e.g. fooled by continuous non-speech sound
    like a TV) cannot make this loop run longer than the cap. On a
    cap-triggered stop, whatever has been captured is written and the
    process exits 0 — never a bare timeout with nothing written.
    """
    cap_seconds = start_timeout_seconds + max_seconds

    if _dry_run():
        if _dry_run_force_timeout():
            return TIMEOUT_EXIT_CODE
        _log_phase("waiting-for-speech")
        if _dry_run_endless_speech():
            # Simulate a VAD permanently fooled by continuous non-speech
            # noise: speech "starts" immediately and never ends. The only
            # thing that can stop this is the wall-clock cap.
            _log_phase("recording")
            rng = np.random.default_rng(0)
            audio = (0.01 * rng.standard_normal(int(samplerate * cap_seconds))).astype(np.float32)
            _save_pcm(out_pcm_path, audio)
            return 0
        seconds = float(os.environ.get("SPEAK_AUDIO_DRY_RUN_SECONDS", "1.0"))
        _log_phase("recording")
        rng = np.random.default_rng(0)
        audio = (0.01 * rng.standard_normal(int(samplerate * seconds))).astype(np.float32)
        _save_pcm(out_pcm_path, audio)
        return 0

    import torch
    import sounddevice as sd
    from silero_vad import VADIterator, load_silero_vad

    vad_model = load_silero_vad()
    vad = VADIterator(
        vad_model,
        threshold=VAD_THRESHOLD,
        sampling_rate=samplerate,
        min_silence_duration_ms=int(silence_seconds * 1000),
        speech_pad_ms=VAD_SPEECH_PAD_MS,
    )

    frames: list[np.ndarray] = []
    speaking = False
    t_open = time.time()

    _log_phase("waiting-for-speech")
    with sd.InputStream(samplerate=samplerate, channels=1, dtype="float32", blocksize=blocksize) as mic:
        while True:
            frame, _ = mic.read(blocksize)
            frame = frame[:, 0]
            frames.append(frame)
            event = vad(torch.from_numpy(frame.copy()), return_seconds=False)
            now = time.time()
            elapsed = now - t_open
            if event and "start" in event and not speaking:
                speaking = True
                _log_phase("recording")
                # keep ~0.5 s of pre-roll before the detected start
                frames = frames[-int(0.5 * samplerate / blocksize):]
            if event and "end" in event and speaking:
                break
            if not speaking and elapsed > start_timeout_seconds:
                return TIMEOUT_EXIT_CODE
            # Hard wall-clock cap, checked unconditionally every iteration
            # (not gated on VAD state) so continuous non-speech noise that
            # the VAD miscounts as speech cannot extend the recording past
            # this bound. Write whatever was captured and succeed.
            if elapsed > cap_seconds:
                break

    _save_pcm(out_pcm_path, np.concatenate(frames))
    return 0


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: speak_audio_worker <play|record> ...", file=sys.stderr)
        return 1
    if _dry_run_force_fail():
        # Still drain stdin for play-stream so a caller blocked on a full
        # pipe write doesn't hang waiting for us.
        if argv[0] == "play-stream":
            while sys.stdin.buffer.read(65536):
                pass
        print("SPEAK_AUDIO_DRY_RUN_FAIL forced this failure", file=sys.stderr)
        return 1
    command, *args = argv
    if command == "play":
        pcm_path, samplerate = args
        cmd_play(pcm_path, int(samplerate))
        return 0
    if command == "play-stream":
        (samplerate,) = args
        cmd_play_stream(int(samplerate))
        return 0
    if command == "record":
        out_pcm_path, samplerate, blocksize, max_seconds, silence_seconds, start_timeout_seconds = args
        return cmd_record(
            out_pcm_path,
            int(samplerate),
            int(blocksize),
            float(max_seconds),
            float(silence_seconds),
            float(start_timeout_seconds),
        )
    print(f"unknown command: {command!r}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
