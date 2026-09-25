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
  play <pcm_path> <samplerate> [output_device_name]
      Play a mono float32 PCM file through the default output device.
      `output_device_name` (optional, defaults to "" = default output
      device): a substring matched against output device names via
      `sd.query_devices(name, kind="output")` -- same no-fallback rule as
      `record`'s input `device_name` (zero or multiple matches raises).

  play-stream <samplerate> [output_device_name]
      Play mono float32 PCM read from stdin as it arrives, so a caller can
      pipe synthesis chunks through as they're produced instead of waiting
      for the whole utterance. Stdin EOF means "no more audio"; the process
      plays out whatever is buffered and exits. `output_device_name` as
      above.

  record <out_pcm_path> <samplerate> <blocksize> <max_seconds>
         <silence_seconds> <start_timeout_seconds> <cue_freq_hz>
         <cue_seconds> <cue_volume> <cue_lead_silence>
         [device_name] [archive_pcm_path] [output_device_name]
      Record from the default input device, gated by Silero VAD: stop
      after `silence_seconds` of quiet following detected speech, or after
      `max_seconds` of speech, whichever comes first. If nobody speaks
      within `start_timeout_seconds` (measured from stream open, i.e. from
      the `phase=stream-open` line below), exit with code TIMEOUT_EXIT_CODE
      and write nothing.

      `device_name` (optional, defaults to "" = current default-device
      behaviour): a substring matched against input device names via
      `sd.query_devices(name, kind="input")` -- `kind="input"` so a device
      that exposes both an input and an output entry under the same name
      (e.g. a USB headset) resolves unambiguously to its input side. No
      fallback: zero or multiple matches raises (sounddevice's own
      ValueError, surfaced as this process's failure) rather than silently
      picking the default device.

      When `device_name` is set AND that device's native sample rate
      (`default_samplerate`) is above `samplerate`, the input stream is
      opened at the device's native rate instead of `samplerate`. Each
      block read from the stream is resampled down to `samplerate` (via
      `scipy.signal.resample_poly`) before it reaches the VAD/recording
      pipeline below, so VAD threshold, blocksize-driven timing, the
      wall-clock cap, and the PCM handed back to the caller are all
      unchanged from the fixed-`samplerate` path -- resampling is an
      internal detail of this device-native-rate branch only.

      `archive_pcm_path` (optional, defaults to "" = no archive): when set
      AND native-rate capture is active (see above), the un-resampled,
      native-rate audio is also accumulated and written to this path on
      success (mono float32 PCM, same raw-file convention as
      `out_pcm_path`) -- the archival copy speak_server.py saves as a
      voice-clone reference. Ignored (nothing written) if native-rate
      capture is not active, since in that case `out_pcm_path` already *is*
      the native-rate capture.

      The "ear open" cue is played from INSIDE this process, right after
      the input stream opens and before the frame-read loop starts -- NOT
      by a separate `play` worker. Two processes each opening one
      direction-split half (input vs output) of the same Bluetooth device
      (e.g. AirPods) at the same time can lose the cue during HFP profile
      renegotiation, silently (see docs -- this is the fix for that). Pass
      `cue_freq_hz <= 0` to skip the cue entirely (used by tests and by any
      future caller that wants a silent start).

      Readiness ordering: torch/sounddevice/silero_vad are imported,
      `load_silero_vad()` runs, and `sd.InputStream` is opened BEFORE
      anything is signalled to the caller. Once the stream is open, the
      process prints `phase=stream-open` to stderr and flushes, then
      starts a background thread that reads and buffers frames from the
      stream immediately -- even while the cue plays -- so frames PortAudio
      delivers during the ~0.5s cue are never dropped to a full internal
      buffer. The cue itself plays synchronously in the main thread
      (`sd.play` + `sd.wait`, a separate short-lived OutputStream from the
      mic's InputStream, both owned by this one process). Once the cue
      finishes, the process prints `phase=cue-played` -- this, not
      `phase=stream-open`, is the readiness signal a caller should wait on
      before treating the user as "being listened to" and starting its own
      `start_timeout_seconds` countdown, so a user speaking during/right
      after the beep is still captured (their frames were already
      buffered by the reader thread) and the timeout clock does not start
      ticking until the beep -- the thing that tells the user to talk --
      has actually finished.

      Hard wall-clock cap: `start_timeout_seconds + max_seconds`, measured
      from stream open. This is the absolute upper bound on how long
      `record` can run, checked on every loop iteration regardless of VAD
      state. It exists because a VAD can be fooled by continuous non-speech
      sound (e.g. a TV in the room) into believing speech never stops, so
      "silence_seconds after speech" may never arrive — no heuristic is
      applied to detect that case; the wall-clock cap is the only guard.
      When the cap is hit, whatever audio has been captured so far is
      written to `out_pcm_path` and the process exits 0 (even if no VAD
      "end" event ever fired) — the caller always gets something rather
      than nothing.

      On success (including a cap-triggered stop), recorded audio (mono
      float32 PCM) is written to `out_pcm_path`. The process prints a
      single status line to stderr on each phase change
      (`phase=stream-open` once the mic is open and the reader thread has
      started, `phase=cue-played` once the ear-open cue has finished
      playing (or immediately, if no cue was requested), `phase=recording`
      on the first detected speech) so a caller that still sees the
      subprocess overrun its own timeout can report which phase it was in.

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

  SPEAK_AUDIO_DRY_RUN_NATIVE_RATE=<hz>
      With dry-run also set: makes `record`'s dry-run path behave as if a
      resolved input device's native rate were <hz> for the purposes of
      exercising the resample-and-archive path deterministically (dry-run
      never calls sd.query_devices, so there is no real device to report a
      native rate). Only takes effect when a `device_name` argument is also
      given. Used by tests only.
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import time

import numpy as np

from speak_tone import TONE_RATE, cue_pcm

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


def _dry_run_native_rate() -> float | None:
    if not _dry_run():
        return None
    value = os.environ.get("SPEAK_AUDIO_DRY_RUN_NATIVE_RATE")
    return float(value) if value else None


def _resolve_input_device(device_name: str) -> tuple[int, float]:
    """Resolve `device_name` (a substring) to an input device index and its
    native sample rate via sounddevice's own lookup. `kind="input"` so a
    device exposing both input and output entries under the same name (a USB
    headset, say) resolves to the input side rather than raising an
    ambiguous-name error against its own output twin. No fallback: zero or
    multiple matches is sounddevice's ValueError, raised as-is -- the caller
    (main()) lets it propagate to a non-zero exit with a stderr message,
    exactly like any other device failure in this process."""
    import sounddevice as sd

    info = sd.query_devices(device_name, kind="input")
    return info["index"], float(info["default_samplerate"])


def _resolve_output_device(device_name: str) -> int:
    """Resolve `device_name` (a substring) to an output device index via
    sounddevice's own lookup. `kind="output"` so a device exposing both
    input and output entries under the same name (a USB headset, say)
    resolves to the output side rather than raising an ambiguous-name error
    against its own input twin. No fallback: zero or multiple matches is
    sounddevice's ValueError, raised as-is -- callers let it propagate to a
    non-zero exit with a stderr message, exactly like any other device
    failure in this process."""
    import sounddevice as sd

    info = sd.query_devices(device_name, kind="output")
    return info["index"]


def _log_phase(phase: str) -> None:
    print(f"phase={phase}", file=sys.stderr, flush=True)


def _load_pcm(path: str) -> np.ndarray:
    return np.fromfile(path, dtype=np.float32)


def _save_pcm(path: str, audio: np.ndarray) -> None:
    audio.astype(np.float32).tofile(path)


def cmd_play(pcm_path: str, samplerate: int, device_name: str = "") -> None:
    audio = _load_pcm(pcm_path).reshape(-1, 1)
    if _dry_run():
        # Dry-run never touches sounddevice, so device_name is accepted but
        # not resolved -- nothing to validate without a real device to ask.
        return
    import sounddevice as sd

    device_index = _resolve_output_device(device_name) if device_name else None
    sd.play(audio, samplerate=samplerate, device=device_index)
    sd.wait()


# Chunk size (frames) read from stdin at a time for play-stream. Small enough
# to keep latency to first sound low, large enough not to spin the read loop.
STREAM_READ_FRAMES = 2400  # 0.1 s at 24 kHz
BYTES_PER_FRAME = 4  # float32


def cmd_play_stream(samplerate: int, device_name: str = "") -> None:
    stdin = sys.stdin.buffer
    if _dry_run():
        while stdin.read(STREAM_READ_FRAMES * BYTES_PER_FRAME):
            pass
        return

    import sounddevice as sd

    device_index = _resolve_output_device(device_name) if device_name else None
    with sd.OutputStream(samplerate=samplerate, channels=1, dtype="float32", device=device_index) as out:
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
    cue_freq_hz: float = 0.0,
    cue_seconds: float = 0.3,
    cue_volume: float = 0.4,
    cue_lead_silence: float = 0.2,
    device_name: str = "",
    archive_pcm_path: str = "",
    output_device_name: str = "",
) -> int:
    """Returns an exit code: 0 on success, TIMEOUT_EXIT_CODE if nobody spoke.

    `output_device_name`, if set, resolves (via `kind="output"`, same
    no-fallback rule as `device_name`'s input-side resolution) the device
    the in-process "ear open" cue plays through -- see SPEAK_OUTPUT_DEVICE
    in speak_server.py. Independent of `device_name` (the input device):
    Frank can point the mic and the cue at the same physical headset by
    setting both to the same substring, but they are resolved separately
    since one is `kind="input"` and the other `kind="output"`.

    `cue_freq_hz <= 0` means "play no cue" (used by callers that don't want
    one, and by tests). Otherwise the "ear open" cue is synthesized and
    played by THIS process, in-process, right after the input stream opens
    -- see the module docstring's `record` entry for why (two processes
    each opening one direction-split half of a Bluetooth device
    concurrently can silently lose the cue during HFP renegotiation).

    Hard wall-clock cap = start_timeout_seconds + max_seconds, measured from
    stream open. Checked every loop iteration regardless of VAD state, so a
    VAD that never emits "end" (e.g. fooled by continuous non-speech sound
    like a TV) cannot make this loop run longer than the cap. On a
    cap-triggered stop, whatever has been captured is written and the
    process exits 0 — never a bare timeout with nothing written.
    """
    cap_seconds = start_timeout_seconds + max_seconds

    # Native-rate capture is only meaningful when a specific device was
    # requested: the default-device path (device_name == "") has never had a
    # "native rate" concept exposed to this function and keeps behaving
    # exactly as before. Resolved once, before the dry-run branch, so a
    # bad/ambiguous device name fails loudly in dry-run tests too rather than
    # only in production.
    device_index: int | None = None
    native_rate: float | None = None
    if device_name:
        if _dry_run():
            native_rate = _dry_run_native_rate()  # None unless a test opts in
        else:
            device_index, native_rate = _resolve_input_device(device_name)
    capture_rate = native_rate if native_rate and native_rate > samplerate else samplerate
    archiving = bool(archive_pcm_path) and capture_rate != samplerate

    # Output device for the ear-open cue, resolved up front (same no-fallback
    # rule as the input device above) so a bad/ambiguous SPEAK_OUTPUT_DEVICE
    # fails loudly before the mic stream ever opens, not mid-cue. Only
    # resolved when a cue will actually be played AND we're not in dry-run
    # (which never touches sounddevice at all).
    cue_output_device_index: int | None = None
    if output_device_name and cue_freq_hz > 0 and not _dry_run():
        cue_output_device_index = _resolve_output_device(output_device_name)

    if _dry_run():
        if _dry_run_force_timeout():
            # Real timeouts always happen AFTER readiness (stream open, then
            # nobody speaks within start_timeout_seconds) -- readiness has
            # already been signalled and a caller may already be relying on
            # it (e.g. having played the "ear open" cue), so this dry-run
            # path must log it too rather than exiting silently before it.
            _log_phase("stream-open")
            _log_phase("cue-played")
            return TIMEOUT_EXIT_CODE
        _log_phase("stream-open")
        if archiving:
            print(f"archive-rate={int(capture_rate)}", file=sys.stderr, flush=True)
        _log_phase("cue-played")
        if _dry_run_endless_speech():
            # Simulate a VAD permanently fooled by continuous non-speech
            # noise: speech "starts" immediately and never ends. The only
            # thing that can stop this is the wall-clock cap.
            _log_phase("recording")
            rng = np.random.default_rng(0)
            audio = (0.01 * rng.standard_normal(int(samplerate * cap_seconds))).astype(np.float32)
            _save_pcm(out_pcm_path, audio)
            if archiving:
                native_audio = (0.01 * rng.standard_normal(int(capture_rate * cap_seconds))).astype(np.float32)
                _save_pcm(archive_pcm_path, native_audio)
            return 0
        seconds = float(os.environ.get("SPEAK_AUDIO_DRY_RUN_SECONDS", "1.0"))
        _log_phase("recording")
        rng = np.random.default_rng(0)
        audio = (0.01 * rng.standard_normal(int(samplerate * seconds))).astype(np.float32)
        _save_pcm(out_pcm_path, audio)
        if archiving:
            native_audio = (0.01 * rng.standard_normal(int(capture_rate * seconds))).astype(np.float32)
            _save_pcm(archive_pcm_path, native_audio)
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

    frames: list[np.ndarray] = []       # at `samplerate` (VAD/Whisper contract, unchanged)
    native_frames: list[np.ndarray] = []  # at `capture_rate`, only kept if archiving
    speaking = False

    # capture_blocksize keeps read cadence proportional to samplerate's
    # `blocksize` when reading at a higher capture_rate -- e.g. a device
    # running at 48kHz with samplerate=16000, blocksize=512 reads 1536-frame
    # (32ms) blocks, matching the VAD's 32ms/16kHz frame duration once
    # resampled down.
    capture_blocksize = blocksize if capture_rate == samplerate else round(blocksize * capture_rate / samplerate)

    # frame_q carries frames from the reader thread to this (main) thread.
    # reader_error carries an exception raised inside the reader thread, if
    # any, so a mid-recording device failure surfaces here instead of the
    # reader thread just going silent forever.
    frame_q: queue.Queue = queue.Queue()
    reader_error: list[BaseException] = []
    reader_stop = threading.Event()

    def read_frames(mic) -> None:
        try:
            while not reader_stop.is_set():
                frame, _ = mic.read(capture_blocksize)
                frame_q.put(frame[:, 0])
        except BaseException as e:  # surfaced in the main thread, never swallowed
            reader_error.append(e)
            frame_q.put(None)  # unblock a main-thread get() waiting on this queue

    with sd.InputStream(
        samplerate=capture_rate, channels=1, dtype="float32", blocksize=capture_blocksize, device=device_index,
    ) as mic:
        # Readiness signal #1: the stream is open. Start the reader thread
        # immediately, BEFORE playing the cue, so frames PortAudio delivers
        # during the ~0.5s cue are pulled off the stream and queued rather
        # than left to a fixed-size internal buffer that could overflow.
        t_open = time.time()
        _log_phase("stream-open")
        if archiving:
            # Tells the caller (speak_server.py, which never imports
            # sounddevice itself) the sample rate of the archive PCM it is
            # about to read back -- the worker is the only place that knows
            # the resolved device's native rate.
            print(f"archive-rate={int(capture_rate)}", file=sys.stderr, flush=True)
        reader_thread = threading.Thread(target=read_frames, args=(mic,), daemon=True)
        reader_thread.start()

        if cue_freq_hz > 0:
            sd.play(cue_pcm(cue_freq_hz, cue_seconds, cue_volume, cue_lead_silence), samplerate=TONE_RATE, device=cue_output_device_index)
            sd.wait()
        # Readiness signal #2: the cue (if any) has finished playing. This,
        # not `stream-open`, is what a caller should measure
        # start_timeout_seconds from -- frames spoken during/right after the
        # cue are already sitting in frame_q from the reader thread, so
        # nothing is lost even though the timeout clock starts only now.
        _log_phase("cue-played")

        while True:
            native_frame = frame_q.get()
            if native_frame is None:
                reader_stop.set()
                raise reader_error[0]
            if capture_rate == samplerate:
                frame = native_frame
            else:
                # Resample this block down to `samplerate` for VAD/Whisper;
                # the pipeline below never sees the native rate. The
                # archive-only copy keeps the un-resampled block untouched.
                import scipy.signal

                frame = scipy.signal.resample_poly(native_frame, samplerate, int(capture_rate)).astype(np.float32)
            frames.append(frame)
            if archiving:
                native_frames.append(native_frame)
            event = vad(torch.from_numpy(frame.copy()), return_seconds=False)
            now = time.time()
            elapsed = now - t_open
            if event and "start" in event and not speaking:
                speaking = True
                _log_phase("recording")
                # keep ~0.5 s of pre-roll before the detected start
                frames = frames[-int(0.5 * samplerate / blocksize):]
                if archiving:
                    native_frames = native_frames[-int(0.5 * samplerate / blocksize):]
            if event and "end" in event and speaking:
                break
            if not speaking and elapsed > start_timeout_seconds:
                reader_stop.set()
                return TIMEOUT_EXIT_CODE
            # Hard wall-clock cap, checked unconditionally every iteration
            # (not gated on VAD state) so continuous non-speech noise that
            # the VAD miscounts as speech cannot extend the recording past
            # this bound. Write whatever was captured and succeed.
            if elapsed > cap_seconds:
                break
        reader_stop.set()

    _save_pcm(out_pcm_path, np.concatenate(frames))
    if archiving:
        _save_pcm(archive_pcm_path, np.concatenate(native_frames) if native_frames else np.zeros(0, dtype=np.float32))
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
        # output_device_name is an optional trailing arg (default "" =
        # default output device) so old-style 2-arg invocations keep working.
        pcm_path, samplerate, *rest = args
        output_device_name = rest[0] if len(rest) > 0 else ""
        cmd_play(pcm_path, int(samplerate), output_device_name)
        return 0
    if command == "play-stream":
        samplerate, *rest = args
        output_device_name = rest[0] if len(rest) > 0 else ""
        cmd_play_stream(int(samplerate), output_device_name)
        return 0
    if command == "record":
        # device_name, archive_pcm_path, and output_device_name are optional
        # trailing args (all default to "" = disabled) so old-style 10-arg
        # invocations -- and every existing test that builds this argv by
        # hand -- keep working unchanged.
        (
            out_pcm_path, samplerate, blocksize, max_seconds, silence_seconds,
            start_timeout_seconds, cue_freq_hz, cue_seconds, cue_volume, cue_lead_silence,
            *rest,
        ) = args
        device_name = rest[0] if len(rest) > 0 else ""
        archive_pcm_path = rest[1] if len(rest) > 1 else ""
        output_device_name = rest[2] if len(rest) > 2 else ""
        return cmd_record(
            out_pcm_path,
            int(samplerate),
            int(blocksize),
            float(max_seconds),
            float(silence_seconds),
            float(start_timeout_seconds),
            float(cue_freq_hz),
            float(cue_seconds),
            float(cue_volume),
            float(cue_lead_silence),
            device_name,
            archive_pcm_path,
            output_device_name,
        )
    print(f"unknown command: {command!r}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
