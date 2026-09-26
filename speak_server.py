"""algolearn-speak: a local ear and voice for Claude Code.

MCP server (stdio) exposing four tools:

  speak(text)     -> synthesise with Kokoro (MLX) and play through the default output
  listen(...)     -> record from the default input until you stop talking (Silero VAD),
                     transcribe with Whisper (MLX), return the text
  converse(text)  -> speak, then listen
  status()        -> report whether speak/listen/converse are busy and how many
                     calls are queued behind the current one

speak/listen/converse are serialized through one process-wide lock (see
_SerializingLock below): a second caller queues and waits rather than being
rejected or talking over the first.

Everything runs on the Mac. No audio leaves the machine.
"""

from __future__ import annotations

import logging
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time

import anyio
import numpy as np
from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

import speak_audio_worker
from speak_tone import cue_pcm, tone

log = logging.getLogger("speak")

# ---------------------------------------------------------------- configuration

WHISPER_MODEL = os.environ.get("SPEAK_WHISPER_MODEL", "mlx-community/whisper-large-v3-turbo")
KOKORO_MODEL = os.environ.get("SPEAK_KOKORO_MODEL", "mlx-community/Kokoro-82M-bf16")
VOICE = os.environ.get("SPEAK_VOICE", "af_heart")
SPEED = float(os.environ.get("SPEAK_SPEED", "1.0"))
LANGUAGE = os.environ.get("SPEAK_LANGUAGE", "en")
ACK_TEXT = os.environ.get("SPEAK_ACK_TEXT", "Processing.")  # spoken after each successful listen

# Opt-in voice-clone archival. Both unset by default -> zero behaviour change.
INPUT_DEVICE = os.environ.get("SPEAK_INPUT_DEVICE", "")   # substring match, resolved in the worker
OUTPUT_DEVICE = os.environ.get("SPEAK_OUTPUT_DEVICE", "")  # substring match, resolved in the worker
SAVE_DIR = os.environ.get("SPEAK_SAVE_DIR", "")            # if set, every successful capture is archived here

MIC_RATE = 16_000          # Whisper and Silero both want 16 kHz mono
VAD_FRAME = 512            # Silero frame size at 16 kHz (32 ms)
TTS_RATE = 24_000          # Kokoro output rate
SENTENCE_SPLIT = r"(?<=[.!?])\s+"

# ---------------------------------------------------------------- engines

class Engines:
    """Holds the three models. Loaded once, in a background thread at startup."""

    def __init__(self) -> None:
        self.ready = threading.Event()
        self.error: BaseException | None = None
        self.kokoro = None

    def load(self) -> None:
        try:
            t0 = time.time()
            import mlx_whisper  # noqa: F401  (import cost is the bulk of the work)
            from mlx_audio.tts.utils import load_model

            self.kokoro = load_model(KOKORO_MODEL)
            # VAD is loaded fresh inside speak_audio_worker per listen() call —
            # the worker is a separate process that never shares state with
            # this one, so there's nothing to warm here for it. That load
            # (~1s) is not free, but it happens entirely before the "ear
            # open" beep: _record_pcm launches the worker and blocks until it
            # signals readiness (VAD loaded, mic stream open and buffering),
            # and only then does _listen_impl play the beep -- so the ~1s
            # cost is hidden behind the beep's own lead-in silence rather
            # than opening the mic after the user has already started
            # answering.
            # Warm the Kokoro pipeline (voice file, G2P, spaCy) so the first speak() is fast.
            for _ in self.kokoro.generate(text="Ready.", voice=VOICE, speed=SPEED, lang_code="a"):
                pass
            # Warm Whisper so the first listen() does not pay the model load.
            mlx_whisper.transcribe(np.zeros(MIC_RATE, dtype=np.float32), path_or_hf_repo=WHISPER_MODEL, language=LANGUAGE)
            log.info("engines ready in %.1fs (whisper=%s kokoro=%s voice=%s)", time.time() - t0, WHISPER_MODEL, KOKORO_MODEL, VOICE)
        except BaseException as e:  # surfaced to every tool call, never swallowed
            self.error = e
            log.exception("engine load failed")
        finally:
            self.ready.set()

    def wait(self) -> None:
        self.ready.wait()
        if self.error is not None:
            raise RuntimeError(f"speech engines failed to load: {self.error!r}") from self.error


engines = Engines()


class _SerializingLock:
    """Process-wide FIFO-ish queue for the three audio tools.

    `speak`, `listen`, and `converse` share one Mac microphone/speaker pair,
    so two calls (e.g. from two agents driving this same MCP server) must
    never run at once -- one waits for the other to finish, then proceeds
    (a queue, never a rejection). This wraps `threading.Lock` (the actual
    mutual exclusion) with a plain counter and the name of the tool
    currently holding it, so `status()` can report `busy`/`current_tool`/
    `waiting` without changing the underlying exclusion semantics: a bare
    Lock already blocks every other waiter until release, which is exactly
    the "wait then proceed" behaviour asked for -- this just makes that
    state observable.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state_lock = threading.Lock()  # protects the two fields below only
        self._current_tool: str | None = None
        self._waiting = 0

    def acquire(self, tool_name: str) -> None:
        with self._state_lock:
            self._waiting += 1
        try:
            self._lock.acquire()
        finally:
            with self._state_lock:
                self._waiting -= 1
        with self._state_lock:
            self._current_tool = tool_name

    def release(self) -> None:
        with self._state_lock:
            self._current_tool = None
        self._lock.release()

    def snapshot(self) -> tuple[bool, str | None, int]:
        """Returns (busy, current_tool, waiting) without blocking."""
        with self._state_lock:
            return self._current_tool is not None, self._current_tool, self._waiting


audio_lock = _SerializingLock()   # speak, listen, and converse are serialized through this -- see class docstring

# ---------------------------------------------------------------- helpers

def _tone(freq_hz: float, seconds: float, volume: float = 0.2) -> np.ndarray:
    # Thin wrapper kept for existing call sites and tests -- the actual
    # generator lives in speak_tone.py so speak_audio_worker.py's record
    # worker can synthesize the "ear open" cue in-process without importing
    # this module (see speak_tone.py's module docstring).
    return tone(freq_hz, seconds, volume)


def _cue(freq_hz: float, seconds: float = 0.12, volume: float = 0.2, lead_silence: float = 0.0) -> None:
    _play_pcm(cue_pcm(freq_hz, seconds, volume, lead_silence), TTS_RATE)


def _ack() -> None:
    """Rising two-note chime then a spoken word: the transcript is in hand and
    is being sent to the model. Distinct from the single ear-open/ear-closed cues."""
    _play_pcm(np.concatenate([_tone(660.0, 0.10), _tone(990.0, 0.14)]), TTS_RATE)
    _speak_impl(ACK_TEXT)


def _plain(text: str) -> str:
    """Strip the markdown that would otherwise be read aloud."""
    text = re.sub(r"```.*?```", " code block omitted ", text, flags=re.S)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"[`*_#>|]", "", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def _worker_command(*args: str) -> list[str]:
    return [sys.executable, "-m", "speak_audio_worker", *(str(a) for a in args)]


def _run_worker_once(*args: str, timeout: float) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            _worker_command(*args),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        # The worker itself enforces a wall-clock cap on every command that
        # can run long (see speak_audio_worker.py) and is expected to exit
        # on its own well inside `timeout`. Getting here means the worker
        # overran its own cap -- a bug in the worker, not an expected path
        # -- so name the phase it was stuck in (from its last stderr status
        # line) to make that bug diagnosable instead of a bare "timed out".
        phase = _last_phase(e.stderr)
        raise RuntimeError(
            f"audio worker {args[0]!r} exceeded its own budget and was killed "
            f"after {timeout:.0f}s (last phase: {phase}); this means the "
            f"worker's internal wall-clock cap did not fire -- see "
            f"speak_audio_worker.py's cmd_record cap logic"
        ) from e


def _last_phase(stderr: bytes | str | None) -> str:
    if not stderr:
        return "unknown (worker produced no stderr before being killed)"
    text = stderr.decode("utf-8", "replace") if isinstance(stderr, bytes) else stderr
    phases = [line.split("=", 1)[1] for line in text.splitlines() if line.startswith("phase=")]
    return phases[-1] if phases else "unknown (no phase status line seen)"


def _last_archive_rate(stderr_lines: list[str]) -> int:
    """Reads the worker's `archive-rate=<hz>` stderr line (see
    speak_audio_worker.py's cmd_record) -- the worker is the only place that
    resolves the input device's native rate, since this process never
    imports sounddevice. Raises if archiving audio was returned but no rate
    line was ever seen: that would mean stale/incoherent state between this
    process and the worker, and saving a WAV with a guessed rate is worse
    than failing loudly."""
    for line in stderr_lines:
        line = line.strip()
        if line.startswith("archive-rate="):
            return int(line.split("=", 1)[1])
    raise RuntimeError("worker produced archive audio but never reported archive-rate=<hz> on stderr")


def _stderr_tail(proc: subprocess.CompletedProcess, lines: int = 20) -> str:
    tail = "\n".join(proc.stderr.strip().splitlines()[-lines:])
    return tail or "(worker produced no stderr)"


def _run_worker(*args: str, timeout: float) -> subprocess.CompletedProcess:
    """Run the audio helper subprocess and return on success. Every actual
    PortAudio open happens in that short-lived process (see
    speak_audio_worker.py / docs/AUDIO_LIFECYCLE_AUDIT.md) — a fresh process
    never inherits a stale device snapshot, so retrying here just means
    launching a second fresh process, not re-touching bad state in this one.
    One immediate retry on a non-timeout-exit-code failure; a readable
    ToolError-friendly RuntimeError on the second failure."""
    proc = _run_worker_once(*args, timeout=timeout)
    if proc.returncode == 0 or proc.returncode == speak_audio_worker.TIMEOUT_EXIT_CODE:
        return proc
    log.warning("audio worker %r failed (exit %d); retrying once:\n%s", args[0], proc.returncode, _stderr_tail(proc))
    proc = _run_worker_once(*args, timeout=timeout)
    if proc.returncode == 0 or proc.returncode == speak_audio_worker.TIMEOUT_EXIT_CODE:
        return proc
    raise RuntimeError(
        f"audio worker {args[0]!r} failed twice (exit {proc.returncode}):\n{_stderr_tail(proc)}"
    )


def _play_pcm(audio: np.ndarray, samplerate: int) -> None:
    with tempfile.NamedTemporaryFile(prefix="speak-play-", suffix=".pcm", delete=False) as f:
        path = f.name
    try:
        audio.astype(np.float32).tofile(path)
        # generous fixed budget: playback itself has no fixed duration bound
        # here since callers pass short cues and full TTS chunks alike.
        duration = len(audio) / samplerate
        _run_worker("play", path, str(samplerate), OUTPUT_DEVICE, timeout=duration + 30.0)
    finally:
        os.unlink(path)


def _play_pcm_stream(chunks: list[np.ndarray], samplerate: int, timeout: float) -> None:
    """Play chunks (already produced, kept in memory) through a play-stream
    worker fed over stdin, so the worker starts playing the first chunk
    without waiting for a temp file of the whole utterance. On failure,
    retry the whole utterance against a fresh worker (a fresh process is the
    fix for a stale PortAudio snapshot; replaying the same bytes is cheap).

    Uses Popen.communicate(input=...) rather than writing to proc.stdin
    directly: writing our own stdin while the worker's stdout/stderr pipes
    can fill is a classic subprocess deadlock (each side blocked on the
    other's pipe); communicate() feeds stdin and drains stdout/stderr
    concurrently on our behalf.
    """
    payload = b"".join(chunk.astype(np.float32).tobytes() for chunk in chunks)
    for attempt in (1, 2):
        proc = subprocess.Popen(
            _worker_command("play-stream", str(samplerate), OUTPUT_DEVICE),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            _, stderr = proc.communicate(input=payload, timeout=timeout)
            returncode = proc.returncode
        except subprocess.TimeoutExpired:
            proc.kill()
            _, stderr = proc.communicate()
            returncode = -1
        if returncode == 0:
            return
        stderr_text = stderr.decode("utf-8", "replace") if stderr else ""
        tail = "\n".join(stderr_text.strip().splitlines()[-20:]) or "(worker produced no stderr)"
        if attempt == 1:
            log.warning("audio worker 'play-stream' failed (exit %s); retrying once:\n%s", returncode, tail)
            continue
        raise RuntimeError(f"audio worker 'play-stream' failed twice (exit {returncode}):\n{tail}")


READY_PHASE = "cue-played"  # mic open AND the ear-open cue (if any) has finished playing
READY_WAIT_SECONDS = 10.0  # bounded wait for the record worker's readiness line


class _RecordHandle:
    """A launched-but-not-yet-awaited record worker: the mic is already open
    and buffering by the time this is returned to the caller (readiness was
    already confirmed). Call `_finish_record_worker` to play out the rest of
    the recording and collect its result."""

    def __init__(self, proc: subprocess.Popen, path: str, stderr_lines: list[str], stderr_thread: threading.Thread) -> None:
        self.proc = proc
        self.path = path
        self.stderr_lines = stderr_lines
        self.stderr_thread = stderr_thread


def _drain_stderr(pipe, lines: list[str], ready_event: threading.Event) -> None:
    """Runs in a background thread for the lifetime of the record worker.
    Reads stderr line by line (never blocking the main thread on it, so a
    worker that writes a lot to stderr can't deadlock us on a full pipe),
    appending each line and setting `ready_event` the moment the readiness
    phase line appears. Keeps draining after readiness so `_last_phase` can
    still see later phase transitions (e.g. `phase=recording`) if the
    worker later overruns its own wall-clock cap."""
    try:
        for line in iter(pipe.readline, ""):
            lines.append(line)
            if line.strip() == f"phase={READY_PHASE}":
                ready_event.set()
    finally:
        pipe.close()


def _start_record_worker(
    path: str, samplerate: int, blocksize: int, max_seconds: float, silence_seconds: float, start_timeout_seconds: float,
    cue_freq_hz: float = 0.0, cue_seconds: float = 0.3, cue_volume: float = 0.4, cue_lead_silence: float = 0.2,
    device_name: str = "", archive_path: str = "", output_device_name: str = "",
) -> _RecordHandle:
    """Launch the record worker and block only until it signals readiness
    (mic open, and the ear-open cue -- if `cue_freq_hz > 0` -- already
    played) -- NOT until it finishes recording. See speak_audio_worker.py's
    cmd_record docstring for the readiness contract.

    The cue is played by the worker itself, in the same process that owns
    the input stream, so passing `cue_freq_hz > 0` here is what makes the
    "ear open" cue play at all; a caller that wants no cue passes
    `cue_freq_hz <= 0` (the default) and readiness fires as soon as the
    stream opens (the worker logs `phase=cue-played` immediately in that
    case -- see cmd_record).

    No silent fallback: if readiness never arrives within
    READY_WAIT_SECONDS, the worker is killed and a clear RuntimeError is
    raised. One retry (a fresh process) is allowed if the worker exits
    before ever reaching readiness -- a fresh process is the documented fix
    for a stale PortAudio device snapshot (see
    docs/AUDIO_LIFECYCLE_AUDIT.md), so this mirrors `_run_worker`'s
    retry-once policy rather than adding new fallback behaviour.
    """
    for attempt in (1, 2):
        proc = subprocess.Popen(
            _worker_command(
                "record", path, str(samplerate), str(blocksize),
                str(max_seconds), str(silence_seconds), str(start_timeout_seconds),
                str(cue_freq_hz), str(cue_seconds), str(cue_volume), str(cue_lead_silence),
                device_name, archive_path, output_device_name,
            ),
            # No stdout pipe: `record` never writes to stdout (see
            # speak_audio_worker.py), so there's nothing to drain and no
            # deadlock risk in leaving it inherited. stderr is piped and
            # owned exclusively by the drain thread below -- never touched
            # via Popen.communicate(), which would race the thread's reads
            # against its own and raise "Bad file descriptor" on double-close.
            stderr=subprocess.PIPE,
            text=True,
        )
        stderr_lines: list[str] = []
        ready_event = threading.Event()
        stderr_thread = threading.Thread(
            target=_drain_stderr, args=(proc.stderr, stderr_lines, ready_event), daemon=True,
        )
        stderr_thread.start()

        # Poll in small increments rather than one blocking wait(timeout=...)
        # so a worker that exits early (fast failure) is noticed and retried
        # right away instead of waiting out the full READY_WAIT_SECONDS.
        deadline = time.time() + READY_WAIT_SECONDS
        while not ready_event.is_set() and proc.poll() is None and time.time() < deadline:
            ready_event.wait(timeout=0.05)
        if ready_event.is_set():
            return _RecordHandle(proc, path, stderr_lines, stderr_thread)

        # Not ready in time: either the worker exited early (fast failure,
        # worth one retry with a fresh process) or it is hung past
        # READY_WAIT_SECONDS (not worth retrying -- retrying a hang just
        # doubles the wait for the same outcome).
        exited_early = proc.poll() is not None
        if exited_early and attempt == 1:
            log.warning(
                "record worker exited before signalling readiness (exit %s); retrying once:\n%s",
                proc.returncode, "".join(stderr_lines[-20:]) or "(worker produced no stderr)",
            )
            stderr_thread.join(timeout=5.0)
            continue

        proc.kill()
        proc.wait()
        stderr_thread.join(timeout=5.0)
        tail = "".join(stderr_lines[-20:]) or "(worker produced no stderr)"
        if exited_early:
            raise RuntimeError(f"record worker failed twice before signalling readiness (exit {proc.returncode}):\n{tail}")
        raise RuntimeError(
            f"record worker did not signal readiness (phase={READY_PHASE}) within "
            f"{READY_WAIT_SECONDS:.0f}s; killed it rather than treat a mic that might "
            f"not be listening as ready. Last stderr:\n{tail}"
        )
    raise AssertionError("unreachable")  # pragma: no cover


def _finish_record_worker(handle: _RecordHandle, cap_seconds: float, start_timeout_seconds: float) -> np.ndarray:
    """Wait for a record worker (already confirmed ready by
    `_start_record_worker`) to finish, and return its captured audio.

    Timeout budget: the worker's own wall-clock cap is
    start_timeout_seconds + max_seconds measured from stream open (already
    passed by the time this is called), plus a small fixed margin for
    process start/exit overhead -- a fixed margin, not a multiple, so a
    slow VAD model load can't silently double the wait. Readiness has
    already been confirmed, so this wait is just "cap_seconds forward from
    here, plus margin", not cap_seconds plus the readiness wait again.
    """
    timeout = cap_seconds + 15.0
    try:
        handle.proc.wait(timeout=timeout)
        returncode = handle.proc.returncode
    except subprocess.TimeoutExpired:
        handle.proc.kill()
        handle.proc.wait()
        returncode = None
    # The drain thread's own `for line in iter(pipe.readline, "")` returns
    # (closing the pipe) once the process exits and stderr hits EOF, so this
    # join is bounded by the wait/kill above, not an independent hang risk.
    handle.stderr_thread.join(timeout=5.0)

    if returncode is None:
        phase = _last_phase("".join(handle.stderr_lines))
        raise RuntimeError(
            f"audio worker 'record' exceeded its own budget and was killed after "
            f"{timeout:.0f}s (last phase: {phase}); this means the worker's internal "
            f"wall-clock cap did not fire -- see speak_audio_worker.py's cmd_record cap logic"
        )
    if returncode == speak_audio_worker.TIMEOUT_EXIT_CODE:
        raise TimeoutError(f"no speech detected within {start_timeout_seconds:.0f}s")
    if returncode != 0:
        tail = "".join(handle.stderr_lines[-20:]) or "(worker produced no stderr)"
        raise RuntimeError(f"audio worker 'record' failed (exit {returncode}) after signalling readiness:\n{tail}")
    return np.fromfile(handle.path, dtype=np.float32)


def _record_pcm(
    samplerate: int, blocksize: int, max_seconds: float, silence_seconds: float, start_timeout_seconds: float,
    cue_freq_hz: float = 0.0, cue_seconds: float = 0.3, cue_volume: float = 0.4, cue_lead_silence: float = 0.2,
    device_name: str = "", output_device_name: str = "",
) -> np.ndarray:
    """Record via the worker subprocess. If `cue_freq_hz > 0`, the worker
    plays the "ear open" cue itself, in-process, right after the mic opens
    and before it starts reading frames for real -- see
    speak_audio_worker.py's cmd_record docstring for why the cue must be
    played by the same process that owns the input stream (playing it via a
    separate `play` subprocess can silently lose the cue on a Bluetooth
    device mid-HFP-renegotiation).

    `device_name`, if non-empty, is resolved in the worker (this process
    never touches sounddevice) -- see speak_audio_worker.py's
    `_resolve_input_device`. Always returns the 16kHz stream -- unchanged
    contract regardless of `device_name`. Callers that also want the
    native-rate archive copy (SPEAK_SAVE_DIR) use `_record_pcm_with_archive`
    instead."""
    audio, _archive_audio, _archive_rate = _record_pcm_with_archive(
        samplerate, blocksize, max_seconds, silence_seconds, start_timeout_seconds,
        cue_freq_hz, cue_seconds, cue_volume, cue_lead_silence,
        device_name, want_archive=False, output_device_name=output_device_name,
    )
    return audio


def _record_pcm_with_archive(
    samplerate: int, blocksize: int, max_seconds: float, silence_seconds: float, start_timeout_seconds: float,
    cue_freq_hz: float = 0.0, cue_seconds: float = 0.3, cue_volume: float = 0.4, cue_lead_silence: float = 0.2,
    device_name: str = "", want_archive: bool = False, output_device_name: str = "",
) -> tuple[np.ndarray, np.ndarray | None, int]:
    """Same recording as `_record_pcm`, plus (when `want_archive` is True and
    `device_name` resolves to a native rate above `samplerate`) the
    un-resampled native-rate copy the worker captured alongside it -- see
    speak_audio_worker.py's cmd_record `archive_pcm_path` argument.

    Returns `(audio, archive_audio, archive_rate)`: `audio` is always the
    16kHz stream (unchanged contract). `archive_audio`/`archive_rate` are
    non-None only when `want_archive` is True AND the worker actually
    captured at a native rate above `samplerate` -- otherwise the archive
    would be identical to `audio` at `samplerate`, which the caller can use
    directly instead."""
    with tempfile.NamedTemporaryFile(prefix="speak-record-", suffix=".pcm", delete=False) as f:
        path = f.name
    os.unlink(path)  # the worker creates it; we just need a unique name
    archive_path = ""
    if want_archive and device_name:
        with tempfile.NamedTemporaryFile(prefix="speak-record-archive-", suffix=".pcm", delete=False) as f:
            archive_path = f.name
        os.unlink(archive_path)
    try:
        handle = _start_record_worker(
            path, samplerate, blocksize, max_seconds, silence_seconds, start_timeout_seconds,
            cue_freq_hz, cue_seconds, cue_volume, cue_lead_silence,
            device_name, archive_path, output_device_name,
        )
        cap_seconds = start_timeout_seconds + max_seconds
        audio = _finish_record_worker(handle, cap_seconds, start_timeout_seconds)
        archive_audio = None
        archive_rate = samplerate
        if archive_path and os.path.exists(archive_path):
            archive_audio = np.fromfile(archive_path, dtype=np.float32)
            if archive_audio.size == 0:
                archive_audio = None
            else:
                archive_rate = _last_archive_rate(handle.stderr_lines)
        return audio, archive_audio, archive_rate
    finally:
        if os.path.exists(path):
            os.unlink(path)
        if archive_path and os.path.exists(archive_path):
            os.unlink(archive_path)


def _speak_impl(text: str) -> float:
    engines.wait()
    text = _plain(text)
    if not text:
        raise ValueError("speak() called with empty text")

    chunks: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=4)
    failure: list[BaseException] = []

    def produce() -> None:
        try:
            for result in engines.kokoro.generate(text=text, voice=VOICE, speed=SPEED, lang_code="a", split_pattern=SENTENCE_SPLIT):
                chunks.put(np.asarray(result.audio, dtype=np.float32).reshape(-1, 1))
        except BaseException as e:
            failure.append(e)
        finally:
            chunks.put(None)

    threading.Thread(target=produce, name="kokoro", daemon=True).start()
    samples = 0
    played: list[np.ndarray] = []
    while (chunk := chunks.get()) is not None:
        played.append(chunk)
        samples += len(chunk)
    if failure:
        raise RuntimeError(f"Kokoro synthesis failed: {failure[0]!r}") from failure[0]
    if played:
        # A generous fixed budget for the worker's whole life: startup + the
        # audio's own duration + margin for the retry path.
        duration = samples / TTS_RATE
        _play_pcm_stream(played, TTS_RATE, timeout=duration + 30.0)
    return samples / TTS_RATE


_save_counter_lock = threading.Lock()
_save_counter = 0


def _check_save_dir(path: str) -> None:
    """No fallback: SPEAK_SAVE_DIR must already exist and be writable. Never
    created here -- an agent or a fat-fingered path silently spawning a new
    directory on disk is worse than a clear startup failure."""
    if not os.path.isdir(path):
        raise RuntimeError(f"SPEAK_SAVE_DIR={path!r} does not exist or is not a directory; create it first (it is never created automatically)")
    if not os.access(path, os.W_OK):
        raise RuntimeError(f"SPEAK_SAVE_DIR={path!r} is not writable")


def _save_capture(audio: np.ndarray, samplerate: int, text: str) -> None:
    """Write `<SAVE_DIR>/<UTC timestamp>_<n>.wav` (mono int16 at `samplerate`)
    plus a sibling `.txt` with the Whisper transcript, for use as voice-clone
    reference material. Only called for captures that already produced a
    non-empty transcript (see _listen_impl) -- silence/noise clips are never
    archived. `_check_save_dir` has already validated SAVE_DIR at startup;
    this still fails loudly (no try/except) if the write itself fails, e.g.
    the directory was removed after startup."""
    global _save_counter
    with _save_counter_lock:
        _save_counter += 1
        n = _save_counter
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    base = os.path.join(SAVE_DIR, f"{stamp}_{n}")
    pcm16 = np.clip(audio, -1.0, 1.0)
    pcm16 = (pcm16 * 32767.0).astype(np.int16)
    _write_wav(f"{base}.wav", pcm16, samplerate)
    with open(f"{base}.txt", "w", encoding="utf-8") as f:
        f.write(text)
    log.info("saved capture to %s.wav (%.1fs at %dHz)", base, len(audio) / samplerate, samplerate)


def _write_wav(path: str, pcm16: np.ndarray, samplerate: int) -> None:
    """Minimal mono 16-bit PCM WAV writer -- no extra dependency for
    something this small (44-byte header, then raw samples)."""
    import struct

    data = pcm16.tobytes()
    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + len(data)))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<IHHIIHH", 16, 1, 1, samplerate, samplerate * 2, 2, 16))
        f.write(b"data")
        f.write(struct.pack("<I", len(data)))
        f.write(data)


def _listen_impl(max_seconds: float, silence_seconds: float, start_timeout_seconds: float) -> str:
    engines.wait()
    import mlx_whisper

    # The "ear open" cue is played by the record worker itself, from inside
    # the same process that holds the input stream open (see
    # speak_audio_worker.py's cmd_record and _record_pcm's docstring) --
    # NOT by a separate `play` subprocess. On a Bluetooth device (e.g.
    # AirPods) that splits into direction-specific CoreAudio entries, two
    # processes each opening one half of the device at the same time can
    # silently lose the cue during HFP profile renegotiation. Playing it
    # in-process also means the mic is already buffering (via a background
    # reader thread, started before the cue plays) for the cue's own
    # duration, so speech spoken during/right after the beep is never lost.
    # start_timeout_seconds is enforced by the worker from the moment the
    # cue finishes playing (`phase=cue-played`), not from stream-open.
    try:
        # ear open: a short gap so it doesn't blend into the tail of speak(), then a longer, louder beep
        audio, archive_audio, archive_rate = _record_pcm_with_archive(
            MIC_RATE, VAD_FRAME, max_seconds, silence_seconds, start_timeout_seconds,
            cue_freq_hz=880.0, cue_seconds=0.3, cue_volume=0.4, cue_lead_silence=0.2,
            device_name=INPUT_DEVICE, want_archive=bool(SAVE_DIR), output_device_name=OUTPUT_DEVICE,
        )
    except TimeoutError:
        _cue(330.0)
        raise
    _cue(440.0)  # ear closed

    t0 = time.time()
    text = mlx_whisper.transcribe(audio, path_or_hf_repo=WHISPER_MODEL, language=LANGUAGE)["text"].strip()
    log.info("listen: %.1fs of audio transcribed in %.1fs: %r", len(audio) / MIC_RATE, time.time() - t0, text)
    if not text:
        raise RuntimeError(f"speech was detected ({len(audio) / MIC_RATE:.1f}s) but Whisper returned no text")
    if SAVE_DIR:
        # Prefer the native-rate archive (better clone reference material)
        # when one was actually captured; otherwise the 16kHz stream already
        # in hand is the only copy that exists.
        if archive_audio is not None:
            _save_capture(archive_audio, archive_rate, text)
        else:
            _save_capture(audio, MIC_RATE, text)
    _ack()
    return text

def _speak_sync(text: str) -> float:
    audio_lock.acquire("speak")
    try:
        return _speak_impl(text)
    finally:
        audio_lock.release()


def _listen_sync(max_seconds: float, silence_seconds: float, start_timeout_seconds: float) -> str:
    audio_lock.acquire("listen")
    try:
        return _listen_impl(max_seconds, silence_seconds, start_timeout_seconds)
    finally:
        audio_lock.release()


def _converse_sync(text: str, max_seconds: float, silence_seconds: float, start_timeout_seconds: float) -> str:
    audio_lock.acquire("converse")
    try:
        _speak_impl(text)
        return _listen_impl(max_seconds, silence_seconds, start_timeout_seconds)
    finally:
        audio_lock.release()

# ---------------------------------------------------------------- MCP surface

mcp = MCPServer("algolearn-speak")


async def _run(fn, *args):
    """Run audio work off the event loop. Expected failures (silence, empty text,
    engines not loaded) are re-raised as ToolError so the model sees the reason;
    the SDK masks any other exception as a generic crash."""
    try:
        return await anyio.to_thread.run_sync(fn, *args)
    except (TimeoutError, RuntimeError, ValueError) as e:
        raise ToolError(str(e)) from e
    except Exception as e:
        # Anything else (a PortAudioError mid-stream, a model load failure)
        # must reach the caller with its type and message, not as a bare
        # "Error executing tool" with nothing to act on.
        log.exception("%s failed", getattr(fn, "__name__", fn))
        raise ToolError(f"{type(e).__name__}: {e}") from e


@mcp.tool()
async def speak(text: str) -> str:
    """Say `text` out loud through the Mac's speakers and return once playback finishes.

    Write it as spoken prose: short sentences, no markdown, no code. Returns the spoken duration.

    Status: calls to speak/listen/converse are serialized through one process-wide
    lock -- there is only one microphone and one speaker. If another call is in
    progress, this one queues and waits for it to finish, then proceeds; it is
    never rejected. Use `status()` to see what is currently running and how many
    calls are waiting.
    """
    seconds = await _run(_speak_sync, text)
    return f"spoke for {seconds:.1f}s"


@mcp.tool()
async def listen(max_seconds: float = 120.0, silence_seconds: float = 1.2, start_timeout_seconds: float = 45.0) -> str:
    """Listen on the Mac's microphone until the user finishes talking and return what they said.

    A high beep means the ear is open, a low beep means it closed, and a rising chime plus
    the word "Processing" means the transcript was captured and is on its way to you. Recording ends after
    `silence_seconds` of quiet following speech, or at `max_seconds` of speech. Raises
    TimeoutError if nobody speaks within `start_timeout_seconds`.

    Status: calls to speak/listen/converse are serialized through one process-wide
    lock -- there is only one microphone and one speaker. If another call is in
    progress, this one queues and waits for it to finish, then proceeds; it is
    never rejected. Use `status()` to see what is currently running and how many
    calls are waiting.
    """
    return await _run(_listen_sync, max_seconds, silence_seconds, start_timeout_seconds)


@mcp.tool()
async def converse(text: str, max_seconds: float = 120.0, silence_seconds: float = 1.2, start_timeout_seconds: float = 45.0) -> str:
    """Say `text`, then listen for the reply. One spoken round trip; returns the user's words.

    Status: calls to speak/listen/converse are serialized through one process-wide
    lock -- there is only one microphone and one speaker. If another call is in
    progress, this one queues and waits for it to finish, then proceeds; it is
    never rejected. Use `status()` to see what is currently running and how many
    calls are waiting.
    """
    return await _run(_converse_sync, text, max_seconds, silence_seconds, start_timeout_seconds)


@mcp.tool()
async def status() -> dict:
    """Report whether speak/listen/converse are busy right now.

    Returns `{"busy": bool, "current_tool": str | None, "waiting": int}`.
    `current_tool` is the name of the tool currently holding the audio lock
    (or null if idle). `waiting` is how many other calls are queued behind
    it -- they will run in the order they arrived, one at a time, once the
    current call finishes. Never raises and never blocks on the audio lock.
    """
    busy, current_tool, waiting = audio_lock.snapshot()
    return {"busy": busy, "current_tool": current_tool, "waiting": waiting}


def main() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    if SAVE_DIR:
        _check_save_dir(SAVE_DIR)
    threading.Thread(target=engines.load, name="engine-load", daemon=True).start()
    mcp.run()


if __name__ == "__main__":
    main()
