"""algolearn-speak: a local ear and voice for Claude Code.

MCP server (stdio) exposing three tools:

  speak(text)     -> synthesise with Kokoro (MLX) and play through the default output
  listen(...)     -> record from the default input until you stop talking (Silero VAD),
                     transcribe with Whisper (MLX), return the text
  converse(text)  -> speak, then listen

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

log = logging.getLogger("speak")

# ---------------------------------------------------------------- configuration

WHISPER_MODEL = os.environ.get("SPEAK_WHISPER_MODEL", "mlx-community/whisper-large-v3-turbo")
KOKORO_MODEL = os.environ.get("SPEAK_KOKORO_MODEL", "mlx-community/Kokoro-82M-bf16")
VOICE = os.environ.get("SPEAK_VOICE", "af_heart")
SPEED = float(os.environ.get("SPEAK_SPEED", "1.0"))
LANGUAGE = os.environ.get("SPEAK_LANGUAGE", "en")
ACK_TEXT = os.environ.get("SPEAK_ACK_TEXT", "Processing.")  # spoken after each successful listen

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
            # it's cheap (~1s) and the worker is a separate process that never
            # shares state with this one. Nothing to warm here for it.
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
audio_lock = threading.Lock()   # speak and listen never overlap

# ---------------------------------------------------------------- helpers

def _tone(freq_hz: float, seconds: float, volume: float = 0.2) -> np.ndarray:
    t = np.arange(int(TTS_RATE * seconds)) / TTS_RATE
    env = np.minimum(1.0, np.minimum(t, seconds - t) / 0.01)  # 10 ms fade in/out
    return (volume * env * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def _cue(freq_hz: float, seconds: float = 0.12, volume: float = 0.2, lead_silence: float = 0.0) -> None:
    tone = _tone(freq_hz, seconds, volume)
    if lead_silence:
        tone = np.concatenate([np.zeros(int(TTS_RATE * lead_silence), dtype=np.float32), tone])
    _play_pcm(tone, TTS_RATE)


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
        _run_worker("play", path, str(samplerate), timeout=duration + 30.0)
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
            _worker_command("play-stream", str(samplerate)),
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


def _record_pcm(samplerate: int, blocksize: int, max_seconds: float, silence_seconds: float, start_timeout_seconds: float) -> np.ndarray:
    with tempfile.NamedTemporaryFile(prefix="speak-record-", suffix=".pcm", delete=False) as f:
        path = f.name
    os.unlink(path)  # the worker creates it; we just need a unique name
    try:
        # The worker's own wall-clock cap is start_timeout_seconds +
        # max_seconds, measured from stream open (see cmd_record in
        # speak_audio_worker.py) -- it always stops and writes whatever it
        # captured by then, even if a VAD is fooled into never reporting
        # silence (e.g. a TV in the room). This subprocess timeout is that
        # same budget plus a small fixed margin for process start/exit
        # overhead; a fixed margin, not a multiple, so a slow VAD model
        # load can't silently double the wait.
        cap_seconds = start_timeout_seconds + max_seconds
        timeout = cap_seconds + 15.0
        proc = _run_worker(
            "record", path, str(samplerate), str(blocksize),
            str(max_seconds), str(silence_seconds), str(start_timeout_seconds),
            timeout=timeout,
        )
        if proc.returncode == speak_audio_worker.TIMEOUT_EXIT_CODE:
            raise TimeoutError(f"no speech detected within {start_timeout_seconds:.0f}s")
        return np.fromfile(path, dtype=np.float32)
    finally:
        if os.path.exists(path):
            os.unlink(path)


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


def _listen_impl(max_seconds: float, silence_seconds: float, start_timeout_seconds: float) -> str:
    engines.wait()
    import mlx_whisper

    # ear open: a short gap so it doesn't blend into the tail of speak(), then a longer, louder beep
    _cue(880.0, seconds=0.3, volume=0.4, lead_silence=0.2)
    try:
        audio = _record_pcm(MIC_RATE, VAD_FRAME, max_seconds, silence_seconds, start_timeout_seconds)
    except TimeoutError:
        _cue(330.0)
        raise
    _cue(440.0)  # ear closed

    t0 = time.time()
    text = mlx_whisper.transcribe(audio, path_or_hf_repo=WHISPER_MODEL, language=LANGUAGE)["text"].strip()
    log.info("listen: %.1fs of audio transcribed in %.1fs: %r", len(audio) / MIC_RATE, time.time() - t0, text)
    if not text:
        raise RuntimeError(f"speech was detected ({len(audio) / MIC_RATE:.1f}s) but Whisper returned no text")
    _ack()
    return text

def _speak_sync(text: str) -> float:
    with audio_lock:
        return _speak_impl(text)


def _listen_sync(max_seconds: float, silence_seconds: float, start_timeout_seconds: float) -> str:
    with audio_lock:
        return _listen_impl(max_seconds, silence_seconds, start_timeout_seconds)


def _converse_sync(text: str, max_seconds: float, silence_seconds: float, start_timeout_seconds: float) -> str:
    with audio_lock:
        _speak_impl(text)
        return _listen_impl(max_seconds, silence_seconds, start_timeout_seconds)

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
    """
    return await _run(_listen_sync, max_seconds, silence_seconds, start_timeout_seconds)


@mcp.tool()
async def converse(text: str, max_seconds: float = 120.0, silence_seconds: float = 1.2, start_timeout_seconds: float = 45.0) -> str:
    """Say `text`, then listen for the reply. One spoken round trip; returns the user's words."""
    return await _run(_converse_sync, text, max_seconds, silence_seconds, start_timeout_seconds)


def main() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    threading.Thread(target=engines.load, name="engine-load", daemon=True).start()
    mcp.run()


if __name__ == "__main__":
    main()
