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
import sys
import threading
import time

import anyio
import numpy as np
import sounddevice as sd
from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

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
        self.vad = None

    def load(self) -> None:
        try:
            t0 = time.time()
            import mlx_whisper  # noqa: F401  (import cost is the bulk of the work)
            from mlx_audio.tts.utils import load_model
            from silero_vad import load_silero_vad

            self.kokoro = load_model(KOKORO_MODEL)
            self.vad = load_silero_vad()
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
    sd.play(tone, samplerate=TTS_RATE)
    sd.wait()


def _ack() -> None:
    """Rising two-note chime then a spoken word: the transcript is in hand and
    is being sent to the model. Distinct from the single ear-open/ear-closed cues."""
    sd.play(np.concatenate([_tone(660.0, 0.10), _tone(990.0, 0.14)]), samplerate=TTS_RATE)
    sd.wait()
    _speak_impl(ACK_TEXT)


def _plain(text: str) -> str:
    """Strip the markdown that would otherwise be read aloud."""
    text = re.sub(r"```.*?```", " code block omitted ", text, flags=re.S)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"[`*_#>|]", "", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def _audio_stream(kind, **kw):
    """Open a sounddevice stream. A PortAudioError here means the device is
    busy or the device list is stale (default output/input changed since this
    process started — a headset connecting, another session holding the mic).
    Re-initialize PortAudio so the NEXT call sees the current devices, and
    raise a readable RuntimeError instead of the SDK's bare "Error executing
    tool" crash."""
    try:
        return kind(**kw)
    except sd.PortAudioError as first:
        # A first open after the process has sat idle often fails once
        # (-9986) and succeeds on an immediate retry; try once before giving up.
        log.warning("audio device error opening %s (%s); retrying once", kind.__name__, first)
        time.sleep(1.0)
    try:
        return kind(**kw)
    except sd.PortAudioError as e:
        # Re-initializing PortAudio in-process (sd._terminate/_initialize) was
        # tried on 2026-09-04 and crashed the process on the next InputStream
        # open (client saw "Connection closed" mid-listen). A stale device list
        # is only reliably cleared by a fresh process, and Claude Code respawns
        # a stdio MCP server that exits — so report, then exit after the error
        # has been sent; the next call lands on a fresh server.
        log.error("audio device error opening %s: %s — exiting so the client respawns a fresh server", kind.__name__, e)
        threading.Timer(1.5, lambda: os._exit(3)).start()
        raise RuntimeError(
            f"audio device error opening {kind.__name__}: {e}. "
            "Server restarting; retry the call in a few seconds."
        ) from e


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
    with _audio_stream(sd.OutputStream, samplerate=TTS_RATE, channels=1, dtype="float32") as out:
        while (chunk := chunks.get()) is not None:
            out.write(chunk)
            samples += len(chunk)
    if failure:
        raise RuntimeError(f"Kokoro synthesis failed: {failure[0]!r}") from failure[0]
    return samples / TTS_RATE


def _listen_impl(max_seconds: float, silence_seconds: float, start_timeout_seconds: float) -> str:
    engines.wait()
    import torch
    import mlx_whisper
    from silero_vad import VADIterator

    vad = VADIterator(engines.vad, threshold=0.5, sampling_rate=MIC_RATE, min_silence_duration_ms=int(silence_seconds * 1000), speech_pad_ms=300)
    frames: list[np.ndarray] = []
    speaking = False
    t_open = time.time()
    t_speech = None

    # ear open: a short gap so it doesn't blend into the tail of speak(), then a longer, louder beep
    _cue(880.0, seconds=0.3, volume=0.4, lead_silence=0.2)
    with _audio_stream(sd.InputStream, samplerate=MIC_RATE, channels=1, dtype="float32", blocksize=VAD_FRAME) as mic:
        while True:
            frame, _ = mic.read(VAD_FRAME)
            frame = frame[:, 0]
            frames.append(frame)
            event = vad(torch.from_numpy(frame.copy()), return_seconds=False)
            now = time.time()
            if event and "start" in event:
                speaking = True
                t_speech = now
                # keep ~0.5 s of pre-roll before the detected start
                frames = frames[-int(0.5 * MIC_RATE / VAD_FRAME):]
            if event and "end" in event and speaking:
                break
            if not speaking and now - t_open > start_timeout_seconds:
                _cue(330.0)
                raise TimeoutError(f"no speech detected within {start_timeout_seconds:.0f}s")
            if speaking and now - t_speech > max_seconds:
                log.info("listen: hit max_seconds=%.0f, transcribing what we have", max_seconds)
                break
    _cue(440.0)  # ear closed

    audio = np.concatenate(frames)
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
