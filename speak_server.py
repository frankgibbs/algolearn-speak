"""algolearn-speak: a Telegram voice for Claude Code.

MCP server (stdio) exposing eight tools. All audio goes over a Telegram call
placed by the speak-telegram daemon; there is no local microphone or speaker
path (owner decision 2026-10-15: Telegram only).

  speak(text)     -> synthesise with Kokoro and play into the connected call
  listen(...)     -> capture the owner's reply on the call (the daemon runs Silero
                     VAD), transcribe with Whisper, return the text
  converse(text)  -> speak, then listen
  call(greeting)  -> Telegram voice call to the owner; speak/listen/converse need it
                     connected (docs/DESIGN_TELEGRAM_CALL.md)
  hang_up()       -> end that call
  status()        -> report whether speak/listen/converse are busy and how many
                     calls are queued behind the current one
  call_debug()    -> the speak-telegram daemon's recent call events and log tail
  set_call_debug()-> toggle the daemon's DEBUG logging at runtime

speak/listen/converse/call are serialized through one lock that is BOTH
process-wide and cross-process (see _SerializingLock below): a second
caller -- whether another thread in this process or a second
speak_server.py process started by a second Claude Code session --
queues and waits rather than being rejected or talking over the first
on the shared call.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import queue
import re
import sys
import threading
import time

import datetime

import anyio
import numpy as np
from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

import speak_telegram as tg
from speak_telegram_daemon import NOT_RUNNING, TelegramClient
from speak_tone import cue_pcm, tone

log = logging.getLogger("speak")

# ---------------------------------------------------------------- configuration

WHISPER_MODEL = os.environ.get("SPEAK_WHISPER_MODEL", "mlx-community/whisper-large-v3-turbo")
KOKORO_MODEL = os.environ.get("SPEAK_KOKORO_MODEL", "mlx-community/Kokoro-82M-bf16")
VOICE = os.environ.get("SPEAK_VOICE", "af_heart")
SPEED = float(os.environ.get("SPEAK_SPEED", "1.0"))
LANGUAGE = os.environ.get("SPEAK_LANGUAGE", "en")
ACK_TEXT = os.environ.get("SPEAK_ACK_TEXT", "Processing.")  # spoken after each successful listen

# Telegram calls (docs/DESIGN_TELEGRAM_CALL.md section 7). Validated in main().
QUIET_HOURS = os.environ.get("SPEAK_QUIET_HOURS", "22:00-07:00")  # no-call window, local time

MIC_RATE = 16_000          # the daemon returns the owner's audio as 16 kHz mono (Whisper's rate)
TTS_RATE = 24_000          # Kokoro output rate
SENTENCE_SPLIT = r"(?<=[.!?])\s+"

# ---------------------------------------------------------------- the call
#
# Every speak/listen/converse needs a call that the speak-telegram daemon
# reports as connected (docs/DESIGN_TELEGRAM_CALL.md section 6). `telegram` is
# a client of that daemon; this process never touches Telegram itself. With no
# connected call the tools raise: there is no local device to fall back to.

telegram = TelegramClient()
# True from an answered `call` until hang_up / a reported drop: while set, a call
# that is no longer connected raises "call ended ... call back" instead of
# "no call connected". Per process: the session that placed (or joined) the call.
_call_expected = False
_call_generation: int | None = None   # the daemon's generation of the call this session expects

AUDIO_NOTE = "\n(audio: telegram)"   # ends every speak/listen/converse result and an answered call()


def _call_ended_error(tool: str, detail: str) -> RuntimeError:
    """Clears the expectation: the drop is reported once, then the session calls back."""
    global _call_expected
    _call_expected = False
    return RuntimeError(f"Telegram call ended during {tool}; call back with the call tool ({detail})")


def _require_call(tool: str) -> None:
    """Raise unless a call is connected. Any session that talks on the call
    expects it from then on (the call is shared), so a later drop raises
    "call ended ... call back" in every such session. A call ended by hang_up
    -- from any session, on the owner's request -- is not a drop."""
    global _call_expected, _call_generation
    info = telegram.info()
    state = info["state"]
    if state == tg.CONNECTED:
        _call_expected, _call_generation = True, info["generation"]
        return
    if _call_expected and info["end_reason"] == tg.END_HUNG_UP and info["generation"] == _call_generation:
        _call_expected = False   # hung up on request: no call-back
    if _call_expected:
        raise _call_ended_error(tool, f"call state is {state}")
    raise RuntimeError(f"{tool} needs a connected Telegram call (call state is {state}); place one with the call tool")


def _guard_call(tool: str, fn):
    """A call that is no longer connected when `fn` finishes (or fails) makes
    the tool raise. A transcript captured before the drop is included in the
    message; a failure to read the call state is reported together with the
    original error, never instead of it."""
    try:
        result = fn()
    except Exception as e:
        try:
            state = telegram.state()
        except Exception as check:
            raise RuntimeError(f"{tool} failed ({e}) and the Telegram call state could not be read ({check})") from e
        if state != tg.CONNECTED:
            raise _call_ended_error(tool, str(e)) from e
        raise
    partial = f"transcript before the drop: {result!r}" if isinstance(result, str) else "no transcript"
    try:
        state = telegram.state()
    except Exception as check:
        raise RuntimeError(f"{tool} finished ({partial}) but the Telegram call state could not be read ({check})") from check
    if state != tg.CONNECTED:
        raise _call_ended_error(tool, f"call state is {state}; {partial}")
    return result


_QUIET_RE = re.compile(r"^(\d{2}):(\d{2})-(\d{2}):(\d{2})$")


def _parse_quiet_hours(window: str) -> tuple[int, int]:
    """`HH:MM-HH:MM` -> (start, end) in minutes after midnight. May wrap midnight."""
    m = _QUIET_RE.match(window)
    if not m:
        raise RuntimeError(f"SPEAK_QUIET_HOURS={window!r} is not HH:MM-HH:MM")
    sh, sm, eh, em = (int(g) for g in m.groups())
    if sh > 23 or eh > 23 or sm > 59 or em > 59:
        raise RuntimeError(f"SPEAK_QUIET_HOURS={window!r} has an out-of-range time")
    start, end = sh * 60 + sm, eh * 60 + em
    if start == end:
        raise RuntimeError(f"SPEAK_QUIET_HOURS={window!r} is an empty window")
    return start, end


def _in_quiet_hours(window: tuple[int, int], now: datetime.time) -> bool:
    start, end = window
    t = now.hour * 60 + now.minute
    if start < end:
        return start <= t < end
    return t >= start or t < end  # wraps midnight


# ---------------------------------------------------------------- engines

class Engines:
    """Holds Kokoro (TTS) and Whisper (STT). Loaded once, in a background thread at startup."""

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
            # Silero VAD runs in the speak-telegram daemon (speak_telegram.py), not here.
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

# ---------------------------------------------------------------- cross-process lock
#
# Two Claude Code sessions each launch their OWN speak_server.py process, so
# an in-process threading.Lock (below) only serializes calls within one
# process -- it cannot stop two processes from talking on the shared call at
# once. LOCK_DIR/LOCK_PATH are a well-known filesystem location every
# speak_server.py instance on this host agrees on, so flock() on LOCK_PATH is
# the actual cross-process mutex; SIDECAR_PATH is written by whoever holds it
# so a process that does NOT hold the lock can identify who does (pid, tool,
# since) for status() without needing IPC.

LOCK_DIR = os.path.expanduser("~/.algolearn-speak")
LOCK_PATH = os.path.join(LOCK_DIR, "audio.lock")
SIDECAR_PATH = os.path.join(LOCK_DIR, "audio.lock.json")


class _CrossProcessLock:
    """An advisory, cross-process exclusive lock via fcntl.flock on a
    well-known lockfile, plus a small JSON sidecar naming the current holder.

    `acquire()` opens (creating if needed) LOCK_PATH and blocks on
    `flock(LOCK_EX)` -- the kernel queues concurrent acquirers and wakes them
    one at a time on release, which is the "wait then proceed, never reject"
    behaviour asked for, now across processes rather than just threads.
    While held, the sidecar file holds `{"pid", "tool", "since"}` so
    `external_holder()` (called by a process that does NOT hold the lock) can
    report who does, without flock() itself exposing that -- flock only ever
    tells a non-holder "locked or not", never by whom.

    Not reentrant, and not meant to be: exactly one `acquire`/`release` pair
    per tool call, exactly like the in-process lock it sits beside.
    """

    def __init__(self, lock_path: str, sidecar_path: str) -> None:
        self._lock_path = lock_path
        self._sidecar_path = sidecar_path
        self._fd: int | None = None

    def acquire(self, tool_name: str) -> None:
        os.makedirs(os.path.dirname(self._lock_path), exist_ok=True)
        fd = os.open(self._lock_path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)  # blocks until free; queues fairly at the kernel level
        except BaseException:
            os.close(fd)
            raise
        self._fd = fd
        sidecar = {"pid": os.getpid(), "tool": tool_name, "since": time.time()}
        with open(self._sidecar_path, "w", encoding="utf-8") as f:
            json.dump(sidecar, f)

    def release(self) -> None:
        if self._fd is None:
            return
        try:
            if os.path.exists(self._sidecar_path):
                os.unlink(self._sidecar_path)
        finally:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
            self._fd = None

    def external_holder(self) -> dict | None:
        """Called only when THIS process does not currently hold the lock
        (see _SerializingLock.snapshot). Probes with a non-blocking
        LOCK_EX|LOCK_NB on a fresh fd: if that succeeds, nothing else holds
        it, so release immediately and return None. If it fails (EWOULDBLOCK/
        EAGAIN), some other process holds it -- read the sidecar it wrote to
        report who. The sidecar can legitimately be missing or mid-write
        (created just after the flock, removed just before releasing it), so
        a missing/unparseable sidecar under a held lock is reported as
        unknown rather than raised -- this is a best-effort status probe, not
        the correctness mechanism (flock is)."""
        if not os.path.exists(self._lock_path):
            return None
        fd = os.open(self._lock_path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(fd, fcntl.LOCK_UN)
            return None
        except OSError:
            try:
                with open(self._sidecar_path, encoding="utf-8") as f:
                    return json.load(f)
            except (OSError, ValueError):
                return {"pid": None, "tool": None, "since": None}
        finally:
            os.close(fd)


class _SerializingLock:
    """Process-wide (in-process) AND cross-process FIFO-ish queue for the
    three audio tools.

    `speak`, `listen`, `converse`, `call` and `hang_up` share one Telegram
    call, so two tool calls -- from two threads in this process, OR from two separate
    speak_server.py processes started by two Claude Code sessions -- must
    never run at once. One waits for the other to finish, then proceeds (a
    queue, never a rejection). The in-process `threading.Lock` handles the
    first case; `_CrossProcessLock` (flock on a well-known lockfile) handles
    the second. `acquire()` takes the in-process lock first (so this
    process's own waiter count/current-tool bookkeeping stays correct even
    while blocked on the cross-process lock), then the file lock; `release()`
    reverses the order.
    """

    def __init__(self, lock_path: str = LOCK_PATH, sidecar_path: str = SIDECAR_PATH) -> None:
        self._lock = threading.Lock()
        self._state_lock = threading.Lock()  # protects the fields below only
        self._current_tool: str | None = None
        self._waiting = 0
        self._cross = _CrossProcessLock(lock_path, sidecar_path)

    def acquire(self, tool_name: str) -> None:
        with self._state_lock:
            self._waiting += 1
        try:
            self._lock.acquire()
        finally:
            with self._state_lock:
                self._waiting -= 1
        try:
            self._cross.acquire(tool_name)
        except BaseException:
            self._lock.release()
            raise
        with self._state_lock:
            self._current_tool = tool_name

    def release(self) -> None:
        with self._state_lock:
            self._current_tool = None
        self._cross.release()
        self._lock.release()

    def snapshot(self) -> tuple[bool, str | None, int]:
        """Returns (busy, current_tool, waiting) without blocking. `busy`/
        `current_tool` reflect this process's own in-process state first; if
        this process is idle, checks whether another process holds the
        cross-process lock and reports that instead (current_tool becomes
        "<tool> (pid <n>, other process)"), so status() is honest across
        processes, not just within one."""
        with self._state_lock:
            current_tool, waiting = self._current_tool, self._waiting
        if current_tool is not None:
            return True, current_tool, waiting
        holder = self._cross.external_holder()
        if holder is None:
            return False, None, waiting
        tool = holder.get("tool")
        pid = holder.get("pid")
        label = f"{tool or 'unknown'} (pid {pid or '?'}, other process)"
        return True, label, waiting


audio_lock = _SerializingLock()   # every tool that uses the call is serialized through this -- see class docstring

# ---------------------------------------------------------------- helpers

def _tone(freq_hz: float, seconds: float, volume: float = 0.2) -> np.ndarray:
    return tone(freq_hz, seconds, volume)


def _play_pcm(audio: np.ndarray, samplerate: int) -> None:
    """Into the call. The daemon's pump paces it; play returns once every frame has gone out."""
    telegram.play(audio, samplerate)


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


def _record(max_seconds: float, silence_seconds: float, start_timeout_seconds: float) -> np.ndarray:
    """The owner's reply on the call: the daemon plays the ear-open cue into the
    call, runs Silero VAD on the owner's audio, and returns 16 kHz float32."""
    audio = telegram.record(max_seconds, silence_seconds, start_timeout_seconds,
                            cue_pcm(880.0, 0.5, 0.5, 0.2), TTS_RATE)
    if audio is tg.TIMEOUT:
        raise TimeoutError(f"no speech detected within {start_timeout_seconds:.0f}s")
    return audio


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
        _play_pcm(np.concatenate([chunk.reshape(-1) for chunk in played]), TTS_RATE)
    return samples / TTS_RATE


def _listen_impl(max_seconds: float, silence_seconds: float, start_timeout_seconds: float) -> str:
    engines.wait()
    import mlx_whisper

    try:
        audio = _record(max_seconds, silence_seconds, start_timeout_seconds)
    except TimeoutError:
        _cue(330.0)
        raise
    _after_capture(lambda: _cue(440.0))  # ear closed

    t0 = time.time()
    text = mlx_whisper.transcribe(audio, path_or_hf_repo=WHISPER_MODEL, language=LANGUAGE)["text"].strip()
    log.info("listen: %.1fs of audio transcribed in %.1fs: %r", len(audio) / MIC_RATE, time.time() - t0, text)
    if not text:
        raise RuntimeError(f"speech was detected ({len(audio) / MIC_RATE:.1f}s) but Whisper returned no text")
    _after_capture(_ack)
    return text


def _after_capture(play) -> None:
    """Cues played after the owner's words are captured. A call that has just ended (the owner said their piece and hung up) makes them
    fail; the words must still be transcribed and returned, so that failure is
    skipped here and _guard_call then raises "call ended" WITH the transcript.
    Any failure while the call is still connected raises as usual."""
    try:
        play()
    except RuntimeError:
        if telegram.state() != tg.CONNECTED:
            log.info("call ended right after the capture; keeping the words for the drop report")
            return
        raise

def _speak_sync(text: str) -> float:
    audio_lock.acquire("speak")
    try:
        _require_call("speak")
        return _guard_call("speak", lambda: _speak_impl(text))
    finally:
        audio_lock.release()


def _listen_sync(max_seconds: float, silence_seconds: float, start_timeout_seconds: float) -> str:
    audio_lock.acquire("listen")
    try:
        _require_call("listen")
        return _guard_call("listen", lambda: _listen_impl(max_seconds, silence_seconds, start_timeout_seconds))
    finally:
        audio_lock.release()


def _converse_sync(text: str, max_seconds: float, silence_seconds: float, start_timeout_seconds: float) -> str:
    audio_lock.acquire("converse")
    try:
        _require_call("converse")

        def round_trip() -> str:
            _speak_impl(text)
            return _listen_impl(max_seconds, silence_seconds, start_timeout_seconds)

        return _guard_call("converse", round_trip)
    finally:
        audio_lock.release()


def _synthesize(text: str) -> np.ndarray:
    """Kokoro audio for `text` at TTS_RATE, all at once (the call's opening greeting)."""
    engines.wait()
    text = _plain(text)
    if not text:
        raise ValueError("call() needs a non-empty greeting")
    return np.concatenate([np.asarray(r.audio, dtype=np.float32).reshape(-1)
                           for r in engines.kokoro.generate(text=text, voice=VOICE, speed=SPEED, lang_code="a",
                                                            split_pattern=SENTENCE_SPLIT)])


def _call_sync(greeting: str, override_quiet_hours: bool, max_seconds: float, silence_seconds: float,
               start_timeout_seconds: float) -> str:
    global _call_expected, _call_generation
    audio_lock.acquire("call")
    try:
        # Checked under the lock: a call queued behind a long converse is judged when it would dial.
        if not override_quiet_hours and _in_quiet_hours(_parse_quiet_hours(QUIET_HOURS), datetime.datetime.now().time()):
            return f"not called: quiet hours ({QUIET_HOURS})"
        info = telegram.info()
        if info["state"] == NOT_RUNNING:
            raise RuntimeError("the speak-telegram daemon is not running (see CLAUDE.md, \"Telegram calls\")")
        if info["state"] == tg.CONNECTED:
            _call_expected, _call_generation = True, info["generation"]
            return tg.ALREADY_CONNECTED
        # Last of the pre-dial checks, so quiet hours / daemon down / already connected keep
        # their outcomes. Measured on the synthesized audio, not guessed from word count.
        audio = _synthesize(greeting)
        seconds = len(audio) / TTS_RATE
        if seconds < tg.OPENER_MIN_SECONDS:
            raise ValueError(
                f"call() greeting is only {seconds:.1f}s when spoken; it must be at least {tg.OPENER_MIN_SECONDS:g}s. "
                "Nothing was dialed. The greeting is the whole opening: say who is calling, why, and the context, "
                "and end with a question. Lengthen it with the context of why you are calling. If this is a fresh "
                "conversation with nothing to report, open with something playful instead: a joke, something "
                "funny or flirty.")
        outcome = telegram.call(audio, TTS_RATE)
        _call_expected = outcome in (tg.ANSWERED, tg.ALREADY_CONNECTED)
        if _call_expected:
            _call_generation = telegram.info()["generation"]
        if outcome != tg.ANSWERED:
            return outcome
        # Answered: the greeting has played; listen for the owner's reply under the
        # same lock hold (as converse does) so no other session slips in between.
        # A listen timeout or a drop raises, exactly as in listen(); on a timeout
        # the call stays connected.
        _require_call("call")
        text = _guard_call("call", lambda: _listen_impl(max_seconds, silence_seconds, start_timeout_seconds))
        return f"{tg.ANSWERED}\n{text}{AUDIO_NOTE}"
    finally:
        audio_lock.release()


def _hang_up_sync() -> str:
    global _call_expected
    audio_lock.acquire("hang_up")
    try:
        if telegram.state() == NOT_RUNNING:
            return tg.NO_ACTIVE_CALL
        return telegram.hang_up()
    finally:
        _call_expected = False   # the owner asked to hang up: never a call-back, even if this raised
        audio_lock.release()


def _check_call_config() -> None:
    """Fail at startup, not on the first call."""
    _parse_quiet_hours(QUIET_HOURS)

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
        # Anything else (a daemon socket error, a model load failure)
        # must reach the caller with its type and message, not as a bare
        # "Error executing tool" with nothing to act on.
        log.exception("%s failed", getattr(fn, "__name__", fn))
        raise ToolError(f"{type(e).__name__}: {e}") from e


@mcp.tool()
async def speak(text: str) -> str:
    """Say `text` to the owner over the connected Telegram call and return once playback finishes.

    Write it as spoken prose: short sentences, no markdown, no code. Returns the spoken duration.

    Status: speak/listen/converse/call are serialized through one lock that is
    both process-wide and cross-process (a second Claude Code session running
    its own speak_server.py is included) -- there is only one Telegram call.
    If another tool call is in progress anywhere, this one queues and waits
    for it to finish, then proceeds; it is never rejected. Use `status()` to
    see what is currently running (including in another process) and how
    many calls are waiting.

    Needs a Telegram call placed by `call()` and still connected; there is no
    local microphone or speaker. With no connected call this raises "...
    needs a connected Telegram call"; place one with `call()`. The result
    ends with `(audio: telegram)`. If the call ends mid-way this raises
    "Telegram call ended during ..."; the rule of engagement is to call back
    with `call()`.
    """
    seconds = await _run(_speak_sync, text)
    return f"spoke for {seconds:.1f}s{AUDIO_NOTE}"


@mcp.tool()
async def listen(max_seconds: float = 120.0, silence_seconds: float = 2.0, start_timeout_seconds: float = 45.0) -> str:
    """Listen to the owner on the connected Telegram call until they finish talking and return what they said.

    A high beep means the ear is open, a low beep means it closed, and a rising chime plus
    the word "Processing" means the transcript was captured and is on its way to you. Recording ends after
    `silence_seconds` of quiet following speech, or at `max_seconds` of speech. Raises
    TimeoutError if nobody speaks within `start_timeout_seconds` (the call stays connected).

    Status: speak/listen/converse/call are serialized through one lock that is
    both process-wide and cross-process (a second Claude Code session running
    its own speak_server.py is included) -- there is only one Telegram call.
    If another tool call is in progress anywhere, this one queues and waits
    for it to finish, then proceeds; it is never rejected. Use `status()` to
    see what is currently running (including in another process) and how
    many calls are waiting.

    Needs a Telegram call placed by `call()` and still connected; there is no
    local microphone or speaker. With no connected call this raises "...
    needs a connected Telegram call"; place one with `call()`. The result
    ends with `(audio: telegram)`. If the call ends mid-way this raises
    "Telegram call ended during ..."; the rule of engagement is to call back
    with `call()`.
    """
    text = await _run(_listen_sync, max_seconds, silence_seconds, start_timeout_seconds)
    return text + AUDIO_NOTE


@mcp.tool()
async def converse(text: str, max_seconds: float = 120.0, silence_seconds: float = 2.0, start_timeout_seconds: float = 45.0) -> str:
    """Say `text` over the connected Telegram call, then listen for the reply. One spoken round trip; returns the owner's words.

    Status: speak/listen/converse/call are serialized through one lock that is
    both process-wide and cross-process (a second Claude Code session running
    its own speak_server.py is included) -- there is only one Telegram call.
    If another tool call is in progress anywhere, this one queues and waits
    for it to finish, then proceeds; it is never rejected. Use `status()` to
    see what is currently running (including in another process) and how
    many calls are waiting.

    Needs a Telegram call placed by `call()` and still connected; there is no
    local microphone or speaker. With no connected call this raises "...
    needs a connected Telegram call"; place one with `call()`. The result
    ends with `(audio: telegram)`. If the call ends mid-way this raises
    "Telegram call ended during ..."; the rule of engagement is to call back
    with `call()`.
    """
    reply = await _run(_converse_sync, text, max_seconds, silence_seconds, start_timeout_seconds)
    return reply + AUDIO_NOTE


@mcp.tool()
async def call(greeting: str, override_quiet_hours: bool = False, max_seconds: float = 120.0,
               silence_seconds: float = 2.0, start_timeout_seconds: float = 45.0) -> str:
    """Phone the owner on Telegram (from the algolearn.ai account), speak the greeting
    once they answer, then listen for their reply.

    `greeting` is your whole opening, spoken the moment they pick up: who is
    calling, why, and the context, ending with a question or invitation,
    because call() listens right after it (ear-open beep, then the same capture as `listen`; the listen
    parameters and defaults are converse's). It must be at least 6 s when
    spoken (the call needs a 6 s opener); a shorter one raises ValueError
    BEFORE dialing, so the phone never rings. Lengthen it with the context of
    why you are calling; with nothing to report on a fresh conversation, open
    with something playful (a joke, something funny or flirty). Example:
    "Hi Frank, it's Angie. The backtest finished a few minutes ago and the
    holdout came back clean, but there is one result I want your read on
    before I do anything else. Do you have a minute?". Works wherever their phone has signal,
    with the Mac locked or unlocked. Once answered, speak/listen/converse run
    over the call until it ends; without a connected call they raise. Returns one of:
      "answered\n<their words>\n(audio: telegram)"
                                              -- they picked up, heard the greeting and
                                                 replied; continue with listen or converse.
                                                 A listen timeout or a drop raises as in
                                                 listen (a timeout leaves the call connected)
      "already connected"                     -- a call is already up (greeting not
                                                 spoken); just talk
      "not answered (declined or no answer)"  -- what you have can wait; do not
                                                 call again for at least one hour
      "answered, then hung up during the greeting"
                                              -- they picked up but ended the call before
                                                 the greeting finished (often because they
                                                 heard nothing). You may place ONE new call
                                                 right away; if that one also ends this way
                                                 or is not answered, wait an hour.
      "hung up"                               -- another session's hang_up ended the call
                                                 during the greeting (the owner asked for
                                                 it); do not call again
      "not called: quiet hours (22:00-07:00)" -- nothing was dialed; no calls in
                                                 that window
      "answered but audio failed to connect: <error type/message>"
                                              -- the owner picked up but the media
                                                 connection failed (they hear beeping,
                                                 then the call drops). This is NOT
                                                 "not answered": do not wait an hour.
                                                 The call was torn down and the daemon
                                                 restarts itself (launchd) a few
                                                 seconds later with fresh state. Call
                                                 `call_debug()` to see why; once it
                                                 shows the daemon back up (small
                                                 uptime_seconds, call state "none"),
                                                 you may place ONE new call. Do not
                                                 loop.
    Quiet hours are a hard rule. Pass `override_quiet_hours=True` ONLY when the
    owner has said in this conversation that they are up and want calls.

    Rule of engagement: if speak/listen/converse raises "Telegram call ended
    during ...", the call dropped mid-conversation -- call back with this tool.
    Hang up with `hang_up()` only when the owner asks; they can also hang up
    themselves.
    """
    return await _run(_call_sync, greeting, override_quiet_hours, max_seconds, silence_seconds, start_timeout_seconds)


_MAX_DEBUG_LINES = 1000


def _call_debug_sync(lines: int) -> dict:
    if isinstance(lines, bool) or not isinstance(lines, int) or not 1 <= lines <= _MAX_DEBUG_LINES:
        raise ValueError(f"lines must be an integer from 1 to {_MAX_DEBUG_LINES}, got {lines!r}")
    return telegram.debug(lines)   # never takes the audio lock: answers during a ringing or connected call


def _set_call_debug_sync(enabled: bool) -> dict:
    if not isinstance(enabled, bool):
        raise ValueError(f"enabled must be a boolean, got {enabled!r}")
    return telegram.set_debug(enabled)


@mcp.tool()
async def call_debug(lines: int = 50) -> dict:
    """Diagnose Telegram calls in real time: what the speak-telegram daemon just did.

    Use it during a call, or right after `call()` returned "answered but audio
    failed to connect: ..." or an unexpected "not answered". Returns
    `{"daemon": "running", "pid", "uptime_seconds", "debug": bool, "call":
    {"state", "generation", "end_reason"}, "events": [...], "log": [...]}`:
    `events` are the last `lines` structured call events (call_id, ts, event
    such as requested/ringing/busy/audio_failed/ended, elapsed_s, error),
    `log` the last `lines` daemon log lines (set_call_debug(True) adds
    py-tgcalls/Telethon DEBUG output). A small uptime_seconds right after a
    failed call is the daemon's self-restart. When no daemon answers it
    returns `{"daemon": "daemon not running"}` and nothing else. Never blocks
    on the audio lock. `lines` is 1 to 1000.
    """
    return await _run(_call_debug_sync, lines)


@mcp.tool()
async def set_call_debug(enabled: bool) -> dict:
    """Turn the daemon's DEBUG logging (speak, py-tgcalls, Telethon) on or off at
    runtime, no restart. Turn it on before a call you expect to fail, read the
    result with `call_debug()`, turn it off after (it is noisy). Resets when the
    daemon restarts. Returns `{"daemon": "running", "debug": bool}` or
    `{"daemon": "daemon not running"}`. Never blocks on the audio lock.
    """
    return await _run(_set_call_debug_sync, enabled)


@mcp.tool()
async def hang_up() -> str:
    """End the Telegram call placed by `call()`. Use only when the owner asks you to.

    Returns "hung up" or "no active call".
    """
    return await _run(_hang_up_sync)


@mcp.tool()
async def status() -> dict:
    """Report whether speak/listen/converse/call are busy right now, in this
    process or in another one (e.g. a second Claude Code session's own
    speak_server.py), and the Telegram call state.

    Returns `{"busy": bool, "current_tool": str | None, "waiting": int, "call": str}`.
    `call` is the daemon's call state ("none", "ringing", "connected"),
    "daemon not running", or "error: <reason>" if it could not be read;
    speak/listen/converse work only while it is "connected".
    `current_tool` is the name of the tool currently holding the lock, or
    `"<tool> (pid <n>, other process)"` when a different process holds it,
    or null if idle. `waiting` counts only callers queued in THIS process
    (this process's own threads waiting on the in-process lock); it cannot
    see how many callers another process has queued. Never raises and never
    blocks.
    """
    busy, current_tool, waiting = audio_lock.snapshot()
    try:
        call_state = telegram.state()
    except Exception as e:  # reported in the result, never hidden: status must not raise
        call_state = f"error: {e}"
    return {"busy": busy, "current_tool": current_tool, "waiting": waiting, "call": call_state}


def main() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    _check_call_config()
    threading.Thread(target=engines.load, name="engine-load", daemon=True).start()
    mcp.run()


if __name__ == "__main__":
    main()
