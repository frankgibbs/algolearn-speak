"""The speak daemon (docs/DESIGN_REMOTE_DAEMON.md): one per LAN, on patrick, in a
container on GPU0.

`speak-telegram` (this module's `main`) owns the algolearn.ai Telegram login and
the call (speak_telegram.TelegramLink), Kokoro and Whisper (speak_engines), the
one audio lock, and all of the tool logic. It listens on TCP 0.0.0.0:8765. Every
Claude Code session on every LAN computer runs the thin MCP client
(speak_server.py), which sends one request per tool call over the framing in
speak_protocol.py. Only text crosses the wire; audio never leaves this process.

Tool ops (`speak`, `listen`, `converse`, `call`, `hang_up`) wait in a FIFO for the
audio lock, get `{"ok": true, "phase": "started"}` the moment they hold it, then
their final reply. Quick ops (`ping`, `status`, `debug`, `set_debug`) never
touch the lock, so they answer while a call is ringing or connected.

A second listener on HTTP 0.0.0.0:9000 replaces the retired whisper-service with
its API (owner decision 2026-10-16): `POST /` takes raw 16 kHz mono int16 PCM and
returns `{"text": ...}`, `GET /health` returns `{"status": "ok", "model", "device"}`.
It reuses the loaded Whisper engine, and every transcription waits in the same FIFO
audio lock as the tool ops (status shows it as `transcribe (<ip>)`), so inference
never overlaps call audio.

Single instance: the TCP bind is the guard (a second daemon gets EADDRINUSE),
and compose's fixed container_name refuses a second container.

A call that was answered but whose media connection failed makes the daemon
exit 1 right after the caller has its reply, so compose's restart policy
restarts it with fresh ntgcalls state.

A native crash (SIGSEGV, SIGABRT, ...) in ntgcalls, torch or ctranslate2 dumps every
Python thread's stack to stderr before the process dies with the signal (exit
139 for SIGSEGV), so `docker logs speak-telegram` shows what each thread was
doing: the telegram-loop thread (pump, record, play) or a request thread inside
Engines.synthesize/transcribe or the Silero VAD.

No fallbacks: every setting is required, a model or login failure stops the
daemon before the port is bound, and every failure raises.
"""

from __future__ import annotations

import datetime
import faulthandler
import json
import logging
import math
import os
import re
import secrets
import select
import signal
import socket
import socketserver
import sys
import threading
import time
import zoneinfo
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable
from urllib.parse import urlsplit

import numpy as np

import speak_protocol as proto
import speak_telegram as tg
from speak_engines import STT_RATE, TTS_RATE, WHISPER_DEVICE, Engines
from speak_tone import cue_pcm, tone

log = logging.getLogger("speak.daemon")

LOG_FORMAT = "%(asctime)s %(name)s %(levelname)s %(message)s"
LOG_RING_SIZE = 2000
MAX_DEBUG_LINES = 1000
DEBUG_LOGGERS = ("speak", "pytgcalls", "ntgcalls", "telethon")   # ntgcalls' native C++ log has no Python hook
REQUIRED_TELEGRAM_ENV = ("TELEGRAM_API_ID", "TELEGRAM_API_HASH", "TELEGRAM_SESSION", "TELEGRAM_CALL_TARGET")
REQUIRED_SPEAK_ENV = ("SPEAK_KOKORO_MODEL", "SPEAK_VOICE", "SPEAK_SPEED", "SPEAK_WHISPER_MODEL",
                      "SPEAK_LANGUAGE", "SPEAK_ACK_TEXT", "SPEAK_QUIET_HOURS", "TZ")
BIND_HOST = "0.0.0.0"   # open to the LAN by design (owner decision 4); no auth, no TLS
WHISPER_HTTP_PORT = 9000   # the retired whisper-service's port, so its clients need no change
TRANSCRIBE_TOOL = "transcribe"   # how status names an HTTP transcription holding the audio lock
MAX_TRANSCRIBE_SECONDS = 15 * 60   # owner decision 2026-10-16: the longest audio port 9000 accepts
MAX_BODY_BYTES = MAX_TRANSCRIBE_SECONDS * STT_RATE * np.dtype(np.int16).itemsize   # 28,800,000: mono int16 PCM
# Port 9000's per-recv socket timeout. A LAN upload's gaps are milliseconds, and the
# known client (httpx) drops an idle pooled connection after 5 s, so it always closes
# first; an idle keep-alive or slowloris connection is closed after this.
HTTP_TIMEOUT_SECONDS = 30


# ------------------------------------------------------------------ errors

class NoSpeech(TimeoutError):
    """listen: nobody spoke within start_timeout_seconds. The only error sent as KIND_TIMEOUT,
    so the client never mistakes any other timeout for "no speech"."""


class CallEndedError(RuntimeError):
    """The call this session was using is no longer connected: call back. Sent as KIND_CALL_ENDED."""


# ------------------------------------------------------------------ settings

_QUIET_RE = re.compile(r"^(\d{2}):(\d{2})-(\d{2}):(\d{2})$")


def parse_quiet_hours(window: str) -> tuple[int, int]:
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


def in_quiet_hours(window: tuple[int, int], now: datetime.time) -> bool:
    start, end = window
    t = now.hour * 60 + now.minute
    if start < end:
        return start <= t < end
    return t >= start or t < end  # wraps midnight


def _require_env(names: tuple[str, ...]) -> dict[str, str]:
    env = {name: os.environ.get(name, "") for name in names}
    missing = [name for name, value in env.items() if not value]
    if missing:
        raise RuntimeError(f"required environment variables are not set: {', '.join(missing)}")
    return env


def load_settings() -> dict:
    """SPEAK_* and TZ from the compose environment block. All required, no defaults. TZ must
    name a zone in the image's tzdata: glibc would silently run a TZ it cannot find as UTC,
    which would move quiet hours by seven or eight hours."""
    env = _require_env(REQUIRED_SPEAK_ENV)
    try:
        tz = zoneinfo.ZoneInfo(env["TZ"])
    except (zoneinfo.ZoneInfoNotFoundError, ValueError) as e:
        raise RuntimeError(f"TZ={env['TZ']!r} is not a zone in this image's tzdata: {e}") from None
    try:
        speed = float(env["SPEAK_SPEED"])
    except ValueError:
        raise RuntimeError(f"SPEAK_SPEED={env['SPEAK_SPEED']!r} is not a number") from None
    if speed <= 0:
        raise RuntimeError(f"SPEAK_SPEED={speed} must be positive")
    return {**env, "SPEAK_SPEED": speed, "tz": tz, "quiet_window": parse_quiet_hours(env["SPEAK_QUIET_HOURS"])}


def load_env() -> dict[str, str]:
    """TELEGRAM_* from the environment (compose `env_file: .env`); a missing one stops the daemon."""
    return _require_env(REQUIRED_TELEGRAM_ENV)


# ------------------------------------------------------------------ debug support

class LogRing(logging.Handler):
    """Keeps the last LOG_RING_SIZE formatted log lines in memory for the `debug` op."""

    def __init__(self, size: int = LOG_RING_SIZE) -> None:
        super().__init__()
        self.setFormatter(logging.Formatter(LOG_FORMAT))
        self._lines: deque[str] = deque(maxlen=size)

    def emit(self, record: logging.LogRecord) -> None:
        self._lines.append(self.format(record))

    def tail(self, n: int) -> list[str]:
        return list(self._lines)[-n:]


class DebugControl:
    """Runtime switch for DEBUG logging of this daemon and the call libraries, plus the log ring.
    Setting a named logger's level (NOTSET to undo) is all it takes: the handlers on the root
    logger have no level of their own, so no restart is needed."""

    def __init__(self) -> None:
        self.ring = LogRing()
        self.enabled = False
        self._previous: dict[str, int] = {}

    def install(self) -> None:
        logging.getLogger().addHandler(self.ring)

    def set(self, enabled: bool) -> None:
        for name in DEBUG_LOGGERS:
            lg = logging.getLogger(name)
            if enabled:
                if not self.enabled:
                    self._previous[name] = lg.level
                lg.setLevel(logging.DEBUG)
            elif name in self._previous:
                lg.setLevel(self._previous.pop(name))
        self.enabled = enabled
        log.warning("debug logging %s", "ON" if enabled else "OFF")


# ------------------------------------------------------------------ the audio lock

class AudioLock:
    """FIFO ticket lock (design section 4.3): every tool op from every session on every
    computer queues here, in arrival order, and never gets rejected."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._queue: deque[object] = deque()
        self._held = False
        self._current: str | None = None

    def acquire(self, tool: str, peer: str) -> None:
        ticket = object()
        with self._cond:
            self._queue.append(ticket)
            self._cond.wait_for(lambda: not self._held and self._queue[0] is ticket)
            self._queue.popleft()
            self._held = True
            self._current = f"{tool} ({peer})"

    def release(self) -> None:
        with self._cond:
            if not self._held:
                raise RuntimeError("audio lock released while not held")
            self._held = False
            self._current = None
            self._cond.notify_all()

    def snapshot(self) -> tuple[bool, str | None, int]:
        with self._cond:
            return self._held, self._current, len(self._queue)


# ------------------------------------------------------------------ the tools

class Session:
    """One request's view of the client session's call expectation (design section 4.4).
    `generation` is the call the session expects, as "<boot>:<link generation>" (None: no
    expectation); the boot id makes a key from before a daemon restart never match a call
    placed after it. `changed` says whether the reply must carry the new value back."""

    def __init__(self, generation: str | None) -> None:
        self.generation = generation
        self.changed = False

    def expect(self, generation: str | None) -> None:
        self.generation = generation
        self.changed = True


class SpeakService:
    """The tool logic moved from speak_server.py, now calling TelegramLink directly.
    Every method runs with the audio lock held."""

    def __init__(self, link, engines: Engines, settings: dict) -> None:
        self.link = link
        self.engines = engines
        self.ack_text = settings["SPEAK_ACK_TEXT"]
        self.quiet_hours = settings["SPEAK_QUIET_HOURS"]
        self.quiet_window = settings["quiet_window"]
        self.tz = settings["tz"]
        self.boot = secrets.token_hex(4)   # per process: generations restart at 0 after a restart

    def key(self, generation: int) -> str:
        return f"{self.boot}:{generation}"

    # -- call checks

    def _call_ended(self, sess: Session, tool: str, detail: str) -> CallEndedError:
        """Clears the expectation: the drop is reported once, then the session calls back."""
        sess.expect(None)
        return CallEndedError(f"Telegram call ended during {tool}; call back with the call tool ({detail})")

    def require_call(self, sess: Session, tool: str) -> None:
        """Raise unless a call is connected. Any session that talks on the call expects it
        from then on (the call is shared), so a later drop raises "call ended ... call back"
        in every such session. A call ended by hang_up, from any session, on the owner's
        request, is not a drop."""
        info = self.link.info()
        state = info["state"]
        if state == tg.CONNECTED:
            sess.expect(self.key(info["generation"]))
            return
        if (sess.generation is not None and info["end_reason"] == tg.END_HUNG_UP
                and self.key(info["generation"]) == sess.generation):
            sess.expect(None)   # hung up on request: no call-back
        if sess.generation is not None:
            raise self._call_ended(sess, tool, f"call state is {state}")
        raise RuntimeError(f"{tool} needs a connected Telegram call (call state is {state}); "
                           "place one with the call tool")

    def guard_call(self, sess: Session, tool: str, fn: Callable):
        """A call that is no longer connected when `fn` finishes (or fails) makes the tool
        raise. A transcript captured before the drop is included in the message."""
        try:
            result = fn()
        except Exception as e:
            if self.link.state() != tg.CONNECTED:
                raise self._call_ended(sess, tool, str(e)) from e
            raise
        if self.link.state() != tg.CONNECTED:
            partial = f"transcript before the drop: {result!r}" if isinstance(result, str) else "no transcript"
            raise self._call_ended(sess, tool, f"call state is {self.link.state()}; {partial}")
        return result

    def _after_capture(self, play: Callable[[], None]) -> None:
        """Cues played after the owner's words are captured. A call that has just ended (the
        owner said their piece and hung up) makes them fail; the words must still be
        transcribed and returned, so that failure is skipped here and guard_call then raises
        "call ended" WITH the transcript. Any failure while the call is connected raises."""
        try:
            play()
        except RuntimeError:
            if self.link.state() != tg.CONNECTED:
                log.info("call ended right after the capture; keeping the words for the drop report")
                return
            raise

    # -- audio

    def _cue(self, freq_hz: float, seconds: float = 0.12, volume: float = 0.2, lead_silence: float = 0.0) -> None:
        self.link.play(cue_pcm(freq_hz, seconds, volume, lead_silence), TTS_RATE)

    def _ack(self) -> None:
        """Rising two-note chime then a spoken word: the transcript is in hand and is being
        sent to the model. Distinct from the single ear-open/ear-closed cues."""
        self.link.play(np.concatenate([tone(660.0, 0.10), tone(990.0, 0.14)]), TTS_RATE)
        self._speak(self.ack_text)

    @staticmethod
    def plain(text: str) -> str:
        """Strip the markdown that would otherwise be read aloud."""
        text = re.sub(r"```.*?```", " code block omitted ", text, flags=re.S)
        text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
        text = re.sub(r"[`*_#>|]", "", text)
        return re.sub(r"[ \t]+", " ", text).strip()

    def _speak(self, text: str) -> float:
        """Synthesize all of it, then play it: inference never overlaps outgoing speech
        (design section 4.6). Returns the spoken seconds."""
        text = self.plain(text)
        if not text:
            raise ValueError("speak() called with empty text")
        audio = self.engines.synthesize(text)
        self.link.play(audio, TTS_RATE)
        return len(audio) / TTS_RATE

    def _listen(self, max_seconds: float, silence_seconds: float, start_timeout_seconds: float) -> str:
        audio = self.link.record(max_seconds, silence_seconds, start_timeout_seconds,
                                 cue_pcm(880.0, 0.5, 0.5, 0.2), TTS_RATE)
        if audio is tg.TIMEOUT:
            self._cue(330.0)
            raise NoSpeech(f"no speech detected within {start_timeout_seconds:.0f}s")
        self._after_capture(lambda: self._cue(440.0))   # ear closed; Whisper runs only after it has played
        t0 = time.time()
        text = self.engines.transcribe(audio)
        log.info("listen: %.1fs of audio transcribed in %.1fs: %r", len(audio) / STT_RATE, time.time() - t0, text)
        if not text:
            raise RuntimeError(f"speech was detected ({len(audio) / STT_RATE:.1f}s) but Whisper returned no text")
        self._after_capture(self._ack)
        return text

    # -- tool ops (lock held)

    def speak(self, sess: Session, text: str) -> dict:
        self.require_call(sess, "speak")
        return {"seconds": self.guard_call(sess, "speak", lambda: self._speak(text))}

    def listen(self, sess: Session, max_seconds: float, silence_seconds: float, start_timeout_seconds: float) -> dict:
        self.require_call(sess, "listen")
        return {"text": self.guard_call(sess, "listen",
                                        lambda: self._listen(max_seconds, silence_seconds, start_timeout_seconds))}

    def converse(self, sess: Session, text: str, max_seconds: float, silence_seconds: float,
                 start_timeout_seconds: float) -> dict:
        self.require_call(sess, "converse")

        def round_trip() -> str:
            self._speak(text)
            return self._listen(max_seconds, silence_seconds, start_timeout_seconds)

        return {"text": self.guard_call(sess, "converse", round_trip)}

    def call(self, sess: Session, greeting: str, override_quiet_hours: bool, max_seconds: float,
             silence_seconds: float, start_timeout_seconds: float) -> dict:
        # Checked under the lock: a call queued behind a long converse is judged when it would dial.
        # Local time in the required TZ (America/Los_Angeles, compose environment).
        if not override_quiet_hours and in_quiet_hours(self.quiet_window, datetime.datetime.now(self.tz).time()):
            return {"outcome": f"not called: quiet hours ({self.quiet_hours})"}
        info = self.link.info()
        if info["state"] == tg.CONNECTED:
            sess.expect(self.key(info["generation"]))
            return {"outcome": tg.ALREADY_CONNECTED}
        # Last of the pre-dial checks, so quiet hours / already connected keep their outcomes.
        # Measured on the synthesized audio, not guessed from word count.
        text = self.plain(greeting)
        if not text:
            raise ValueError("call() needs a non-empty greeting")
        audio = self.engines.synthesize(text)
        seconds = len(audio) / TTS_RATE
        if seconds < tg.OPENER_MIN_SECONDS:
            raise ValueError(
                f"call() greeting is only {seconds:.1f}s when spoken; it must be at least {tg.OPENER_MIN_SECONDS:g}s. "
                "Nothing was dialed. The greeting is the whole opening: say who is calling, why, and the context, "
                "and end with a question. Lengthen it with the context of why you are calling. If this is a fresh "
                "conversation with nothing to report, open with something playful instead: a joke, something "
                "funny or flirty.")
        outcome = self.link.call(audio, TTS_RATE)
        if outcome in (tg.ANSWERED, tg.ALREADY_CONNECTED):
            sess.expect(self.key(self.link.info()["generation"]))
        else:
            sess.expect(None)
        if outcome != tg.ANSWERED:
            return {"outcome": outcome}
        # Answered: the greeting has played; listen for the owner's reply under the same lock
        # hold (as converse does) so no other session slips in between. A listen timeout or a
        # drop raises, exactly as in listen; on a timeout the call stays connected.
        self.require_call(sess, "call")
        text = self.guard_call(sess, "call", lambda: self._listen(max_seconds, silence_seconds, start_timeout_seconds))
        return {"outcome": tg.ANSWERED, "text": text}

    def hang_up(self, sess: Session) -> dict:
        try:
            return {"outcome": self.link.hang_up()}
        finally:
            sess.expect(None)   # the owner asked to hang up: never a call-back, even if this raised


# ------------------------------------------------------------------ request validation

def _number(req: dict, key: str) -> float:
    value = req.get(key)
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            or not value > 0):
        raise ValueError(f"{key} must be a finite positive number, got {value!r}")
    return float(value)


def _text(req: dict, key: str) -> str:
    value = req.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string, got {value!r}")
    return value


def _flag(req: dict, key: str) -> bool:
    value = req.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean, got {value!r}")
    return value


def _generation(req: dict) -> str | None:
    if "expect_generation" not in req:
        raise ValueError("expect_generation is required (a string from an earlier reply, or null)")
    value = req["expect_generation"]
    if value is not None and not isinstance(value, str):
        raise ValueError(f"expect_generation must be a string or null, got {value!r}")
    return value


def _debug_lines(req: dict) -> int:
    lines = req.get("lines")
    if not isinstance(lines, int) or isinstance(lines, bool) or not 1 <= lines <= MAX_DEBUG_LINES:
        raise ValueError(f"lines must be an integer from 1 to {MAX_DEBUG_LINES}, got {lines!r}")
    return lines


def _listen_args(req: dict) -> tuple[float, float, float]:
    return _number(req, "max_seconds"), _number(req, "silence_seconds"), _number(req, "start_timeout_seconds")


def _error_reply(e: BaseException) -> dict:
    if isinstance(e, NoSpeech):
        kind = proto.KIND_TIMEOUT
    elif isinstance(e, ValueError):
        kind = proto.KIND_VALUE
    elif isinstance(e, CallEndedError):
        kind = proto.KIND_CALL_ENDED
    else:
        kind = proto.KIND_RUNTIME
    if isinstance(e, (RuntimeError, ValueError, TimeoutError)):
        message = str(e) or type(e).__name__
    else:   # reported to the caller with its type, never swallowed
        log.error("request failed", exc_info=e)
        message = f"{type(e).__name__}: {e}"
    return {"ok": False, "error": message, "kind": kind}


# ------------------------------------------------------------------ TCP server

class _Handler(socketserver.BaseRequestHandler):
    def setup(self) -> None:
        proto.enable_keepalive(self.request)

    def handle(self) -> None:
        daemon: SpeakDaemon = self.server.daemon  # type: ignore[attr-defined]
        peer = self.client_address[0]
        try:
            req, _payload = proto.recv_msg(self.request)
        except (ConnectionError, OSError, ValueError) as e:
            log.warning("request from %s could not be read: %s", peer, e)
            return
        op = req.get("op")
        if op in proto.QUICK_OPS:
            try:
                reply = {"ok": True, **daemon.quick(op, req)}
            except Exception as e:
                reply = _error_reply(e)
            self._send(peer, reply)
            return
        if op not in proto.LOCKED_OPS:
            self._send(peer, _error_reply(ValueError(f"unknown op {op!r}")))
            return
        restart_reason: str | None = None
        daemon.lock.acquire(op, peer)
        try:
            if _peer_left(self.request):
                log.warning("%s from %s: the client left while queued; skipped", op, peer)
                return
            if daemon.closing.is_set():
                # A restart or stop is under way: never touch the link (no dial into a dying daemon).
                reply = _error_reply(RuntimeError("speak daemon is restarting; try again in a minute"))
            else:
                try:
                    proto.send_msg(self.request, {"ok": True, "phase": proto.PHASE_STARTED})
                except OSError as e:
                    log.warning("%s from %s: the client left while queued (%s); skipped", op, peer, e)
                    return
                log.info("%s from %s: started", op, peer)
                reply = daemon.run_locked(op, req)
                if op == proto.OP_CALL and reply.get("ok") and proto.is_audio_failure(reply["outcome"]):
                    restart_reason = reply["outcome"]
                    daemon.closing.set()   # before the release: no queued op reaches the link
        finally:
            daemon.lock.release()
        self._send(peer, reply)
        if restart_reason is not None:
            daemon.request_restart(restart_reason)   # after the reply: the caller already has its answer

    def _send(self, peer: str, reply: dict) -> None:
        try:
            proto.send_msg(self.request, reply)
        except (TypeError, ValueError) as e:   # a reply that is not JSON: report it, never a bare close
            log.exception("reply to %s could not be encoded", peer)
            self._send(peer, _error_reply(RuntimeError(f"daemon reply could not be encoded: {e}")))
        except OSError as e:
            log.warning("could not send the reply to %s: %s", peer, e)


def _peer_left(sock: socket.socket) -> bool:
    """True when the client closed its end while its request was queued. A client sends one
    request and then only reads, so a readable socket here means EOF (or a reset)."""
    readable, _, _ = select.select([sock], [], [], 0)
    if not readable:
        return False
    try:
        return sock.recv(1, socket.MSG_PEEK) == b""
    except (ConnectionResetError, BrokenPipeError):
        return True


class _TCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    daemon_threads = True
    allow_reuse_address = True   # TIME_WAIT only; a live listener still refuses a second bind (EADDRINUSE)


# ------------------------------------------------------------------ HTTP whisper API (port 9000)

def _json_bytes(obj: dict) -> bytes:
    """Encoded exactly as the old service's Starlette JSONResponse: compact, UTF-8, no NaN."""
    return json.dumps(obj, ensure_ascii=False, allow_nan=False, indent=None, separators=(",", ":")).encode("utf-8")


class _BadRequest(ValueError):
    """The request's framing could not be read (bad Content-Length or chunk size)."""


class _TooLarge(ValueError):
    """The request body is larger than MAX_BODY_BYTES; raised before the excess is read."""


class ClientLeft(Exception):
    """transcribe_pcm: the HTTP client closed its connection while the request was queued
    for the audio lock; inference was skipped."""


class DaemonClosing(RuntimeError):
    """transcribe_pcm: a restart or stop is under way; inference was skipped."""


class _WhisperHandler(BaseHTTPRequestHandler):
    """The retired whisper-service's API (scratch/whisper-service/server.py, FastAPI under
    uvicorn), reproduced: the same routes, status codes, bodies and content types, including
    FastAPI's 404/405 `{"detail": ...}` replies and its redirect of a trailing slash."""

    protocol_version = "HTTP/1.1"   # keep-alive, as uvicorn; every reply carries Content-Length
    ROUTES = {"/": "POST", "/health": "GET"}
    timeout = HTTP_TIMEOUT_SECONDS   # StreamRequestHandler.setup applies it as the socket timeout

    def setup(self) -> None:
        super().setup()
        proto.enable_keepalive(self.request)   # a dead peer is detected while its request waits for the lock

    def do_GET(self) -> None:
        self._route()

    do_POST = do_PUT = do_DELETE = do_PATCH = do_OPTIONS = do_HEAD = do_GET

    def log_message(self, fmt: str, *args) -> None:   # access log through logging (stderr), not raw stderr
        log.info("http %s: %s", self.client_address[0], fmt % args)

    def log_error(self, fmt: str, *args) -> None:   # e.g. "Request timed out" on an idle connection
        log.warning("http %s: %s", self.client_address[0], fmt % args)

    def _route(self) -> None:
        try:
            body = self._read_body()   # always read, so a kept-alive connection stays in step
        except _BadRequest as e:
            log.warning("http %s: %s", self.client_address[0], e)
            self.close_connection = True
            self._reply(400, b"Invalid HTTP request received.", "text/plain; charset=utf-8")
            return
        except _TooLarge as e:   # the rest of the body is never read, so the connection cannot stay in step
            log.warning("http %s: %s", self.client_address[0], e)
            self.close_connection = True
            self._reply(413, _json_bytes({"detail": str(e)}), "application/json", {"Connection": "close"})
            return
        except TimeoutError as e:   # the body stalled for HTTP_TIMEOUT_SECONDS
            log.warning("http %s: body read timed out after %ss: %s", self.client_address[0], self.timeout, e)
            self.close_connection = True
            self._reply(408, _json_bytes({"detail": "Request Timeout"}), "application/json", {"Connection": "close"})
            return
        path = urlsplit(self.path).path
        method = self.command
        if path in self.ROUTES:
            if self.ROUTES[path] != method:
                self._reply(405, _json_bytes({"detail": "Method Not Allowed"}), "application/json",
                            {"Allow": self.ROUTES[path]})
            elif path == "/health":
                self._health()
            else:
                self._transcribe(body)
            return
        if path != "/":
            other = path[:-1] if path.endswith("/") else path + "/"
            if other in self.ROUTES:   # Starlette's redirect_slashes: 307 to the other spelling
                query = urlsplit(self.path).query
                host = self.headers.get("Host", f"{self.server.server_address[0]}:{self.server.server_address[1]}")
                self._reply(307, b"", None, {"Location": f"http://{host}{other}" + (f"?{query}" if query else "")})
                return
        self._reply(404, _json_bytes({"detail": "Not Found"}), "application/json")

    @staticmethod
    def _too_large(size: int) -> _TooLarge:
        return _TooLarge(f"request body of {size} bytes exceeds the limit of {MAX_BODY_BYTES} bytes "
                         f"({MAX_TRANSCRIBE_SECONDS // 60} minutes of 16 kHz mono int16 PCM)")

    def _read_body(self) -> bytes:
        """The whole body, never more than MAX_BODY_BYTES: a larger declared size raises
        _TooLarge before the body (or the chunk that passes the limit) is read."""
        if "chunked" in self.headers.get("Transfer-Encoding", "").lower():
            chunks = []
            total = 0
            while True:
                line = self.rfile.readline(65537)
                try:
                    size = int(line.split(b";", 1)[0].strip(), 16)
                except ValueError:
                    raise _BadRequest(f"bad chunk size line {line[:40]!r}") from None
                if size == 0:
                    while self.rfile.readline(65537) not in (b"\r\n", b"\n", b""):
                        pass   # trailers
                    return b"".join(chunks)
                total += size
                if total > MAX_BODY_BYTES:
                    raise self._too_large(total)
                chunks.append(self.rfile.read(size))
                self.rfile.readline()   # the CRLF after the chunk
        length = self.headers.get("Content-Length")
        if length is None:
            return b""
        if not length.strip().isdigit():
            raise _BadRequest(f"bad Content-Length {length!r}")
        if int(length) > MAX_BODY_BYTES:
            raise self._too_large(int(length))
        return self.rfile.read(int(length))

    def _health(self) -> None:
        daemon: SpeakDaemon = self.server.daemon  # type: ignore[attr-defined]
        self._reply(200, _json_bytes({"status": "ok", "model": daemon.service.engines.whisper_model,
                                      "device": WHISPER_DEVICE}), "application/json")

    def _transcribe(self, body: bytes) -> None:
        daemon: SpeakDaemon = self.server.daemon  # type: ignore[attr-defined]
        peer = self.client_address[0]
        try:
            text = daemon.transcribe_pcm(body, peer, self.request)
        except ClientLeft:
            log.warning("http transcribe from %s: the client left while queued; skipped", peer)
            self.close_connection = True   # nobody to reply to
            return
        except DaemonClosing as e:
            log.warning("http transcribe from %s: refused, %s", peer, e)
            self.close_connection = True
            self._reply(503, _json_bytes({"detail": str(e)}), "application/json", {"Connection": "close"})
            return
        except Exception:   # the old service's unhandled exception: uvicorn's plain 500
            log.exception("http %s: transcription failed", self.client_address[0])
            self._reply(500, b"Internal Server Error", "text/plain; charset=utf-8")
            return
        self._reply(200, _json_bytes({"text": text}), "application/json")

    def _reply(self, status: int, body: bytes, content_type: str | None, headers: dict | None = None) -> None:
        try:
            self.send_response(status)
            if content_type is not None:
                self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            for name, value in (headers or {}).items():
                self.send_header(name, value)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)
        except OSError as e:
            self.close_connection = True
            log.warning("http %s: could not send the reply: %s", self.client_address[0], e)


class _WhisperHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True   # TIME_WAIT only; a live listener still refuses a second bind (EADDRINUSE)


class SpeakDaemon:
    """Owns the link, the engines, the audio lock, the TCP server and the HTTP whisper API."""

    def __init__(self, link, engines: Engines, settings: dict, debug: DebugControl, port: int = proto.DAEMON_PORT,
                 on_restart: Callable[[str], None] | None = None, http_port: int = WHISPER_HTTP_PORT) -> None:
        self.link = link
        self.service = SpeakService(link, engines, settings)
        self.lock = AudioLock()
        self.port = port
        self.http_port = http_port
        self.on_restart = on_restart   # main(): exit 1 so compose restarts the container
        self.debug = debug   # installed on the root logger by main()
        self.closing = threading.Event()   # set on restart or stop: queued tool ops are refused
        self.started_at = time.monotonic()
        self._server: _TCPServer | None = None
        self._http: _WhisperHTTPServer | None = None

    def start(self) -> None:
        """Telegram (Silero preload, login) first, then the binds: a refused connection on
        either port always means "not ready or not running", never "running without models
        or a login". Both ports bind before either serves, so a failed bind serves nothing."""
        self.link.start()
        server = _TCPServer((BIND_HOST, self.port), _Handler)
        try:
            http = _WhisperHTTPServer((BIND_HOST, self.http_port), _WhisperHandler)
        except BaseException:
            server.server_close()
            raise
        server.daemon = self  # type: ignore[attr-defined]
        http.daemon = self  # type: ignore[attr-defined]
        self._server, self._http = server, http
        threading.Thread(target=server.serve_forever, name="speak-tcp", daemon=True).start()
        threading.Thread(target=http.serve_forever, name="speak-http", daemon=True).start()
        log.info("speak daemon: listening on %s:%d (tools) and http %s:%d (whisper API)",
                 BIND_HOST, server.server_address[1], BIND_HOST, http.server_address[1])

    def transcribe_pcm(self, body: bytes, peer: str, sock: socket.socket) -> str:
        """POST / on port 9000: raw 16 kHz mono int16 PCM -> text, converted as the retired
        whisper-service did (an odd byte count raises, which the handler sends as a 500).
        Inference waits in the audio lock, FIFO with the tool ops, so it never overlaps call
        audio. No "no text" error here, unlike listen: the old API returned an empty text.
        Once the lock is held, as for the tool ops: a client that left while queued (`sock`
        at EOF; httpx does not pipeline, so nothing else is readable) raises ClientLeft, and
        a restart or stop under way raises DaemonClosing; neither runs inference."""
        audio = np.frombuffer(body, dtype=np.int16).astype(np.float32) / 32768.0
        self.lock.acquire(TRANSCRIBE_TOOL, peer)
        try:
            if _peer_left(sock):
                raise ClientLeft(f"http transcribe from {peer}: the client left while queued")
            if self.closing.is_set():
                raise DaemonClosing("speak daemon is restarting")
            t0 = time.time()
            text = self.service.engines.transcribe(audio)
        finally:
            self.lock.release()
        log.info("http transcribe from %s: %.1fs of audio in %.1fs: %r", peer, len(audio) / STT_RATE,
                 time.time() - t0, text)
        return text

    def quick(self, op: str, req: dict) -> dict:
        link = self.link
        if op == proto.OP_PING:
            return {}
        if op == proto.OP_STATUS:
            busy, current_tool, waiting = self.lock.snapshot()
            return {"busy": busy, "current_tool": current_tool, "waiting": waiting, "call": link.info()}
        if op == proto.OP_DEBUG:
            n = _debug_lines(req)
            return {"daemon": "running", "pid": os.getpid(),
                    "uptime_seconds": round(time.monotonic() - self.started_at, 1),
                    "debug": self.debug.enabled, "call": link.info(),
                    "events": link.events.recent(n), "log": self.debug.ring.tail(n)}
        if op == proto.OP_SET_DEBUG:
            enabled = _flag(req, "enabled")
            self.debug.set(enabled)
            link.audio_stats = enabled   # audio-level events only while debugging
            return {"daemon": "running", "debug": self.debug.enabled}
        raise ValueError(f"unknown op {op!r}")

    def run_locked(self, op: str, req: dict) -> dict:
        """One tool op with the lock held. The reply carries "expect" when the session's
        call expectation changed (an int, or null to clear)."""
        svc = self.service
        sess: Session | None = None
        try:
            if op == proto.OP_HANG_UP:
                sess = Session(None)
                result = svc.hang_up(sess)
            else:
                sess = Session(_generation(req))
                if op == proto.OP_SPEAK:
                    result = svc.speak(sess, _text(req, "text"))
                elif op == proto.OP_LISTEN:
                    result = svc.listen(sess, *_listen_args(req))
                elif op == proto.OP_CONVERSE:
                    result = svc.converse(sess, _text(req, "text"), *_listen_args(req))
                elif op == proto.OP_CALL:
                    result = svc.call(sess, _text(req, "greeting"), _flag(req, "override_quiet_hours"),
                                      *_listen_args(req))
                else:
                    raise ValueError(f"unknown op {op!r}")
            reply = {"ok": True, **result}
        except Exception as e:
            reply = _error_reply(e)
        if sess is not None and sess.changed:
            reply["expect"] = sess.generation
        return reply

    def request_restart(self, reason: str) -> None:
        if self.on_restart is None:
            raise RuntimeError(f"daemon restart requested ({reason}) but no on_restart handler is set")
        log.error("restarting the daemon (exit 1, compose's restart policy restarts it): %s", reason)
        self.on_restart(reason)

    def stop(self) -> None:
        self.closing.set()
        for server in (self._server, self._http):
            if server is not None:
                server.shutdown()
                server.server_close()
        self._server = self._http = None
        self.link.stop()


# ------------------------------------------------------------------ entry point

def build_link(env: dict[str, str]) -> tg.TelegramLink:
    from pytgcalls import PyTgCalls
    from telethon import TelegramClient as TelethonClient
    from telethon.sessions import StringSession

    return tg.TelegramLink(
        env["TELEGRAM_CALL_TARGET"],
        client_factory=lambda: TelethonClient(StringSession(env["TELEGRAM_SESSION"]),
                                              int(env["TELEGRAM_API_ID"]), env["TELEGRAM_API_HASH"]),
        calls_factory=PyTgCalls,
    )


def enable_crash_dumps() -> None:
    """On a fatal signal, write every Python thread's stack to stderr, then die by that
    signal as before (compose still sees exit 139 for SIGSEGV and restarts the daemon).
    The SIGSEGV of 2026-10-16 09:17 left no trace in the log; the kernel log named only
    ntgcalls and its native thread."""
    faulthandler.enable(file=sys.stderr, all_threads=True)


def main() -> None:
    enable_crash_dumps()   # first: a crash while the engines load is dumped too
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format=LOG_FORMAT)
    code = 0
    try:
        stop = threading.Event()
        exit_code = [0]

        def restart(_reason: str) -> None:
            exit_code[0] = 1
            stop.set()

        signal.signal(signal.SIGTERM, lambda *_: stop.set())
        signal.signal(signal.SIGINT, lambda *_: stop.set())
        debug = DebugControl()
        debug.install()
        settings = load_settings()
        log.info("quiet hours %s in %s (local time now %s)", settings["SPEAK_QUIET_HOURS"], settings["TZ"],
                 datetime.datetime.now(settings["tz"]).strftime("%Y-%m-%d %H:%M %Z"))
        telegram_env = load_env()
        # Startup order (design section 6.2): engines, then Silero + Telegram (link.start), then the binds.
        engines = Engines(settings["SPEAK_KOKORO_MODEL"], settings["SPEAK_VOICE"], settings["SPEAK_SPEED"],
                          settings["SPEAK_WHISPER_MODEL"], settings["SPEAK_LANGUAGE"])
        engines.load()
        daemon = SpeakDaemon(build_link(telegram_env), engines, settings, debug, on_restart=restart)
        daemon.start()
        stop.wait()
        daemon.stop()
        code = exit_code[0]
    except BaseException:
        log.exception("speak daemon failed")
        code = 1
    finally:
        # Every exit path skips the interpreter's teardown: ntgcalls' C++ global destructors
        # abort() while its WebRTC thread is alive (macOS crash report 2026-10-09).
        sys.stderr.flush()
        os._exit(code)


if __name__ == "__main__":
    main()
