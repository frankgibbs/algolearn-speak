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
their final reply. Quick ops (`ping`, `status`, `debug`, `set_debug`, and the
persona ops) never touch the lock, so they answer while a call is ringing or connected.

Personas (docs/DESIGN_PERSONAS.md, owner decisions 2026-10-22): every project speaks as
a persona (a name, a voice blend, a speed and a personality) kept in the PersonaStore
(speak_personas.py, the file SPEAK_PERSONAS_FILE on the speak_state volume). Before the
lock, a project with no persona is assigned to the house persona, and a speak, converse
or call whose persona_seen is stale (the project moved to another persona, or its
persona's name or personality changed) is refused with the persona block, so nothing is
spoken or dialed in words written for another persona. A voice or speed change never
refuses a turn: it is applied per sentence.

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
Engines.synthesize_segments/synthesize/transcribe or the Silero VAD.

No fallbacks: every setting is required, a model or login failure stops the
daemon before the port is bound, and every failure raises.
"""

from __future__ import annotations

import dataclasses
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
from speak_engines import STT_RATE, TTS_RATE, WHISPER_DEVICE, Engines, Voice, split_sentences
from speak_personas import PersonaStore
from speak_tone import chime_pcm, cue_pcm, switch_pcm

log = logging.getLogger("speak.daemon")

LOG_FORMAT = "%(asctime)s %(name)s %(levelname)s %(message)s"
LOG_RING_SIZE = 2000
MAX_DEBUG_LINES = 1000
DEBUG_LOGGERS = ("speak", "pytgcalls", "ntgcalls", "telethon")   # ntgcalls' native C++ log has no Python hook
REQUIRED_TELEGRAM_ENV = ("TELEGRAM_API_ID", "TELEGRAM_API_HASH", "TELEGRAM_SESSION", "TELEGRAM_CALL_TARGET")
REQUIRED_SPEAK_ENV = ("SPEAK_KOKORO_MODEL", "SPEAK_VOICE", "SPEAK_SPEED", "SPEAK_WHISPER_MODEL",
                      "SPEAK_LANGUAGE", "SPEAK_QUIET_HOURS", "SPEAK_HOLD_PHRASE", "SPEAK_RESUME_PHRASE",
                      "SPEAK_PERSONAS_FILE", "TZ")
BIND_HOST = "0.0.0.0"   # open to the LAN by design (owner decision 4); no auth, no TLS
WHISPER_HTTP_PORT = 9000   # the retired whisper-service's port, so its clients need no change
TRANSCRIBE_TOOL = "transcribe"   # how status names an HTTP transcription holding the audio lock
SESSION_SWITCH_EVENT = "session_switch"   # call_debug: a switch of session was announced on the call
PERSONA_REFUSED_EVENT = "persona_refused"   # call_debug: a stale persona_seen was refused before the lock
# call_debug: call() returned or raised without dialing (owner decision 2026-10-23), and its reasons.
CALL_REFUSED_EVENT = "call_refused"
REFUSED_QUIET_HOURS, REFUSED_EMPTY_GREETING, REFUSED_SHORT_GREETING = "quiet_hours", "empty_greeting", "greeting_too_short"
# The floor (docs/DESIGN_FLOOR.md, owner decisions 2026-10-20): call_debug events and the
# floor_release reasons.
FLOOR_TAKE_EVENT, FLOOR_RELEASE_EVENT = "floor_take", "floor_release"
FLOOR_SPEAK, FLOOR_IDLE, FLOOR_HANG_UP = "speak", "idle", "hang_up"
FLOOR_HOLD_EXPIRED, FLOOR_CLIENT_LEFT = "hold_expired", "client_left"
# A switch caused by a bare listen says this after the project name (owner decision 2026-10-20):
# the ear opens right after, so a name alone would give the owner no context.
SWITCH_LISTEN_SUFFIX = ", still listening"
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


class PersonaRefused(ValueError):
    """speak, converse or call from a session that has not seen its project's current persona,
    name and personality (the key "<id>:<words_revision>", docs/DESIGN_PERSONAS.md 4.7):
    refused before the lock, nothing spoken or dialed. Sent as
    KIND_VALUE with the persona block (REPLY_PERSONA)."""

    def __init__(self, message: str, block: dict) -> None:
        super().__init__(message)
        self.block = block


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


def normalize_phrase(text: str) -> str:
    """For the hold/resume match (docs/DESIGN_INTERRUPT.md section 4.2): lowercase, every
    character but letters, digits and apostrophes becomes a space, spaces collapse. Whisper's
    "Hold on." and the setting "hold on" both become "hold on"."""
    return " ".join(re.sub(r"[^a-z0-9']+", " ", text.lower()).split())


def parse_phrases(hold: str, resume: str) -> dict[str, str]:
    """SPEAK_HOLD_PHRASE / SPEAK_RESUME_PHRASE -> {normalized phrase: command}. Both must
    still say something once normalized, and they must differ."""
    norm = {proto.CMD_HOLD: normalize_phrase(hold), proto.CMD_RESUME: normalize_phrase(resume)}
    for name, raw, value in (("SPEAK_HOLD_PHRASE", hold, norm[proto.CMD_HOLD]),
                             ("SPEAK_RESUME_PHRASE", resume, norm[proto.CMD_RESUME])):
        if not value:
            raise RuntimeError(f"{name}={raw!r} has no letters or digits")
    if norm[proto.CMD_HOLD] == norm[proto.CMD_RESUME]:
        raise RuntimeError(f"SPEAK_HOLD_PHRASE and SPEAK_RESUME_PHRASE are the same phrase ({norm[proto.CMD_HOLD]!r})")
    return {phrase: command for command, phrase in norm.items()}


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
    if not proto.PERSONA_SPEED_MIN <= speed <= proto.PERSONA_SPEED_MAX:   # the speed every new persona starts with
        raise RuntimeError(f"SPEAK_SPEED={speed} must be from {proto.PERSONA_SPEED_MIN} to {proto.PERSONA_SPEED_MAX}")
    return {**env, "SPEAK_SPEED": speed, "tz": tz, "quiet_window": parse_quiet_hours(env["SPEAK_QUIET_HOURS"]),
            "phrases": parse_phrases(env["SPEAK_HOLD_PHRASE"], env["SPEAK_RESUME_PHRASE"])}


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

class _Ticket:
    """One queued acquire: the session id of its op, or None for an HTTP transcription.
    Compared by identity (deque.remove)."""

    __slots__ = ("session_id",)

    def __init__(self, session_id: str | None) -> None:
        self.session_id = session_id


class AudioLock:
    """Ticket lock (design section 4.3): every tool op from every session on every computer
    queues here and never gets rejected, with the floor (docs/DESIGN_FLOOR.md, owner
    decisions 2026-10-20). The session that last played audio on the call holds the floor;
    while the floor is live, the free lock goes only to its ops (and to HTTP transcriptions,
    which carry no session), in arrival order; every other session keeps waiting. Without a
    live floor: plain FIFO. The floor lapses `idle_seconds` after the holder's last op
    released the lock; no timer thread: a waiter that is not eligible waits until that
    deadline, and the first thread that finds it past records the lapse and wakes the rest.
    `record_event` is the call's event log (link.record_event): FLOOR_TAKE_EVENT and
    FLOOR_RELEASE_EVENT."""

    def __init__(self, record_event: Callable[..., None], idle_seconds: float = proto.FLOOR_IDLE_SECONDS) -> None:
        self._cond = threading.Condition()
        self._queue: deque[_Ticket] = deque()
        self._held = False
        self._current: str | None = None
        self._record_event = record_event
        self._idle_seconds = idle_seconds
        self._floor: Speaker | None = None
        self._floor_since = 0.0
        self._idle_deadline: float | None = None   # None while the holder's op runs
        self._current_session: str | None = None   # the session of the op holding the lock
        self._closed = False   # close(): the daemon is going away, plain FIFO from then on

    def acquire(self, tool: str, peer: str, speaker: Speaker | None) -> None:
        """`speaker` is the tool op's validated identity; None only for an HTTP transcription
        (owner decision 8: it skips the floor and waits only for the op holding the lock)."""
        ticket = _Ticket(None if speaker is None else speaker.session_id)
        with self._cond:
            self._lapse_if_due()
            self._queue.append(ticket)
            if self._floor is not None and ticket.session_id == self._floor.session_id:
                # The holder is active from the moment its op arrives, not when it gets the
                # lock: an op holding the lock past the deadline (a long transcription) must
                # not let the floor lapse under the holder's queued op.
                self._idle_deadline = None
            try:
                while True:
                    self._lapse_if_due()
                    if not self._held and self._first_eligible() is ticket:
                        break
                    if self._floor is not None and self._idle_deadline is not None:
                        self._cond.wait(self._idle_deadline - time.monotonic())
                    else:
                        self._cond.wait()
            except BaseException:
                self._queue.remove(ticket)   # never strand a ticket that would block the queue
                floor = self._floor
                if (floor is not None and ticket.session_id == floor.session_id
                        and self._current_session != floor.session_id
                        and not any(t.session_id == floor.session_id for t in self._queue)):
                    # Its arrival stopped the idle count; with no other op of the holder's
                    # queued or running, nothing would restart it and the floor would stick.
                    self._idle_deadline = time.monotonic() + self._idle_seconds
                self._cond.notify_all()
                raise
            self._queue.remove(ticket)
            self._held = True
            self._current = f"{tool} ({peer})"
            self._current_session = ticket.session_id
            if self._floor is not None and ticket.session_id == self._floor.session_id:
                self._idle_deadline = None   # the holder is active again

    def release(self) -> None:
        """The holder's op releasing starts the idle count; a transcription's (no session)
        or another session's leaves the deadline as it was."""
        with self._cond:
            if not self._held:
                raise RuntimeError("audio lock released while not held")
            if self._floor is not None and self._current_session == self._floor.session_id:
                self._idle_deadline = time.monotonic() + self._idle_seconds
            self._held = False
            self._current = None
            self._current_session = None
            self._cond.notify_all()

    def take_floor(self, speaker: Speaker) -> None:
        """The session that just played audio on the call (SpeakService._take_turn, an
        answered call) holds the floor. By the current holder: no change, no event. Only the
        op holding the lock may take it, as `speaker`."""
        with self._cond:
            if not self._held or self._current_session != speaker.session_id:
                raise RuntimeError(f"take_floor by {speaker.project!r} without holding the audio lock as that session")
            self._lapse_if_due()
            if self._floor is not None and self._floor.session_id == speaker.session_id:
                return
            previous = self._floor
            self._floor = speaker
            self._floor_since = time.monotonic()
            self._idle_deadline = None
            self._record_event(FLOOR_TAKE_EVENT, project=speaker.project,
                               from_project=None if previous is None else previous.project)
            log.info("floor: %r takes it (from %r)", speaker.project, None if previous is None else previous.project)

    def yield_floor(self, speaker: Speaker, reason: str) -> None:
        """Clears the floor when `speaker` holds it; the release() that follows then wakes
        the queue in plain FIFO order. Not the holder (or no floor): nothing to yield."""
        with self._cond:
            if self._floor is None or self._floor.session_id != speaker.session_id:
                return
            self._clear_floor(reason)

    def close(self) -> None:
        """The daemon is restarting or stopping: the floor no longer orders the queue, so
        every queued op reaches the lock in arrival order and gets the "restarting" refusal
        at once instead of waiting out the idle time. No floor event: nothing yields it."""
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    def snapshot(self) -> tuple[bool, str | None, int]:
        with self._cond:
            return self._held, self._current, len(self._queue)

    # -- under self._cond

    def _first_eligible(self) -> _Ticket | None:
        """The first ticket that may take the free lock: with a live floor, the holder's own
        and every transcription's; without one, or once closed, the head of the queue."""
        for ticket in self._queue:
            if self._closed or self._floor is None or ticket.session_id is None or ticket.session_id == self._floor.session_id:
                return ticket
        return None

    def _lapse_if_due(self) -> None:
        """A floor past its idle deadline lapses here, observed by whichever thread looks
        first (a waiter woken by its timed wait, a later acquire or take_floor)."""
        if (self._floor is not None and self._idle_deadline is not None
                and time.monotonic() >= self._idle_deadline):
            self._clear_floor(FLOOR_IDLE)
            self._cond.notify_all()

    def _clear_floor(self, reason: str) -> None:
        floor = self._floor
        self._floor = None
        self._idle_deadline = None
        self._record_event(FLOOR_RELEASE_EVENT, project=floor.project, reason=reason,
                           held_s=round(time.monotonic() - self._floor_since, 1), waiting=len(self._queue))
        log.info("floor: %r releases it (%s), %d waiting", floor.project, reason, len(self._queue))


# ------------------------------------------------------------------ the tools

@dataclasses.dataclass(frozen=True)
class Speaker:
    """The client session a tool op comes from (protocol 3, owner decision 2026-10-20): a
    session id the client generates once per process, and its project, the basename of the
    client's working folder."""
    session_id: str
    project: str


class Session:
    """One request's view of the client session's call expectation (design section 4.4).
    `generation` is the call the session expects, as "<boot>:<link generation>" (None: no
    expectation); the boot id makes a key from before a daemon restart never match a call
    placed after it. `changed` says whether the reply must carry the new value back.
    `speaker` is who sent the request; `switched_from` is set to the previous session's
    project once a switch of session has been announced (the reply then carries it)."""

    def __init__(self, generation: str | None, speaker: Speaker) -> None:
        self.generation = generation
        self.changed = False
        self.speaker = speaker
        self.switched_from: str | None = None

    def expect(self, generation: str | None) -> None:
        self.generation = generation
        self.changed = True


@dataclasses.dataclass
class Spoken:
    """What _speak did: the seconds of speech sent, the seconds the owner held it, and,
    when a hold expired, the interruption for the reply (REPLY_INTERRUPTION)."""
    seconds: float
    held_seconds: float
    interruption: dict | None = None


class _Segments:
    """Kokoro's (sentence index, audio) segments for one speech, pulled lazily and kept, so
    resuming after a hold replays the cut sentence and anything already made without
    synthesizing it again."""

    def __init__(self, segments) -> None:
        self._it = iter(segments)
        self._cache: list[tuple[int, np.ndarray]] = []
        self._done = False

    def _pull(self) -> tuple[int, np.ndarray] | None:
        if self._done:
            return None
        item = next(self._it, None)
        if item is None:
            self._done = True
            return None
        self._cache.append(item)
        return item

    def from_sentence(self, start: int):
        """Every segment of sentence `start` and after, cached ones first."""
        i = 0
        while True:
            if i < len(self._cache):
                item = self._cache[i]
            else:
                item = self._pull()
                if item is None:
                    return
            i += 1
            if item[0] >= start:
                yield item

    def next_index(self) -> int | None:
        """The sentence of the first segment not yet played after a cut that fell between
        segments (everything queued had played): pulled now. None: nothing is left."""
        item = self._pull()
        return None if item is None else item[0]


class SpeakService:
    """The tool logic moved from speak_server.py, now calling TelegramLink directly.
    Every method runs with the audio lock held; `lock` is that AudioLock, for the floor
    (take_floor where the turn moves, yield_floor where an op gives the floor up)."""

    def __init__(self, link, engines: Engines, personas: PersonaStore, settings: dict, lock: AudioLock) -> None:
        self.link = link
        self.engines = engines
        self.personas = personas
        self.lock = lock
        self.quiet_hours = settings["SPEAK_QUIET_HOURS"]
        self.quiet_window = settings["quiet_window"]
        self.tz = settings["tz"]
        self.phrases: dict[str, str] = settings["phrases"]   # normalized phrase -> CMD_HOLD / CMD_RESUME
        self.boot = secrets.token_hex(4)   # per process: generations restart at 0 after a restart
        # Who played audio to the owner last, and on which call: (call key, Speaker), or None
        # before the first call. Owner decision 2026-10-20: tied to the call, so a new call
        # starts from the session that placed it (its greeting introduced it) and a session's
        # turn on an earlier call is never compared with one on the current call.
        self._turn: tuple[str, Speaker] | None = None

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
        """A cue played after the owner's words are captured. A call that has just ended (the
        owner said their piece and hung up) makes it fail; the words must still be returned,
        so that failure is skipped here and guard_call then raises "call ended" WITH the
        transcript. Any failure while the call is connected raises."""
        try:
            play()
        except RuntimeError:
            if self.link.state() != tg.CONNECTED:
                log.info("call ended right after the capture; keeping the words for the drop report")
                return
            raise

    # -- personas (docs/DESIGN_PERSONAS.md)

    def _voice(self, project: str) -> Voice:
        """The project's persona's voice now: read for each sentence of a speech, so a persona
        change lands on the next sentence synthesized (owner decision 7)."""
        persona = self.personas.persona_of(project)
        return Voice(self.engines.voice_pack(persona.voice), persona.speed)

    # -- session switch (owner decisions 2026-10-20)

    def _take_turn(self, sess: Session, tool: str) -> None:
        """Before a speak/listen/converse plays anything: when the previous audio on this call
        came from another session, announce the switch (the switch cue, then "<persona name>,
        <project name>" in the incoming persona's voice, owner decisions 6 and 13 of
        docs/DESIGN_PERSONAS.md; for a bare listen "..., still listening"), record
        SESSION_SWITCH_EVENT and set sess.switched_from to the previous session's project. The
        first audio on a call from the session that placed it is not a switch (call() makes it
        the call's first turn). The announcement is one plain play, like the greeting and the
        cues: it cannot be held. A call that drops during it
        raises "call ended" and leaves the turn unchanged."""
        key = self.key(self.link.info()["generation"])
        previous = self._turn
        if previous is not None and previous[0] == key and previous[1].session_id != sess.speaker.session_id:
            persona = self.personas.persona_of(sess.speaker.project)
            spoken = f"{persona.name}, {proto.speakable_name(sess.speaker.project)}"
            if tool == "listen":
                spoken += SWITCH_LISTEN_SUFFIX
            voice = Voice(self.engines.voice_pack(persona.voice), persona.speed)
            audio = np.concatenate([switch_pcm(), self.engines.synthesize(spoken, voice)])
            self.guard_call(sess, tool, lambda: self.link.play(audio, TTS_RATE))
            self.link.record_event(SESSION_SWITCH_EVENT, tool=tool, from_project=previous[1].project,
                                   to_project=sess.speaker.project, to_persona=persona.name, spoken=spoken,
                                   announce_s=round(len(audio) / TTS_RATE, 2))
            log.info("%s: session switch on call %s, %r -> %r announced", tool, key, previous[1].project,
                     sess.speaker.project)
            sess.switched_from = previous[1].project
        self._turn = (key, sess.speaker)
        self.lock.take_floor(sess.speaker)   # the floor moves with the turn

    # -- audio

    def _cue(self, freq_hz: float, seconds: float = 0.12, volume: float = 0.2, lead_silence: float = 0.0) -> None:
        self.link.play(cue_pcm(freq_hz, seconds, volume, lead_silence), TTS_RATE)

    @staticmethod
    def plain(text: str) -> str:
        """Strip the markdown that would otherwise be read aloud."""
        text = re.sub(r"```.*?```", " code block omitted ", text, flags=re.S)
        text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
        text = re.sub(r"[`*_#>|]", "", text)
        return re.sub(r"[ \t]+", " ", text).strip()

    def classify(self, audio: np.ndarray) -> tg.Classified:
        """One watch candidate: Whisper, then an exact match of the WHOLE normalized
        transcript against the two phrases (no names, no fuzzy match; docs/DESIGN_INTERRUPT.md section 4.2)."""
        text = self.engines.transcribe(audio)
        return tg.Classified(self.phrases.get(normalize_phrase(text)), text)

    def _speak(self, speaker: Speaker, text: str, client_left: Callable[[], bool]) -> Spoken:
        """Stream it: the first Kokoro segment starts playing as soon as it is synthesized,
        and the rest are synthesized while earlier ones play (owner decision 2026-10-16,
        docs/DESIGN_STREAMED_SPEECH.md). While it plays, the owner's audio is watched for
        the hold phrase (docs/DESIGN_INTERRUPT.md, Release 2): "hold on" cuts the speech at
        once and holds (silence out) until "continue", which restarts the cut sentence from
        its beginning. The holds of one speech may add up to HOLD_MAX_SECONDS; then the hold
        has expired and the result carries the interruption (the caller ends the call). A
        hold whose client goes away (`client_left`) ends at once and raises
        (_client_left_during_hold). Returns once all of it has played (or the hold expired), so the listen that follows
        never overlaps outgoing speech."""
        text = self._spoken_text(text)
        pieces = split_sentences(text)
        # The voice is asked per sentence: a persona change lands on the next sentence, and a
        # sentence replayed after "continue" comes from the cache in the voice it was made in.
        segments = _Segments(self.engines.synthesize_segments(text, lambda: self._voice(speaker.project)))
        start, seconds, held = 0, 0.0, 0.0
        while True:
            out = self.link.play_watched(segments.from_sentence(start), TTS_RATE, self.classify, proto.CMD_HOLD)
            seconds += out.seconds
            if out.command is None:
                return Spoken(seconds, held)
            cut = out.cut_tag if out.cut_tag is not None else segments.next_index()
            if cut is None:   # held after the last sentence had all been sent: nothing is left to say
                log.info("hold after the whole speech had played; nothing to resume")
                return Spoken(seconds, held)
            t0 = time.monotonic()
            got = self.link.watch_for(proto.HOLD_MAX_SECONDS - held, self.classify, proto.CMD_RESUME, client_left)
            held += time.monotonic() - t0
            if got is tg.CLIENT_LEFT:
                self._client_left_during_hold(speaker, held)
            if got is tg.TIMEOUT:
                return Spoken(seconds, held, {
                    "command": proto.INTERRUPTION_HOLD_EXPIRED, "cut_index": cut + 1,
                    "sentence_count": len(pieces), "cut_sentence": pieces[cut],
                    "unspoken": " ".join(pieces[cut:])})
            log.info("resumed after %.1fs on hold, at sentence %d of %d", held, cut + 1, len(pieces))
            start = cut

    def _spoken_text(self, text: str) -> str:
        """`text` without markdown; raises ValueError when nothing is left to say. Checked
        before a switch announcement too, so an empty speak announces nothing."""
        text = self.plain(text)
        if not text:
            raise ValueError("speak() called with empty text")
        return text

    def _client_left_during_hold(self, speaker: Speaker, held: float) -> None:
        """The request's client went away while the speech was held (its tool call was cancelled
        or its session closed): nobody waits for the reply, so the hold ends now and the lock
        goes to the next request (the handler releases it when this raises). Owner decision
        2026-10-16: the call is NOT hung up; it stays connected. Owner decision 2026-10-20:
        the floor is released at once (FLOOR_CLIENT_LEFT), so the head of the queue goes."""
        self.lock.yield_floor(speaker, FLOOR_CLIENT_LEFT)
        log.warning("the client left during a hold (held %.1fs); ending the hold, the call stays up", held)
        raise RuntimeError(f"the client left during a hold after {held:.0f}s; the hold was ended")

    def _hold_expired(self, sess: Session, tool: str, spoken: Spoken, reply: dict) -> dict:
        """Owner decision 2026-10-16: a hold that reaches HOLD_MAX_SECONDS ends the call. The
        reply says so and carries the unspoken text; the session expects no call afterwards
        (this is not a drop to call back on). If the hang-up itself fails, that failure is
        recorded (hold_expire_hangup_failed) and raised, and the call may still be up. Either
        way the floor is released (FLOOR_HOLD_EXPIRED, owner decision 2026-10-20)."""
        log.info("%s: hold expired after %.0fs; hanging up", tool, spoken.held_seconds)
        try:
            self.link.hang_up()
        except Exception as e:
            self.link.record_event("hold_expire_hangup_failed", tool=tool, error=tg.describe(e))
            raise
        finally:
            sess.expect(None)
            self.lock.yield_floor(sess.speaker, FLOOR_HOLD_EXPIRED)
        return {**reply, proto.REPLY_HELD_SECONDS: round(spoken.held_seconds, 1),
                proto.REPLY_INTERRUPTION: spoken.interruption}

    def _listen(self, max_seconds: float, silence_seconds: float, start_timeout_seconds: float) -> str:
        """The ear-open beep, the capture, Whisper, then the rising two-note chime.

        Owner decisions 2026-10-16: no ear-closed beep and no spoken ack; the rising chime
        is back, played once Whisper has the transcript, so the owner hears "you're done, I
        heard you" (it is distinct from the ear-open beep). Whisper never overlaps outgoing
        audio: record() returns only after the ear-open beep has played and the capture is
        done, nothing is queued before Whisper finishes, and the chime is queued only after.
        A call that ends right after the capture still has its words transcribed: the chime's
        failure on that dropped call is skipped (_after_capture) and guard_call then raises
        "call ended" with the transcript."""
        audio = self.link.record(max_seconds, silence_seconds, start_timeout_seconds,
                                 cue_pcm(880.0, 0.5, 0.5, 0.2), TTS_RATE)
        if audio is tg.TIMEOUT:
            self._cue(330.0)
            raise NoSpeech(f"no speech detected within {start_timeout_seconds:.0f}s")
        t0 = time.time()
        text = self.engines.transcribe(audio)
        log.info("listen: %.1fs of audio transcribed in %.1fs: %r", len(audio) / STT_RATE, time.time() - t0, text)
        if not text:
            raise RuntimeError(f"speech was detected ({len(audio) / STT_RATE:.1f}s) but Whisper returned no text")
        self._after_capture(lambda: self.link.play(chime_pcm(), TTS_RATE))   # captured
        return text

    # -- tool ops (lock held)

    def speak(self, sess: Session, text: str, client_left: Callable[[], bool]) -> dict:
        self.require_call(sess, "speak")
        self._spoken_text(text)
        self._take_turn(sess, "speak")
        # Owner decision 2026-10-20: a speak gives up the floor when it ends, however it ends,
        # except when the call ended under it: then the floor stays with this session for the
        # idle window, so it can call back first.
        call_ended = False
        try:
            spoken = self.guard_call(sess, "speak", lambda: self._speak(sess.speaker, text, client_left))
            reply = {"seconds": round(spoken.seconds, 3)}
            if spoken.interruption is not None:
                return self._hold_expired(sess, "speak", spoken, reply)
            return {**reply, proto.REPLY_HELD_SECONDS: round(spoken.held_seconds, 1)}
        except CallEndedError:
            call_ended = True
            raise
        finally:
            if not call_ended:
                self.lock.yield_floor(sess.speaker, FLOOR_SPEAK)

    def listen(self, sess: Session, max_seconds: float, silence_seconds: float, start_timeout_seconds: float) -> dict:
        self.require_call(sess, "listen")
        self._take_turn(sess, "listen")
        return {"text": self.guard_call(sess, "listen",
                                        lambda: self._listen(max_seconds, silence_seconds, start_timeout_seconds))}

    def converse(self, sess: Session, text: str, max_seconds: float, silence_seconds: float,
                 start_timeout_seconds: float, client_left: Callable[[], bool]) -> dict:
        self.require_call(sess, "converse")
        self._spoken_text(text)
        self._take_turn(sess, "converse")

        box: dict[str, Spoken] = {}

        def round_trip() -> str | None:
            box["spoken"] = spoken = self._speak(sess.speaker, text, client_left)
            if spoken.interruption is not None:
                return None   # owner decision 2026-10-16: no listen after an expired hold
            return self._listen(max_seconds, silence_seconds, start_timeout_seconds)

        heard = self.guard_call(sess, "converse", round_trip)
        spoken = box["spoken"]
        if spoken.interruption is not None:
            return self._hold_expired(sess, "converse", spoken, {})
        return {"text": heard, proto.REPLY_HELD_SECONDS: round(spoken.held_seconds, 1)}

    def call(self, sess: Session, greeting: str, override_quiet_hours: bool, max_seconds: float,
             silence_seconds: float, start_timeout_seconds: float) -> dict:
        # Checked under the lock: a call queued behind a long converse is judged when it would dial.
        # A connected call first (owner decision 2026-10-23): the owner is on the line, so a
        # session coming back with a result gets "already connected" whatever the time.
        info = self.link.info()
        if info["state"] == tg.CONNECTED:
            sess.expect(self.key(info["generation"]))
            return {"outcome": tg.ALREADY_CONNECTED}
        # Local time in the required TZ (America/Los_Angeles, compose environment).
        if not override_quiet_hours and in_quiet_hours(self.quiet_window, datetime.datetime.now(self.tz).time()):
            self.link.record_event(CALL_REFUSED_EVENT, project=sess.speaker.project, reason=REFUSED_QUIET_HOURS)
            return {"outcome": f"not called: quiet hours ({self.quiet_hours})"}
        # Last of the pre-dial checks, so already connected / quiet hours keep their outcomes.
        # Measured on the synthesized audio, not guessed from word count.
        text = self.plain(greeting)
        if not text:
            self.link.record_event(CALL_REFUSED_EVENT, project=sess.speaker.project, reason=REFUSED_EMPTY_GREETING)
            raise ValueError("call() needs a non-empty greeting")
        audio = self.engines.synthesize(text, self._voice(sess.speaker.project))   # the caller's persona
        seconds = len(audio) / TTS_RATE
        if seconds < tg.OPENER_MIN_SECONDS:
            self.link.record_event(CALL_REFUSED_EVENT, project=sess.speaker.project, reason=REFUSED_SHORT_GREETING,
                                   greeting_s=round(seconds, 1))
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
        # The greeting is this call's first turn: the caller's next speak is not a switch.
        self._turn = (self.key(self.link.info()["generation"]), sess.speaker)
        self.lock.take_floor(sess.speaker)   # owner decision 2026-10-20: an answered call gives the caller the floor
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
            self.lock.yield_floor(sess.speaker, FLOOR_HANG_UP)   # a deliberate end (owner decision 2026-10-20)


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


def _speaker(req: dict) -> Speaker:
    """The client's identity on a tool op (protocol 3). Missing, not a string or empty is
    refused like a missing protocol version; so is a project with no letter or digit (it
    would announce nothing)."""
    values = {}
    for key in (proto.FIELD_SESSION_ID, proto.FIELD_PROJECT):
        value = req.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"speak client sent no {key} (got {value!r}); every tool op of protocol "
                             f"{proto.PROTOCOL_VERSION} carries a non-empty {proto.FIELD_SESSION_ID} and "
                             f"{proto.FIELD_PROJECT}; update the client (git pull) and reconnect /mcp")
        values[key] = value
    if not proto.speakable_name(values[proto.FIELD_PROJECT]):
        raise ValueError(f"{proto.FIELD_PROJECT} {values[proto.FIELD_PROJECT]!r} has no letters or digits to speak")
    return Speaker(values[proto.FIELD_SESSION_ID], values[proto.FIELD_PROJECT])


def _persona_seen(req: dict) -> str | None:
    """FIELD_PERSONA_SEEN on a tool op (protocol 4): required, null or a string."""
    if proto.FIELD_PERSONA_SEEN not in req:
        raise ValueError(f"speak client sent no {proto.FIELD_PERSONA_SEEN}; every tool op of protocol "
                         f"{proto.PROTOCOL_VERSION} carries it; update the client (git pull) and reconnect /mcp")
    value = req[proto.FIELD_PERSONA_SEEN]
    if value is not None and not isinstance(value, str):
        raise ValueError(f"{proto.FIELD_PERSONA_SEEN} must be a string or null, got {value!r}")
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
    reply = {"ok": False, "error": message, "kind": kind}
    if isinstance(e, PersonaRefused):
        reply[proto.REPLY_PERSONA] = e.block
    return reply


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
        if op not in proto.VERSIONLESS_OPS and (op in proto.QUICK_OPS or op in proto.LOCKED_OPS):
            try:
                daemon.admit(op, req)
            except Exception as e:   # a refusal, or a failed persona write: sent, never swallowed
                log.warning("%s from %s refused: %s", op, peer, e)
                self._send(peer, _error_reply(e))
                return
        if op in proto.QUICK_OPS:
            try:
                reply = {"ok": True, proto.FIELD_PROTOCOL: proto.PROTOCOL_VERSION, **daemon.quick(op, req)}
            except Exception as e:
                reply = _error_reply(e)
            self._send(peer, reply)
            return
        if op not in proto.LOCKED_OPS:
            self._send(peer, _error_reply(ValueError(f"unknown op {op!r}")))
            return
        restart_reason: str | None = None
        daemon.lock.acquire(op, peer, _speaker(req))   # validated above; the floor orders the queue by it
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
                reply = daemon.run_locked(op, req, lambda: _peer_left(self.request))
                if op == proto.OP_CALL and reply.get("ok") and proto.is_audio_failure(reply["outcome"]):
                    restart_reason = reply["outcome"]
                    daemon.closing.set()   # before the release: no queued op reaches the link
                    daemon.lock.close()    # and every queued op reaches its refusal at once
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


def _check_protocol(req: dict) -> None:
    """Raises ValueError unless the request carries this daemon's PROTOCOL_VERSION: an older
    or newer client would misread the replies (docs: CLAUDE.md "Deploying a protocol change")."""
    if proto.FIELD_PROTOCOL not in req:
        raise ValueError(f"speak client sent no protocol version (a client older than protocol "
                         f"{proto.PROTOCOL_VERSION}); update the client (git pull) and reconnect /mcp")
    version = req[proto.FIELD_PROTOCOL]
    if version != proto.PROTOCOL_VERSION:
        raise ValueError(f"speak client protocol {version} does not match daemon protocol "
                         f"{proto.PROTOCOL_VERSION}; update the client (git pull) and reconnect /mcp")


def _peer_left(sock: socket.socket) -> bool:
    """True when the client closed its end while its request was queued or held. A client sends one
    request and then only reads, so a readable socket here means EOF, a reset, or a socket
    error such as keepalive's ETIMEDOUT or EHOSTUNREACH: in every case the peer is gone."""
    readable, _, _ = select.select([sock], [], [], 0)
    if not readable:
        return False
    try:
        return sock.recv(1, socket.MSG_PEEK) == b""
    except OSError:
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
                if size < 0:   # int() accepts "-1", and read(-1) would read to EOF past the limit
                    raise _BadRequest(f"bad chunk size line {line[:40]!r}")
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

    def __init__(self, link, engines: Engines, personas: PersonaStore, settings: dict, debug: DebugControl,
                 port: int = proto.DAEMON_PORT, on_restart: Callable[[str], None] | None = None,
                 http_port: int = WHISPER_HTTP_PORT) -> None:
        self.link = link
        self.personas = personas
        self.lock = AudioLock(link.record_event)
        self.service = SpeakService(link, engines, personas, settings, self.lock)
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
        self.lock.acquire(TRANSCRIBE_TOOL, peer, None)   # no session: skips the floor (owner decision 8)
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

    def admit(self, op: str, req: dict) -> None:
        """Before the lock, for every versioned op: the protocol version; the identity (tool
        ops and persona ops); persona_seen (tool ops); the automatic assignment of a new
        project to the house persona (owner decision 12); and the refusal of a speak, converse
        or call whose persona_seen is not the project's current "<id>:<words_revision>"
        (PersonaRefused: nothing is spoken or dialed). A voice or speed change does not change
        that key, so it refuses nobody (Design 2026-10-22). listen and hang_up carry no words and are never refused
        for it. Raises; the handler sends the error."""
        _check_protocol(req)
        if op not in proto.LOCKED_OPS and op not in proto.PERSONA_OPS:
            return
        speaker = _speaker(req)
        seen = _persona_seen(req) if op in proto.LOCKED_OPS else None
        self.personas.ensure(speaker.project)
        if op not in proto.VOICED_OPS:
            return
        current = self.personas.key(speaker.project)
        if seen != current:
            block = self.personas.block(speaker.project)
            self.link.record_event(PERSONA_REFUSED_EVENT, tool=op, project=speaker.project, seen=seen,
                                   current=current)
            raise PersonaRefused(f"your persona is {block['name']} and you have not seen it yet "
                                 f"(persona_seen {seen!r}, current {current!r})", block)

    def quick(self, op: str, req: dict) -> dict:
        link = self.link
        if op in proto.PERSONA_OPS:
            project = _speaker(req).project   # validated in admit
            if op == proto.OP_PERSONA:
                return self.personas.overview(project)
            if op == proto.OP_PERSONA_NEW:
                return {proto.REPLY_PERSONA: self.personas.new(project)}
            if op == proto.OP_PERSONA_SET:
                return {proto.REPLY_PERSONA: self.personas.set(
                    project, name=req.get("name"), voice=req.get("voice"), speed=req.get("speed"),
                    personality=req.get("personality"))}
            return {proto.REPLY_PERSONA: self.personas.add(project, req.get("note"))}
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

    def run_locked(self, op: str, req: dict, client_left: Callable[[], bool]) -> dict:
        """One tool op with the lock held. The reply carries "expect" when the session's
        call expectation changed (an int, or null to clear), REPLY_SWITCHED_FROM when a
        switch of session was announced before the op's audio, and REPLY_PERSONA when the
        project's "<id>:<words_revision>" is not the one the request's persona_seen names
        (another persona, or its name or personality changed while the op was queued or
        playing). `client_left` says whether
        the request's client has gone away; a hold polls it (speak, converse)."""
        svc = self.service
        sess: Session | None = None
        try:
            if op == proto.OP_HANG_UP:
                sess = Session(None, _speaker(req))
                result = svc.hang_up(sess)
            else:
                sess = Session(_generation(req), _speaker(req))
                if op == proto.OP_SPEAK:
                    result = svc.speak(sess, _text(req, "text"), client_left)
                elif op == proto.OP_LISTEN:
                    result = svc.listen(sess, *_listen_args(req))
                elif op == proto.OP_CONVERSE:
                    result = svc.converse(sess, _text(req, "text"), *_listen_args(req), client_left)
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
        if sess is not None and sess.switched_from is not None:
            reply[proto.REPLY_SWITCHED_FROM] = sess.switched_from   # on a failure too: the owner heard it
        project = _speaker(req).project   # validated in admit, as is persona_seen
        if _persona_seen(req) != self.personas.key(project):
            reply[proto.REPLY_PERSONA] = self.personas.block(project)
        return reply

    def request_restart(self, reason: str) -> None:
        if self.on_restart is None:
            raise RuntimeError(f"daemon restart requested ({reason}) but no on_restart handler is set")
        log.error("restarting the daemon (exit 1, compose's restart policy restarts it): %s", reason)
        self.on_restart(reason)

    def stop(self) -> None:
        self.closing.set()
        self.lock.close()
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
        # Startup order (docs/DESIGN_REMOTE_DAEMON.md 6.2, docs/DESIGN_PERSONAS.md 4.2): the personas directory, engines (with the voice
        # list), the persona file (validated against that list), then Silero + Telegram
        # (link.start), then the binds.
        PersonaStore.check_dir(settings["SPEAK_PERSONAS_FILE"])
        engines = Engines(settings["SPEAK_KOKORO_MODEL"], settings["SPEAK_WHISPER_MODEL"], settings["SPEAK_LANGUAGE"])
        engines.load(settings["SPEAK_VOICE"], settings["SPEAK_SPEED"])
        link = build_link(telegram_env)
        personas = PersonaStore.open(settings["SPEAK_PERSONAS_FILE"], engines.voices, settings["SPEAK_VOICE"],
                                     settings["SPEAK_SPEED"], settings["tz"], link.record_event)
        daemon = SpeakDaemon(link, engines, personas, settings, debug, on_restart=restart)
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
