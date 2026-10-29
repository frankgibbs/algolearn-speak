"""algolearn-speak: a Telegram voice for Claude Code. The thin MCP client.

MCP server (stdio), one per Claude Code session on any LAN computer. It holds no
models and no audio: every tool is one request over TCP to the speak daemon on
patrick (speak_telegram_daemon.py, docs/DESIGN_REMOTE_DAEMON.md), named by
`SPEAK_DAEMON` (`host:port`, required, no default). All audio goes over a
Telegram call placed by that daemon; there is no local microphone or speaker.

  speak(text)     -> the daemon synthesises with Kokoro and plays into the call
  listen(...)     -> the daemon captures the owner's reply (Silero VAD) and
                     transcribes it with Whisper; returns the text
  converse(text)  -> speak, then listen under one lock hold: one call per turn
                     (speak and converse can be held by the owner: "hold on" /
                     "continue", docs/DESIGN_INTERRUPT.md)
  call(greeting)  -> Telegram voice call to the owner; speak/listen/converse need it
                     connected (docs/DESIGN_TELEGRAM_CALL.md)
  hang_up()       -> end that call
  persona()       -> this project's persona (name, voice, speed, personality),
                     every persona, the voice ids and speed range
  new_persona(), set_persona(...), add_to_personality(note)
                  -> change this project's persona (docs/DESIGN_PERSONAS.md)
  status()        -> whether the tools are busy, how many requests are queued,
                     and the call state
  call_debug()    -> the daemon's recent call events and log tail
  set_call_debug()-> toggle the daemon's DEBUG logging at runtime

Imports only mcp, anyio, the standard library and speak_protocol, so a client
computer needs no numpy, torch or ntgcalls.

The one thing this process remembers is the session's call expectation
(design section 4.4): the generation of the call this session talked on. It is
sent with every tool op, and the daemon's reply says how it changes, so a later
drop raises "call ended ... call back" in every session that used the call.

Every tool op also carries this client's identity (protocol 3, owner decision
2026-10-20): a session id generated once per process and the project, the
basename of the working folder at startup. The daemon uses it to announce a
switch of session on the shared call; a result that follows a switch starts
with a "[took over the call from ...]" line.

Personas (protocol 4, owner decisions 2026-10-22, docs/DESIGN_PERSONAS.md): the
project speaks as a persona kept by the daemon. This process also remembers
the "<id>:<words_revision>" of the persona it last saw (`_persona_seen`) and
sends it with every tool op; a reply that carries a new persona, or one whose
name or personality changed, puts a "[persona: ...]" line in the result, and a
speak, converse or call sent before the session saw its persona's current name
and personality is refused (nothing spoken or dialed) with the persona in the
error. A voice or speed change never refuses a turn: the daemon applies it per
sentence.
"""

from __future__ import annotations

import contextvars
import logging
import os
import secrets
import socket
import sys
import threading
import time

import anyio
from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

import speak_protocol as proto
from speak_protocol import AUDIO_NOTE

log = logging.getLogger("speak")

CONNECT_TIMEOUT_SECONDS = 5.0
QUICK_REPLY_TIMEOUT_SECONDS = 20.0
MAX_DEBUG_LINES = 1000
# Extra budget after "started" for a session-switch announcement (synthesis, the cue and the
# spoken "<persona name>, <project>"), plus (longest name + 2 + len(project)) / 5 s as the
# other budgets count text: a fixed bound, whether or not the persona has been seen yet.
ANNOUNCE_BUDGET_SECONDS = 10.0

_daemon: tuple[str, int] | None = None   # set in main() from SPEAK_DAEMON
# This client's identity on every tool op: {FIELD_SESSION_ID, FIELD_PROJECT}. Set once per
# process by init_identity() (main(), or scripts/drive_daemon.py).
_identity: dict[str, str] | None = None

# The session's call expectation: the daemon's generation of the call this session
# expects, or None. Updated only from a reply's "expect" key.
_expect_lock = threading.Lock()
_call_generation: str | None = None   # opaque "<daemon boot>:<call generation>"
# The "<persona id>:<words_revision>" of this project's persona as last reported by the daemon, or
# None before any report. Sent as FIELD_PERSONA_SEEN on every tool op; updated from any reply
# (or error) that carries REPLY_PERSONA.
_persona_seen: str | None = None


class DaemonUnreachable(RuntimeError):
    """The TCP connect to the daemon failed: it is not running, not ready yet, or not reachable."""


class _Abort:
    """Lets a cancelled tool call close its request's socket from the event loop, so the
    daemon sees the client leave (a held speak/converse then ends and frees the lock).
    The worker thread attaches the socket for the life of the request."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sock: socket.socket | None = None
        self.cancelled = False

    def attach(self, sock: socket.socket) -> None:
        with self._lock:
            if self.cancelled:
                raise RuntimeError("the tool call was cancelled before its request was sent")
            self._sock = sock

    def detach(self) -> None:
        with self._lock:
            self._sock = None

    def cancel(self) -> None:
        with self._lock:
            self.cancelled = True
            if self._sock is None:
                return
            try:
                self._sock.shutdown(socket.SHUT_RDWR)
            except OSError as e:   # already closed by the peer: the cancellation itself stands
                log.warning("closing the cancelled request's socket failed: %s", e)


# The running tool call's _Abort; None when a _sync function is driven directly
# (scripts/drive_daemon.py), where there is no cancellation to forward.
_abort_var: contextvars.ContextVar[_Abort | None] = contextvars.ContextVar("speak_abort", default=None)


def _address() -> str:
    if _daemon is None:
        raise RuntimeError("SPEAK_DAEMON was not read; main() sets it at startup")
    return f"{_daemon[0]}:{_daemon[1]}"


def init_identity(cwd: str) -> dict[str, str]:
    """Generate this process's session id and take the project from `cwd` (the working
    folder at startup). A folder whose name has no letter or digit (e.g. "/") is refused:
    there would be nothing to announce."""
    global _identity
    project = os.path.basename(os.path.normpath(cwd))
    if not proto.speakable_name(project):
        raise RuntimeError(f"the speak client's working folder {cwd!r} has no name to announce (its basename "
                           f"{project!r} has no letters or digits); start Claude Code in a project folder")
    _identity = {proto.FIELD_SESSION_ID: secrets.token_hex(8), proto.FIELD_PROJECT: project}
    return _identity


def _identity_fields() -> dict[str, str]:
    if _identity is None:
        raise RuntimeError("the speak client's identity was not set; main() calls init_identity at startup")
    return _identity


def _apply_expect(reply: dict) -> None:
    global _call_generation
    if "expect" in reply:
        with _expect_lock:
            _call_generation = reply["expect"]


def _current_generation() -> str | None:
    with _expect_lock:
        return _call_generation


def _apply_persona(reply: dict) -> None:
    """Move `_persona_seen` to the reply's persona block. A refusal for a stale persona (an error
    reply of kind value with a block) states the daemon's current key, so it applies
    unconditionally, even when it is lower than the key seen. An ordinary reply moves only
    forward: a different persona id always applies; the same id only at a higher
    words_revision, so a reply that arrives after a newer one (two requests in flight) never
    moves it back."""
    global _persona_seen
    if proto.REPLY_PERSONA not in reply:
        return
    block = reply[proto.REPLY_PERSONA]
    refused = not reply.get("ok") and reply.get("kind") == proto.KIND_VALUE
    with _expect_lock:
        if _persona_seen is not None and not refused:
            seen_id, seen_revision = _persona_seen.rsplit(":", 1)
            if block["id"] == seen_id and block["words_revision"] <= int(seen_revision):
                return
        _persona_seen = proto.persona_key(block)


def _current_persona_seen() -> str | None:
    with _expect_lock:
        return _persona_seen


def _request(op: str, fields: dict, budget: float) -> dict:
    """One request on its own connection. Tool ops wait for the daemon's "started" with no
    timeout (the queue never rejects), then for the final reply within `budget` seconds;
    quick ops get one reply within `budget`. A failure reply is raised as the error its kind
    names; a stalled daemon raises RuntimeError, never TimeoutError, so it is never mistaken
    for "no speech detected"."""
    addr = _address()
    if op in proto.LOCKED_OPS:
        fields = {**_identity_fields(), proto.FIELD_PERSONA_SEEN: _current_persona_seen(), **fields}
    elif op in proto.PERSONA_OPS:
        fields = {**_identity_fields(), **fields}
    started = False
    try:
        sock = socket.create_connection(_daemon, timeout=CONNECT_TIMEOUT_SECONDS)
    except OSError as e:
        raise DaemonUnreachable(f"speak daemon at {addr} is not reachable: {e}") from e
    abort = _abort_var.get()
    with sock:
        if abort is not None:
            abort.attach(sock)
        try:
            proto.enable_keepalive(sock)
            proto.send_msg(sock, {"op": op, proto.FIELD_PROTOCOL: proto.PROTOCOL_VERSION, **fields})
            if op in proto.LOCKED_OPS:
                sock.settimeout(None)
                reply, _ = proto.recv_msg(sock)
                if reply.get("ok"):
                    if reply.get("phase") != proto.PHASE_STARTED:
                        raise RuntimeError(f"speak daemon at {addr} sent {reply!r} for {op!r} "
                                           "where it must send 'started' or an error")
                    started = True
                    reply, _ = proto.recv_msg(sock, deadline=time.monotonic() + budget)
            else:
                reply, _ = proto.recv_msg(sock, deadline=time.monotonic() + budget)
        except TimeoutError:
            raise RuntimeError(f"speak daemon at {addr} did not reply to {op!r} within {budget:.0f}s") from None
        except (OSError, ValueError) as e:
            raise RuntimeError(f"speak daemon at {addr} failed during {op!r}: {type(e).__name__}: {e}") from e
        finally:
            if abort is not None:
                abort.detach()
    _apply_expect(reply)
    _apply_persona(reply)
    if not reply.get("ok"):
        message = reply.get("error") or f"speak daemon reported an error without a message: {reply!r}"
        kind = reply.get("kind")
        if op in proto.PERSONA_OPS and message == f"unknown op {op!r}":
            # A daemon older than protocol 4 does not know the persona ops (and sends no version).
            message = (f"protocol mismatch: speak daemon at {addr} does not know {op!r}, so it is older than "
                       f"this client's protocol {proto.PROTOCOL_VERSION}; deploy the daemon, or use the client "
                       "that matches it (git pull) and reconnect /mcp")
        elif op in proto.VOICED_OPS and not started and kind == proto.KIND_VALUE and proto.REPLY_PERSONA in reply:
            # Refused before the lock: this session had not seen its current persona.
            message = (f"{_persona_line(reply)}Nothing was spoken (or dialed). Rewrite your words as "
                       f"{reply[proto.REPLY_PERSONA]['name']} and send again.")
        else:   # the owner heard the announcement before the failure; a persona change is news either way
            if proto.REPLY_CALL_ENDED in reply:   # only a speak that played through ends the call
                raise RuntimeError(f"speak daemon sent call_ended {reply[proto.REPLY_CALL_ENDED]!r} "
                                   f"on an error reply; its error: {message}")
            message = f"{_switch_line(reply)}{_persona_line(reply)}{message}"
        if kind == proto.KIND_TIMEOUT:
            raise TimeoutError(message)
        if kind == proto.KIND_VALUE:
            raise ValueError(message)
        raise RuntimeError(message)
    return reply


# ---------------------------------------------------------------- budgets after "started"

def _speak_budget(text: str) -> float:
    """Synthesis and playback, plus the longest the owner may hold the speech (the holds of one
    op add up to at most HOLD_MAX_SECONDS; then the daemon ends the call and replies)."""
    return 60.0 + len(text) / 5 + proto.HOLD_MAX_SECONDS


def _listen_budget(max_seconds: float, start_timeout_seconds: float) -> float:
    return start_timeout_seconds + max_seconds + 60.0


def _announce_budget() -> float:
    return ANNOUNCE_BUDGET_SECONDS + (proto.PERSONA_NAME_MAX + 2 + len(_identity_fields()[proto.FIELD_PROJECT])) / 5


def _listen_fields(max_seconds: float, silence_seconds: float, start_timeout_seconds: float) -> dict:
    return {"max_seconds": max_seconds, "silence_seconds": silence_seconds,
            "start_timeout_seconds": start_timeout_seconds}


# ---------------------------------------------------------------- sync bodies (worker thread)

def _switch_line(reply: dict) -> str:
    """'[took over the call from "<project>"; ...]\n' when the daemon announced a switch of
    session before this op's audio (the previous audio on the call came from another
    session); "" otherwise. Always the result's first line."""
    if proto.REPLY_SWITCHED_FROM not in reply:
        return ""
    return (f"[took over the call from \"{reply[proto.REPLY_SWITCHED_FROM]}\"; the owner heard the switch "
            f"cue and your persona's name and this project's name first]\n")


def _persona_line(reply: dict) -> str:
    """'[persona: you are <Name>; voice ...; speed ...; personality: ...]\n' when the reply
    carries this project's persona (new to this session, or its name or personality changed);
    "" otherwise. On the house persona it says "you are <Name>, the house persona", so the agent
    knows not to add to its personality on its own. Comes after the switch line and before the
    held line."""
    if proto.REPLY_PERSONA not in reply:
        return ""
    p = reply[proto.REPLY_PERSONA]
    voice = ", ".join(f"{vid} {weight:g}" for vid, weight in p["voice"].items())
    personality = " / ".join(p["personality"]) if p["personality"] else "(none yet)"
    shared = f"; shared with {', '.join(p['shared_with'])}" if p["shared_with"] else ""
    house = ", the house persona" if p["house"] else ""
    return f"[persona: you are {p['name']}{house}; voice {voice}; speed {p['speed']:g}; personality: {personality}{shared}]\n"


def _ended_line(reply: dict) -> str:
    """'\n[call ended: nobody else was waiting]' when the daemon ended the call after this
    speak, the sign-off, because no other session was queued (owner decision 2026-10-28); ""
    otherwise. It follows the result's own text, before the audio note. An error reply never
    carries the field (_request raises if one does, with the daemon's error)."""
    if proto.REPLY_CALL_ENDED not in reply:
        return ""
    if reply[proto.REPLY_CALL_ENDED] != proto.CALL_ENDED_SIGNOFF:
        raise RuntimeError(f"speak daemon sent an unknown call_ended {reply[proto.REPLY_CALL_ENDED]!r} "
                           f"on a speak that played for {reply['seconds']:.1f}s")
    return "\n[call ended: nobody else was waiting]"


def _lead(reply: dict) -> str:
    """The lines every tool result starts with: the switch line, then the persona line."""
    return _switch_line(reply) + _persona_line(reply)


def _held_prefix(reply: dict) -> str:
    """"[held 42s, then resumed]\n" when the owner held the speech and it finished; "" otherwise."""
    held = reply[proto.REPLY_HELD_SECONDS]
    return f"[held {held:.0f}s, then resumed]\n" if held > 0 else ""


def _expired_result(reply: dict) -> str:
    """The result of a speak/converse whose hold expired (the daemon ended the call)."""
    i = reply[proto.REPLY_INTERRUPTION]
    if i["command"] != proto.INTERRUPTION_HOLD_EXPIRED:
        raise RuntimeError(f"speak daemon sent an unknown interruption: {i!r}")
    return (f"[hold expired after {reply[proto.REPLY_HELD_SECONDS]:.0f}s; the call was ended]\n"
            f"cut off in sentence {i['cut_index']} of {i['sentence_count']}: \"{i['cut_sentence']}\"\n"
            f"not spoken (from the cut sentence): \"{i['unspoken']}\"{AUDIO_NOTE}")


def _speak_result(reply: dict) -> str:
    if proto.REPLY_INTERRUPTION in reply:
        return _lead(reply) + _expired_result(reply)
    return f"{_lead(reply)}{_held_prefix(reply)}spoke for {reply['seconds']:.1f}s{_ended_line(reply)}{AUDIO_NOTE}"


def _converse_result(reply: dict) -> str:
    if proto.REPLY_INTERRUPTION in reply:
        return _lead(reply) + _expired_result(reply)
    return f"{_lead(reply)}{_held_prefix(reply)}{reply['text']}{AUDIO_NOTE}"


def _listen_result(reply: dict) -> str:
    return f"{_lead(reply)}{reply['text']}"


def _speak_sync(text: str) -> str:
    reply = _request(proto.OP_SPEAK, {"text": text, "expect_generation": _current_generation()},
                     _speak_budget(text) + _announce_budget())
    return _speak_result(reply)


def _listen_sync(max_seconds: float, silence_seconds: float, start_timeout_seconds: float) -> str:
    reply = _request(proto.OP_LISTEN,
                     {**_listen_fields(max_seconds, silence_seconds, start_timeout_seconds),
                      "expect_generation": _current_generation()},
                     _listen_budget(max_seconds, start_timeout_seconds) + _announce_budget())
    return _listen_result(reply)


def _converse_sync(text: str, max_seconds: float, silence_seconds: float, start_timeout_seconds: float) -> str:
    reply = _request(proto.OP_CONVERSE,
                     {"text": text, **_listen_fields(max_seconds, silence_seconds, start_timeout_seconds),
                      "expect_generation": _current_generation()},
                     _speak_budget(text) + _listen_budget(max_seconds, start_timeout_seconds) + _announce_budget())
    return _converse_result(reply)


def _call_sync(greeting: str, override_quiet_hours: bool, max_seconds: float, silence_seconds: float,
               start_timeout_seconds: float) -> str:
    budget = (proto.REDIAL_SETTLE_SECONDS + proto.RING_TIMEOUT_SECONDS + proto.CONNECT_GRACE_SECONDS + len(greeting) / 5
              + _listen_budget(max_seconds, start_timeout_seconds))
    reply = _request(proto.OP_CALL,
                     {"greeting": greeting, "override_quiet_hours": override_quiet_hours,
                      **_listen_fields(max_seconds, silence_seconds, start_timeout_seconds),
                      "expect_generation": _current_generation()},
                     budget)
    if reply["outcome"] == proto.ANSWERED:
        return f"{_persona_line(reply)}{proto.ANSWERED}\n{reply['text']}{AUDIO_NOTE}"
    return f"{_persona_line(reply)}{reply['outcome']}"


def _hang_up_sync() -> str:
    global _call_generation
    try:
        reply = _request(proto.OP_HANG_UP, {}, QUICK_REPLY_TIMEOUT_SECONDS)
        return f"{_persona_line(reply)}{reply['outcome']}"
    finally:
        with _expect_lock:
            _call_generation = None   # a deliberate end (the owner asked): never a call-back, even if this raised


def _strip(reply: dict) -> dict:
    return {k: v for k, v in reply.items() if k not in ("ok", "payload_len")}


def _call_debug_sync(lines: int) -> dict:
    if isinstance(lines, bool) or not isinstance(lines, int) or not 1 <= lines <= MAX_DEBUG_LINES:
        raise ValueError(f"lines must be an integer from 1 to {MAX_DEBUG_LINES}, got {lines!r}")
    try:
        return _strip(_request(proto.OP_DEBUG, {"lines": lines}, QUICK_REPLY_TIMEOUT_SECONDS))
    except DaemonUnreachable as e:
        return {"daemon": f"unreachable: {e}"}


def _set_call_debug_sync(enabled: bool) -> dict:
    if not isinstance(enabled, bool):
        raise ValueError(f"enabled must be a boolean, got {enabled!r}")
    try:
        return _strip(_request(proto.OP_SET_DEBUG, {"enabled": enabled}, QUICK_REPLY_TIMEOUT_SECONDS))
    except DaemonUnreachable as e:
        return {"daemon": f"unreachable: {e}"}


def _persona_sync() -> dict:
    return _strip(_request(proto.OP_PERSONA, {}, QUICK_REPLY_TIMEOUT_SECONDS))


def _new_persona_sync() -> dict:
    return _strip(_request(proto.OP_PERSONA_NEW, {}, QUICK_REPLY_TIMEOUT_SECONDS))


def _set_persona_sync(name: str | None, voice: dict[str, float] | None, speed: float | None,
                      personality: list[str] | None) -> dict:
    return _strip(_request(proto.OP_PERSONA_SET, {"name": name, "voice": voice, "speed": speed,
                                                  "personality": personality}, QUICK_REPLY_TIMEOUT_SECONDS))


def _add_to_personality_sync(note: str) -> dict:
    return _strip(_request(proto.OP_PERSONA_ADD, {"note": note}, QUICK_REPLY_TIMEOUT_SECONDS))


def _status_sync() -> dict:
    try:
        reply = _request(proto.OP_STATUS, {}, QUICK_REPLY_TIMEOUT_SECONDS)
    except DaemonUnreachable as e:   # reported in the result, never hidden: status must not raise
        return {"busy": None, "current_tool": None, "waiting": None, "call": f"daemon unreachable: {e}"}
    except (RuntimeError, ValueError) as e:   # reached but failed: also reported in the result, and logged
        log.exception("status request failed")
        return {"busy": None, "current_tool": None, "waiting": None, "call": f"error: {e}"}
    daemon_protocol = reply.get(proto.FIELD_PROTOCOL)   # absent: a daemon older than versioning
    if daemon_protocol != proto.PROTOCOL_VERSION:
        return {"busy": None, "current_tool": None, "waiting": None,
                "call": f"protocol mismatch: speak client protocol {proto.PROTOCOL_VERSION} does not match "
                        f"daemon protocol {daemon_protocol}; update the client (git pull) and reconnect /mcp, "
                        "or deploy the daemon"}
    return {"busy": reply["busy"], "current_tool": reply["current_tool"], "waiting": reply["waiting"],
            "call": reply["call"]["state"]}


# ---------------------------------------------------------------- MCP surface

# Claude Code cuts MCP server instructions off at 4096 characters (verified 2026-10-26: the
# 6,891-character text before this one lost every floor and coming-back rule past the cut).
# Order: the floor and coming back first, then calls, persona, speaking. Details live in the
# tool docstrings, which are not cut.
INSTRUCTIONS_MAX_CHARS = 4096

_INSTRUCTIONS = """\
You talk with the owner (Frank) over one Telegram call, shared by every session on every
computer. His rules for you:

The floor (one conversation at a time):
- The session that last played audio holds the floor. converse() and listen() keep it;
  speak() gives it up. Other sessions queue until then.
- While you and the owner work the problem out (understanding it, deciding what to do),
  stay in converse().
- When you are about to go do the work, whatever its length (even two minutes), ask him
  whether he is ready to let you go. Once he says yes, sign off with ONE speak() that
  promises to come back on the call, never "by message".
- After your sign-off, do nothing more on the call: with nobody else waiting the daemon
  hangs up ("[call ended: nobody else was waiting]"), otherwise the next session goes.
- Coming back is owed, by voice. When the work is done, report it on the call before
  anything else, even after your context was summarized; a text is only in addition.
  Never ask whether he wants a call back. First status(). "connected": send ONE
  converse() with your report and leave it waiting in the queue, however long, even
  behind another session's conversation. Being backgrounded after ~120 s is normal.
  While it waits, do not cancel it, resend it, text instead or call(). If it returns
  "Telegram call ended during ..." or "needs a connected Telegram call", call() back as
  Calls says.
  "ringing": another session is dialing; status() again. "none": call().
- '[took over the call from ...]': open with one sentence recapping where you left off
  and your last question, unless his reply answers it.

Calls:
- When he asks you to call, he is up: pass override_quiet_hours=True without asking.
  Calls you start on your own respect quiet hours; they never apply while a call is
  connected.
- "not answered": wait an hour. "busy": call once more, then wait an hour. He hung up,
  or "answered, then hung up during the greeting": call back once right away; if that is
  not answered or ends the same way, wait an hour. "Telegram call ended during ...":
  call back. hang_up() only when he asks.
- Whenever you must wait (quiet hours, the hour above, an expired hold), say in this
  terminal when you will try again and start a background timer (a background sleep).

Your persona:
- Call persona() before your first call or converse. On the call you are its name, never
  Claude, with its personality. A "[persona: ...]" line: you are now what it says. A
  ValueError carrying a persona: nothing was spoken or dialed; rewrite as that persona
  and send again.
- Add one-sentence lines with add_to_personality() as you learn your role, except on
  the house persona, which changes only when he asks. Change persona, name, voice or
  speed only when he asks. "Persona review": go item by item, then
  set_persona(personality=[...]).

Speaking:
- Plain conversational sentences, no markdown, lists or code. One topic, one aspect and
  one question at a time.
- Every turn is ONE converse(text), never speak() then listen() (dead air). listen()
  alone only to reopen the ear after a TimeoutError or with nothing to say first.
- TimeoutError: nobody spoke, the call is up; listen() again.
- Go straight into the answer, never "heard you" or a restatement. Once he lets you
  go, hand the work to agents.
- "[held Ns, then resumed]": all was said. "[hold expired ...; the call was ended]":
  the reply was NOT finished and converse did not listen; keep the "not spoken" text
  and come back with it after an hour, by timer. Never speak "hold on" or "continue" as a sentence by itself.
- A call seems off (silence, drops): without asking, set_call_debug(True), place a test
  call, read call_debug(), instrument rather than retry blind; debug off once calls
  work.
"""


def check_instructions_length(text: str) -> None:
    """Raise unless `text` is shorter than INSTRUCTIONS_MAX_CHARS (at most 4095 characters)."""
    if len(text) >= INSTRUCTIONS_MAX_CHARS:
        raise RuntimeError(f"the speak server instructions are {len(text)} characters; they must be under "
                           f"{INSTRUCTIONS_MAX_CHARS} (Claude Code truncates them at {INSTRUCTIONS_MAX_CHARS}), "
                           "so shorten them (move detail into the tool docstrings)")


check_instructions_length(_INSTRUCTIONS)

mcp = MCPServer("algolearn-speak", instructions=_INSTRUCTIONS)


async def _run(fn, *args):
    """Run the request off the event loop. Expected failures (no speech, empty text, no
    call, daemon unreachable) are re-raised as ToolError so the model sees the reason;
    the SDK masks any other exception as a generic crash.

    A cancelled tool call (the user interrupts, the session closes) closes the request's
    socket and leaves the worker thread behind: the daemon sees the client leave, and a
    held speak/converse ends instead of keeping the lock."""
    abort = _Abort()
    token = _abort_var.set(abort)   # copied into the worker thread's context by anyio
    try:
        return await anyio.to_thread.run_sync(fn, *args, abandon_on_cancel=True)
    except anyio.get_cancelled_exc_class():
        abort.cancel()
        raise
    except (TimeoutError, RuntimeError, ValueError) as e:
        raise ToolError(str(e)) from e
    except Exception as e:
        log.exception("%s failed", getattr(fn, "__name__", fn))
        raise ToolError(f"{type(e).__name__}: {e}") from e
    finally:
        _abort_var.reset(token)


_QUEUE_NOTE = """
    Status: speak/listen/converse/call/hang_up are serialized through one
    lock in the speak daemon, shared by every Claude Code session on every
    computer -- there is only one Telegram call. If another tool call is in
    progress anywhere, this one queues and waits; it is never rejected. Use
    `status()` to see what is running and how many calls are waiting; `busy`
    false with `waiting` above 0 means another session holds the floor. Your
    request still queues behind it: leave it waiting.

    Switch of session: if the previous audio on the call came from another
    session (the greeting counts as the calling session's), the daemon first
    plays a double beep and speaks your persona's name and your project's name,
    and the result (or the error) starts with the line '[took over the call from
    "<other project>"; ...]'. Recap where you left off and your last question in
    one sentence in your next turn.
"""


def _doc(doc: str):
    """Set a tool's docstring before @mcp.tool() reads it (shared paragraphs are composed)."""
    def apply(fn):
        fn.__doc__ = doc
        return fn
    return apply


_HOLD_NOTE = """
    Hold: while the speech plays, the owner saying "hold on" by itself stops it at once
    and holds (silence); "continue" restarts the sentence that was cut off and finishes.
    The result then starts with "[held Ns, then resumed]" and is otherwise the normal
    result. If no "continue" comes within 300 s (holds of one call added up), the daemon
    ENDS THE CALL and the result is:
      [hold expired after 300s; the call was ended]
      cut off in sentence 3 of 5: "<the cut sentence>"
      not spoken (from the cut sentence): "<everything not said>"
      (audio: telegram)
    The reply was not finished (and converse did not listen). Do not call back right
    away; keep the unspoken text and bring it up later. Never speak "hold on" or
    "continue" as a sentence by itself.
"""


_CALL_NOTE = """
    Needs a Telegram call placed by `call()` and still connected; there is no
    local microphone or speaker. With no connected call this raises "...
    needs a connected Telegram call"; place one with `call()`. The result
    ends with `(audio: telegram)`. If the call ends mid-way this raises
    "Telegram call ended during ..."; the rule of engagement is to call back
    with `call()`. If the speak daemon cannot be reached this raises "speak
    daemon at <addr> is not reachable: ...".
"""


@mcp.tool()
@_doc("""Say `text` to the owner over the connected Telegram call and return once playback finishes.

    Write it as spoken prose: plain sentences, no markdown, no code. Returns the spoken duration.

    speak() gives up the floor. Use it only for your sign-off, once the owner has said you may
    go do the work: one speak() that promises to come back on the call. Then do nothing more
    on the call: when no other request is queued as it ends, the daemon hangs up and the
    result's last line before the audio note is "[call ended: nobody else was waiting]";
    otherwise the call stays up and the next session takes the floor. Do not call hang_up()
    or call back for it. If the sign-off raises an error other than "Telegram call ended
    during ...", the call is still up: send the speak() again. When the work is done, come
    back on the call: status() first.
""" + _HOLD_NOTE + _QUEUE_NOTE + _CALL_NOTE)
async def speak(text: str) -> str:
    return await _run(_speak_sync, text)


@mcp.tool()
@_doc("""Listen to the owner on the connected Telegram call until they finish talking and return what they said.

    A high beep means the ear is open; a rising two-note chime, played once the transcript
    is captured, tells the owner their words were heard. There is no spoken acknowledgement,
    so answer straight away. Recording ends after
    `silence_seconds` of quiet following speech, or at `max_seconds` of speech. Raises
    TimeoutError if nobody speaks within `start_timeout_seconds` (the call stays connected).

    Use it alone only to reopen the ear after a TimeoutError or when there is nothing to
    say first; a conversational turn is one `converse(text)`, never speak() then listen().
    After a TimeoutError the call is still up: listen again; do not assume the owner left
    and do not hang up.

    listen() keeps the floor.
""" + _QUEUE_NOTE + _CALL_NOTE)
async def listen(max_seconds: float = 120.0, silence_seconds: float = 2.0, start_timeout_seconds: float = 45.0) -> str:
    text = await _run(_listen_sync, max_seconds, silence_seconds, start_timeout_seconds)
    return text + AUDIO_NOTE


@mcp.tool()
@_doc("""Say `text` over the connected Telegram call, then listen for the reply. One spoken round trip; returns the owner's words.

    Every conversational turn is ONE converse call: speak and listen run under one lock
    hold, so the ear-open beep follows the last word at once (speak() then listen() leaves
    seconds of dead air). One topic, one aspect and one question per turn. Takes listen's
    parameters and raises TimeoutError as listen does if nobody speaks within
    `start_timeout_seconds`; the call is still up, so reopen the ear with `listen()`; do
    not assume the owner left and do not hang up.

    converse() keeps the floor. Coming back with a report, it may wait minutes in the queue
    behind another session; after about 120 s Claude Code moves it to the background and
    notifies you when it completes. While it waits, leave it: do not cancel it, resend it,
    text instead or call(). If it returns "Telegram call ended during ..." or "needs a
    connected Telegram call", call() back.
""" + _HOLD_NOTE + _QUEUE_NOTE + _CALL_NOTE)
async def converse(text: str, max_seconds: float = 120.0, silence_seconds: float = 2.0, start_timeout_seconds: float = 45.0) -> str:
    return await _run(_converse_sync, text, max_seconds, silence_seconds, start_timeout_seconds)


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
    "Hi Frank, it's <your persona's name>. The backtest finished a few minutes ago and the
    holdout came back clean, but there is one result I want your read on
    before I do anything else. Do you have a minute?". Works wherever their
    phone has signal; the call is placed by the speak daemon on patrick, so it
    does not depend on this computer's audio. Once answered, speak/listen/converse run
    over the call until it ends; without a connected call they raise. Returns one of:
      "answered\\n<their words>\\n(audio: telegram)"
                                              -- they picked up, heard the greeting and
                                                 replied; reply with converse (not speak
                                                 then listen). A listen timeout or a drop
                                                 raises as in listen (a timeout leaves the
                                                 call connected: reopen the ear with listen)
      "already connected"                     -- a call is already up (greeting not
                                                 spoken), at any hour; just talk
      "not answered (declined or no answer)"  -- what you have can wait; do not
                                                 call again for at least one hour
      "busy (the owner's Telegram reports a call in progress)"
                                              -- Telegram discarded the call as busy (the
                                                 owner is on another call, or their app is
                                                 still ending the previous one); the phone
                                                 did not ring. This is NOT "not answered".
                                                 Call once more (the daemon already waits
                                                 10 s after any call ends); if that one is
                                                 busy or not answered too, wait an hour.
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
                                                 that window. Never returned while a
                                                 call is connected: that is "already
                                                 connected" at any hour
      "answered but audio failed to connect: <error type/message>"
                                              -- the owner picked up but the media
                                                 connection failed (they hear beeping,
                                                 then the call drops). This is NOT
                                                 "not answered": do not wait an hour.
                                                 The call was torn down and the daemon
                                                 container restarts itself a few
                                                 seconds later with fresh state. Call
                                                 `call_debug()` to see why; once it
                                                 shows the daemon back up (small
                                                 uptime_seconds, call state "none"),
                                                 you may place ONE new call. Do not
                                                 loop.
    Quiet hours are a hard rule for placing a call. They never apply while a call is
    connected: call() then returns "already connected", whatever the time. Pass
    `override_quiet_hours=True` ONLY when the owner has said in this conversation that
    they are up and want calls.

    Rule of engagement: if speak/listen/converse raises "Telegram call ended
    during ...", the call dropped mid-conversation -- call back with this tool.
    The daemon never dials sooner than 10 s after the previous call ended,
    whatever the reason (a redial 1.4 s after a hang-up left the owner's
    Telegram app dropping every incoming call until it was force-quit); a
    call placed sooner waits out the rest of those 10 s, then dials.
    Hang up with `hang_up()` only when the owner asks; the owner can also hang
    up themselves, and the daemon hangs up after a speak() sign-off when no
    other request is queued ("[call ended: nobody else was waiting]"). If the
    speak daemon cannot be reached this raises "speak daemon at <addr> is not
    reachable: ...".

    To come back with a result, status() first; while the call is connected use converse(), not call().
    """
    return await _run(_call_sync, greeting, override_quiet_hours, max_seconds, silence_seconds, start_timeout_seconds)


_READING_EVENTS = """
    Reading the events:
    - `opener_s` (on answered_media_connected and recording) is the greeting's
      spoken length: the greeting is played as the call's opener file. It is
      NOT a delay. `raw_frames` and `connected` follow once the opener
      finishes (opener_s plus about 1 s).
    - `first_incoming_frame` only means frames arrive from the owner, not that
      they carry sound. It fires once per call, during the first listen or the
      first watched speech (`during`: listen or watch), whichever comes first.
    - With debug on, `audio_in` / `audio_out` summarize each window (every 1 s
      for the first 10 s after answer, then every 10 s; `window_s`, `phase`
      opener or raw). `audio_in` with `zero_frames == frames` and `peak` 0 is
      exact digital silence: the call's media is dead (in practice both ways).
      It is not a muted mic and not VAD.
    - Real `audio_out` levels (`audio_peak`, `audio_frames` > 0) while
      `audio_in` is dead and the owner says they hear nothing: the call's media
      path is dead. Ask the owner to hang up (or `hang_up()` if they ask), then
      place ONE new call. A listen TimeoutError after the ear-open beep with
      dead `audio_in` is the same fault, not the owner staying quiet.
"""


@mcp.tool()
@_doc("""Diagnose Telegram calls in real time: what the speak daemon just did.

    Use it during a call, or right after `call()` returned "answered but audio
    failed to connect: ..." or an unexpected "not answered". Returns
    `{"daemon": "running", "pid", "uptime_seconds", "debug": bool, "call":
    {"state", "generation", "end_reason"}, "events": [...], "log": [...]}`:
    `events` are the last `lines` structured call events (call_id, ts, event
    such as requested/ringing/busy/audio_failed/ended, elapsed_s, error),
    `log` the last `lines` daemon log lines (set_call_debug(True) adds
    py-tgcalls/Telethon DEBUG output and the audio-level events). A small
    uptime_seconds right after a failed call is the daemon's self-restart.
    When the daemon cannot be reached it returns `{"daemon": "unreachable:
    <error>"}` and nothing else. Never blocks on the audio lock. `lines` is 1
    to 1000.
""" + _READING_EVENTS)
async def call_debug(lines: int = 50) -> dict:
    return await _run(_call_debug_sync, lines)


@mcp.tool()
async def set_call_debug(enabled: bool) -> dict:
    """Turn the daemon's DEBUG logging (speak, py-tgcalls, Telethon) on or off at
    runtime, no restart. Turn it on before a call you expect to fail, read the
    result with `call_debug()`, turn it off after (it is noisy). Resets when the
    daemon restarts. Returns `{"daemon": "running", "debug": bool}` or
    `{"daemon": "unreachable: <error>"}`. Never blocks on the audio lock.
    """
    return await _run(_set_call_debug_sync, enabled)


@mcp.tool()
async def hang_up() -> str:
    """End the Telegram call placed by `call()`. Use it only when the owner asks you to.
    After your speak() sign-off the daemon itself hangs up when no other request is
    queued, so never call it for that.

    Returns "hung up" or "no active call". Waits its turn in the audio lock
    queue like the other call tools.
    """
    return await _run(_hang_up_sync)


@mcp.tool()
@_doc("""Your project's persona (name, voice blend, speed, personality, the projects sharing it), every persona, and the voice ids and speed range you may use.

    Returns `{"persona": {...}, "personas": [{"name", "house", "voice", "speed", "projects"}],
    "voices": [ids], "speed_range": [min, max], "pool_left": n}`. Never waits for the call.
""")
async def persona() -> dict:
    return await _run(_persona_sync)


@mcp.tool()
@_doc("""Give your project a brand-new persona with an unused name from the pool and the default voice; returns it.

    Raises ValueError "every pool name is in use; ask the owner for a name" when the pool
    is used up.
""")
async def new_persona() -> dict:
    return await _run(_new_persona_sync)


@mcp.tool()
@_doc("""Change your persona: name alone moves your project onto the persona of that name, or renames yours (for every project sharing it) when no persona has it; voice {id: weight} and speed change how it sounds for every project using it from the next sentence; personality [lines] replaces the note (persona review).

    `name` (one word of 2 to 12 letters) cannot be combined with the others. `voice` takes
    1 to 4 English voice ids from `persona()` with positive weights (normalized to sum 1);
    `speed` is within `persona()`'s speed_range.
""")
async def set_persona(name: str | None = None, voice: dict[str, float] | None = None, speed: float | None = None,
                      personality: list[str] | None = None) -> dict:
    return await _run(_set_persona_sync, name, voice, speed, personality)


@mcp.tool()
@_doc("""Add one line to your persona's personality.

    Raises ValueError "the personality is full; ask the owner for a persona review" past
    4000 characters.
""")
async def add_to_personality(note: str) -> dict:
    return await _run(_add_to_personality_sync, note)


@mcp.tool()
async def status() -> dict:
    """Report whether speak/listen/converse/call/hang_up are busy right now, in any
    session on any computer, and the Telegram call state.

    Returns `{"busy": bool, "current_tool": str | None, "waiting": int, "call": str}`.
    `call` is the daemon's call state ("none", "ringing", "connected"),
    "daemon unreachable: <error>", "protocol mismatch: <detail>" (this client and the
    daemon speak different wire versions: update the client and reconnect /mcp, or
    deploy the daemon), or "error: <reason>" if it could not be read; when the daemon
    is not reached or mismatched, busy/current_tool/waiting are null (unknown).
    speak/listen/converse work only while `call` is "connected".
    `current_tool` is `"<tool> (<client ip>)"` for the tool holding the lock
    ("transcribe" is a request to the daemon's HTTP whisper API on port 9000),
    or null if idle. `waiting` counts every queued caller from every session.
    Never raises and never blocks on the audio lock.

    Check it first when you come back with a report. "connected" means converse(), even
    when `busy` is true or `waiting` is above 0: your converse queues and you leave it
    waiting. This `busy` is not call()'s "busy" outcome. "ringing" means another session is
    dialing: check again until "connected" (converse) or "none" (call).
    """
    return await _run(_status_sync)


def main() -> None:
    global _daemon
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    value = os.environ.get("SPEAK_DAEMON")
    if not value:
        raise RuntimeError("SPEAK_DAEMON is required (host:port of the speak daemon, e.g. 192.168.86.31:8765)")
    _daemon = proto.parse_address(value)
    init_identity(os.getcwd())   # the project is the folder Claude Code started this client in
    mcp.run()


if __name__ == "__main__":
    main()
