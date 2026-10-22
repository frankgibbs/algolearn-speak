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
# spoken project name), plus len(project) / 5 s as the other budgets count text.
ANNOUNCE_BUDGET_SECONDS = 10.0

_daemon: tuple[str, int] | None = None   # set in main() from SPEAK_DAEMON
# This client's identity on every tool op: {FIELD_SESSION_ID, FIELD_PROJECT}. Set once per
# process by init_identity() (main(), or scripts/drive_daemon.py).
_identity: dict[str, str] | None = None

# The session's call expectation: the daemon's generation of the call this session
# expects, or None. Updated only from a reply's "expect" key.
_expect_lock = threading.Lock()
_call_generation: str | None = None   # opaque "<daemon boot>:<call generation>"


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


def _request(op: str, fields: dict, budget: float) -> dict:
    """One request on its own connection. Tool ops wait for the daemon's "started" with no
    timeout (the queue never rejects), then for the final reply within `budget` seconds;
    quick ops get one reply within `budget`. A failure reply is raised as the error its kind
    names; a stalled daemon raises RuntimeError, never TimeoutError, so it is never mistaken
    for "no speech detected"."""
    addr = _address()
    if op in proto.LOCKED_OPS:
        fields = {**_identity_fields(), **fields}
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
    if not reply.get("ok"):
        message = reply.get("error") or f"speak daemon reported an error without a message: {reply!r}"
        if proto.REPLY_SWITCHED_FROM in reply:   # the owner heard the announcement before the failure
            message = f"{_switch_line(reply)}{message}"
        kind = reply.get("kind")
        if kind == proto.KIND_TIMEOUT:
            raise TimeoutError(message)
        if kind == proto.KIND_VALUE:
            raise ValueError(message)
        raise RuntimeError(message)
    return reply


# ---------------------------------------------------------------- budgets after "started"

def _speak_budget(text: str) -> float:
    """Synthesis and playback, plus the longest the owner may hold Angie (the holds of one
    op add up to at most HOLD_MAX_SECONDS; then the daemon ends the call and replies)."""
    return 60.0 + len(text) / 5 + proto.HOLD_MAX_SECONDS


def _listen_budget(max_seconds: float, start_timeout_seconds: float) -> float:
    return start_timeout_seconds + max_seconds + 60.0


def _announce_budget() -> float:
    return ANNOUNCE_BUDGET_SECONDS + len(_identity_fields()[proto.FIELD_PROJECT]) / 5


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
            f"cue and this project's name first]\n")


def _held_prefix(reply: dict) -> str:
    """"[held 42s, then resumed]\n" when the owner held Angie and she finished; "" otherwise."""
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
        return _switch_line(reply) + _expired_result(reply)
    return f"{_switch_line(reply)}{_held_prefix(reply)}spoke for {reply['seconds']:.1f}s{AUDIO_NOTE}"


def _converse_result(reply: dict) -> str:
    if proto.REPLY_INTERRUPTION in reply:
        return _switch_line(reply) + _expired_result(reply)
    return f"{_switch_line(reply)}{_held_prefix(reply)}{reply['text']}{AUDIO_NOTE}"


def _listen_result(reply: dict) -> str:
    return f"{_switch_line(reply)}{reply['text']}"


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
        return f"{proto.ANSWERED}\n{reply['text']}{AUDIO_NOTE}"
    return reply["outcome"]


def _hang_up_sync() -> str:
    global _call_generation
    try:
        return _request(proto.OP_HANG_UP, {}, QUICK_REPLY_TIMEOUT_SECONDS)["outcome"]
    finally:
        with _expect_lock:
            _call_generation = None   # the owner asked to hang up: never a call-back, even if this raised


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

_INSTRUCTIONS = """\
Voice tools for talking with the owner (Frank) over a Telegram call. Owner conventions:

- On calls and in anything spoken you are "Angie", never Claude ("Hi Frank, it's Angie").
- Spoken text is conversational: plain sentences, no markdown, lists or code.
- One topic at a time, one aspect at a time, and one question at a time, so the
  conversation flows linearly.
- Every conversational turn is ONE converse(text) call. Never speak() then listen() for
  a turn: the gap between the two calls is seconds of dead air before the ear-open beep;
  converse() opens the ear right after the last word. listen() alone only to reopen the
  ear after a TimeoutError or when there is nothing to say first. speak() alone only for
  the final sign-off (see "the floor" below).
- A listen/converse TimeoutError means nobody spoke in time; the call is still up.
  Reopen the ear with listen(). Do not assume the owner left, do not hang up.
- Go straight into the answer. Never prefix a reply with "heard you" or a restatement;
  listen's rising two-note chime (played once your transcript is captured) is the
  acknowledgement, and there is no spoken one. So answer first and hand real work to
  agents (subagents, background tasks) rather than going silent while you do it.
- The owner can hold you mid-sentence by saying "hold on" alone; you stop at once and
  wait in silence, and "continue" restarts the sentence you were cut off in. A result
  starting "[held Ns, then resumed]" means everything was said; carry on as normal.
  A result starting "[hold expired after ...; the call was ended]" means the owner never
  said "continue" within 5 minutes, so the daemon hung up: the reply was NOT finished
  and converse did NOT listen. Do not assume he heard it and do not call back right
  away (he ignores calls after this); keep the "not spoken" text and pick it up later.
- Never speak "hold on" or "continue" as a sentence by itself: on a phone without echo
  cancellation you would hear yourself and hold or resume your own speech.
- When the owner asks you to call, they are up: pass override_quiet_hours=True without
  asking. Calls you start on your own respect quiet hours.
- Call etiquette: "not answered" -> wait an hour. "busy" -> call once more, then wait an
  hour. Owner hung up -> call back once right away; if that is not answered, wait an hour.
  "Telegram call ended during ..." -> call back. hang_up only when the owner asks.
- When a call seems off (silence, no speech heard, drops): set_call_debug(True), place
  or keep the call, read call_debug (its docstring says how to read the events), and add
  instrumentation rather than retrying blind. No need to ask first. Turn debug off once
  calls work.
- There is one call, shared by every session on every computer; tools queue for it.
  The owner may also be reading this terminal.
- The floor: other sessions may be waiting for the call. The session that last played
  audio holds the floor, so the owner hears one conversation at a time. converse() and
  listen() keep the floor; speak() gives it up.
- Stay in converse() as long as the topic is alive, including while an agent you started
  does a few minutes of research: tell the owner what you started and keep the
  conversation going. He will likely have more questions, or a related thread on the
  same topic.
- Before leaving, ask whether he has anything else.
- End with ONE speak() sign-off (for example "Will do, talk soon.") only when the topic
  has closed, the owner has said he has nothing else, and the remaining work is long
  (around 15 minutes or more). Never use speak() for a quick check or a short task.
- Do not offer to call back; say you will come back with the result. When the work is
  done, check status(): if its call field says the call is still connected, converse()
  with the result (it waits its turn, and the owner hears the switch cue and your
  project's name first). Use call() only if the call has ended.
- When your speak/listen/converse is the first audio on the call after another
  session's, the daemon first plays a double beep and speaks your project's name, and
  your result's first line is '[took over the call from "<other project>"; ...]'. The
  owner was just talking with that other session, and your words (written before you
  knew) followed the announcement. In your next spoken turn, open with ONE sentence that
  recaps where you left off and your last question (unless the owner's reply already
  answers it), then carry on.
"""

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
    false with `waiting` above 0 means another session holds the floor.

    Switch of session: if the previous audio on the call came from another
    session (the greeting counts as the calling session's), the daemon first
    plays a double beep and speaks this project's name, and the result (or the
    error) starts with the line '[took over the call from "<other project>";
    the owner heard the switch cue and this project's name first]'. Recap where
    you left off and your last question in one sentence in your next turn.
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

    speak() gives up the floor; use it only for the final sign-off. For a
    conversational turn use `converse(text)`, never speak() then listen(): the gap
    between the two calls is seconds of dead air before the ear-open beep.
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

    converse() keeps the floor.
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
    "Hi Frank, it's Angie. The backtest finished a few minutes ago and the
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
                                                 spoken); just talk
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
                                                 that window
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
    Quiet hours are a hard rule. Pass `override_quiet_hours=True` ONLY when the
    owner has said in this conversation that they are up and want calls.

    Rule of engagement: if speak/listen/converse raises "Telegram call ended
    during ...", the call dropped mid-conversation -- call back with this tool.
    The daemon never dials sooner than 10 s after the previous call ended,
    whatever the reason (a redial 1.4 s after a hang-up left the owner's
    Telegram app dropping every incoming call until it was force-quit); a
    call placed sooner waits out the rest of those 10 s, then dials.
    Hang up with `hang_up()` only when the owner asks; they can also hang up
    themselves. If the speak daemon cannot be reached this raises "speak daemon
    at <addr> is not reachable: ...".
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
    """End the Telegram call placed by `call()`. Use only when the owner asks you to.

    Returns "hung up" or "no active call". Waits its turn in the audio lock
    queue like the other call tools.
    """
    return await _run(_hang_up_sync)


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
