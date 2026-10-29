"""The wire between the thin MCP client (speak_server.py) and the speak daemon
(speak_telegram_daemon.py), plus the outcome and state strings both sides use
(docs/DESIGN_REMOTE_DAEMON.md section 4.2).

Standard library only: the client imports this module and nothing heavier, so a
client computer needs no numpy, torch or ntgcalls. speak_telegram.py imports its
outcome and state strings from here, so each one is defined once.

Framing (both directions): a 4-byte big-endian header length, a UTF-8 JSON
header, then `header["payload_len"]` raw bytes. Only text crosses the wire, so
payloads are empty in practice; the field stays for the framing's own checks.

One request per TCP connection. Quick ops (`ping`, `status`, `debug`,
`set_debug`) get one reply. Tool ops (`speak`, `listen`, `converse`, `call`,
`hang_up`) wait for the daemon's audio lock and get two: `{"ok": true,
"phase": "started"}` the moment the daemon holds the lock, then the final
reply. A failure is `{"ok": false, "error": "...", "kind": KIND_*}`.

A tool reply (success or failure) may carry `"expect"`: the call generation the
session should now expect (an int), or null for none. When the key is absent the
session's expectation is unchanged.

Every request carries FIELD_PROTOCOL = PROTOCOL_VERSION. The daemon refuses `ping`
and every tool op whose version is missing or differs (a `value` error telling the
session to update its client and reconnect /mcp). `status`, `debug` and `set_debug`
are served to any client and carry the daemon's version in their reply, so an old
client can still see what is wrong.

Every tool op also carries the client's identity (protocol 3, owner decision 2026-10-20):
FIELD_SESSION_ID (generated once per client process) and FIELD_PROJECT (the basename of
the client's working folder). The daemon refuses a tool op whose identity is missing,
not a string, empty, or (project) has no letter or digit, before the lock, like a
missing version. A speak/listen/converse reply (success or failure) carries
REPLY_SWITCHED_FROM, the previous session's project, when the daemon announced a switch
of session before the op's own audio; when the key is absent there was no switch.

Personas (protocol 4, docs/DESIGN_PERSONAS.md, owner decisions 2026-10-22): every tool op
also carries FIELD_PERSONA_SEEN, the "<persona id>:<words_revision>" of the project's
persona this session last saw (null at first; a missing field is refused like a missing
version). words_revision rises only when the persona's name or personality changes, the
things an agent's words depend on; a voice or speed change leaves the key as it is (Design
2026-10-22). A reply to a tool op carries REPLY_PERSONA, the project's persona block,
whenever the project's key differs from it; a persona op's reply always carries it. Before
the lock, speak, converse and call with a stale FIELD_PERSONA_SEEN are refused (kind value,
with the block): nothing is spoken or dialed. The persona ops (PERSONA_OPS) are versioned and
carry the identity like tool ops, but never touch the audio lock.
"""

from __future__ import annotations

import json
import re
import socket
import struct
import time

DAEMON_PORT = 8765
MAX_HEADER_BYTES = 1 << 20
MAX_PAYLOAD_BYTES = 1 << 20   # text only crosses the wire

# The wire version. Bump it whenever a request or reply changes in a way an older peer
# would misread. The unversioned protocol before it counts as 1; 2 is Release 2 of
# docs/DESIGN_INTERRUPT.md (hold replies, longer speak budgets); 3 adds the client's
# identity to every tool op and the session-switch reply field (owner decision 2026-10-20);
# 4 adds personas: persona_seen on every tool op, the persona reply block and the persona
# ops (owner decisions 2026-10-22, docs/DESIGN_PERSONAS.md).
PROTOCOL_VERSION = 4
FIELD_PROTOCOL = "protocol"

# The client's identity on every tool op, and the session-switch announcement (owner
# decisions 2026-10-20). When a speak/listen/converse on the connected call comes from
# another session than the previous audio on that call (the greeting counts as the calling
# session's), the daemon plays the switch cue and speaks this op's project name first, and
# the reply carries REPLY_SWITCHED_FROM = the previous session's project.
FIELD_SESSION_ID, FIELD_PROJECT = "session_id", "project"
REPLY_SWITCHED_FROM = "switched_from"

# Personas (docs/DESIGN_PERSONAS.md, owner decisions 2026-10-22). The block in REPLY_PERSONA:
# {"id", "words_revision", "name", "house" (true on the house persona, which never grows its
# personality on its own), "voice": {id: weight}, "speed", "personality": [text, ...],
# "shared_with": [other projects]}.
FIELD_PERSONA_SEEN = "persona_seen"
REPLY_PERSONA = "persona"
PERSONA_SPEED_MIN, PERSONA_SPEED_MAX = 0.5, 2.0
PERSONA_POOL = ("Angie", "Nova", "Ruby", "Hazel", "Paige", "Stella", "Gwen", "Willow", "Margo", "Juno",
                "Tess", "Fiona")
PERSONA_SEED_NAME = "Angie"
PERSONA_NAME_MAX = 12            # a name is one word of 2 to 12 letters
PERSONALITY_MAX_CHARS = 4000
PERSONA_MAX_VOICES = 4
PERSONA_VOICE_PREFIXES = ("af_", "am_", "bf_", "bm_")   # English voices only

# ops
OP_PING, OP_STATUS, OP_DEBUG, OP_SET_DEBUG = "ping", "status", "debug", "set_debug"
OP_SPEAK, OP_LISTEN, OP_CONVERSE, OP_CALL, OP_HANG_UP = "speak", "listen", "converse", "call", "hang_up"
OP_PERSONA, OP_PERSONA_NEW, OP_PERSONA_SET, OP_PERSONA_ADD = "persona", "persona_new", "persona_set", "persona_add"
PERSONA_OPS = frozenset({OP_PERSONA, OP_PERSONA_NEW, OP_PERSONA_SET, OP_PERSONA_ADD})   # versioned, identity, no lock
QUICK_OPS = frozenset({OP_PING, OP_STATUS, OP_DEBUG, OP_SET_DEBUG}) | PERSONA_OPS
VERSIONLESS_OPS = frozenset({OP_STATUS, OP_DEBUG, OP_SET_DEBUG})   # served to a client of any protocol
LOCKED_OPS = frozenset({OP_SPEAK, OP_LISTEN, OP_CONVERSE, OP_CALL, OP_HANG_UP})
VOICED_OPS = frozenset({OP_SPEAK, OP_CONVERSE, OP_CALL})   # refused before the lock on a stale persona_seen
PHASE_STARTED = "started"

# error kinds: the client re-raises timeout as TimeoutError (no speech), value as
# ValueError, and call_ended / runtime as RuntimeError
KIND_TIMEOUT, KIND_VALUE, KIND_CALL_ENDED, KIND_RUNTIME = "timeout", "value", "call_ended", "runtime"

# call outcomes (speak_telegram.TelegramLink and the call tool)
ANSWERED = "answered"
ALREADY_CONNECTED = "already connected"
NOT_ANSWERED = "not answered (declined or no answer)"
BUSY = "busy (the owner's Telegram reports a call in progress)"
HUNG_UP_DURING_GREETING = "answered, then hung up during the greeting"
AUDIO_FAILED_PREFIX = "answered but audio failed to connect: "
HUNG_UP = "hung up"
NO_ACTIVE_CALL = "no active call"

# call states and end reasons
NONE, RINGING, CONNECTED = "none", "ringing", "connected"
END_REMOTE, END_HUNG_UP, END_FAILED, END_NOT_ANSWERED = "remote", "hung_up", "failed", "not_answered"
END_BUSY = "busy"

RING_TIMEOUT_SECONDS = 60
# Minimum time from the end of one call (any reason) to the next dial. A redial 1.4 s after
# hang_up got CallBusy and then left the owner's Telegram app silently dropping every incoming
# call (half-second ring, no discard sent) until it was force-quit; a redial 10 s after
# hang_up rang and was answered (2026-10-15).
REDIAL_SETTLE_SECONDS = 10.0
CONNECT_GRACE_SECONDS = 30   # py-tgcalls waits for the media connection with no timeout of its own
OPENER_MIN_SECONDS = 6.0     # 2 s failed, 6 s worked (2026-10-09). The one definition: the daemon rejects shorter greetings before dialing, write_opener raises on them

# Holding Angie by spoken command (docs/DESIGN_INTERRUPT.md, Release 2). A speak/converse
# reply always carries REPLY_HELD_SECONDS; after a hold expiry it also carries
# REPLY_INTERRUPTION = {"command": INTERRUPTION_HOLD_EXPIRED, "cut_index" (1-based),
# "sentence_count", "cut_sentence", "unspoken" (from the start of the cut sentence)}.
# Owner decision 2026-10-16: a hold that reaches HOLD_MAX_SECONDS (the holds of one op added
# up) ends the call. The client's speak/converse budgets include it.
HOLD_MAX_SECONDS = 300.0
# The floor (docs/DESIGN_FLOOR.md, owner decisions 2026-10-20): the session that last played
# audio on the call goes ahead of every other session until it ends with a speak, or sends no
# op for this long after its last op ended. Not a wire field: the client's instruction text
# and the daemon's AudioLock use the same number.
FLOOR_IDLE_SECONDS = 45.0
CMD_HOLD, CMD_RESUME = "hold", "resume"
INTERRUPTION_HOLD_EXPIRED = "hold_expired"
REPLY_HELD_SECONDS, REPLY_INTERRUPTION = "held_seconds", "interruption"
# The sign-off hang-up (owner decision 2026-10-28, docs/DESIGN_FLOOR.md): when a speak gives up
# the floor (the sign-off) and no other tool op is queued, the daemon ends the call, and the
# speak reply (success or failure) carries REPLY_CALL_ENDED = CALL_ENDED_SIGNOFF, with "expect"
# null. Absent: the call was not ended by the sign-off. No version bump: a protocol 4 client
# that predates it ignores the key and still clears its expectation from "expect".
REPLY_CALL_ENDED, CALL_ENDED_SIGNOFF = "call_ended", "signoff"

AUDIO_NOTE = "\n(audio: telegram)"   # ends every speak/listen/converse result and an answered call()


def speakable_name(project: str) -> str:
    """A project (folder) name as it is spoken: every run of characters that are not letters or
    digits becomes one space ("algolearn-speak" -> "algolearn speak"). "" when nothing is left,
    which the client refuses at startup and the daemon refuses in a request."""
    return " ".join(re.sub(r"[\W_]+", " ", project).split())


def persona_key(block: dict) -> str:
    """The "<persona id>:<words_revision>" a client sends as FIELD_PERSONA_SEEN after seeing `block`."""
    return f"{block['id']}:{block['words_revision']}"


def is_audio_failure(outcome: str) -> bool:
    return outcome.startswith(AUDIO_FAILED_PREFIX)


def enable_keepalive(sock: socket.socket, idle: int = 30, interval: int = 10, count: int = 3) -> None:
    """TCP keepalive, so a dead peer is detected while a request waits for the audio lock.
    Linux names the idle option TCP_KEEPIDLE, macOS TCP_KEEPALIVE."""
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    idle_opt = socket.TCP_KEEPIDLE if hasattr(socket, "TCP_KEEPIDLE") else socket.TCP_KEEPALIVE
    sock.setsockopt(socket.IPPROTO_TCP, idle_opt, idle)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, interval)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, count)


def _recv_exact(sock: socket.socket, n: int, deadline: float | None) -> bytes:
    """`deadline` (time.monotonic()) bounds the whole read, not each recv; None leaves the
    socket's own timeout in force. Past it, raises TimeoutError."""
    buf = bytearray()
    while len(buf) < n:
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("deadline passed while reading a message")
            sock.settimeout(remaining)
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("the peer closed the connection")
        buf += chunk
    return bytes(buf)


def send_msg(sock: socket.socket, header: dict, payload: bytes = b"") -> None:
    header = {**header, "payload_len": len(payload)}
    raw = json.dumps(header).encode()
    sock.sendall(struct.pack(">I", len(raw)) + raw + payload)


def recv_msg(sock: socket.socket, deadline: float | None = None) -> tuple[dict, bytes]:
    (hlen,) = struct.unpack(">I", _recv_exact(sock, 4, deadline))
    if hlen > MAX_HEADER_BYTES:
        raise ValueError(f"message header too large ({hlen} bytes)")
    header = json.loads(_recv_exact(sock, hlen, deadline))
    if not isinstance(header, dict):
        raise ValueError(f"message header is not a JSON object: {header!r}")
    plen = header.get("payload_len", 0)
    if not isinstance(plen, int) or plen < 0 or plen > MAX_PAYLOAD_BYTES:
        raise ValueError(f"bad payload_len {plen!r}")
    return header, _recv_exact(sock, plen, deadline) if plen else b""


def parse_address(value: str) -> tuple[str, int]:
    """`host:port` -> (host, port). Raises ValueError on anything else."""
    host, sep, port = value.rpartition(":")
    if not sep or not host or not port.isdigit() or not 0 < int(port) < 65536:
        raise ValueError(f"{value!r} is not host:port")
    return host, int(port)
