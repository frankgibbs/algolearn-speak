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
"""

from __future__ import annotations

import json
import socket
import struct

DAEMON_PORT = 8765
MAX_HEADER_BYTES = 1 << 20
MAX_PAYLOAD_BYTES = 1 << 20   # text only crosses the wire

# ops
OP_PING, OP_STATUS, OP_DEBUG, OP_SET_DEBUG = "ping", "status", "debug", "set_debug"
OP_SPEAK, OP_LISTEN, OP_CONVERSE, OP_CALL, OP_HANG_UP = "speak", "listen", "converse", "call", "hang_up"
QUICK_OPS = frozenset({OP_PING, OP_STATUS, OP_DEBUG, OP_SET_DEBUG})
LOCKED_OPS = frozenset({OP_SPEAK, OP_LISTEN, OP_CONVERSE, OP_CALL, OP_HANG_UP})
PHASE_STARTED = "started"

# error kinds: the client re-raises timeout as TimeoutError (no speech), value as
# ValueError, and call_ended / runtime as RuntimeError
KIND_TIMEOUT, KIND_VALUE, KIND_CALL_ENDED, KIND_RUNTIME = "timeout", "value", "call_ended", "runtime"

# call outcomes (speak_telegram.TelegramLink and the call tool)
ANSWERED = "answered"
ALREADY_CONNECTED = "already connected"
NOT_ANSWERED = "not answered (declined or no answer)"
HUNG_UP_DURING_GREETING = "answered, then hung up during the greeting"
AUDIO_FAILED_PREFIX = "answered but audio failed to connect: "
HUNG_UP = "hung up"
NO_ACTIVE_CALL = "no active call"

# call states and end reasons
NONE, RINGING, CONNECTED = "none", "ringing", "connected"
END_REMOTE, END_HUNG_UP, END_FAILED, END_NOT_ANSWERED = "remote", "hung_up", "failed", "not_answered"

RING_TIMEOUT_SECONDS = 60
CONNECT_GRACE_SECONDS = 30   # py-tgcalls waits for the media connection with no timeout of its own
OPENER_MIN_SECONDS = 6.0     # 2 s failed, 6 s worked (2026-10-09). The one definition: the daemon rejects shorter greetings before dialing, write_opener raises on them

AUDIO_NOTE = "\n(audio: telegram)"   # ends every speak/listen/converse result and an answered call()


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


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("the peer closed the connection")
        buf += chunk
    return bytes(buf)


def send_msg(sock: socket.socket, header: dict, payload: bytes = b"") -> None:
    header = {**header, "payload_len": len(payload)}
    raw = json.dumps(header).encode()
    sock.sendall(struct.pack(">I", len(raw)) + raw + payload)


def recv_msg(sock: socket.socket) -> tuple[dict, bytes]:
    (hlen,) = struct.unpack(">I", _recv_exact(sock, 4))
    if hlen > MAX_HEADER_BYTES:
        raise ValueError(f"message header too large ({hlen} bytes)")
    header = json.loads(_recv_exact(sock, hlen))
    if not isinstance(header, dict):
        raise ValueError(f"message header is not a JSON object: {header!r}")
    plen = header.get("payload_len", 0)
    if not isinstance(plen, int) or plen < 0 or plen > MAX_PAYLOAD_BYTES:
        raise ValueError(f"bad payload_len {plen!r}")
    return header, _recv_exact(sock, plen) if plen else b""


def parse_address(value: str) -> tuple[str, int]:
    """`host:port` -> (host, port). Raises ValueError on anything else."""
    host, sep, port = value.rpartition(":")
    if not sep or not host or not port.isdigit() or not 0 < int(port) < 65536:
        raise ValueError(f"{value!r} is not host:port")
    return host, int(port)
