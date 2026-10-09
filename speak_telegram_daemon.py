"""The always-on Telegram link, one per Mac (docs/DESIGN_TELEGRAM_CALL.md section 3).

`speak-telegram` (this module's `main`) owns the algolearn.ai Telegram login and
the call through speak_telegram.TelegramLink, and listens on a local Unix
socket. Every speak server (one per Claude Code session) uses `TelegramClient`
to talk to it. Only one daemon may run: a Telegram login must be used by one
connection at a time, so the daemon holds an exclusive lock file for its life.

IPC framing (both directions, unchanged from the retired speak-phone daemon,
commit b719b7a): 4-byte big-endian header length, a UTF-8 JSON header, then
`header["payload_len"]` raw bytes (float32 little-endian PCM). One request per
connection, one reply: `{"ok": true, ...}` or `{"ok": false, "error": "..."}`;
the client re-raises the message as RuntimeError. Requests: `ping`, `state`,
`call`, `play`, `record`, `hang_up`.

No fallbacks: a daemon that is not running means no call (`state()` returns
NOT_RUNNING, which speak_server routes to the Mac unless a call is expected);
every other operation against a missing daemon raises.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import re
import signal
import socket
import socketserver
import struct
import sys
import threading

import numpy as np

import speak_telegram as tg

log = logging.getLogger("speak.telegram-daemon")

CONNECT_TIMEOUT_SECONDS = 5.0
QUICK_REPLY_TIMEOUT_SECONDS = 20.0
MAX_HEADER_BYTES = 1 << 20
MAX_PAYLOAD_BYTES = 1 << 28   # 256 MB: far above any utterance
NOT_RUNNING = "daemon not running"
ENV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
REQUIRED_ENV = ("TELEGRAM_API_ID", "TELEGRAM_API_HASH", "TELEGRAM_SESSION", "TELEGRAM_CALL_TARGET")


def default_socket_path() -> str:
    return os.environ.get("SPEAK_TELEGRAM_SOCKET") or os.path.expanduser("~/.algolearn-speak/telegram.sock")


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("telegram daemon closed the connection")
        buf += chunk
    return bytes(buf)


def send_msg(sock: socket.socket, header: dict, payload: bytes = b"") -> None:
    header = {**header, "payload_len": len(payload)}
    raw = json.dumps(header).encode()
    sock.sendall(struct.pack(">I", len(raw)) + raw + payload)


def recv_msg(sock: socket.socket) -> tuple[dict, bytes]:
    (hlen,) = struct.unpack(">I", _recv_exact(sock, 4))
    if hlen > MAX_HEADER_BYTES:
        raise ValueError(f"IPC header too large ({hlen} bytes)")
    header = json.loads(_recv_exact(sock, hlen))
    plen = header.get("payload_len", 0)
    if not isinstance(plen, int) or plen < 0 or plen > MAX_PAYLOAD_BYTES:
        raise ValueError(f"bad IPC payload_len {plen!r}")
    return header, _recv_exact(sock, plen) if plen else b""


def _f32(payload: bytes) -> np.ndarray:
    return np.frombuffer(payload, dtype="<f4").copy()


# ------------------------------------------------------------------ daemon side

class _Handler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        link = self.server.link  # type: ignore[attr-defined]
        try:
            req, payload = recv_msg(self.request)
        except (ConnectionError, OSError, ValueError) as e:
            log.warning("IPC request could not be read: %s", e)
            return
        try:
            reply, out = self._dispatch(link, req, payload)
        except (RuntimeError, ValueError, TimeoutError) as e:
            reply, out = {"ok": False, "error": str(e) or type(e).__name__}, b""
        except Exception as e:  # reported to the caller with its type, never swallowed
            log.exception("IPC request failed")
            reply, out = {"ok": False, "error": f"{type(e).__name__}: {e}"}, b""
        try:
            send_msg(self.request, reply, out)
        except OSError as e:
            log.warning("could not send IPC reply: %s", e)

    @staticmethod
    def _dispatch(link, req: dict, payload: bytes) -> tuple[dict, bytes]:
        op = req.get("op")
        if op == "ping":
            return {"ok": True}, b""
        if op == "state":
            return {"ok": True, **link.info()}, b""
        if op == "call":
            return {"ok": True, "outcome": link.call(_f32(payload), req["rate"])}, b""
        if op == "play":
            link.play(_f32(payload), req["rate"])
            return {"ok": True}, b""
        if op == "record":
            result = link.record(req["max_seconds"], req["silence_seconds"], req["start_timeout_seconds"],
                                 _f32(payload) if payload else None, req["cue_rate"])
            if result is tg.TIMEOUT:
                return {"ok": True, "timeout": True}, b""
            return {"ok": True, "timeout": False}, np.asarray(result, dtype="<f4").tobytes()
        if op == "hang_up":
            return {"ok": True, "outcome": link.hang_up()}, b""
        raise ValueError(f"unknown IPC op {op!r}")


class _IPCServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True


class TelegramDaemon:
    """Owns the link, the single-instance lock and the control socket."""

    def __init__(self, link, socket_path: str) -> None:
        self.link = link
        self.socket_path = socket_path
        self._lock_file = None
        self._ipc: _IPCServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        os.makedirs(os.path.dirname(self.socket_path), mode=0o700, exist_ok=True)
        self._lock_file = open(self.socket_path + ".lock", "w")
        try:
            fcntl.flock(self._lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._lock_file.close()
            self._lock_file = None
            raise RuntimeError(f"another speak-telegram daemon holds {self.socket_path}.lock; only one may use the Telegram login")
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)   # we hold the lock, so no live daemon owns it
        self.link.start()
        self._ipc = _IPCServer(self.socket_path, _Handler)
        os.chmod(self.socket_path, 0o600)
        self._ipc.link = self.link  # type: ignore[attr-defined]
        self._thread = threading.Thread(target=self._ipc.serve_forever, name="telegram-ipc", daemon=True)
        self._thread.start()
        log.info("speak-telegram: control socket %s", self.socket_path)

    def stop(self) -> None:
        if self._ipc is not None:
            self._ipc.shutdown()
            self._ipc.server_close()
            self._ipc = None
        try:
            self.link.stop()
        finally:
            if os.path.exists(self.socket_path):
                os.unlink(self.socket_path)
            if self._lock_file is not None:
                self._lock_file.close()
                self._lock_file = None


# ------------------------------------------------------------------ client side

class TelegramClient:
    """What a speak server uses: synchronous calls forwarded to the daemon."""

    def __init__(self, socket_path: str | None = None) -> None:
        self.socket_path = socket_path or default_socket_path()

    def _request(self, header: dict, payload: bytes = b"", reply_timeout: float | None = None) -> tuple[dict, bytes]:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            s.settimeout(CONNECT_TIMEOUT_SECONDS)
            s.connect(self.socket_path)
            s.settimeout(reply_timeout)
            send_msg(s, header, payload)
            reply, data = recv_msg(s)
        except TimeoutError:
            # socket.timeout is TimeoutError in 3.12; re-raised as RuntimeError so a stalled
            # daemon is never mistaken for listen()'s "no speech detected" TimeoutError.
            raise RuntimeError(f"telegram daemon did not reply to {header['op']!r} within "
                               f"{s.gettimeout():.0f}s") from None
        finally:
            s.close()
        if not reply.get("ok"):
            raise RuntimeError(reply.get("error", "telegram daemon reported an error"))
        return reply, data

    def info(self) -> dict:
        """`{"state", "generation", "end_reason"}`, or state NOT_RUNNING when no daemon answers."""
        try:
            reply, _ = self._request({"op": "state"}, reply_timeout=QUICK_REPLY_TIMEOUT_SECONDS)
        except (FileNotFoundError, ConnectionRefusedError):
            return {"state": NOT_RUNNING, "generation": None, "end_reason": None}
        return {"state": reply["state"], "generation": reply["generation"], "end_reason": reply["end_reason"]}

    def state(self) -> str:
        """The call state, or NOT_RUNNING when no daemon answers (no socket / refused)."""
        return self.info()["state"]

    def call(self, greeting: np.ndarray, rate: int) -> str:
        pcm = np.asarray(greeting, dtype="<f4").reshape(-1)
        budget = tg.RING_TIMEOUT_SECONDS + tg.OPENER_MIN_SECONDS + len(pcm) / rate + 60.0
        reply, _ = self._request({"op": "call", "rate": rate}, pcm.tobytes(), reply_timeout=budget)
        return reply["outcome"]

    def play(self, audio: np.ndarray, rate: int) -> None:
        pcm = np.asarray(audio, dtype="<f4").reshape(-1)
        self._request({"op": "play", "rate": rate}, pcm.tobytes(), reply_timeout=len(pcm) / rate + 60.0)

    def record(self, max_seconds: float, silence_seconds: float, start_timeout_seconds: float,
               cue: np.ndarray | None, cue_rate: int):
        cue_pcm = b"" if cue is None else np.asarray(cue, dtype="<f4").reshape(-1).tobytes()
        cue_seconds = len(cue_pcm) / 4 / cue_rate
        reply, data = self._request(
            {"op": "record", "max_seconds": max_seconds, "silence_seconds": silence_seconds,
             "start_timeout_seconds": start_timeout_seconds, "cue_rate": cue_rate},
            cue_pcm, reply_timeout=cue_seconds + start_timeout_seconds + max_seconds + 60.0,
        )
        if reply["timeout"]:
            return tg.TIMEOUT
        return np.frombuffer(data, dtype="<f4").copy()

    def hang_up(self) -> str:
        reply, _ = self._request({"op": "hang_up"}, reply_timeout=QUICK_REPLY_TIMEOUT_SECONDS)
        return reply["outcome"]


# ------------------------------------------------------------------ entry point

def load_env(path: str = ENV_FILE) -> dict[str, str]:
    """TELEGRAM_* values from the repo's git-ignored .env; a missing one stops the daemon."""
    with open(path) as f:
        env = {k: v.strip().strip("\r").strip("\"'") for k, v in re.findall(r"^(TELEGRAM_[A-Z_]+)=(.*)$", f.read(), re.M)}
    missing = [k for k in REQUIRED_ENV if not env.get(k)]
    if missing:
        raise RuntimeError(f"{path} is missing {', '.join(missing)}")
    return env


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


def main() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    code = 0
    try:
        daemon = TelegramDaemon(build_link(load_env()), default_socket_path())
        daemon.start()
        stop = threading.Event()
        signal.signal(signal.SIGTERM, lambda *_: stop.set())
        signal.signal(signal.SIGINT, lambda *_: stop.set())
        stop.wait()
        daemon.stop()
    except BaseException:
        log.exception("speak-telegram failed")
        code = 1
    finally:
        # Every exit path skips the interpreter's teardown: ntgcalls' C++ global destructors
        # abort() while its WebRTC thread is alive (macOS crash report 2026-10-09).
        sys.stderr.flush()
        os._exit(code)


if __name__ == "__main__":
    main()
