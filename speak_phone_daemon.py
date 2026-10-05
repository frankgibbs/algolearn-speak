"""The always-on phone link, split out of the per-session speak servers
(docs/DESIGN_PHONE_AUDIO.md section 10).

`speak-phone` (this module's `main`) is ONE process per Mac. It owns TCP 8772
through speak_phone_audio.PhoneAudioServer (protocol to the phone unchanged),
holds the single phone connection and runs Silero VAD on the phone's mic
blocks. It also listens on a local Unix socket for control requests.

Every speak server (one per Claude Code session) uses `PhoneClient`, which has
the same synchronous methods as PhoneAudioServer (`connected`, `set_state`,
`play`, `play_segments`, `record`) but forwards them to the daemon over that
socket. A speak server never binds 8772.

IPC framing (both directions): 4-byte big-endian header length, a UTF-8 JSON
header, then `header["payload_len"]` raw bytes (float32 PCM, little-endian).
One request per connection, one reply. Requests: `ping`, `connected`,
`set_state`, `play_segments`, `record`. A reply is `{"ok": true, ...}` or
`{"ok": false, "error": "<message>"}`; the client re-raises the message as
RuntimeError so the phone-disconnect errors read exactly as before.

No fallbacks: a daemon that is not running means no phone is connected
(`connected()` is False, which is the documented Mac route); any other
operation against a missing daemon raises.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import socketserver
import struct
import sys
import threading

import numpy as np

import speak_phone_audio as pa

log = logging.getLogger("speak.phone-daemon")

PHONE_PORT_DEFAULT = 8772
CONNECT_TIMEOUT_SECONDS = 5.0     # connect + quick requests (ping/connected/set_state)
QUICK_REPLY_TIMEOUT_SECONDS = 15.0  # set_state is bounded by the daemon's own 5 s send bound
MAX_HEADER_BYTES = 1 << 20
MAX_PAYLOAD_BYTES = 1 << 28        # 256 MB: far above any utterance


def default_socket_path() -> str:
    return os.environ.get("SPEAK_PHONE_SOCKET") or os.path.expanduser("~/.algolearn-speak/phone.sock")


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("phone daemon closed the connection")
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


# ------------------------------------------------------------------ daemon side

class _Handler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        phone: pa.PhoneAudioServer = self.server.phone  # type: ignore[attr-defined]
        try:
            req, payload = recv_msg(self.request)
            reply, out = self._dispatch(phone, req, payload)
        except (RuntimeError, ValueError) as e:
            reply, out = {"ok": False, "error": str(e)}, b""
        except (ConnectionError, OSError) as e:
            log.warning("IPC connection error: %s", e)
            return
        try:
            send_msg(self.request, reply, out)
        except OSError as e:
            log.warning("could not send IPC reply: %s", e)

    @staticmethod
    def _dispatch(phone: pa.PhoneAudioServer, req: dict, payload: bytes) -> tuple[dict, bytes]:
        op = req.get("op")
        if op == "ping":
            return {"ok": True}, b""
        if op == "connected":
            return {"ok": True, "connected": phone.connected()}, b""
        if op == "set_state":
            phone.set_state(req["value"])
            return {"ok": True}, b""
        if op == "play_segments":
            data = np.frombuffer(payload, dtype="<f4")
            segments, pos = [], 0
            for n in req["lengths"]:
                segments.append(data[pos:pos + n].copy())
                pos += n
            if pos != len(data):
                raise ValueError("play_segments payload does not match lengths")
            phone.play_segments(segments, req["rate"], state=req.get("state", "speaking"))
            return {"ok": True}, b""
        if op == "record":
            cue = np.frombuffer(payload, dtype="<f4").copy() if payload else None
            result = phone.record(req["max_seconds"], req["silence_seconds"], req["start_timeout_seconds"], cue=cue)
            if result is pa.TIMEOUT:
                return {"ok": True, "timeout": True}, b""
            return {"ok": True, "timeout": False}, np.asarray(result, dtype="<f4").tobytes()
        raise ValueError(f"unknown IPC op {op!r}")


class _IPCServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True


class PhoneDaemon:
    """Owns a PhoneAudioServer (TCP) and the local control socket."""

    def __init__(self, phone: pa.PhoneAudioServer, socket_path: str) -> None:
        self.phone = phone
        self.socket_path = socket_path
        self._ipc: _IPCServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.phone.start()  # raises if the TCP port cannot be bound
        os.makedirs(os.path.dirname(self.socket_path), mode=0o700, exist_ok=True)
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)  # only reached after we own the TCP port, so no live daemon owns it
        self._ipc = _IPCServer(self.socket_path, _Handler)
        os.chmod(self.socket_path, 0o600)
        self._ipc.phone = self.phone  # type: ignore[attr-defined]
        self._thread = threading.Thread(target=self._ipc.serve_forever, name="phone-ipc", daemon=True)
        self._thread.start()
        log.info("speak-phone: phone port %d, control socket %s", self.phone.port, self.socket_path)

    def stop(self) -> None:
        if self._ipc is not None:
            self._ipc.shutdown()
            self._ipc.server_close()
            self._ipc = None
        self.phone.stop()
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)


# ------------------------------------------------------------------ client side

class PhoneClient:
    """Drop-in for PhoneAudioServer's synchronous API, forwarding to the daemon."""

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
        finally:
            s.close()
        if not reply.get("ok"):
            raise RuntimeError(reply.get("error", "phone daemon reported an error"))
        return reply, data

    def daemon_reachable(self) -> bool:
        """True iff the daemon answers a ping. Not running (no socket / refused) is False."""
        try:
            self._request({"op": "ping"}, reply_timeout=CONNECT_TIMEOUT_SECONDS)
        except (FileNotFoundError, ConnectionRefusedError):
            return False
        return True

    def connected(self) -> bool:
        """Is a phone connected? Daemon not running means no phone: False."""
        try:
            reply, _ = self._request({"op": "connected"}, reply_timeout=CONNECT_TIMEOUT_SECONDS)
        except (FileNotFoundError, ConnectionRefusedError):
            return False
        return bool(reply["connected"])

    def set_state(self, value: str) -> None:
        self._request({"op": "set_state", "value": value}, reply_timeout=QUICK_REPLY_TIMEOUT_SECONDS)

    def play(self, pcm_f32: np.ndarray, rate: int) -> None:
        self.play_segments([pcm_f32], rate)

    def play_segments(self, segments: list[np.ndarray], rate: int, state: str = "speaking") -> None:
        segs = [np.asarray(seg, dtype="<f4").reshape(-1) for seg in segments]
        self._request(
            {"op": "play_segments", "rate": rate, "state": state, "lengths": [len(x) for x in segs]},
            b"".join(x.tobytes() for x in segs),
        )

    def record(
        self, max_seconds: float, silence_seconds: float, start_timeout_seconds: float, cue: np.ndarray | None = None,
    ) -> np.ndarray | object:
        reply, data = self._request(
            {"op": "record", "max_seconds": max_seconds, "silence_seconds": silence_seconds,
             "start_timeout_seconds": start_timeout_seconds},
            b"" if cue is None else np.asarray(cue, dtype="<f4").reshape(-1).tobytes(),
        )
        if reply["timeout"]:
            return pa.TIMEOUT
        return np.frombuffer(data, dtype="<f4").copy()


# ------------------------------------------------------------------ entry point

def main() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    port = int(os.environ.get("SPEAK_PHONE_PORT", PHONE_PORT_DEFAULT))
    daemon = PhoneDaemon(pa.PhoneAudioServer(host="0.0.0.0", port=port), default_socket_path())
    daemon.start()
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        daemon.stop()


if __name__ == "__main__":
    main()
