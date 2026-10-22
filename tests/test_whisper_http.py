"""Tests for the daemon's HTTP whisper API on port 9000 (speak_telegram_daemon._WhisperHandler),
which replaces the retired whisper-service (scratch/whisper-service/server.py) with its API.

A real SpeakDaemon on ephemeral ports, with a fake link and fake engines: no CUDA, no Telegram.
The known client (scratch/voice-claude/livekit_agent.py, WhisperServiceSTT) POSTs raw PCM with
httpx and Content-Type application/octet-stream; test_livekit_client_call does exactly that.
"""

from __future__ import annotations

import errno
import http.client
import json
import os
import socket
import threading
import time
import unittest
from unittest import mock

import httpx
import numpy as np

import speak_protocol as proto
import speak_telegram_daemon as d
from test_hold import temp_personas


class FakeEngines:
    whisper_model = "large-v3-turbo"

    def __init__(self) -> None:
        self.calls: list[np.ndarray] = []
        self.text = "hello there"
        self.gate: threading.Event | None = None   # set: transcribe blocks until it is set
        self.entered = threading.Event()

    def transcribe(self, audio: np.ndarray) -> str:
        self.calls.append(audio)
        self.entered.set()
        if self.gate is not None:
            self.gate.wait(10)
        return self.text


class FakeLink:
    def __init__(self) -> None:
        self.daemon: d.SpeakDaemon | None = None
        self.bound_during_start: bool | None = None

    def start(self) -> None:
        self.bound_during_start = self.daemon._http is not None or self.daemon._server is not None

    def stop(self) -> None:
        pass

    def info(self) -> dict:
        return {"state": proto.NONE, "generation": 0, "end_reason": None}

    def state(self) -> str:
        return proto.NONE

    def record_event(self, event: str, **fields) -> None:
        pass


OWNER_SESSION = d.Speaker("s-test", "algolearn-speak")   # the tool op these tests hold the lock with
SETTINGS = {"SPEAK_QUIET_HOURS": "22:00-07:00", "quiet_window": (1320, 420),
            "tz": None, "phrases": {"hold on": "hold", "continue": "resume"}}


class HttpCase(unittest.TestCase):
    def setUp(self):
        self.engines = FakeEngines()
        self.link = FakeLink()
        self.daemon = d.SpeakDaemon(self.link, self.engines, temp_personas(), SETTINGS, d.DebugControl(), port=0, http_port=0)
        self.link.daemon = self.daemon
        self.daemon.start()
        self.addCleanup(self.daemon.stop)
        self.http_port = self.daemon._http.server_address[1]
        self.tcp_port = self.daemon._server.server_address[1]
        self.url = f"http://127.0.0.1:{self.http_port}"

    def request(self, method: str, path: str, body: bytes | None = None, headers: dict | None = None):
        conn = http.client.HTTPConnection("127.0.0.1", self.http_port, timeout=10)
        self.addCleanup(conn.close)
        conn.request(method, path, body=body, headers=headers or {})
        resp = conn.getresponse()
        return resp, resp.read()


class TestRoutes(HttpCase):
    def test_health(self):
        resp, body = self.request("GET", "/health")
        self.assertEqual(resp.status, 200)
        self.assertEqual(resp.getheader("Content-Type"), "application/json")
        self.assertEqual(body, b'{"status":"ok","model":"large-v3-turbo","device":"cuda"}')

    def test_transcribe_converts_int16_exactly_as_the_old_service(self):
        pcm = np.array([0, 16384, -32768, 32767, -1], dtype=np.int16)
        resp, body = self.request("POST", "/", pcm.tobytes(), {"Content-Type": "application/octet-stream"})
        self.assertEqual(resp.status, 200)
        self.assertEqual(resp.getheader("Content-Type"), "application/json")
        self.assertEqual(json.loads(body), {"text": "hello there"})
        (audio,) = self.engines.calls
        self.assertEqual(audio.dtype, np.float32)
        np.testing.assert_array_equal(audio, pcm.astype(np.float32) / 32768.0)

    def test_livekit_client_call(self):
        """WhisperServiceSTT's request, verbatim: httpx POST, octet-stream, raise_for_status, .get("text")."""
        pcm = (np.sin(np.arange(16000) / 10) * 1000).astype(np.int16).tobytes()
        resp = httpx.post(self.url + "/", content=pcm, headers={"Content-Type": "application/octet-stream"},
                          timeout=30.0)
        resp.raise_for_status()
        self.assertEqual(resp.json().get("text", "").strip(), "hello there")
        self.assertEqual(len(self.engines.calls[0]), 16000)

    def test_empty_text_is_returned_not_raised(self):
        self.engines.text = ""
        resp, body = self.request("POST", "/", b"\x00\x00" * 100)
        self.assertEqual((resp.status, body), (200, b'{"text":""}'))

    def test_empty_body_reaches_the_engine(self):
        resp, _ = self.request("POST", "/", b"")
        self.assertEqual(resp.status, 200)
        self.assertEqual(len(self.engines.calls[0]), 0)

    def test_utf8_text_is_not_escaped(self):
        self.engines.text = "café"
        _, body = self.request("POST", "/", b"\x00\x00")
        self.assertEqual(body, '{"text":"café"}'.encode())

    def test_odd_byte_count_is_a_plain_500(self):
        resp, body = self.request("POST", "/", b"\x00\x00\x00")
        self.assertEqual(resp.status, 500)
        self.assertEqual(resp.getheader("Content-Type"), "text/plain; charset=utf-8")
        self.assertEqual(body, b"Internal Server Error")
        self.assertEqual(self.engines.calls, [])
        self.assertEqual(self.daemon.lock.snapshot(), (False, None, 0))

    def test_engine_failure_is_a_plain_500_and_releases_the_lock(self):
        def boom(_audio):
            raise RuntimeError("cuda out of memory")
        self.engines.transcribe = boom
        resp, body = self.request("POST", "/", b"\x00\x00")
        self.assertEqual((resp.status, body), (500, b"Internal Server Error"))
        self.assertEqual(self.daemon.lock.snapshot(), (False, None, 0))

    def test_wrong_method_is_405_with_allow(self):
        for method, path, allow in (("GET", "/", "POST"), ("POST", "/health", "GET"), ("HEAD", "/health", "GET"),
                                    ("PUT", "/", "POST")):
            with self.subTest(method=method, path=path):
                resp, body = self.request(method, path)
                self.assertEqual(resp.status, 405)
                self.assertEqual(resp.getheader("Allow"), allow)
                if method != "HEAD":
                    self.assertEqual(body, b'{"detail":"Method Not Allowed"}')
        self.assertEqual(self.engines.calls, [])

    def test_unknown_path_is_404(self):
        resp, body = self.request("GET", "/transcribe")
        self.assertEqual(resp.status, 404)
        self.assertEqual(resp.getheader("Content-Type"), "application/json")
        self.assertEqual(body, b'{"detail":"Not Found"}')

    def test_trailing_slash_redirects(self):
        resp, body = self.request("GET", "/health/?x=1", headers={"Host": "patrick:9000"})
        self.assertEqual(resp.status, 307)
        self.assertEqual(resp.getheader("Location"), "http://patrick:9000/health?x=1")
        self.assertEqual(body, b"")

    def test_query_string_is_ignored_by_routing(self):
        resp, _ = self.request("GET", "/health?verbose=1")
        self.assertEqual(resp.status, 200)

    def test_chunked_body(self):
        pcm = np.arange(10, dtype=np.int16).tobytes()
        resp, _ = self.request("POST", "/", iter([pcm[:7], pcm[7:]]))   # http.client sends an iterable chunked
        self.assertEqual(resp.status, 200)
        np.testing.assert_array_equal(self.engines.calls[0], np.arange(10, dtype=np.float32) / 32768.0)

    def test_bad_content_length_is_400(self):
        with socket.create_connection(("127.0.0.1", self.http_port), timeout=10) as s:
            s.sendall(b"POST / HTTP/1.1\r\nHost: x\r\nContent-Length: abc\r\n\r\n")
            data = _recv_all(s)   # headers and body arrive in separate sends; the server then closes
        self.assertTrue(data.startswith(b"HTTP/1.1 400"), data)
        self.assertTrue(data.endswith(b"Invalid HTTP request received."), data)

    def test_keep_alive_serves_several_requests_on_one_connection(self):
        conn = http.client.HTTPConnection("127.0.0.1", self.http_port, timeout=10)
        self.addCleanup(conn.close)
        for path, method, body in (("/health", "GET", None), ("/", "POST", b"\x01\x00"), ("/nope", "POST", b"xx"),
                                   ("/health", "GET", None)):
            conn.request(method, path, body=body)
            resp = conn.getresponse()
            resp.read()
            self.assertIn(resp.status, (200, 404))
        self.assertEqual(len(self.engines.calls), 1)


class TestAudioLock(HttpCase):
    def test_transcription_waits_for_a_tool_op_holding_the_lock(self):
        self.daemon.lock.acquire(proto.OP_CONVERSE, "10.0.0.5", OWNER_SESSION)
        result: dict = {}

        def post():
            result["resp"] = httpx.post(self.url + "/", content=b"\x00\x00" * 16, timeout=10)

        t = threading.Thread(target=post)
        t.start()
        deadline = time.time() + 5
        while self.daemon.lock.snapshot()[2] == 0 and time.time() < deadline:
            time.sleep(0.01)
        self.assertEqual(self.daemon.lock.snapshot(), (True, "converse (10.0.0.5)", 1))
        self.assertEqual(self.engines.calls, [])   # no inference while the tool op holds the lock
        self.daemon.lock.release()
        t.join(10)
        self.assertEqual(result["resp"].status_code, 200)
        self.assertEqual(len(self.engines.calls), 1)

    def test_transcription_holds_the_lock_and_status_names_it(self):
        self.engines.gate = threading.Event()
        t = threading.Thread(target=lambda: httpx.post(self.url + "/", content=b"\x00\x00", timeout=10))
        t.start()
        self.assertTrue(self.engines.entered.wait(5))
        with socket.create_connection(("127.0.0.1", self.tcp_port), timeout=10) as s:
            proto.send_msg(s, {"op": proto.OP_STATUS})
            status, _ = proto.recv_msg(s)
        self.assertEqual((status["busy"], status["current_tool"]), (True, "transcribe (127.0.0.1)"))
        self.engines.gate.set()
        t.join(10)
        self.assertEqual(self.daemon.lock.snapshot(), (False, None, 0))


def _recv_all(s: socket.socket) -> bytes:
    """Everything the server sends until it closes the connection."""
    data = b""
    while chunk := s.recv(65536):
        data += chunk
    return data


class TestHardening(HttpCase):
    def raw(self, request: bytes) -> bytes:
        with socket.create_connection(("127.0.0.1", self.http_port), timeout=10) as s:
            s.sendall(request)
            return _recv_all(s)

    def assert_413(self, data: bytes) -> None:
        self.assertTrue(data.startswith(b"HTTP/1.1 413"), data)
        self.assertIn(b"Connection: close", data)
        head, _, body = data.partition(b"\r\n\r\n")
        self.assertIn(b"exceeds the limit of", json.loads(body)["detail"].encode())
        self.assertEqual(self.engines.calls, [])

    def test_limit_is_15_minutes_of_16khz_mono_int16(self):
        self.assertEqual(d.MAX_BODY_BYTES, 15 * 60 * 16000 * 2)
        self.assertEqual(d.MAX_BODY_BYTES, 28_800_000)

    def test_content_length_over_the_limit_is_413_before_the_body_is_read(self):
        with self.assertLogs("speak.daemon", "WARNING"):
            data = self.raw(b"POST / HTTP/1.1\r\nHost: x\r\nContent-Length: %d\r\n\r\n" % (d.MAX_BODY_BYTES + 1))
        self.assert_413(data)   # answered and closed with no body bytes sent

    def test_content_length_at_the_limit_is_accepted(self):
        with mock.patch.object(d, "MAX_BODY_BYTES", 100):
            resp, _ = self.request("POST", "/", b"\x00" * 100)
        self.assertEqual(resp.status, 200)
        self.assertEqual(len(self.engines.calls[0]), 50)

    def test_chunked_body_over_the_limit_is_413_once_the_running_total_passes_it(self):
        with mock.patch.object(d, "MAX_BODY_BYTES", 100), self.assertLogs("speak.daemon", "WARNING"):
            # 60 + 60 > 100: refused at the second chunk's size line, whose bytes are never sent
            data = self.raw(b"POST / HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n\r\n"
                            b"3c\r\n" + b"\x00" * 60 + b"\r\n3c\r\n")
        self.assert_413(data)

    def test_one_huge_chunk_is_413_before_it_is_read(self):
        data = self.raw(b"POST / HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n\r\n"
                        + b"%x\r\n" % (d.MAX_BODY_BYTES + 1))
        self.assert_413(data)

    def test_negative_chunk_size_is_400_not_a_read_to_eof(self):
        # int("-1", 16) parses; read(-1) would read to EOF, past the size limit
        with mock.patch.object(d, "MAX_BODY_BYTES", 100), self.assertLogs("speak.daemon", "WARNING") as logs:
            with socket.create_connection(("127.0.0.1", self.http_port), timeout=10) as s:
                s.sendall(b"POST / HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n\r\n-1\r\n"
                          + b"\x00" * 200)   # the connection stays open: a read to EOF would hang here
                data = _recv_all(s)
        self.assertTrue(data.startswith(b"HTTP/1.1 400"), data)
        self.assertTrue(data.endswith(b"Invalid HTTP request received."), data)
        self.assertTrue(any("bad chunk size line b'-1\\r\\n'" in line for line in logs.output), logs.output)
        self.assertEqual(self.engines.calls, [])

    def test_client_that_left_while_queued_is_skipped(self):
        self.daemon.lock.acquire(proto.OP_CONVERSE, "10.0.0.5", OWNER_SESSION)
        s = socket.create_connection(("127.0.0.1", self.http_port), timeout=10)
        s.sendall(b"POST / HTTP/1.1\r\nHost: x\r\nContent-Length: 4\r\n\r\n\x00\x00\x00\x00")
        deadline = time.time() + 5
        while self.daemon.lock.snapshot()[2] == 0 and time.time() < deadline:
            time.sleep(0.01)
        self.assertEqual(self.daemon.lock.snapshot()[2], 1)   # queued behind the tool op
        s.close()   # the client gives up
        time.sleep(0.1)   # let the FIN arrive
        with self.assertLogs("speak.daemon", "WARNING") as logs:
            self.daemon.lock.release()
            deadline = time.time() + 5
            while self.daemon.lock.snapshot() != (False, None, 0) and time.time() < deadline:
                time.sleep(0.01)
            time.sleep(0.1)   # the handler logs after releasing the lock
        self.assertEqual(self.daemon.lock.snapshot(), (False, None, 0))
        self.assertEqual(self.engines.calls, [])
        self.assertTrue(any("left while queued; skipped" in line for line in logs.output), logs.output)

    def test_closing_daemon_is_503_without_inference(self):
        self.daemon.closing.set()
        resp, body = self.request("POST", "/", b"\x00\x00")
        self.assertEqual(resp.status, 503)
        self.assertEqual(resp.getheader("Content-Type"), "application/json")
        self.assertEqual(body, b'{"detail":"speak daemon is restarting"}')
        self.assertEqual(self.engines.calls, [])
        self.assertEqual(self.daemon.lock.snapshot(), (False, None, 0))

    def test_keepalive_is_enabled_on_the_connection(self):
        with mock.patch.object(d.proto, "enable_keepalive", wraps=proto.enable_keepalive) as ka:
            resp, _ = self.request("GET", "/health")
        self.assertEqual(resp.status, 200)
        ka.assert_called_once()

    def test_timeout_is_comfortably_above_upload_gaps(self):
        self.assertEqual(d._WhisperHandler.timeout, d.HTTP_TIMEOUT_SECONDS)
        self.assertGreater(d.HTTP_TIMEOUT_SECONDS, 5)   # httpx's keep-alive expiry: the client closes first


class TestTimeouts(HttpCase):
    def setUp(self):
        patcher = mock.patch.object(d._WhisperHandler, "timeout", 0.3)
        patcher.start()
        self.addCleanup(patcher.stop)
        super().setUp()

    def test_idle_connection_is_closed(self):
        with self.assertLogs("speak.daemon", "WARNING") as logs:
            with socket.create_connection(("127.0.0.1", self.http_port), timeout=10) as s:
                t0 = time.time()
                self.assertEqual(s.recv(1), b"")   # EOF: the server closed the idle connection
                self.assertLess(time.time() - t0, 5)
            time.sleep(0.1)
        self.assertTrue(any("timed out" in line for line in logs.output), logs.output)

    def test_idle_keep_alive_connection_is_closed_after_a_request(self):
        with socket.create_connection(("127.0.0.1", self.http_port), timeout=10) as s:
            s.sendall(b"GET /health HTTP/1.1\r\nHost: x\r\n\r\n")
            data = _recv_all(s)   # the reply, then EOF once the connection sits idle
        self.assertTrue(data.startswith(b"HTTP/1.1 200"), data)

    def test_stalled_body_is_logged_408_and_closed(self):
        with self.assertLogs("speak.daemon", "WARNING") as logs:
            with socket.create_connection(("127.0.0.1", self.http_port), timeout=10) as s:
                s.sendall(b"POST / HTTP/1.1\r\nHost: x\r\nContent-Length: 10\r\n\r\n\x00\x00")
                data = _recv_all(s)
        self.assertTrue(data.startswith(b"HTTP/1.1 408"), data)
        self.assertTrue(data.endswith(b'{"detail":"Request Timeout"}'), data)
        self.assertTrue(any("body read timed out" in line for line in logs.output), logs.output)
        self.assertEqual(self.engines.calls, [])


    def test_request_queued_longer_than_the_timeout_still_gets_its_text(self):
        # the socket timeout covers reads from the client, not the wait for the audio lock
        self.daemon.lock.acquire(proto.OP_CONVERSE, "10.0.0.5", OWNER_SESSION)
        result: dict = {}

        def post():
            result["resp"] = httpx.post(self.url + "/", content=b"\x00\x00" * 16, timeout=30)

        t = threading.Thread(target=post)
        t.start()
        time.sleep(2.0)   # queued for about 7x the handler timeout
        self.assertEqual(self.daemon.lock.snapshot(), (True, "converse (10.0.0.5)", 1))
        self.daemon.lock.release()
        t.join(10)
        self.assertEqual(result["resp"].status_code, 200)
        self.assertEqual(result["resp"].json(), {"text": "hello there"})
        self.assertEqual(len(self.engines.calls), 1)
        self.assertEqual(self.daemon.lock.snapshot(), (False, None, 0))


class TestPeerLeft(unittest.TestCase):
    """_peer_left is shared by the 8765 tool handler and the port-9000 handler."""

    def setUp(self):
        self.ours, self.theirs = socket.socketpair()
        self.addCleanup(self.ours.close)
        self.addCleanup(self.theirs.close)

    def test_quiet_peer_is_still_there(self):
        self.assertFalse(d._peer_left(self.ours))

    def test_closed_peer_is_gone(self):
        self.theirs.close()
        self.assertTrue(d._peer_left(self.ours))

    def test_a_socket_error_on_the_peek_means_the_peer_is_gone(self):
        # a dead peer found by TCP keepalive: select reports readable, recv raises ETIMEDOUT or EHOSTUNREACH
        self.theirs.sendall(b"x")   # readable, so _peer_left peeks
        for err in (errno.ETIMEDOUT, errno.EHOSTUNREACH):
            with self.subTest(errno=errno.errorcode[err]):
                sock = mock.Mock(fileno=self.ours.fileno, recv=mock.Mock(side_effect=OSError(err, os.strerror(err))))
                self.assertTrue(d._peer_left(sock))
                sock.recv.assert_called_once_with(1, socket.MSG_PEEK)


class TestStartup(unittest.TestCase):
    def test_nothing_binds_before_the_link_has_started(self):
        link = FakeLink()
        daemon = d.SpeakDaemon(link, FakeEngines(), temp_personas(), SETTINGS, d.DebugControl(), port=0, http_port=0)
        link.daemon = daemon
        daemon.start()
        self.addCleanup(daemon.stop)
        self.assertIs(link.bound_during_start, False)

    def test_a_link_failure_binds_nothing(self):
        class FailingLink(FakeLink):
            def start(self):
                raise RuntimeError("telegram login revoked")
        daemon = d.SpeakDaemon(FailingLink(), FakeEngines(), temp_personas(), SETTINGS, d.DebugControl(), port=0, http_port=0)
        with self.assertRaises(RuntimeError):
            daemon.start()
        self.assertIsNone(daemon._server)
        self.assertIsNone(daemon._http)

    def test_an_http_port_in_use_fails_the_start_and_releases_the_tool_port(self):
        holder = socket.socket()
        holder.bind(("0.0.0.0", 0))
        holder.listen()
        self.addCleanup(holder.close)
        taken = holder.getsockname()[1]
        with socket.socket() as probe:   # a free port for 8765's stand-in, checked free after the failure
            probe.bind(("0.0.0.0", 0))
            tool_port = probe.getsockname()[1]
        daemon = d.SpeakDaemon(FakeLink(), FakeEngines(), temp_personas(), SETTINGS, d.DebugControl(), port=tool_port,
                               http_port=taken)
        daemon.link.daemon = daemon
        with self.assertRaises(OSError):
            daemon.start()
        self.assertIsNone(daemon._server)
        with socket.socket() as again:
            again.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            again.bind(("0.0.0.0", tool_port))   # released, not left listening

    def test_the_tool_port_still_answers(self):
        link = FakeLink()
        daemon = d.SpeakDaemon(link, FakeEngines(), temp_personas(), SETTINGS, d.DebugControl(), port=0, http_port=0)
        link.daemon = daemon
        daemon.start()
        self.addCleanup(daemon.stop)
        with socket.create_connection(("127.0.0.1", daemon._server.server_address[1]), timeout=10) as s:
            proto.send_msg(s, {"op": proto.OP_PING, proto.FIELD_PROTOCOL: proto.PROTOCOL_VERSION})
            reply, _ = proto.recv_msg(s)
        self.assertIs(reply["ok"], True)

    def test_default_ports(self):
        daemon = d.SpeakDaemon(FakeLink(), FakeEngines(), temp_personas(), SETTINGS, d.DebugControl())
        self.assertEqual((daemon.port, daemon.http_port), (8765, 9000))


if __name__ == "__main__":
    unittest.main()
