"""The wire between the thin client and a real SpeakDaemon (fake link and engines; no CUDA,
no Telegram): the protocol version check, and a hold that outlives its client (the client's
cancelled tool call closes its socket; the daemon ends the hold and frees the lock).
"""

from __future__ import annotations

import os
import socket
import sys
import threading
import time
import unittest
from unittest import mock

import anyio
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import speak_protocol as proto  # noqa: E402
import speak_server as client  # noqa: E402
import speak_telegram as tg  # noqa: E402
import speak_telegram_daemon as d  # noqa: E402
from test_hold import SETTINGS, FakeEngines  # noqa: E402


class HoldingLink:
    """A connected call. play_watched is cut by "hold on" in its first segment; watch_for
    then holds until the request's client leaves (or 10 s, which fails the test)."""

    def __init__(self) -> None:
        self._state = tg.CONNECTED
        self.held = threading.Event()
        self.hold_result = None
        self.hung_up = 0
        self.events = tg.CallEventLog()

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def info(self) -> dict:
        return {"state": self._state, "generation": 1, "end_reason": None}

    def state(self) -> str:
        return self._state

    def record_event(self, event: str, **fields) -> None:
        self.events.record(None, event, **fields)

    def play_watched(self, segments, rate, classify, accept):
        tag, _audio = next(iter(segments))
        return tg.PlayOutcome(proto.CMD_HOLD, 100, tag, 0.1)

    def watch_for(self, timeout_s, classify, accept, client_left):
        self.held.set()
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if client_left():
                self.hold_result = tg.CLIENT_LEFT
                return tg.CLIENT_LEFT
            time.sleep(0.02)
        self.hold_result = tg.TIMEOUT
        return tg.TIMEOUT

    def hang_up(self) -> str:
        self.hung_up += 1
        self._state = tg.NONE
        return tg.HUNG_UP


class DaemonCase(unittest.TestCase):
    def setUp(self):
        self.link = HoldingLink()
        self.daemon = d.SpeakDaemon(self.link, FakeEngines([]), SETTINGS, d.DebugControl(), port=0, http_port=0)
        self.daemon.start()
        self.addCleanup(self.daemon.stop)
        self.addr = ("127.0.0.1", self.daemon._server.server_address[1])

    def exchange(self, header: dict) -> dict:
        with socket.create_connection(self.addr, timeout=10) as s:
            proto.send_msg(s, header)
            reply, _ = proto.recv_msg(s)
        return reply

    def wait_free(self) -> None:
        deadline = time.monotonic() + 5
        while self.daemon.lock.snapshot() != (False, None, 0):
            if time.monotonic() > deadline:
                self.fail(f"the lock was not released: {self.daemon.lock.snapshot()}")
            time.sleep(0.02)


class TestProtocolVersion(DaemonCase):
    V = {proto.FIELD_PROTOCOL: proto.PROTOCOL_VERSION}

    def test_a_tool_op_without_a_version_is_refused_before_the_lock(self):
        reply = self.exchange({"op": proto.OP_SPEAK, "text": "One.", "expect_generation": None})
        self.assertEqual(reply["kind"], proto.KIND_VALUE)
        self.assertEqual(reply["error"], f"speak client sent no protocol version (a client older than protocol "
                                         f"{proto.PROTOCOL_VERSION}); update the client (git pull) and reconnect /mcp")
        self.assertFalse(self.link.held.is_set())
        self.assertEqual(self.daemon.lock.snapshot(), (False, None, 0))

    def test_a_mismatched_version_is_refused(self):
        reply = self.exchange({"op": proto.OP_CONVERSE, proto.FIELD_PROTOCOL: 1, "text": "One."})
        self.assertEqual(reply["kind"], proto.KIND_VALUE)
        self.assertEqual(reply["error"], f"speak client protocol 1 does not match daemon protocol "
                                         f"{proto.PROTOCOL_VERSION}; update the client (git pull) and reconnect /mcp")

    def test_ping_checks_the_version(self):
        self.assertEqual(self.exchange({"op": proto.OP_PING})["kind"], proto.KIND_VALUE)
        self.assertIs(self.exchange({"op": proto.OP_PING, **self.V})["ok"], True)

    def test_the_diagnostic_ops_serve_any_client_and_name_the_daemon_version(self):
        for header in ({"op": proto.OP_STATUS}, {"op": proto.OP_STATUS, proto.FIELD_PROTOCOL: 1},
                       {"op": proto.OP_DEBUG, "lines": 5}, {"op": proto.OP_SET_DEBUG, "enabled": False}):
            reply = self.exchange(header)
            self.assertIs(reply["ok"], True, header)
            self.assertEqual(reply[proto.FIELD_PROTOCOL], proto.PROTOCOL_VERSION, header)

    def test_the_client_sends_its_version_on_every_op(self):
        seen = []
        real = proto.send_msg

        def spy(sock, header, payload=b""):
            seen.append(header)
            return real(sock, header, payload)
        with mock.patch.object(client, "_daemon", self.addr), mock.patch.object(proto, "send_msg", spy):
            client._request(proto.OP_PING, {}, 5)
            self.assertEqual(client._status_sync()["call"], tg.CONNECTED)
            client._call_debug_sync(5)
        self.assertEqual([h[proto.FIELD_PROTOCOL] for h in seen if "op" in h], [proto.PROTOCOL_VERSION] * 3)


class TestStatusAgainstAnOtherDaemon(unittest.TestCase):
    """The client's status reports a daemon of another protocol in its `call` field."""

    def serve(self, reply: dict) -> tuple[str, int]:
        srv = socket.create_server(("127.0.0.1", 0))
        self.addCleanup(srv.close)

        def run():
            conn, _ = srv.accept()
            with conn:
                proto.recv_msg(conn)
                proto.send_msg(conn, reply)
        threading.Thread(target=run, daemon=True).start()
        return srv.getsockname()

    def test_an_unversioned_daemon(self):
        addr = self.serve({"ok": True, "busy": False, "current_tool": None, "waiting": 0,
                           "call": {"state": tg.NONE}})
        with mock.patch.object(client, "_daemon", addr):
            got = client._status_sync()
        self.assertEqual((got["busy"], got["current_tool"], got["waiting"]), (None, None, None))
        self.assertTrue(got["call"].startswith(f"protocol mismatch: speak client protocol {proto.PROTOCOL_VERSION} "
                                               "does not match daemon protocol None"), got)

    def test_a_newer_daemon(self):
        addr = self.serve({"ok": True, proto.FIELD_PROTOCOL: proto.PROTOCOL_VERSION + 1, "busy": False,
                           "current_tool": None, "waiting": 0, "call": {"state": tg.NONE}})
        with mock.patch.object(client, "_daemon", addr):
            got = client._status_sync()
        self.assertIn(f"daemon protocol {proto.PROTOCOL_VERSION + 1}", got["call"])


class TestHoldOutlivingItsClient(DaemonCase):
    def test_a_client_that_closes_during_the_hold_frees_the_lock_without_hanging_up(self):
        s = socket.create_connection(self.addr, timeout=10)
        proto.send_msg(s, {"op": proto.OP_SPEAK, proto.FIELD_PROTOCOL: proto.PROTOCOL_VERSION,
                           "text": "One. Two.", "expect_generation": None})
        started, _ = proto.recv_msg(s)
        self.assertEqual(started["phase"], proto.PHASE_STARTED)
        self.assertTrue(self.link.held.wait(5))
        self.assertEqual(self.daemon.lock.snapshot()[:2], (True, "speak (127.0.0.1)"))
        s.close()
        self.wait_free()
        self.assertIs(self.link.hold_result, tg.CLIENT_LEFT)
        self.assertEqual(self.link.hung_up, 0)
        self.assertEqual(self.link.state(), tg.CONNECTED)

    def test_a_cancelled_tool_call_closes_its_socket_and_ends_the_hold(self):
        async def cancelled_converse():
            with anyio.move_on_after(0.5) as scope:
                await client._run(client._converse_sync, "One. Two.", 120, 2.0, 45)
            return scope.cancelled_caught

        # On its own thread: anyio.run on the main thread would replace (and orphan, unclosed)
        # the main-thread loop pytgcalls creates at import (pytgcalls/sync.py, get_event_loop).
        got: list[bool] = []
        with mock.patch.object(client, "_daemon", self.addr):
            t = threading.Thread(target=lambda: got.append(anyio.run(cancelled_converse)))
            t.start()
            t.join(10)
        self.assertEqual(got, [True])
        self.wait_free()
        self.assertIs(self.link.hold_result, tg.CLIENT_LEFT)
        self.assertEqual(self.link.hung_up, 0)

    def test_a_held_op_that_is_not_cancelled_keeps_the_lock(self):
        s = socket.create_connection(self.addr, timeout=10)
        self.addCleanup(s.close)
        proto.send_msg(s, {"op": proto.OP_SPEAK, proto.FIELD_PROTOCOL: proto.PROTOCOL_VERSION,
                           "text": "One. Two.", "expect_generation": None})
        proto.recv_msg(s)
        self.assertTrue(self.link.held.wait(5))
        time.sleep(0.3)
        self.assertIsNone(self.link.hold_result)
        self.assertTrue(self.daemon.lock.snapshot()[0])


class TestAbort(unittest.TestCase):
    def test_a_cancel_before_the_request_is_sent_stops_it(self):
        abort = client._Abort()
        abort.cancel()
        a, b = socket.socketpair()
        self.addCleanup(a.close)
        self.addCleanup(b.close)
        with self.assertRaisesRegex(RuntimeError, "cancelled before its request was sent"):
            abort.attach(a)

    def test_cancel_shuts_the_attached_socket(self):
        abort = client._Abort()
        a, b = socket.socketpair()
        self.addCleanup(a.close)
        self.addCleanup(b.close)
        abort.attach(a)
        abort.cancel()
        b.settimeout(5)
        self.assertEqual(b.recv(1), b"")   # the peer sees EOF

    def test_a_direct_sync_call_has_no_abort(self):
        self.assertIsNone(client._abort_var.get())


if __name__ == "__main__":
    unittest.main()
