"""The wire between the thin client and a real SpeakDaemon (fake link and engines; no CUDA,
no Telegram): the protocol version check, the client's identity (protocol 3), the
session-switch announcement end to end, and a hold that outlives its client (the client's
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
from test_hold import SETTINGS, FakeEngines, temp_personas  # noqa: E402

ID_A = {proto.FIELD_SESSION_ID: "session-a", proto.FIELD_PROJECT: "algolearn-speak"}
ID_B = {proto.FIELD_SESSION_ID: "session-b", proto.FIELD_PROJECT: "algolearn-hedge"}


def as_session(ident: dict, fn, *args):
    """Run a client sync body as one session: its identity (the keys of `ident` without a
    leading "_"), its call expectation (ident["_expect"]) and the persona it last saw
    (ident["_seen"]), both saved back into `ident` afterwards."""
    with mock.patch.object(client, "_identity", {k: v for k, v in ident.items() if not k.startswith("_")}), \
            mock.patch.object(client, "_call_generation", ident.get("_expect")), \
            mock.patch.object(client, "_persona_seen", ident.get("_seen")):
        try:
            return fn(*args)
        finally:
            ident["_expect"] = client._call_generation
            ident["_seen"] = client._persona_seen


def seen_key(daemon: d.SpeakDaemon, ident: dict) -> str:
    """The persona_seen of a session that has seen its project's current persona (assigning
    the project first, as the daemon's admit does)."""
    project = ident[proto.FIELD_PROJECT]
    daemon.personas.ensure(project)
    return daemon.personas.key(project)


def short_floor(daemon: d.SpeakDaemon, link, idle_seconds: float) -> None:
    """Before start(): the daemon's lock (and its service's) with a floor idle time of
    `idle_seconds` instead of FLOOR_IDLE_SECONDS, so a test does not wait 45 s for a lapse."""
    daemon.lock = daemon.service.lock = d.AudioLock(link.record_event, idle_seconds=idle_seconds)


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

    def record_unbound_event(self, event: str, **fields) -> None:
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
        self.daemon = d.SpeakDaemon(self.link, FakeEngines([]), temp_personas(), SETTINGS, d.DebugControl(), port=0, http_port=0)
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

    def test_a_persona_op_on_a_daemon_that_does_not_know_it_reports_the_protocol_mismatch(self):
        bodies = {proto.OP_PERSONA: (client._persona_sync,), proto.OP_PERSONA_NEW: (client._new_persona_sync,),
                  proto.OP_PERSONA_SET: (client._set_persona_sync, None, None, 1.1, None),
                  proto.OP_PERSONA_ADD: (client._add_to_personality_sync, "Dry humor.")}
        for op, (fn, *args) in bodies.items():
            # What a protocol-3 daemon sends for an op it does not know (no version in an error).
            addr = self.serve({"ok": False, "error": f"unknown op {op!r}", "kind": proto.KIND_VALUE})
            with mock.patch.object(client, "_daemon", addr):
                with self.assertRaises(ValueError) as caught:
                    as_session(dict(ID_A), fn, *args)
            message = str(caught.exception)
            self.assertTrue(message.startswith(f"protocol mismatch: speak daemon at 127.0.0.1:{addr[1]} does not "
                                               f"know {op!r}"), message)
            self.assertIn(f"this client's protocol {proto.PROTOCOL_VERSION}; deploy the daemon", message)


class TestHoldOutlivingItsClient(DaemonCase):
    def test_a_client_that_closes_during_the_hold_frees_the_lock_without_hanging_up(self):
        s = socket.create_connection(self.addr, timeout=10)
        proto.send_msg(s, {"op": proto.OP_SPEAK, proto.FIELD_PROTOCOL: proto.PROTOCOL_VERSION,
                           **ID_A, "text": "One. Two.", "expect_generation": None,
                           proto.FIELD_PERSONA_SEEN: seen_key(self.daemon, ID_A)})
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
        with mock.patch.object(client, "_daemon", self.addr), mock.patch.object(client, "_identity", ID_A), \
                mock.patch.object(client, "_persona_seen", seen_key(self.daemon, ID_A)):
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
                           **ID_A, "text": "One. Two.", "expect_generation": None,
                           proto.FIELD_PERSONA_SEEN: seen_key(self.daemon, ID_A)})
        proto.recv_msg(s)
        self.assertTrue(self.link.held.wait(5))
        time.sleep(0.3)
        self.assertIsNone(self.link.hold_result)
        self.assertTrue(self.daemon.lock.snapshot()[0])


class TestIdentity(DaemonCase):
    """Protocol 3: every tool op names its session and project; a missing, empty or
    unspeakable one is refused before the lock, as a missing protocol version is."""
    V = {proto.FIELD_PROTOCOL: proto.PROTOCOL_VERSION}

    def refused(self, header: dict) -> str:
        reply = self.exchange(header)
        self.assertIs(reply["ok"], False, header)
        self.assertEqual(reply["kind"], proto.KIND_VALUE, header)
        self.assertFalse(self.link.held.is_set())
        self.assertEqual(self.daemon.lock.snapshot(), (False, None, 0))
        return reply["error"]

    def test_a_tool_op_without_a_session_id_is_refused_before_the_lock(self):
        error = self.refused({"op": proto.OP_SPEAK, **self.V, proto.FIELD_PROJECT: "algolearn-speak",
                              "text": "One.", "expect_generation": None})
        self.assertEqual(error, f"speak client sent no session_id (got None); every tool op of protocol "
                                f"{proto.PROTOCOL_VERSION} carries a non-empty session_id and project; "
                                "update the client (git pull) and reconnect /mcp")

    def test_an_empty_or_missing_project_is_refused(self):
        for project in ("", "   ", None, 5):
            header = {"op": proto.OP_CONVERSE, **self.V, proto.FIELD_SESSION_ID: "s", "text": "One.",
                      "expect_generation": None}
            if project is not None:
                header[proto.FIELD_PROJECT] = project
            self.assertIn("speak client sent no project", self.refused(header), project)

    def test_a_project_with_nothing_to_speak_is_refused(self):
        error = self.refused({"op": proto.OP_LISTEN, **self.V, proto.FIELD_SESSION_ID: "s",
                              proto.FIELD_PROJECT: "--", "expect_generation": None})
        self.assertEqual(error, "project '--' has no letters or digits to speak")

    def test_every_tool_op_needs_it_but_the_quick_ops_do_not(self):
        for op in proto.LOCKED_OPS:
            self.assertIn("session_id", self.refused({"op": op, **self.V}), op)
        self.assertIs(self.exchange({"op": proto.OP_PING, **self.V})["ok"], True)

    def test_the_client_sends_its_identity_on_every_tool_op(self):
        seen = []
        real = proto.send_msg

        def spy(sock, header, payload=b""):
            seen.append(header)
            return real(sock, header, payload)
        with mock.patch.object(client, "_daemon", self.addr), mock.patch.object(client, "_identity", ID_A), \
                mock.patch.object(proto, "send_msg", spy):
            client._request(proto.OP_PING, {}, 5)
            client._hang_up_sync()
        ping, hang_up = (h for h in seen if "op" in h)
        self.assertNotIn(proto.FIELD_SESSION_ID, ping)
        self.assertEqual({k: hang_up[k] for k in ID_A}, ID_A)


class TurnLink:
    """A call that plays everything and hears the owner each listen (or nobody, `silent`).
    call() connects a new generation; every played buffer is logged."""

    def __init__(self) -> None:
        self._state = tg.NONE
        self.generation = 0
        self.log: list = []
        self.silent = False
        self.events = tg.CallEventLog()

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def info(self) -> dict:
        return {"state": self._state, "generation": self.generation, "end_reason": None}

    def state(self) -> str:
        return self._state

    def record_event(self, event: str, **fields) -> None:
        self.events.record(None, event, **fields)

    def record_unbound_event(self, event: str, **fields) -> None:
        self.events.record(None, event, **fields)

    def call(self, audio, rate) -> str:
        self.generation += 1
        self._state = tg.CONNECTED
        self.log.append("greeting")
        return tg.ANSWERED

    def play(self, audio, rate) -> None:
        self.log.append(("play", len(audio)))

    def play_watched(self, segments, rate, classify, accept):
        n = sum(len(a) for _tag, a in segments)
        self.log.append("speech")
        return tg.PlayOutcome(None, n, None, n / rate)

    def record(self, max_seconds, silence_seconds, start_timeout_seconds, cue, rate):
        self.log.append("record")
        return tg.TIMEOUT if self.silent else np.zeros(16000, dtype=np.float32)

    def hang_up(self) -> str:
        self._state = tg.NONE
        return tg.HUNG_UP


class TestSessionSwitchOverTheWire(unittest.TestCase):
    """Two client sessions on one call, through the real daemon and the client's own code."""

    def setUp(self):
        self.link = TurnLink()
        self.engines = FakeEngines([])
        self.daemon = d.SpeakDaemon(self.link, self.engines, temp_personas(), SETTINGS, d.DebugControl(), port=0, http_port=0)
        short_floor(self.daemon, self.link, 0.2)   # B's ops wait out A's floor (docs/DESIGN_FLOOR.md)
        self.daemon.start()
        self.addCleanup(self.daemon.stop)
        p = mock.patch.object(client, "_daemon", ("127.0.0.1", self.daemon._server.server_address[1]))
        p.start()
        self.addCleanup(p.stop)
        self.generation = None

    def session(self, ident: dict) -> dict:
        """A session that has read its persona (the persona tool), as an agent does first."""
        ident = dict(ident)
        as_session(ident, client._persona_sync)
        return ident

    def as_session(self, ident: dict, fn, *args):
        return as_session(ident, fn, *args)

    def test_the_session_that_called_talks_on_then_another_takes_over(self):
        a, b = self.session(ID_A), self.session(ID_B)
        self.assertTrue(self.as_session(a, client._call_sync, "Hi Frank, it's Angie.", True, 120, 2.0, 45)
                        .startswith(proto.ANSWERED))
        self.assertEqual(self.as_session(a, client._converse_sync, "One.", 120, 2.0, 45),
                         "hi Angie" + proto.AUDIO_NOTE)   # the greeting introduced it: no switch
        self.assertEqual(self.engines.log.count(("synthesize", "Angie, algolearn hedge")), 0)
        got = self.as_session(b, client._converse_sync, "Two.", 120, 2.0, 45)
        self.assertEqual(got, '[took over the call from "algolearn-speak"; the owner heard the switch cue and '
                              "your persona's name and this project's name first]\nhi Angie" + proto.AUDIO_NOTE)
        self.assertIn(("synthesize", "Angie, algolearn hedge"), self.engines.log)
        self.assertEqual(self.as_session(b, client._speak_sync, "Three."), "spoke for 0.5s" + proto.AUDIO_NOTE)
        got = self.as_session(a, client._listen_sync, 120, 2.0, 45)
        self.assertEqual(got, '[took over the call from "algolearn-hedge"; the owner heard the switch cue and '
                              "your persona's name and this project's name first]\nhi Angie")
        (first, second) = [e for e in self.link.events.recent(50) if e["event"] == d.SESSION_SWITCH_EVENT]
        self.assertEqual((first["from_project"], first["to_project"], first["tool"], first["spoken"]),
                         ("algolearn-speak", "algolearn-hedge", "converse", "Angie, algolearn hedge"))
        self.assertEqual((second["from_project"], second["to_project"], second["tool"], second["spoken"]),
                         ("algolearn-hedge", "algolearn-speak", "listen", "Angie, algolearn speak, still listening"))
        self.assertIn(("synthesize", "Angie, algolearn speak, still listening"), self.engines.log)

    def test_already_connected_during_quiet_hours(self):
        """Owner decision 2026-10-23: a connected call is checked before quiet hours, so a session
        coming back without override gets "already connected" and talks on the call."""
        a, b = self.session(ID_A), self.session(ID_B)
        self.as_session(a, client._call_sync, "Hi Frank, it's Angie.", True, 120, 2.0, 45)
        with mock.patch.object(d, "in_quiet_hours", lambda window, now: True):
            self.assertEqual(self.as_session(b, client._call_sync, "Hi Frank, it's Angie again.", False, 120, 2.0, 45),
                             proto.ALREADY_CONNECTED)
        self.assertEqual(b["_expect"], a["_expect"])   # b now expects the connected call
        self.assertEqual([e for e in self.link.events.recent(100) if e["event"] == d.CALL_REFUSED_EVENT], [])
        self.assertEqual(self.link.log.count("greeting"), 1)   # dialed once, by a
        self.assertEqual(self.as_session(a, client._converse_sync, "One.", 120, 2.0, 45),
                         "hi Angie" + proto.AUDIO_NOTE)   # the turn is still a's: no switch

    def test_a_failure_after_the_announcement_still_says_who_took_over(self):
        a, b = self.session(ID_A), self.session(ID_B)
        self.as_session(a, client._call_sync, "Hi Frank, it's Angie.", True, 120, 2.0, 45)
        self.link.silent = True
        with self.assertRaises(TimeoutError) as cm:
            self.as_session(b, client._listen_sync, 120, 2.0, 45)
        self.assertEqual(str(cm.exception), '[took over the call from "algolearn-speak"; the owner heard the switch '
                                            "cue and your persona's name and this project's name first]\n"
                                            "no speech detected within 45s")
        with self.assertRaises(TimeoutError) as cm:   # the owner already heard it: no second announcement
            self.as_session(b, client._listen_sync, 120, 2.0, 45)
        self.assertEqual(str(cm.exception), "no speech detected within 45s")


class TestPersonasOverTheWire(unittest.TestCase):
    """Protocol 4 (docs/DESIGN_PERSONAS.md): persona_seen on every tool op, the automatic
    assignment of a new project, the refusal of words written for another persona, and the
    persona ops, through the real daemon and the client's own code."""

    def setUp(self):
        self.link = TurnLink()
        self.engines = FakeEngines([])
        self.daemon = d.SpeakDaemon(self.link, self.engines, temp_personas(), SETTINGS, d.DebugControl(), port=0,
                                    http_port=0)
        self.daemon.start()
        self.addCleanup(self.daemon.stop)
        self.addr = ("127.0.0.1", self.daemon._server.server_address[1])
        p = mock.patch.object(client, "_daemon", self.addr)
        p.start()
        self.addCleanup(p.stop)

    def exchange(self, header: dict) -> dict:
        with socket.create_connection(self.addr, timeout=10) as s:
            proto.send_msg(s, {proto.FIELD_PROTOCOL: proto.PROTOCOL_VERSION, **header})
            reply, _ = proto.recv_msg(s)
        return reply

    def connect(self, ident: dict) -> None:
        """`ident` reads its persona and places the call."""
        as_session(ident, client._persona_sync)
        self.assertTrue(as_session(ident, client._call_sync, "Hi Frank, it's Angie.", True, 120, 2.0, 45)
                        .startswith(proto.ANSWERED))

    def test_the_protocol_is_4(self):
        self.assertEqual(proto.PROTOCOL_VERSION, 4)

    def test_a_tool_op_without_persona_seen_is_refused_before_the_lock(self):
        reply = self.exchange({"op": proto.OP_LISTEN, **ID_A, "expect_generation": None})
        self.assertEqual(reply["kind"], proto.KIND_VALUE)
        self.assertEqual(reply["error"], f"speak client sent no persona_seen; every tool op of protocol "
                                         f"{proto.PROTOCOL_VERSION} carries it; update the client (git pull) "
                                         "and reconnect /mcp")
        self.assertEqual(self.daemon.lock.snapshot(), (False, None, 0))

    def test_a_new_projects_first_words_are_refused_with_the_house_persona(self):
        a = dict(ID_A)
        with self.assertRaises(ValueError) as cm:
            as_session(a, client._call_sync, "Hi Frank, it's me, calling about the build.", True, 120, 2.0, 45)
        self.assertEqual(str(cm.exception),
                         "[persona: you are Angie, the house persona; voice af_heart 1; speed 1; personality: (none yet)]\n"
                         "Nothing was spoken (or dialed). Rewrite your words as Angie and send again.")
        self.assertEqual((self.link.generation, self.link.log), (0, []))   # never dialed
        self.assertEqual(a["_seen"], self.daemon.personas.key("algolearn-speak"))
        refused = [e for e in self.link.events.recent(50) if e["event"] == d.PERSONA_REFUSED_EVENT]
        self.assertEqual([(e["tool"], e["seen"]) for e in refused], [(proto.OP_CALL, None)])
        self.assertTrue(as_session(a, client._call_sync, "Hi Frank, it's Angie, calling about the build.", True,
                                   120, 2.0, 45).startswith(proto.ANSWERED))   # rewritten, sent again

    def test_a_stale_persona_seen_refuses_speak_converse_and_call_with_nothing_played(self):
        a, b = dict(ID_A), dict(ID_A, **{proto.FIELD_SESSION_ID: "session-a2"})
        self.connect(a)
        as_session(b, client._persona_sync)
        as_session(b, client._set_persona_sync, "Nova", None, None, None)   # a's persona changes under it
        played = list(self.link.log)
        for fn, args in ((client._speak_sync, ("One.",)), (client._converse_sync, ("One.", 120, 2.0, 45)),
                         (client._call_sync, ("Hi Frank, it's Angie again.", True, 120, 2.0, 45))):
            stale = dict(a)
            with self.assertRaises(ValueError) as cm:
                as_session(stale, fn, *args)
            self.assertTrue(str(cm.exception).startswith("[persona: you are Nova, the house persona; "), fn)
            self.assertTrue(str(cm.exception).endswith("Rewrite your words as Nova and send again."), fn)
        self.assertEqual((self.link.log, self.link.generation), (played, 1))

    def test_a_refusal_resets_a_persona_seen_that_is_ahead_of_the_daemon(self):
        a = dict(ID_A)
        self.connect(a)
        pid, revision = a["_seen"].rsplit(":", 1)
        current = a["_seen"]
        a["_seen"] = f"{pid}:{int(revision) + 2}"   # seen A:n+2 while the daemon is at A:n
        with self.assertRaises(ValueError) as cm:
            as_session(a, client._speak_sync, "One.")
        self.assertTrue(str(cm.exception).startswith("[persona: "))
        self.assertEqual(a["_seen"], current)
        self.assertEqual(as_session(a, client._speak_sync, "One.").split("\n")[-1], "(audio: telegram)")

    def test_listen_and_hang_up_are_not_refused_and_carry_the_new_persona(self):
        a, b = dict(ID_A), dict(ID_A, **{proto.FIELD_SESSION_ID: "session-a2"})
        self.connect(a)
        as_session(b, client._persona_sync)
        as_session(b, client._set_persona_sync, "Nova", None, None, None)
        self.assertEqual(as_session(a, client._listen_sync, 120, 2.0, 45),
                         "[persona: you are Nova, the house persona; voice af_heart 1; speed 1; personality: (none yet)]\n"
                         "hi Angie")
        self.assertEqual(as_session(a, client._listen_sync, 120, 2.0, 45), "hi Angie")   # seen now: no line
        as_session(b, client._add_to_personality_sync, "Dry humor.")
        self.assertEqual(as_session(a, client._hang_up_sync),
                         "[persona: you are Nova, the house persona; voice af_heart 1; speed 1; personality: Dry humor.]\n"
                         + tg.HUNG_UP)

    def test_the_reply_carries_the_persona_only_when_it_changed(self):
        a = dict(ID_A)
        self.connect(a)
        seen = a["_seen"]
        self.assertEqual(as_session(a, client._converse_sync, "One.", 120, 2.0, 45), "hi Angie" + proto.AUDIO_NOTE)
        self.assertEqual(a["_seen"], seen)

    def test_a_voice_or_speed_change_refuses_nobody_and_is_not_pushed(self):
        """Design 2026-10-22: the words do not depend on the sound, so another session's voice
        or speed change neither refuses a's speak, converse or call nor attaches a block."""
        a, b = dict(ID_A), dict(ID_A, **{proto.FIELD_SESSION_ID: "session-a2"})
        self.connect(a)
        seen = a["_seen"]
        as_session(b, client._persona_sync)
        as_session(b, client._set_persona_sync, None, {"bf_emma": 1}, 1.2, None)
        self.assertEqual(as_session(a, client._speak_sync, "One.").split("\n")[-1], "(audio: telegram)")
        self.assertFalse(as_session(a, client._speak_sync, "Two.").startswith("[persona"))
        self.assertEqual(as_session(a, client._converse_sync, "Three.", 120, 2.0, 45), "hi Angie" + proto.AUDIO_NOTE)
        self.assertEqual(as_session(a, client._call_sync, "Hi Frank, it's Angie again.", True, 120, 2.0, 45),
                         proto.ALREADY_CONNECTED)
        self.assertEqual(a["_seen"], seen)
        self.assertEqual(self.daemon.personas.persona_of("algolearn-speak").speed, 1.2)

    def test_persona_ops_answer_while_another_op_holds_the_lock(self):
        a = dict(ID_A)
        self.connect(a)
        release = threading.Event()
        entered = threading.Event()

        def slow_record(*args):
            entered.set()
            release.wait(10)
            return np.zeros(16000, dtype=np.float32)
        self.link.record = slow_record
        t = threading.Thread(target=lambda: as_session(dict(a), client._listen_sync, 120, 2.0, 45))
        t.start()
        self.addCleanup(t.join, 10)
        self.addCleanup(release.set)
        self.assertTrue(entered.wait(5))
        self.assertTrue(self.daemon.lock.snapshot()[0])
        got = as_session(dict(ID_B), client._persona_sync)
        self.assertEqual(got[proto.REPLY_PERSONA]["name"], "Angie")
        self.assertEqual(got[proto.REPLY_PERSONA]["shared_with"], ["algolearn-speak"])
        self.assertTrue(self.daemon.lock.snapshot()[0])   # still held: the persona op did not wait for it

    def test_persona_ops_are_versioned_and_need_the_identity(self):
        for op in proto.PERSONA_OPS:
            reply = self.exchange({"op": op, proto.FIELD_PROTOCOL: 1, **ID_A})
            self.assertEqual(reply["kind"], proto.KIND_VALUE, op)
            self.assertIn("does not match daemon protocol", reply["error"], op)
            reply = self.exchange({"op": op})
            self.assertEqual(reply["kind"], proto.KIND_VALUE, op)
            self.assertIn("speak client sent no session_id", reply["error"], op)

    def test_the_client_sends_persona_seen_on_tool_ops_and_identity_on_persona_ops(self):
        seen = []
        real = proto.send_msg

        def spy(sock, header, payload=b""):
            seen.append(header)
            return real(sock, header, payload)
        a = dict(ID_A)
        with mock.patch.object(proto, "send_msg", spy):
            as_session(a, client._persona_sync)
            as_session(a, client._hang_up_sync)
        persona, hang_up = (h for h in seen if "op" in h)
        self.assertEqual({k: persona[k] for k in ID_A}, ID_A)
        self.assertNotIn(proto.FIELD_PERSONA_SEEN, persona)
        self.assertEqual(hang_up[proto.FIELD_PERSONA_SEEN], a["_seen"])


class TestPersonaLine(unittest.TestCase):
    """The client's result lines: the switch line, then the persona line, then the held line."""

    BLOCK = {"id": "p1", "words_revision": 3, "name": "Nova", "house": False, "voice": {"af_bella": 0.7, "bf_emma": 0.3}, "speed": 1.1,
             "personality": ["Dry humor.", "Short answers."], "shared_with": ["algolearn-hedge"]}

    def test_the_persona_line_comes_after_the_switch_line_and_before_the_held_line(self):
        reply = {"ok": True, "seconds": 2.0, proto.REPLY_HELD_SECONDS: 12.0,
                 proto.REPLY_SWITCHED_FROM: "algolearn-hedge", proto.REPLY_PERSONA: self.BLOCK}
        self.assertEqual(client._speak_result(reply).split("\n"), [
            '[took over the call from "algolearn-hedge"; the owner heard the switch cue and your persona\'s name '
            "and this project's name first]",
            "[persona: you are Nova; voice af_bella 0.7, bf_emma 0.3; speed 1.1; personality: Dry humor. / "
            "Short answers.; shared with algolearn-hedge]",
            "[held 12s, then resumed]",
            "spoke for 2.0s", "(audio: telegram)"])

    def test_persona_seen_follows_replies_and_errors(self):
        with mock.patch.object(client, "_persona_seen", None):
            client._apply_persona({"ok": True})
            self.assertIsNone(client._persona_seen)
            client._apply_persona({"ok": False, "kind": proto.KIND_VALUE, "error": "x", proto.REPLY_PERSONA: self.BLOCK})
            self.assertEqual(client._persona_seen, "p1:3")

    def test_persona_seen_only_moves_forward(self):
        def reply(pid: str, revision: int) -> dict:
            return {"ok": True, proto.REPLY_PERSONA: {**self.BLOCK, "id": pid, "words_revision": revision}}
        with mock.patch.object(client, "_persona_seen", "p1:3"):
            client._apply_persona(reply("p1", 2))   # an older reply arriving late
            self.assertEqual(client._persona_seen, "p1:3")
            client._apply_persona(reply("p1", 3))
            self.assertEqual(client._persona_seen, "p1:3")
            client._apply_persona(reply("p1", 4))
            self.assertEqual(client._persona_seen, "p1:4")
            client._apply_persona(reply("p2", 1))   # another persona: always applies
            self.assertEqual(client._persona_seen, "p2:1")
            client._apply_persona(reply("p1", 1))
            self.assertEqual(client._persona_seen, "p1:1")


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
