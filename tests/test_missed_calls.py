"""Missed calls (owner decision 2026-11-02): each call's events name its caller, a call that
was not answered, was busy, or was hung up during the greeting puts the calling session on
the MissedCalls list (persisted), the session leaves it when it next talks with the owner on
a connected call, and the list is served as the quick op `missed_calls` and GET /missed-calls."""

from __future__ import annotations

import http.client
import json
import os
import shutil
import sys
import tempfile
import unittest
import zoneinfo
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import speak_protocol as proto  # noqa: E402
import speak_server as client  # noqa: E402
import speak_telegram as tg  # noqa: E402
import speak_telegram_daemon as d  # noqa: E402
from test_floor import A, B, FloorServiceCase  # noqa: E402
from test_hold import NEVER_LEFT, SETTINGS, FakeEngines, temp_missed, temp_personas  # noqa: E402
from test_speak_telegram import CALLER, LinkCase, events_named  # noqa: E402

TZ = zoneinfo.ZoneInfo("America/Los_Angeles")
GREETING = "Hi Frank, it's Angie. The backtest finished; do you have a minute?"
MISSES = (tg.NOT_ANSWERED, tg.BUSY, tg.HUNG_UP_DURING_GREETING)


class StoreCase(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="speak-missed-test-")
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.path = os.path.join(self.dir, "missed_calls.json")


class TestStore(StoreCase):
    def test_a_missing_file_is_created_empty(self):
        store = d.MissedCalls.open(self.path, TZ)
        self.assertEqual(store.list(), [])
        with open(self.path) as f:
            self.assertEqual(json.load(f), {"version": 1, "missed": []})

    def test_a_miss_survives_a_restart(self):
        d.MissedCalls.open(self.path, TZ).record(CALLER, tg.NOT_ANSWERED)
        (entry,) = d.MissedCalls.open(self.path, TZ).list()
        self.assertEqual({k: entry[k] for k in ("session_id", "project", "persona", "outcome", "missed")},
                         {"session_id": "feedc0de", "project": "algolearn-speak", "persona": "Angie",
                          "outcome": tg.NOT_ANSWERED, "missed": 1})
        self.assertEqual(entry["time"], entry["first_time"])
        self.assertRegex(entry["time"], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d[-+]\d\d:\d\d$")

    def test_a_second_miss_updates_the_entry(self):
        store = d.MissedCalls.open(self.path, TZ)
        first = store.record(CALLER, tg.NOT_ANSWERED)
        with mock.patch.object(d.datetime, "datetime", wraps=d.datetime.datetime) as dt:
            dt.now.return_value = d.datetime.datetime(2030, 1, 1, 9, 0, tzinfo=TZ)
            store.record(CALLER, tg.BUSY)
        (entry,) = store.list()
        self.assertEqual((entry["missed"], entry["outcome"], entry["first_time"]), (2, tg.BUSY, first["time"]))
        self.assertEqual(entry["time"], "2030-01-01T09:00:00-08:00")

    def test_clear_removes_only_that_session(self):
        store = d.MissedCalls.open(self.path, TZ)
        other = tg.Caller("Nova", "algolearn-speak", "other")   # same project, another session
        store.record(CALLER, tg.NOT_ANSWERED)
        store.record(other, tg.BUSY)
        self.assertEqual(store.clear(d.Speaker("feedc0de", "algolearn-speak"), "test")["persona"], "Angie")
        self.assertEqual([e["session_id"] for e in d.MissedCalls.open(self.path, TZ).list()], ["other"])
        self.assertIsNone(store.clear(d.Speaker("feedc0de", "algolearn-speak"), "test"))

    def test_a_clear_with_no_entry_writes_nothing(self):
        store = d.MissedCalls.open(self.path, TZ)
        with mock.patch.object(store, "_write") as write:
            store.clear(d.Speaker("nobody", "x"), "test")
        write.assert_not_called()

    def test_only_miss_outcomes_are_recorded(self):
        store = d.MissedCalls.open(self.path, TZ)
        for outcome in (tg.ANSWERED, tg.HUNG_UP, tg.ALREADY_CONNECTED):
            with self.assertRaisesRegex(ValueError, "not a missed-call outcome"):
                store.record(CALLER, outcome)

    def test_a_malformed_file_stops_the_daemon(self):
        for body in ('{"version": 2, "missed": []}', '[]', '{"version": 1, "missed": [{"session_id": "x"}]}'):
            with open(self.path, "w") as f:
                f.write(body)
            with self.assertRaises(RuntimeError):
                d.MissedCalls.open(self.path, TZ)
        with open(self.path, "w") as f:
            f.write("not json")
        with self.assertRaises(json.JSONDecodeError):
            d.MissedCalls.open(self.path, TZ)

    def test_a_missing_directory_is_refused(self):
        with self.assertRaisesRegex(RuntimeError, "does not exist"):
            d.MissedCalls.open(os.path.join(self.dir, "nope", "missed.json"), TZ)

    def test_a_failed_write_leaves_memory_as_it_was(self):
        store = d.MissedCalls.open(self.path, TZ)
        with mock.patch.object(d.os, "replace", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                store.record(CALLER, tg.NOT_ANSWERED)
        self.assertEqual(store.list(), [])

    def test_settings_require_the_file(self):
        self.assertIn("SPEAK_MISSED_CALLS_FILE", d.REQUIRED_SPEAK_ENV)

    def test_compose_sets_the_file_on_the_state_volume(self):
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docker-compose.yml")) as f:
            self.assertIn("SPEAK_MISSED_CALLS_FILE: /var/lib/speak/missed_calls.json", f.read())


class TestServiceMissedCalls(FloorServiceCase):
    def setUp(self):
        super().setUp()
        self.link._state = tg.NONE

    def call(self, speaker: d.Speaker, outcome: str) -> dict:
        self.link.outcome = outcome
        return self.svc.call(self.req(speaker), GREETING, True, 120, 2.0, 45)

    def test_each_miss_outcome_records_the_caller(self):
        for outcome in MISSES:
            self.assertEqual(self.call(A, outcome), {"outcome": outcome})
        (entry,) = self.svc.missed.list()
        self.assertEqual({k: entry[k] for k in ("session_id", "project", "persona", "outcome", "missed")},
                         {"session_id": "session-a", "project": "algolearn-speak", "persona": "Angie",
                          "outcome": tg.HUNG_UP_DURING_GREETING, "missed": 3})

    def test_the_link_is_told_who_is_calling(self):
        seen = []
        self.link.call = lambda audio, rate, caller: seen.append(caller) or tg.NOT_ANSWERED
        self.svc.call(self.req(B), GREETING, True, 120, 2.0, 45)
        self.assertEqual(seen, [tg.Caller("Angie", "algolearn-hedge", "session-b")])

    def test_hung_up_by_another_session_is_not_a_miss(self):
        self.call(A, tg.HUNG_UP)
        self.assertEqual(self.svc.missed.list(), [])

    def test_an_answered_call_returns_the_miss(self):
        self.call(A, tg.NOT_ANSWERED)
        self.call(B, tg.BUSY)
        self.assertEqual(self.call(A, tg.ANSWERED)["outcome"], tg.ANSWERED)
        self.assertEqual([e["session_id"] for e in self.svc.missed.list()], ["session-b"])

    def test_talking_on_a_connected_call_returns_the_miss(self):
        self.call(B, tg.NOT_ANSWERED)
        self.call(A, tg.ANSWERED)   # A's call; B queued its converse behind it
        self.assertEqual(len(self.svc.missed.list()), 1)
        self.svc.converse(self.req(B), self.TEXT, 120, 2.0, 45, NEVER_LEFT)
        self.assertEqual(self.svc.missed.list(), [])

    def test_already_connected_alone_does_not_return_the_miss(self):
        self.call(B, tg.NOT_ANSWERED)
        self.call(A, tg.ANSWERED)
        self.assertEqual(self.call(B, tg.ANSWERED), {"outcome": tg.ALREADY_CONNECTED})
        self.assertEqual(len(self.svc.missed.list()), 1)   # the converse that follows returns it

    def test_a_failed_record_still_reports_the_outcome(self):
        with mock.patch.object(self.svc.missed, "_write", side_effect=OSError("read-only")):
            with self.assertRaisesRegex(RuntimeError, r"call outcome: not answered.*recording the missed call "
                                                      r"failed: OSError: read-only"):
                self.call(A, tg.NOT_ANSWERED)


class TestLinkTagsTheCaller(LinkCase):
    def test_every_event_of_the_call_names_the_caller(self):
        self.assertEqual(self.link.call(np.zeros(2400, dtype=np.float32), 24000, CALLER), tg.ANSWERED)
        self.remote_hang_up()
        self.wait_until(lambda: self.link.state() == tg.NONE)
        bound = [e for e in self.link.events.recent(500) if e["call_id"] is not None]
        self.assertGreater(len(bound), 3)
        for event in bound:
            self.assertEqual(event["caller"], {"persona": "Angie", "project": "algolearn-speak",
                                               "session_id": "feedc0de"}, event)
        self.assertEqual(len(events_named(self.link, "requested")), 1)

    def test_a_call_without_a_caller_is_refused(self):
        with self.assertRaises(TypeError):
            self.link.call(np.zeros(2400, dtype=np.float32), 24000, None)


class NoCallLink:
    def info(self) -> dict:
        return {"state": proto.NONE, "generation": 0, "end_reason": None}

    def state(self) -> str:
        return proto.NONE

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def record_event(self, event: str, **fields) -> None:
        pass


class TestServedList(unittest.TestCase):
    def setUp(self):
        self.missed = temp_missed()
        self.missed.record(CALLER, tg.BUSY)
        self.daemon = d.SpeakDaemon(NoCallLink(), FakeEngines([]), temp_personas(), self.missed, SETTINGS,
                                    d.DebugControl(), port=0, http_port=0)
        self.daemon.start()
        self.addCleanup(self.daemon.stop)

    def test_the_mcp_tool_returns_the_list(self):
        with mock.patch.object(client, "_daemon", ("127.0.0.1", self.daemon._server.server_address[1])):
            reply = client._missed_calls_sync()
        self.assertEqual(reply[proto.FIELD_PROTOCOL], proto.PROTOCOL_VERSION)
        self.assertEqual(reply["missed_calls"], self.missed.list())
        self.assertEqual(reply["missed_calls"][0]["outcome"], tg.BUSY)

    def test_any_protocol_may_read_it(self):
        import socket
        with socket.create_connection(("127.0.0.1", self.daemon._server.server_address[1]), timeout=10) as s:
            proto.send_msg(s, {"op": proto.OP_MISSED_CALLS})
            reply, _ = proto.recv_msg(s)
        self.assertTrue(reply["ok"])
        self.assertEqual(len(reply["missed_calls"]), 1)

    def test_http_get_missed_calls(self):
        conn = http.client.HTTPConnection("127.0.0.1", self.daemon._http.server_address[1], timeout=10)
        self.addCleanup(conn.close)
        conn.request("GET", "/missed-calls")
        resp = conn.getresponse()
        self.assertEqual((resp.status, resp.getheader("Content-Type")), (200, "application/json"))
        self.assertEqual(json.loads(resp.read()), {"missed_calls": self.missed.list()})
        conn.request("POST", "/missed-calls", body=b"")
        resp = conn.getresponse()
        resp.read()
        self.assertEqual((resp.status, resp.getheader("Allow")), (405, "GET"))


if __name__ == "__main__":
    unittest.main()
