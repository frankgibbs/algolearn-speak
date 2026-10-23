"""call()'s pre-dial checks (owner decisions 2026-10-23): a connected call comes before quiet
hours, and every refusal without dialing is recorded as a call_refused event."""

from __future__ import annotations

import os
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import speak_telegram as tg  # noqa: E402
import speak_telegram_daemon as d  # noqa: E402
from test_floor import A, B, FloorServiceCase  # noqa: E402
from test_hold import NEVER_LEFT  # noqa: E402
from test_speak_telegram import LinkCase, events_named  # noqa: E402

GREETING = "Hi Frank, it's Angie. The backtest finished; do you have a minute?"


class TestCallChecks(FloorServiceCase):
    def setUp(self):
        super().setUp()
        self.link._state = tg.NONE

    def quiet(self, on: bool):
        p = mock.patch.object(d, "in_quiet_hours", lambda window, now: on)
        p.start()
        self.addCleanup(p.stop)

    def refusals(self) -> list[dict]:
        """call_refused events, recorded unbound (no call's id); never bound to the last call."""
        self.assertEqual([e for e, f in self.link.events if e == d.CALL_REFUSED_EVENT], [])
        return [f for e, f in self.link.unbound if e == d.CALL_REFUSED_EVENT]

    def dials(self) -> int:
        return sum(1 for e in self.log if e[0] == "dial")

    def test_a_connected_call_is_already_connected_in_quiet_hours(self):
        self.quiet(True)
        self.link._state = tg.CONNECTED
        sess = self.req(A)
        self.assertEqual(self.svc.call(sess, GREETING, False, 120, 2.0, 45), {"outcome": tg.ALREADY_CONNECTED})
        self.assertEqual(sess.generation, self.svc.key(1))   # the session now talks on that call
        self.assertEqual(self.refusals(), [])   # "already connected" is not a refusal
        self.assertEqual(self.dials(), 0)

    def test_already_connected_in_quiet_hours_leaves_the_turn_and_the_floor(self):
        self.quiet(True)
        self.link._state = tg.CONNECTED
        self.svc.converse(self.req(A), self.TEXT, 120, 2.0, 45, NEVER_LEFT)   # A holds the turn and the floor
        turn, floor = self.svc._turn, self.svc.lock._floor
        self.assertEqual(self.svc.call(self.req(B), GREETING, False, 120, 2.0, 45), {"outcome": tg.ALREADY_CONNECTED})
        self.assertEqual((self.svc._turn, self.svc.lock._floor), (turn, floor))
        self.assertEqual(floor, A)
        self.assertEqual(self.takes(), ["algolearn-speak"])
        self.assertEqual(self.releases(), [])
        self.assertEqual(self.refusals(), [])

    def test_already_connected_comes_before_the_greeting_length(self):
        self.link._state = tg.CONNECTED
        self.assertEqual(self.svc.call(self.req(A), "Hi.", False, 120, 2.0, 45), {"outcome": tg.ALREADY_CONNECTED})
        self.assertEqual(self.refusals(), [])

    def test_quiet_hours_refuse_without_dialing_and_record_it(self):
        self.quiet(True)
        reply = self.svc.call(self.req(A), GREETING, False, 120, 2.0, 45)
        self.assertEqual(reply, {"outcome": "not called: quiet hours (22:00-07:00)"})
        (refusal,) = self.refusals()
        self.assertRegex(refusal.pop("local_time"), r"^\d\d:\d\d$")
        self.assertEqual(refusal, {"project": "algolearn-speak", "reason": d.REFUSED_QUIET_HOURS})
        self.assertEqual(self.dials(), 0)

    def test_quiet_hours_come_before_the_greeting_length(self):
        self.quiet(True)
        self.svc.call(self.req(A), "Hi.", False, 120, 2.0, 45)
        self.assertEqual([f["reason"] for f in self.refusals()], [d.REFUSED_QUIET_HOURS])

    def test_override_dials_in_quiet_hours(self):
        self.quiet(True)
        self.assertEqual(self.svc.call(self.req(A), GREETING, True, 120, 2.0, 45)["outcome"], tg.ANSWERED)
        self.assertEqual(self.refusals(), [])
        self.assertEqual(self.dials(), 1)

    def test_an_empty_greeting_is_refused_before_synthesis_and_recorded(self):
        self.quiet(False)
        with self.assertRaisesRegex(ValueError, "non-empty greeting"):
            self.svc.call(self.req(A), "  ", False, 120, 2.0, 45)
        self.assertEqual([e for e in self.log if e[0] == "synthesize"], [])
        self.assertEqual(self.refusals(), [{"project": "algolearn-speak", "reason": d.REFUSED_EMPTY_GREETING}])
        self.assertEqual(self.dials(), 0)

    def test_a_short_greeting_is_refused_and_recorded(self):
        self.quiet(False)
        self.engines.synthesize = lambda text, voice: np.zeros(d.TTS_RATE * 2, dtype=np.float32)   # 2 s
        with self.assertRaisesRegex(ValueError, "only 2.0s when spoken"):
            self.svc.call(self.req(A), "Hi.", False, 120, 2.0, 45)
        self.assertEqual(self.refusals(), [{"project": "algolearn-speak", "reason": d.REFUSED_SHORT_GREETING,
                                            "greeting_s": 2.0}])
        self.assertEqual(self.dials(), 0)

    def test_a_call_that_dials_records_no_refusal(self):
        self.quiet(False)
        for outcome in (tg.NOT_ANSWERED, tg.BUSY):
            self.link.outcome = outcome
            self.assertEqual(self.svc.call(self.req(A), GREETING, False, 120, 2.0, 45), {"outcome": outcome})
        self.assertEqual(self.refusals(), [])
        self.assertEqual(self.dials(), 2)


class TestUnboundEvents(LinkCase):
    """The real link: a refusal recorded after a call has ended carries no call_id and no
    elapsed_s, though the link still remembers that call's id."""

    def test_an_unbound_event_after_a_call_carries_no_call_id(self):
        self.assertEqual(self.link.call(np.zeros(2400, dtype=np.float32), 24000), tg.ANSWERED)
        self.remote_hang_up()
        self.wait_until(lambda: self.link.state() == tg.NONE)
        (ended,) = events_named(self.link, "ended")
        self.assertIsNotNone(ended["call_id"])
        self.link.record_unbound_event(d.CALL_REFUSED_EVENT, project="algolearn-speak", reason=d.REFUSED_QUIET_HOURS)
        (refused,) = events_named(self.link, d.CALL_REFUSED_EVENT)
        self.assertIsNone(refused["call_id"])
        self.assertNotIn("elapsed_s", refused)


if __name__ == "__main__":
    unittest.main()
