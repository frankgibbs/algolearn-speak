"""The floor on the shared call (docs/DESIGN_FLOOR.md, owner decisions 2026-10-20).

The session that last played audio on the call holds the floor: its ops go ahead of every
other session's until it ends with a speak, or sends nothing for the idle time after its
last op ended. Three levels: AudioLock with real threads and a short idle time; SpeakService
against the scripted link and fake engines of test_hold; and a real SpeakDaemon driven by
the client's own sync bodies (no CUDA, no Telegram).
"""

from __future__ import annotations

import os
import socket
import sys
import threading
import time
import unittest
from unittest import mock

import httpx
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import speak_protocol as proto  # noqa: E402
import speak_server as client  # noqa: E402
import speak_telegram as tg  # noqa: E402
import speak_telegram_daemon as d  # noqa: E402
from test_daemon_wire import ID_A, ID_B, TurnLink, as_session, seen_key, short_floor  # noqa: E402
from test_hold import NEVER_LEFT, SETTINGS, FakeEngines, HoldServiceCase, ScriptedLink, locked_service, temp_personas  # noqa: E402

A = d.Speaker("session-a", "algolearn-speak")
B = d.Speaker("session-b", "algolearn-hedge")
C = d.Speaker("session-c", "algolearn-forex")


def wait_until(predicate, what: str, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out waiting for {what}")
        time.sleep(0.005)


# ------------------------------------------------------------------ AudioLock

class Events:
    """Stands in for link.record_event."""

    def __init__(self) -> None:
        self.items: list[tuple[str, dict]] = []

    def __call__(self, event: str, **fields) -> None:
        self.items.append((event, fields))

    def of(self, event: str) -> list[dict]:
        return [f for e, f in self.items if e == event]


class Waiter(threading.Thread):
    """Acquires the lock as `speaker` (None: a transcription), notes the grant, holds the lock
    until `finish` is set, then releases."""

    def __init__(self, lock: d.AudioLock, name: str, speaker: d.Speaker | None, order: list) -> None:
        super().__init__(daemon=True)
        self.lock = lock
        self.label = name
        self.speaker = speaker
        self.order = order
        self.granted = threading.Event()
        self.granted_at = 0.0
        self.finish = threading.Event()

    def run(self) -> None:
        tool = d.TRANSCRIBE_TOOL if self.speaker is None else proto.OP_CONVERSE
        self.lock.acquire(tool, "10.0.0.5", self.speaker)
        self.granted_at = time.monotonic()
        self.order.append(self.label)
        self.granted.set()
        self.finish.wait(10)
        self.lock.release()


class LockCase(unittest.TestCase):
    IDLE = 0.3
    LONG_IDLE = 2.0   # where a timing bound must not flake under load

    def setUp(self):
        self.events = Events()
        self.lock = d.AudioLock(self.events, idle_seconds=self.IDLE)
        self.order: list[str] = []
        self.waiters: list[Waiter] = []

    def tearDown(self):
        for w in self.waiters:
            w.finish.set()
        for w in self.waiters:
            w.join(5)

    def queue(self, name: str, speaker: d.Speaker | None) -> Waiter:
        """Start a waiter and return once its ticket is queued (or it holds the lock)."""
        before = self.lock.snapshot()[2]
        w = Waiter(self.lock, name, speaker, self.order)
        self.waiters.append(w)
        w.start()
        wait_until(lambda: self.lock.snapshot()[2] > before or w.granted.is_set(), f"{name} to queue")
        return w

    def hold_floor(self, speaker: d.Speaker) -> None:
        """`speaker` runs an op that plays audio (takes the floor) and is still running."""
        self.lock.acquire(proto.OP_CONVERSE, "10.0.0.5", speaker)
        self.lock.take_floor(speaker)

    def done(self, w: Waiter) -> None:
        """`w`'s op ends."""
        w.finish.set()
        w.join(5)


class TestAudioLockFloor(LockCase):
    def test_the_holders_op_queued_after_another_sessions_goes_first(self):
        self.hold_floor(A)
        b = self.queue("B", B)
        a = self.queue("A", A)
        self.lock.release()
        self.assertTrue(a.granted.wait(2))
        self.assertFalse(b.granted.is_set())
        self.assertEqual(self.order, ["A"])

    def test_without_a_floor_it_is_plain_fifo(self):
        self.lock.acquire(proto.OP_CONVERSE, "10.0.0.5", A)   # no audio played: no floor
        b = self.queue("B", B)
        a = self.queue("A", A)
        c = self.queue("C", C)
        self.lock.release()
        self.assertTrue(b.granted.wait(2))
        self.done(b)
        self.assertTrue(a.granted.wait(2))
        self.done(a)
        self.assertTrue(c.granted.wait(2))
        self.assertEqual(self.order, ["B", "A", "C"])
        self.assertEqual(self.events.items, [])

    def test_a_free_lock_waits_for_the_idle_lapse_then_goes_to_the_next_session(self):
        self.hold_floor(A)
        b = self.queue("B", B)
        released = time.monotonic()
        self.lock.release()
        self.assertFalse(b.granted.wait(self.IDLE * 0.5))
        self.assertEqual(self.lock.snapshot(), (False, None, 1))   # free, and B still waits
        self.assertTrue(b.granted.wait(2))
        self.assertGreaterEqual(b.granted_at - released, self.IDLE * 0.9)
        (release,) = self.events.of(d.FLOOR_RELEASE_EVENT)
        self.assertEqual((release["project"], release["reason"], release["waiting"]), ("algolearn-speak", "idle", 1))

    def test_the_deadline_is_cleared_while_the_holder_runs_and_restarts_at_its_release(self):
        self.hold_floor(A)
        self.lock.release()   # the idle count starts
        self.lock.acquire(proto.OP_CONVERSE, "10.0.0.5", A)   # A is back before the lapse
        b = self.queue("B", B)
        time.sleep(self.IDLE * 2)   # longer than the idle time, while A's op runs
        released = time.monotonic()
        self.lock.release()
        self.assertTrue(b.granted.wait(2))
        self.assertGreaterEqual(b.granted_at - released, self.IDLE * 0.9)   # counted from this release
        self.assertEqual([r["reason"] for r in self.events.of(d.FLOOR_RELEASE_EVENT)], ["idle"])

    def test_yield_floor_hands_the_lock_to_the_head_of_the_queue_at_the_release(self):
        self.lock = d.AudioLock(self.events, idle_seconds=self.LONG_IDLE)
        self.hold_floor(A)
        b = self.queue("B", B)
        c = self.queue("C", C)
        self.lock.yield_floor(A, d.FLOOR_SPEAK)
        released = time.monotonic()
        self.lock.release()
        self.assertTrue(b.granted.wait(2))
        self.assertLess(b.granted_at - released, self.LONG_IDLE * 0.5)   # not the idle time
        self.assertFalse(c.granted.is_set())
        (release,) = self.events.of(d.FLOOR_RELEASE_EVENT)
        self.assertEqual((release["project"], release["reason"], release["waiting"]), ("algolearn-speak", "speak", 2))

    def test_yield_floor_by_a_session_that_does_not_hold_it_changes_nothing(self):
        self.hold_floor(A)
        self.lock.yield_floor(B, d.FLOOR_HANG_UP)
        b = self.queue("B", B)
        self.lock.release()
        self.assertFalse(b.granted.wait(self.IDLE * 0.5))
        self.assertTrue(b.granted.wait(2))
        self.assertEqual([r["reason"] for r in self.events.of(d.FLOOR_RELEASE_EVENT)], ["idle"])

    def test_yield_floor_counts_the_queued_tool_ops_but_not_transcriptions(self):
        # Owner decision 2026-10-28: the count the sign-off hang-up reads, taken with the floor's
        # clear while the speak still holds the lock.
        self.hold_floor(A)
        self.queue("B", B)
        self.queue("transcribe", None)
        self.queue("C", C)
        self.assertEqual(self.lock.yield_floor(A, d.FLOOR_SPEAK), 2)
        self.lock.release()

    def test_yield_floor_with_nothing_queued_counts_zero_and_without_the_floor_none(self):
        self.hold_floor(A)
        self.assertIsNone(self.lock.yield_floor(B, d.FLOOR_SPEAK))   # B does not hold it
        self.queue("transcribe", None)
        self.assertEqual(self.lock.yield_floor(A, d.FLOOR_SPEAK), 0)
        self.assertIsNone(self.lock.yield_floor(A, d.FLOOR_SPEAK))   # already yielded
        self.lock.release()

    def test_a_lapse_with_nobody_waiting_is_recorded_once_by_the_next_acquire(self):
        self.hold_floor(A)
        self.lock.release()
        time.sleep(self.IDLE * 2)
        self.assertEqual(self.events.of(d.FLOOR_RELEASE_EVENT), [])   # nobody looked yet
        self.lock.acquire(proto.OP_CONVERSE, "10.0.0.5", B)   # not delayed: the floor has lapsed
        self.lock.release()
        self.lock.acquire(proto.OP_CONVERSE, "10.0.0.5", B)
        self.lock.release()
        (release,) = self.events.of(d.FLOOR_RELEASE_EVENT)
        self.assertEqual((release["reason"], release["waiting"]), ("idle", 0))   # seen before B's ticket queues
        self.assertGreaterEqual(release["held_s"], round(self.IDLE * 2, 1) - 0.1)

    def test_a_transcription_skips_the_floor_and_leaves_the_deadline_alone(self):
        self.lock = d.AudioLock(self.events, idle_seconds=0.6)
        self.hold_floor(A)
        b = self.queue("B", B)
        released = time.monotonic()
        self.lock.release()   # the idle count starts
        t = self.queue("transcribe", None)
        self.assertTrue(t.granted.wait(2))   # behind B in the queue, but goes first
        self.assertLess(t.granted_at - released, 0.3)
        self.assertFalse(b.granted.is_set())
        time.sleep(0.25)
        t_released = time.monotonic()
        self.done(t)
        self.assertTrue(b.granted.wait(2))
        self.assertGreaterEqual(b.granted_at - released, 0.6 * 0.9)
        self.assertLess(b.granted_at - t_released, 0.6 * 0.9)   # not counted from the transcription's release
        self.assertEqual(self.order, ["transcribe", "B"])

    def test_take_floor_records_a_take_and_a_repeat_records_nothing(self):
        self.hold_floor(A)
        self.lock.take_floor(A)
        self.lock.release()
        self.lock.acquire(proto.OP_CONVERSE, "10.0.0.6", B)   # once A's floor has lapsed
        self.lock.take_floor(B)
        self.lock.release()
        self.assertEqual(self.events.of(d.FLOOR_TAKE_EVENT),
                         [{"project": "algolearn-speak", "from_project": None},
                          {"project": "algolearn-hedge", "from_project": None}])

    def test_take_floor_refuses_a_session_that_does_not_hold_the_lock(self):
        with self.assertRaisesRegex(RuntimeError, "without holding the audio lock"):
            self.lock.take_floor(A)   # the lock is free
        self.lock.acquire(proto.OP_CONVERSE, "10.0.0.5", A)
        with self.assertRaisesRegex(RuntimeError, "without holding the audio lock"):
            self.lock.take_floor(B)   # held, but by A
        self.lock.release()
        t = self.queue("transcribe", None)
        self.assertTrue(t.granted.wait(2))
        with self.assertRaisesRegex(RuntimeError, "without holding the audio lock"):
            self.lock.take_floor(A)   # held by a transcription
        self.done(t)
        self.assertEqual(self.events.of(d.FLOOR_TAKE_EVENT), [])

    def test_the_holders_op_that_arrives_in_time_keeps_the_floor_through_a_long_transcription(self):
        self.lock = d.AudioLock(self.events, idle_seconds=1.0)
        self.hold_floor(A)
        released = time.monotonic()
        self.lock.release()   # the idle count starts
        b = self.queue("B", B)
        t = self.queue("transcribe", None)
        self.assertTrue(t.granted.wait(2))
        a = self.queue("A", A)   # well before the deadline, while the transcription holds the lock
        self.assertLess(time.monotonic() - released, 0.5)
        time.sleep(max(0.0, released + 1.3 - time.monotonic()))   # the transcription outlasts the deadline
        self.done(t)
        self.assertTrue(a.granted.wait(2))
        self.assertFalse(b.granted.is_set())
        self.assertEqual(self.order, ["transcribe", "A"])
        self.assertEqual(self.events.of(d.FLOOR_RELEASE_EVENT), [])

    def raise_in_this_thread(self):
        """A Condition.wait stand-in that raises KeyboardInterrupt in the calling thread only."""
        real_wait = self.lock._cond.wait
        me = threading.current_thread()

        def wait(*a, **k):
            if threading.current_thread() is me:
                raise KeyboardInterrupt
            return real_wait(*a, **k)
        return wait

    def test_an_acquire_whose_wait_raises_leaves_no_ticket(self):
        self.hold_floor(A)
        with mock.patch.object(self.lock._cond, "wait", side_effect=self.raise_in_this_thread()):
            with self.assertRaises(KeyboardInterrupt):
                self.lock.acquire(proto.OP_CONVERSE, "10.0.0.6", B)
        self.assertEqual(self.lock.snapshot(), (True, "converse (10.0.0.5)", 0))
        self.lock.release()
        a = self.queue("A", A)
        self.assertTrue(a.granted.wait(2))

    def test_a_holders_acquire_whose_wait_raises_restarts_the_idle_count(self):
        self.hold_floor(A)
        self.lock.release()   # the idle count starts
        t = self.queue("transcribe", None)
        self.assertTrue(t.granted.wait(2))
        b = self.queue("B", B)
        with mock.patch.object(self.lock._cond, "wait", side_effect=self.raise_in_this_thread()):
            with self.assertRaises(KeyboardInterrupt):
                self.lock.acquire(proto.OP_CONVERSE, "10.0.0.5", A)   # arrives, stops the count, raises
        raised = time.monotonic()
        self.done(t)
        self.assertTrue(b.granted.wait(2))   # the floor did not stick
        self.assertGreaterEqual(b.granted_at - raised, self.IDLE * 0.9)   # counted again from the raise
        self.assertEqual([r["reason"] for r in self.events.of(d.FLOOR_RELEASE_EVENT)], ["idle"])

    def test_close_hands_the_lock_to_the_head_of_the_queue_despite_a_floor(self):
        self.lock = d.AudioLock(self.events, idle_seconds=self.LONG_IDLE)
        self.hold_floor(A)
        b = self.queue("B", B)
        c = self.queue("C", C)
        self.lock.release()
        self.assertFalse(b.granted.wait(0.1))   # behind A's floor
        closed = time.monotonic()
        self.lock.close()
        self.assertTrue(b.granted.wait(2))
        self.assertLess(b.granted_at - closed, self.LONG_IDLE * 0.5)   # not the idle time
        self.done(b)
        self.assertTrue(c.granted.wait(2))
        self.assertEqual(self.order, ["B", "C"])
        self.assertEqual(self.events.of(d.FLOOR_RELEASE_EVENT), [])   # nothing yielded it


# ------------------------------------------------------------------ SpeakService

class DialLink(ScriptedLink):
    """ScriptedLink that can place a call with a scripted outcome."""

    def __init__(self, log, clock) -> None:
        super().__init__(log, clock)
        self.outcome = tg.ANSWERED
        self.no_speech = False

    def call(self, audio, rate) -> str:
        self.log.append(("dial",))
        if self.outcome == tg.ANSWERED:
            self._state = tg.CONNECTED
        return self.outcome

    def record(self, max_seconds, silence_seconds, start_timeout_seconds, cue, rate):
        if self.no_speech:
            self.log.append(("record",))
            return tg.TIMEOUT
        return super().record(max_seconds, silence_seconds, start_timeout_seconds, cue, rate)


class FloorServiceCase(HoldServiceCase):
    TEXT = "One. Two. Three."

    def setUp(self):
        super().setUp()
        self.link = DialLink(self.log, self.clock)
        self.svc = locked_service(self.link, self.engines, SETTINGS, self.link.record_event)

    def floor(self) -> list[tuple[str, dict]]:
        return [(e, f) for e, f in self.link.events if e in (d.FLOOR_TAKE_EVENT, d.FLOOR_RELEASE_EVENT)]

    def takes(self) -> list[str]:
        return [f["project"] for e, f in self.link.events if e == d.FLOOR_TAKE_EVENT]

    def releases(self) -> list[tuple[str, str]]:
        return [(f["project"], f["reason"]) for e, f in self.link.events if e == d.FLOOR_RELEASE_EVENT]

    def req(self, speaker: d.Speaker = A) -> d.Session:   # one Session per request, as run_locked
        return d.Session(None, speaker)


class TestServiceFloor(FloorServiceCase):
    def test_converse_and_listen_take_the_floor_and_keep_it(self):
        self.svc.converse(self.req(), self.TEXT, 120, 2.0, 45, NEVER_LEFT)
        self.svc.listen(self.req(), 120, 2.0, 45)
        self.assertEqual(self.takes(), ["algolearn-speak"])
        self.assertEqual(self.releases(), [])

    def test_a_bare_speak_takes_the_floor_and_gives_it_up_when_it_ends(self):
        self.svc.speak(self.req(), self.TEXT, NEVER_LEFT)
        self.assertEqual([e for e, _ in self.floor()], [d.FLOOR_TAKE_EVENT, d.FLOOR_RELEASE_EVENT])
        self.assertEqual(self.releases(), [("algolearn-speak", "speak")])

    def test_a_speak_that_was_held_and_resumed_gives_up_the_floor(self):
        self.link.cuts = [(2, False)]
        self.link.holds = [(3.0, proto.CMD_RESUME)]
        self.svc.speak(self.req(), self.TEXT, NEVER_LEFT)
        self.assertEqual(self.releases(), [("algolearn-speak", "speak")])

    def test_a_drop_during_speak_keeps_the_floor(self):
        self.link.cuts = [(1, False)]
        self.link.holds = [RuntimeError("the call dropped")]   # ends the call mid-speech
        with self.assertRaisesRegex(d.CallEndedError, "Telegram call ended during speak"):
            self.svc.speak(self.req(), self.TEXT, NEVER_LEFT)
        self.assertEqual(self.takes(), ["algolearn-speak"])
        self.assertEqual(self.releases(), [])

    def test_a_speak_failing_otherwise_after_its_turn_gives_up_the_floor(self):
        with mock.patch.object(self.link, "play_watched", side_effect=RuntimeError("boom")):
            with self.assertRaisesRegex(RuntimeError, "boom"):
                self.svc.speak(self.req(), self.TEXT, NEVER_LEFT)
        self.assertEqual(self.releases(), [("algolearn-speak", "speak")])
        # ... with nobody queued, so the daemon ends the call (owner decision 2026-10-28)
        self.assertEqual((self.link.state(), self.link.hung_up), (tg.NONE, 1))

    def test_an_op_refused_before_it_plays_leaves_the_floor_alone(self):
        with self.assertRaises(ValueError):
            self.svc.speak(self.req(), "  ", NEVER_LEFT)
        with self.assertRaises(ValueError):
            self.svc.converse(self.req(), "**", 120, 2.0, 45, NEVER_LEFT)
        self.link._state = tg.NONE
        with self.assertRaises(RuntimeError):
            self.svc.speak(self.req(), self.TEXT, NEVER_LEFT)
        with self.assertRaises(RuntimeError):
            self.svc.listen(self.req(), 120, 2.0, 45)
        self.assertEqual(self.floor(), [])

    def test_listen_with_no_speech_keeps_the_floor_however_often(self):
        self.link.no_speech = True
        for _ in range(3):   # owner decision 13: reopening the ear keeps the floor
            with self.assertRaises(d.NoSpeech):
                self.svc.listen(self.req(), 120, 2.0, 45)
        self.assertEqual(self.takes(), ["algolearn-speak"])
        self.assertEqual(self.releases(), [])

    def test_hang_up_by_the_holder_releases(self):
        self.svc.converse(self.req(), self.TEXT, 120, 2.0, 45, NEVER_LEFT)
        self.svc.hang_up(self.req())
        self.assertEqual(self.releases(), [("algolearn-speak", "hang_up")])

    def test_hang_up_by_another_session_leaves_the_floor(self):
        self.svc.converse(self.req(), self.TEXT, 120, 2.0, 45, NEVER_LEFT)
        self.svc.hang_up(self.req(B))
        self.assertEqual(self.releases(), [])

    def test_a_hold_expiry_releases(self):
        self.link.cuts = [(2, False)]
        self.link.holds = [(300.0, proto.CMD_RESUME)]
        reply = self.svc.speak(self.req(), self.TEXT, NEVER_LEFT)
        self.assertIn(proto.REPLY_INTERRUPTION, reply)
        self.assertEqual(self.releases(), [("algolearn-speak", "hold_expired")])

    def test_a_converse_hold_expiry_releases(self):
        self.link.cuts = [(2, False)]
        self.link.holds = [(300.0, proto.CMD_RESUME)]
        self.svc.converse(self.req(), self.TEXT, 120, 2.0, 45, NEVER_LEFT)
        self.assertEqual(self.releases(), [("algolearn-speak", "hold_expired")])

    def test_a_hold_expiry_whose_hang_up_fails_still_releases(self):
        self.link.cuts = [(2, False)]
        self.link.holds = [(300.0, proto.CMD_RESUME)]
        self.link.fail_hang_up = RuntimeError("leave_call failed")
        with self.assertRaisesRegex(RuntimeError, "leave_call failed"):
            self.svc.speak(self.req(), self.TEXT, NEVER_LEFT)
        self.assertEqual(self.releases(), [("algolearn-speak", "hold_expired")])

    def test_a_client_that_leaves_during_a_hold_releases_once(self):
        for op in ("speak", "converse"):
            with self.subTest(op=op):
                self.link.events.clear()
                self.link.cuts = [(2, False)]
                self.link.holds = [(3.0, tg.CLIENT_LEFT)]
                with self.assertRaisesRegex(RuntimeError, "the client left during a hold"):
                    if op == "speak":
                        self.svc.speak(self.req(), self.TEXT, NEVER_LEFT)
                    else:
                        self.svc.converse(self.req(), self.TEXT, 120, 2.0, 45, NEVER_LEFT)
                self.assertEqual(self.link.state(), tg.CONNECTED)   # the call stays up
                self.assertEqual(self.releases(), [("algolearn-speak", "client_left")])

    def test_an_answered_call_gives_the_caller_the_floor(self):
        self.link._state = tg.NONE
        reply = self.svc.call(self.req(B), "Hi Frank, it's Angie.", True, 120, 2.0, 45)
        self.assertEqual(reply["outcome"], tg.ANSWERED)
        self.assertEqual(self.floor(), [(d.FLOOR_TAKE_EVENT, {"project": "algolearn-hedge", "from_project": None})])

    def test_a_call_not_answered_busy_or_already_connected_leaves_the_floor(self):
        self.svc.converse(self.req(), self.TEXT, 120, 2.0, 45, NEVER_LEFT)   # A holds the floor
        self.assertEqual(self.svc.call(self.req(B), "Hi Frank, it's Angie.", True, 120, 2.0, 45),
                         {"outcome": tg.ALREADY_CONNECTED})
        self.link._state = tg.NONE
        for outcome in (tg.NOT_ANSWERED, tg.BUSY):
            self.link.outcome = outcome
            self.assertEqual(self.svc.call(self.req(B), "Hi Frank, it's Angie.", True, 120, 2.0, 45),
                             {"outcome": outcome})
        self.assertEqual(self.takes(), ["algolearn-speak"])
        self.assertEqual(self.releases(), [])

    def test_floor_take_names_the_previous_holder_and_the_switch_plays_once_per_change(self):
        self.svc.converse(self.req(A), self.TEXT, 120, 2.0, 45, NEVER_LEFT)
        self.svc.converse(self.req(B), self.TEXT, 120, 2.0, 45, NEVER_LEFT)
        self.svc.converse(self.req(B), self.TEXT, 120, 2.0, 45, NEVER_LEFT)
        self.assertEqual([f for e, f in self.floor()],
                         [{"project": "algolearn-speak", "from_project": None},
                          {"project": "algolearn-hedge", "from_project": "algolearn-speak"}])
        switches = [f for e, f in self.link.events if e == d.SESSION_SWITCH_EVENT]
        self.assertEqual([(s["from_project"], s["to_project"]) for s in switches],
                         [("algolearn-speak", "algolearn-hedge")])


class TestSignOffHangUp(FloorServiceCase):
    """Owner decisions 2026-10-28: a speak that gives up the floor (reason "speak") with no
    other tool op queued ends the call, as a deliberate end."""

    def expecting(self, speaker: d.Speaker = A) -> d.Session:
        """A request from a session that talked on this call."""
        return d.Session(self.svc.key(self.link.info()["generation"]), speaker)

    def signoffs(self) -> list[dict]:
        return [f for e, f in self.link.events if e == d.SIGNOFF_HANGUP_EVENT]

    def test_a_sign_off_with_nobody_queued_hangs_up_and_clears_the_expectation(self):
        sess = self.expecting()
        self.assertEqual(self.svc.speak(sess, self.TEXT, NEVER_LEFT), {"seconds": 1.5, "held_seconds": 0.0})
        self.assertEqual((self.link.state(), self.link.hung_up), (tg.NONE, 1))
        self.assertIsNone(sess.generation)
        self.assertTrue(sess.changed)
        self.assertEqual(sess.call_ended, proto.CALL_ENDED_SIGNOFF)
        kinds = [e for e, _ in self.link.events if e in (d.FLOOR_RELEASE_EVENT, d.SIGNOFF_HANGUP_EVENT)]
        self.assertEqual(kinds, [d.FLOOR_RELEASE_EVENT, d.SIGNOFF_HANGUP_EVENT])
        self.assertEqual(self.signoffs(), [{"project": "algolearn-speak", "waiting": 0, "outcome": tg.HUNG_UP}])

    def test_a_held_and_resumed_sign_off_hangs_up_too(self):
        self.link.cuts = [(2, False)]
        self.link.holds = [(3.0, proto.CMD_RESUME)]
        self.svc.speak(self.expecting(), self.TEXT, NEVER_LEFT)
        self.assertEqual(self.link.hung_up, 1)
        self.assertEqual(len(self.signoffs()), 1)

    def test_converse_and_listen_never_hang_up(self):
        self.svc.converse(self.expecting(), self.TEXT, 120, 2.0, 45, NEVER_LEFT)
        self.svc.listen(self.expecting(), 120, 2.0, 45)
        self.link.no_speech = True
        with self.assertRaises(d.NoSpeech):
            self.svc.listen(self.expecting(), 120, 2.0, 45)
        self.assertEqual((self.link.state(), self.link.hung_up, self.signoffs()), (tg.CONNECTED, 0, []))

    def test_a_drop_during_the_speak_is_no_sign_off(self):
        self.link.cuts = [(1, False)]
        self.link.holds = [RuntimeError("the call dropped")]
        sess = self.expecting()
        with self.assertRaises(d.CallEndedError):
            self.svc.speak(sess, self.TEXT, NEVER_LEFT)
        self.assertEqual((self.link.hung_up, self.signoffs(), sess.call_ended), (0, [], None))

    def test_an_expired_hold_hangs_up_as_before_and_is_no_sign_off(self):
        self.link.cuts = [(2, False)]
        self.link.holds = [(300.0, proto.CMD_RESUME)]
        sess = self.expecting()
        self.svc.speak(sess, self.TEXT, NEVER_LEFT)
        self.assertEqual(self.link.hung_up, 1)   # the hold expiry's own hang-up
        self.assertEqual((self.signoffs(), sess.call_ended), ([], None))

    def test_a_client_that_left_during_a_hold_leaves_the_call_up(self):
        self.link.cuts = [(2, False)]
        self.link.holds = [(3.0, tg.CLIENT_LEFT)]
        with self.assertRaises(RuntimeError):
            self.svc.speak(self.expecting(), self.TEXT, NEVER_LEFT)
        self.assertEqual((self.link.state(), self.link.hung_up, self.signoffs()), (tg.CONNECTED, 0, []))

    def test_a_speak_refused_before_its_turn_does_not_hang_up(self):
        with self.assertRaises(ValueError):
            self.svc.speak(self.expecting(), "  ", NEVER_LEFT)
        self.assertEqual((self.link.state(), self.link.hung_up), (tg.CONNECTED, 0))

    def test_a_failed_sign_off_hang_up_is_recorded_and_raised_and_still_clears(self):
        self.link.fail_hang_up = RuntimeError("leave_call failed")
        sess = self.expecting()
        with self.assertRaisesRegex(RuntimeError, "leave_call failed"):
            self.svc.speak(sess, self.TEXT, NEVER_LEFT)
        self.assertIsNone(sess.generation)
        self.assertIsNone(sess.call_ended)
        (failed,) = [f for e, f in self.link.events if e == d.SIGNOFF_HANGUP_FAILED_EVENT]
        self.assertEqual(failed["project"], "algolearn-speak")
        self.assertIn("leave_call failed", failed["error"])
        self.assertEqual(self.signoffs(), [])

    def run_locked(self, **fields) -> dict:
        daemon = d.SpeakDaemon.__new__(d.SpeakDaemon)
        daemon.service, daemon.personas = self.svc, self.svc.personas
        self.svc.personas.ensure(A.project)
        req = {proto.FIELD_SESSION_ID: A.session_id, proto.FIELD_PROJECT: A.project,
               "expect_generation": self.svc.key(self.link.info()["generation"]),
               proto.FIELD_PERSONA_SEEN: self.svc.personas.key(A.project), "text": self.TEXT, **fields}
        return daemon.run_locked(proto.OP_SPEAK, req, NEVER_LEFT)

    def test_the_reply_says_the_call_ended_and_clears_the_expectation(self):
        reply = self.run_locked()
        self.assertIs(reply["ok"], True)
        self.assertEqual(reply[proto.REPLY_CALL_ENDED], proto.CALL_ENDED_SIGNOFF)
        self.assertIsNone(reply["expect"])

    def test_a_failed_speak_that_hung_up_says_so_too(self):
        with mock.patch.object(self.link, "play_watched", side_effect=RuntimeError("boom")):
            reply = self.run_locked()
        self.assertIs(reply["ok"], False)
        self.assertEqual(reply[proto.REPLY_CALL_ENDED], proto.CALL_ENDED_SIGNOFF)
        self.assertIsNone(reply["expect"])

    def test_without_a_sign_off_hang_up_the_reply_has_no_call_ended(self):
        self.link.cuts = [(1, False)]
        self.link.holds = [RuntimeError("the call dropped")]
        self.assertNotIn(proto.REPLY_CALL_ENDED, self.run_locked())


# ------------------------------------------------------------------ over the wire

class GateLink(TurnLink):
    """TurnLink whose listen waits for `gate` while it is set (an op that keeps running)."""

    def __init__(self) -> None:
        super().__init__()
        self.gate: threading.Event | None = None
        self.speech_gate: threading.Event | None = None   # holds a speak while it plays

    def play_watched(self, segments, rate, classify, accept):
        if self.speech_gate is not None:
            self.speech_gate.wait(10)
        return super().play_watched(segments, rate, classify, accept)

    def record(self, max_seconds, silence_seconds, start_timeout_seconds, cue, rate):
        if self.gate is not None:
            self.gate.wait(10)
        return super().record(max_seconds, silence_seconds, start_timeout_seconds, cue, rate)


class WireCase(unittest.TestCase):
    """Two client sessions on one call, through a real daemon with a short floor idle time."""

    IDLE = 1.0

    def setUp(self):
        self.link = GateLink()
        self.engines = FakeEngines([])
        self.daemon = d.SpeakDaemon(self.link, self.engines, temp_personas(), SETTINGS, d.DebugControl(), port=0,
                                    http_port=0)
        short_floor(self.daemon, self.link, self.IDLE)
        self.daemon.start()
        self.addCleanup(self.daemon.stop)
        p = mock.patch.object(client, "_daemon", ("127.0.0.1", self.daemon._server.server_address[1]))
        p.start()
        self.addCleanup(p.stop)
        self.a, self.b = dict(ID_A), dict(ID_B)
        for ident in (self.a, self.b):   # each session has seen its persona, as after persona()
            self.as_session(ident, client._persona_sync)
        self.order: list[str] = []   # client bodies run in threads, as they return
        self.threads: list[threading.Thread] = []
        self.addCleanup(self.join_all)

    def join_all(self):
        for gate in (self.link.gate, self.link.speech_gate):
            if gate is not None:
                gate.set()
        for t in self.threads:
            t.join(10)

    def as_session(self, ident: dict, fn, *args):
        return as_session(ident, fn, *args)

    def in_thread(self, name: str, ident: dict, fn, *args) -> dict:
        """Run a client body in a thread; the returned box gets "result" (or "error") and
        the order list gets `name` when it returns."""
        box: dict = {}

        def run():
            try:
                box["result"] = self.as_session(ident, fn, *args)
            except Exception as e:   # noqa: BLE001 - handed to the test, which asserts on it
                box["error"] = e
            self.order.append(name)

        t = threading.Thread(target=run, daemon=True)
        self.threads.append(t)
        box["thread"] = t
        t.start()
        return box

    def waiting(self, n: int) -> None:
        wait_until(lambda: self.daemon.lock.snapshot()[2] == n, f"{n} waiting")

    def answered_by_a(self) -> None:
        self.assertTrue(self.as_session(self.a, client._call_sync, "Hi Frank, it's Angie.", True, 120, 2.0, 45)
                        .startswith(proto.ANSWERED))

    def floor_events(self) -> list[dict]:
        return [e for e in self.link.events.recent(100) if e["event"] in (d.FLOOR_TAKE_EVENT, d.FLOOR_RELEASE_EVENT)]


class TestFloorOverTheWire(WireCase):
    def test_the_holders_next_converse_goes_before_a_queued_session_until_it_speaks(self):
        self.answered_by_a()
        self.link.gate = threading.Event()
        a1 = self.in_thread("A1", self.a, client._converse_sync, "One.", 120, 2.0, 45)
        wait_until(lambda: self.daemon.lock.snapshot()[1] is not None, "A's converse to hold the lock")
        b = self.in_thread("B", self.b, client._converse_sync, "Two.", 120, 2.0, 45)
        self.waiting(1)
        a2 = self.in_thread("A2", dict(self.a), client._converse_sync, "Three.", 120, 2.0, 45)
        self.waiting(2)
        self.link.gate.set()
        a1["thread"].join(10)
        a2["thread"].join(10)
        self.assertEqual(self.order, ["A1", "A2"])
        self.assertEqual(self.daemon.lock.snapshot(), (False, None, 1))   # B still waits behind A's floor
        self.assertEqual(self.as_session(self.a, client._speak_sync, "Will do, talk soon."),
                         "spoke for 0.5s" + proto.AUDIO_NOTE)
        b["thread"].join(10)
        self.assertEqual(self.order, ["A1", "A2", "B"])
        self.assertEqual(b["result"], '[took over the call from "algolearn-speak"; the owner heard the switch cue '
                                      "and your persona's name and this project's name first]\nhi Angie" + proto.AUDIO_NOTE)
        self.assertEqual([(e["event"], e["project"], e.get("reason")) for e in self.floor_events()],
                         [(d.FLOOR_TAKE_EVENT, "algolearn-speak", None),
                          (d.FLOOR_RELEASE_EVENT, "algolearn-speak", "speak"),
                          (d.FLOOR_TAKE_EVENT, "algolearn-hedge", None)])

    def test_a_holder_that_goes_quiet_loses_the_floor_after_the_idle_time(self):
        self.answered_by_a()
        self.as_session(self.a, client._converse_sync, "One.", 120, 2.0, 45)
        quiet_since = time.monotonic()
        b = self.in_thread("B", self.b, client._converse_sync, "Two.", 120, 2.0, 45)
        b["thread"].join(10)
        self.assertGreaterEqual(time.monotonic() - quiet_since, self.IDLE * 0.9)
        self.assertIn("result", b)
        events = client._call_debug_sync(50)["events"]
        (release,) = [e for e in events if e["event"] == d.FLOOR_RELEASE_EVENT]
        self.assertEqual((release["project"], release["reason"], release["waiting"]), ("algolearn-speak", "idle", 1))

    def test_after_a_drop_the_holders_call_back_goes_before_a_queued_session(self):
        self.answered_by_a()
        self.as_session(self.a, client._converse_sync, "One.", 120, 2.0, 45)
        self.link._state = tg.NONE   # the owner hung up (link side, between ops)
        b = self.in_thread("B", self.b, client._converse_sync, "Two.", 120, 2.0, 45)
        self.waiting(1)
        self.answered_by_a()   # sent after B's op, runs first
        self.assertEqual(self.link.generation, 2)
        self.assertEqual(self.order, [])
        self.assertEqual(self.daemon.lock.snapshot(), (False, None, 1))   # B still waits
        self.as_session(self.a, client._speak_sync, "Will do, talk soon.")
        b["thread"].join(10)
        self.assertEqual(self.order, ["B"])
        self.assertIn("result", b)

    def test_a_transcription_does_not_wait_for_a_conversation(self):
        self.answered_by_a()   # A holds the floor; the lock is free
        b = self.in_thread("B", self.b, client._converse_sync, "Two.", 120, 2.0, 45)
        self.waiting(1)
        t0 = time.monotonic()
        resp = httpx.post(f"http://127.0.0.1:{self.daemon._http.server_address[1]}/",
                          content=np.zeros(160, dtype=np.int16).tobytes(), timeout=10)
        self.assertEqual(resp.status_code, 200)
        self.assertLess(time.monotonic() - t0, self.IDLE * 0.5)
        self.assertEqual(self.order, [])   # B is still behind A's floor
        self.as_session(self.a, client._speak_sync, "Will do, talk soon.")
        b["thread"].join(10)

    def test_a_session_that_left_while_queued_behind_the_floor_is_skipped_once_it_lapses(self):
        self.answered_by_a()
        played = list(self.link.log)
        s = socket.create_connection(self.daemon._server.server_address, timeout=10)
        proto.send_msg(s, {"op": proto.OP_CONVERSE, proto.FIELD_PROTOCOL: proto.PROTOCOL_VERSION, **ID_B,
                           "text": "Two.", **client._listen_fields(120, 2.0, 45), "expect_generation": None,
                           proto.FIELD_PERSONA_SEEN: seen_key(self.daemon, ID_B)})
        self.waiting(1)
        s.close()   # B's tool call was cancelled while it waited
        # A's floor lapses, B's op reaches the lock and is skipped
        wait_until(lambda: self.daemon.lock.snapshot() == (False, None, 0), "B's op to be skipped")
        self.assertEqual([(e["event"], e["project"], e.get("reason")) for e in self.floor_events()],
                         [(d.FLOOR_TAKE_EVENT, "algolearn-speak", None),
                          (d.FLOOR_RELEASE_EVENT, "algolearn-speak", "idle")])
        self.assertEqual(self.link.log, played)   # B played nothing

    def test_stopping_the_daemon_refuses_a_session_queued_behind_the_floor_at_once(self):
        self.answered_by_a()
        played = list(self.link.log)
        b = self.in_thread("B", self.b, client._converse_sync, "Two.", 120, 2.0, 45)
        self.waiting(1)
        stopped = time.monotonic()
        stopping = threading.Thread(target=self.daemon.stop)   # its server shutdowns poll for 0.5 s each
        stopping.start()
        self.addCleanup(stopping.join, 10)
        b["thread"].join(10)
        self.assertLess(time.monotonic() - stopped, self.IDLE * 0.5)   # not the idle time
        self.assertIsInstance(b["error"], RuntimeError)
        self.assertIn("speak daemon is restarting", str(b["error"]))
        self.assertEqual(self.link.log, played)

    def test_status_shows_a_free_lock_with_a_session_waiting(self):
        self.answered_by_a()
        b = self.in_thread("B", self.b, client._converse_sync, "Two.", 120, 2.0, 45)
        self.waiting(1)
        status = client._status_sync()
        self.assertEqual((status["busy"], status["current_tool"], status["waiting"]), (False, None, 1))
        self.as_session(self.a, client._speak_sync, "Will do, talk soon.")
        b["thread"].join(10)


class TestSignOffHangUpOverTheWire(WireCase):
    """Owner decisions 2026-10-28, through the real daemon and the client's own code."""

    ENDED = "\n[call ended: nobody else was waiting]"

    def signoffs(self) -> list[dict]:
        return [e for e in self.link.events.recent(100) if e["event"] == d.SIGNOFF_HANGUP_EVENT]

    def test_a_sign_off_with_nobody_queued_ends_the_call(self):
        self.answered_by_a()
        self.assertIsNotNone(self.a["_expect"])
        self.assertEqual(self.as_session(self.a, client._speak_sync, "Will do, talk soon."),
                         "spoke for 0.5s" + self.ENDED + proto.AUDIO_NOTE)
        self.assertEqual(self.link.state(), tg.NONE)
        self.assertIsNone(self.a["_expect"])   # a deliberate end: nobody calls back
        (signoff,) = self.signoffs()
        self.assertEqual((signoff["project"], signoff["waiting"]), ("algolearn-speak", 0))
        self.assertEqual([(e["event"], e.get("reason")) for e in self.floor_events()][-1],
                         (d.FLOOR_RELEASE_EVENT, "speak"))
        with self.assertRaisesRegex(RuntimeError, "needs a connected Telegram call"):
            self.as_session(self.a, client._converse_sync, "One.", 120, 2.0, 45)

    def test_a_failed_sign_off_speak_says_after_its_error_that_the_call_ended(self):
        self.answered_by_a()
        with mock.patch.object(self.link, "play_watched", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError) as cm:
                self.as_session(self.a, client._speak_sync, "Will do, talk soon.")
        self.assertEqual(str(cm.exception), "boom" + self.ENDED)
        self.assertIsNone(self.a["_expect"])
        self.assertEqual(self.link.state(), tg.NONE)

    def test_a_sign_off_with_a_session_queued_leaves_the_call_to_it(self):
        self.answered_by_a()
        b = self.in_thread("B", self.b, client._converse_sync, "Two.", 120, 2.0, 45)
        self.waiting(1)
        self.assertEqual(self.as_session(self.a, client._speak_sync, "Will do, talk soon."),
                         "spoke for 0.5s" + proto.AUDIO_NOTE)   # no call-ended line
        b["thread"].join(10)
        self.assertEqual(b["result"], '[took over the call from "algolearn-speak"; the owner heard the switch cue '
                                      "and your persona's name and this project's name first]\nhi Angie"
                                      + proto.AUDIO_NOTE)
        self.assertEqual(self.link.state(), tg.CONNECTED)
        self.assertEqual(self.signoffs(), [])
        self.assertEqual([(e["event"], e["project"]) for e in self.floor_events()][-1],
                         (d.FLOOR_TAKE_EVENT, "algolearn-hedge"))

    def test_a_queued_transcription_does_not_keep_the_call_up_and_runs_after_the_hang_up(self):
        self.answered_by_a()
        self.link.speech_gate = threading.Event()
        states_at_transcription: list[str] = []
        real = self.engines.transcribe

        def transcribe(audio):
            states_at_transcription.append(self.link.state())
            return real(audio)
        self.engines.transcribe = transcribe
        a = self.in_thread("A", self.a, client._speak_sync, "Will do, talk soon.")
        wait_until(lambda: self.daemon.lock.snapshot()[0], "A's speak to hold the lock")
        box: dict = {}

        def post():
            box["resp"] = httpx.post(f"http://127.0.0.1:{self.daemon._http.server_address[1]}/",
                                     content=np.zeros(160, dtype=np.int16).tobytes(), timeout=10)
        t = threading.Thread(target=post, daemon=True)
        t.start()
        self.threads.append(t)
        self.waiting(1)
        self.link.speech_gate.set()
        a["thread"].join(10)
        t.join(10)
        self.assertEqual(a["result"], "spoke for 0.5s" + self.ENDED + proto.AUDIO_NOTE)
        self.assertEqual(box["resp"].status_code, 200)   # not lost, not broken
        self.assertEqual(states_at_transcription, [tg.NONE])   # it ran after the hang-up
        self.assertEqual(len(self.signoffs()), 1)


if __name__ == "__main__":
    unittest.main()
