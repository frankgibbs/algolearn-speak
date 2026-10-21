"""Release 2 of docs/DESIGN_INTERRUPT.md: "hold on" / "continue" (owner approval 2026-10-16).

Three layers, each with fakes as the other test files use (no CUDA, no Telegram):
- TelegramLink.play_watched / watch_for and the watch's utterance guards, against the fake
  py-tgcalls of test_speak_telegram (real pump, StubVAD, scripted owner audio).
- SpeakService's hold loop (cut, resume at the cut sentence, expiry hangs up, converse does
  not listen), against a scripted fake link and fake engines.
- The thin client's budgets and result formatting, and that it still imports no numpy.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import types
import unittest
from unittest import mock

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)   # the link test harness

import speak_engines as eng  # noqa: E402
import speak_protocol as proto  # noqa: E402
import speak_server as client  # noqa: E402
import speak_telegram as tg  # noqa: E402
import speak_telegram_daemon as d  # noqa: E402
from test_speak_telegram import LinkCase, events_named, sine, speech_frames  # noqa: E402

RATE = 24000
SILENCE = bytes(tg.FRAME_BYTES)


def NEVER_LEFT() -> bool:   # noqa: N802 - a constant-like predicate: the request's client stays
    return False


def say(leading: float, speech: float, tail: float = 0.6) -> list[bytes]:
    """Owner audio: `leading` s of quiet, `speech` s of voice, `tail` s of quiet (ends the utterance)."""
    return speech_frames(leading, loud=False) + speech_frames(speech) + speech_frames(tail, loud=False)


class Classifier:
    """Stands in for SpeakService.classify: every candidate is heard as `phrase`."""

    def __init__(self, phrase: str | None, text: str = "hold on") -> None:
        self.phrase = phrase
        self.text = text
        self.calls: list[float] = []   # seconds of audio per candidate

    def __call__(self, audio: np.ndarray) -> tg.Classified:
        self.calls.append(len(audio) / tg.VAD_RATE)
        return tg.Classified(self.phrase, self.text)


# ---------------------------------------------------------------------------- the link

class WatchCase(LinkCase):
    def setUp(self):
        super().setUp()
        self.link.call(np.zeros(100, dtype=np.float32), RATE)

    def loud(self, frames):
        return [f for f in frames if f != SILENCE]

    def owner_says(self, frames: list[bytes]) -> threading.Thread:
        """Push the owner's frames once the watch is on (it drops anything queued before)."""
        def run():
            self.wait_until(lambda: self.link._watching)
            self.push_incoming(frames)
        t = threading.Thread(target=run)
        t.start()
        return t

    def candidates(self):
        return events_named(self.link, "watch_candidate")


class TestPlayWatched(WatchCase):
    def test_everything_plays_when_nobody_speaks(self):
        a, b = sine(0.2), sine(0.3, 660.0)
        n0 = len(self.calls.frames)
        out = self.link.play_watched(iter([("a", a), ("b", b)]), RATE, Classifier("hold"), proto.CMD_HOLD)
        pcm = tg.to_call_pcm(a, RATE) + tg.to_call_pcm(b, RATE)
        self.assertEqual(out, tg.PlayOutcome(None, len(pcm), None, 0.5))
        self.assertEqual(b"".join(self.loud(self.calls.frames[n0:])), pcm)
        self.assertFalse(self.link._watching)
        (done,) = events_named(self.link, "speech_stream_done")
        self.assertNotIn("cut", done)
        self.assertEqual(events_named(self.link, "playback_cut"), [])

    def test_hold_cuts_the_speech_at_once_and_names_the_playing_segment(self):
        classify = Classifier("hold")
        a, b = sine(2.0), sine(2.0, 660.0)
        n0 = len(self.calls.frames)
        t = self.owner_says(say(1.0, 0.5))
        out = self.link.play_watched(iter([("a", a), ("b", b)]), RATE, classify, proto.CMD_HOLD)
        t.join(5)
        self.assertEqual(out.command, proto.CMD_HOLD)
        self.assertEqual(out.cut_tag, "a")   # cut inside the first 2 s segment
        self.assertLess(out.played_bytes, len(tg.to_call_pcm(a, RATE)))
        self.assertAlmostEqual(out.seconds, out.played_bytes / (tg.FRAME_BYTES * 100))
        time.sleep(0.2)
        self.assertEqual(self.link._outgoing, bytearray())   # nothing left to play
        sent = self.loud(self.calls.frames[n0:])
        self.assertEqual(len(sent) * tg.FRAME_BYTES, out.played_bytes)   # silence from the cut on
        (c,) = self.candidates()
        self.assertEqual((c["result"], c["matched"], c["acted"], c["mode"]), ("transcribed", "hold", True, "playing"))
        self.assertNotIn("text", c)   # the transcript only while debug is on
        self.assertTrue(400 <= c["length_ms"] <= 700, c)
        self.assertGreaterEqual(c["leading_silence_ms"], 800)
        (cut,) = events_named(self.link, "playback_cut")
        self.assertEqual(cut["command"], proto.CMD_HOLD)
        self.assertIn("command_latency_ms", cut)
        self.assertGreater(cut["dropped_ms"], 0)
        (done,) = events_named(self.link, "speech_stream_done")
        self.assertEqual((done["cut"], done["aborted"]), (proto.CMD_HOLD, False))
        self.assertFalse(self.link._watching)

    def test_the_transcript_is_recorded_while_debug_is_on(self):
        self.link.audio_stats = True
        t = self.owner_says(say(1.0, 0.5))
        self.link.play_watched(iter([("a", sine(2.0))]), RATE, Classifier(None, "hello there"), proto.CMD_HOLD)
        t.join(5)
        (c,) = self.candidates()
        self.assertEqual((c["text"], c["matched"], c["acted"]), ("hello there", None, False))

    def _dropped(self, frames: list[bytes]) -> tuple[dict, Classifier]:
        classify = Classifier("hold")
        t = self.owner_says(frames)
        out = self.link.play_watched(iter([("a", sine(2.5))]), RATE, classify, proto.CMD_HOLD)
        t.join(5)
        self.assertIsNone(out.command)   # played to the end
        (c,) = self.candidates()
        self.assertEqual(c["result"], "dropped")
        self.assertEqual(classify.calls, [])   # never transcribed
        return c, classify

    def test_no_leading_pause_is_dropped_untranscribed(self):
        c, _ = self._dropped(say(0.3, 0.5))
        self.assertEqual(c["reason"], "no_leading_pause")

    def test_too_short_is_dropped_untranscribed(self):
        c, _ = self._dropped(say(1.0, 0.1))
        self.assertEqual(c["reason"], "too_short")

    def test_too_long_is_dropped_untranscribed(self):
        c, _ = self._dropped(say(1.0, 2.5))
        self.assertEqual(c["reason"], "too_long")

    def test_the_leading_pause_counts_from_the_end_of_the_previous_utterance(self):
        # "... and hold on" mid-sentence: 0.3 s after an utterance is 0.7 s of non-speech
        # (the 0.4 s that ended it plus 0.3 s), under the 0.8 s guard.
        classify = Classifier(None, "so the plan is")
        frames = say(1.0, 1.0, tail=0.0) + speech_frames(0.7, loud=False) + speech_frames(0.5) \
            + speech_frames(0.6, loud=False)
        t = self.owner_says(frames)
        out = self.link.play_watched(iter([("a", sine(3.0))]), RATE, classify, proto.CMD_HOLD)
        t.join(5)
        first, second = self.candidates()
        self.assertEqual(first["result"], "transcribed")   # 1.0 s after 1.0 s of quiet: a candidate
        self.assertEqual((second["result"], second["reason"]), ("dropped", "no_leading_pause"))
        self.assertTrue(600 <= second["leading_silence_ms"] < 800, second)
        self.assertEqual(len(classify.calls), 1)   # the second was never transcribed
        self.assertIsNone(out.command)

    def test_continue_while_playing_does_nothing(self):
        t = self.owner_says(say(1.0, 0.5))
        out = self.link.play_watched(iter([("a", sine(2.0))]), RATE, Classifier(proto.CMD_RESUME, "continue"),
                                     proto.CMD_HOLD)
        t.join(5)
        self.assertIsNone(out.command)
        (c,) = self.candidates()
        self.assertEqual((c["matched"], c["acted"]), (proto.CMD_RESUME, False))

    def test_a_drop_during_watched_playback_raises_call_ended(self):
        threading.Timer(0.3, self.remote_hang_up).start()
        with self.assertRaises(tg.CallEnded):
            self.link.play_watched(iter([("a", sine(2.0))]), RATE, Classifier(None), proto.CMD_HOLD)
        self.assertFalse(self.link._watching)
        (done,) = events_named(self.link, "speech_stream_done")
        self.assertTrue(done["aborted"])

    def test_a_producer_failure_aborts_and_propagates(self):
        def produce():
            yield "a", sine(0.2)
            raise RuntimeError("kokoro failed")

        with self.assertRaisesRegex(RuntimeError, "kokoro failed"):
            self.link.play_watched(produce(), RATE, Classifier(None), proto.CMD_HOLD)
        self.assertFalse(self.link._watching)
        self.assertEqual(self.link._outgoing, bytearray())

    def test_a_cut_in_a_later_segment_names_that_segment(self):
        a, b = sine(0.3), sine(3.0, 660.0)

        def owner():   # speaks once "a" has played and "b" is under way
            self.wait_until(lambda: self.link._watching)
            time.sleep(0.6)
            self.push_incoming(say(1.0, 0.5))
        t = threading.Thread(target=owner)
        t.start()
        out = self.link.play_watched(iter([("a", a), ("b", b)]), RATE, Classifier("hold"), proto.CMD_HOLD)
        t.join(5)
        self.assertEqual((out.command, out.cut_tag), (proto.CMD_HOLD, "b"))
        a_bytes = len(tg.to_call_pcm(a, RATE))
        self.assertGreater(out.played_bytes, a_bytes)
        self.assertLess(out.played_bytes, a_bytes + len(tg.to_call_pcm(b, RATE)))

    def test_a_cut_after_the_last_mark_has_no_cut_segment(self):
        # Everything queued has played (remaining == 0): no segment was cut.
        self.link._run(self.link._stream_begin(), 5, "begin")
        out = self.link._cut(tg._Hit(proto.CMD_HOLD, time.monotonic()), [("a", 100), ("b", 200)], 200)
        self.assertEqual(out, tg.PlayOutcome(proto.CMD_HOLD, 200, None, 200 / (tg.FRAME_BYTES * 100)))
        (cut,) = events_named(self.link, "playback_cut")
        self.assertEqual(cut["dropped_ms"], 0)

    def test_a_drain_that_overruns_its_deadline_raises_and_aborts(self):
        with mock.patch.object(tg, "WATCH_DRAIN_GRACE_SECONDS", 0.1):
            self.calls.stall_send = 0.2   # the pump sends one frame every 0.2 s: 0.2 s of audio takes 4 s
            self.addCleanup(setattr, self.calls, "stall_send", 0.0)
            with self.assertRaisesRegex(TimeoutError, "playing watched audio did not finish within 0s"):
                self.link.play_watched(iter([("a", sine(0.2))]), RATE, Classifier(None), proto.CMD_HOLD)
        self.assertFalse(self.link._watching)
        self.assertEqual(self.link._outgoing, bytearray())
        (done,) = events_named(self.link, "speech_stream_done")
        self.assertTrue(done["aborted"])
        self.assertIn("did not finish", done["error"])

    def test_frames_are_queued_only_while_watching_or_listening(self):
        self.push_incoming(speech_frames(0.05))
        self.assertEqual(self.link._incoming.qsize(), 0)


class TestPlayWatchedWithoutCall(LinkCase):
    def test_raises_before_the_producer_is_touched(self):
        touched = []

        def produce():
            touched.append(True)
            yield "a", sine(0.1)

        with self.assertRaisesRegex(tg.CallEnded, "no connected"):
            self.link.play_watched(produce(), RATE, Classifier(None), proto.CMD_HOLD)
        self.assertEqual(touched, [])
        with self.assertRaises(tg.CallEnded):
            self.link.watch_for(1.0, Classifier(None), proto.CMD_RESUME, NEVER_LEFT)


class TestWatchFor(WatchCase):
    def test_continue_ends_the_hold(self):
        t = self.owner_says(say(1.0, 0.5))
        got = self.link.watch_for(10.0, Classifier(proto.CMD_RESUME, "continue"), proto.CMD_RESUME, NEVER_LEFT)
        t.join(5)
        self.assertEqual(got, proto.CMD_RESUME)
        (end,) = events_named(self.link, "hold_end")
        self.assertEqual(end["how"], proto.CMD_RESUME)
        self.assertEqual(len(events_named(self.link, "hold_start")), 1)
        (c,) = self.candidates()
        self.assertEqual(c["mode"], "held")
        self.assertFalse(self.link._watching)

    def test_hold_on_while_held_does_nothing_and_the_hold_expires(self):
        t = self.owner_says(say(1.0, 0.5))
        t0 = time.monotonic()
        got = self.link.watch_for(2.0, Classifier(proto.CMD_HOLD), proto.CMD_RESUME, NEVER_LEFT)
        t.join(5)
        self.assertIs(got, tg.TIMEOUT)
        self.assertGreaterEqual(time.monotonic() - t0, 2.0)
        (c,) = self.candidates()
        self.assertEqual((c["matched"], c["acted"]), (proto.CMD_HOLD, False))
        (end,) = events_named(self.link, "hold_end")
        self.assertEqual(end["how"], "expired")

    def test_silence_goes_out_while_held(self):
        n0 = len(self.calls.frames)
        self.link.watch_for(0.3, Classifier(None), proto.CMD_RESUME, NEVER_LEFT)
        self.assertEqual(self.loud(self.calls.frames[n0:]), [])

    def test_speech_in_the_first_0_8s_of_the_hold_is_dropped(self):
        # "continue" said straight away (0.3 s into the hold) has no leading pause: dropped.
        classify = Classifier(proto.CMD_RESUME, "continue")
        t = self.owner_says(say(0.3, 0.5))
        got = self.link.watch_for(1.5, classify, proto.CMD_RESUME, NEVER_LEFT)
        t.join(5)
        self.assertIs(got, tg.TIMEOUT)
        (c,) = self.candidates()
        self.assertEqual((c["result"], c["reason"]), ("dropped", "no_leading_pause"))
        self.assertEqual(classify.calls, [])

    def test_frames_queued_before_the_hold_are_discarded(self):
        classify = Classifier(proto.CMD_RESUME, "continue")
        now = time.monotonic()
        for f in say(1.0, 0.5):   # a whole "continue" sitting in the queue from before
            self.link._incoming.put((now, f))
        got = self.link.watch_for(0.5, classify, proto.CMD_RESUME, NEVER_LEFT)
        self.assertIs(got, tg.TIMEOUT)
        self.assertEqual(self.candidates(), [])
        self.assertEqual(classify.calls, [])

    def test_the_hold_ends_when_the_client_leaves(self):
        left = threading.Event()
        threading.Timer(0.2, left.set).start()
        t0 = time.monotonic()
        got = self.link.watch_for(10.0, Classifier(None), proto.CMD_RESUME, left.is_set)
        self.assertIs(got, tg.CLIENT_LEFT)
        self.assertLess(time.monotonic() - t0, 2.0)
        (end,) = events_named(self.link, "hold_end")
        self.assertEqual(end["how"], "client_left")
        self.assertFalse(self.link._watching)
        self.assertEqual(self.link.state(), tg.CONNECTED)   # the link never hangs up on its own

    def test_a_classify_failure_ends_the_hold_and_propagates(self):
        def boom(audio):
            raise RuntimeError("whisper failed")
        t = self.owner_says(say(1.0, 0.5))
        with self.assertRaisesRegex(RuntimeError, "whisper failed"):
            self.link.watch_for(10.0, boom, proto.CMD_RESUME, NEVER_LEFT)
        t.join(5)
        (end,) = events_named(self.link, "hold_end")
        self.assertEqual(end["how"], "error")
        self.assertIn("whisper failed", end["error"])
        self.assertFalse(self.link._watching)

    def test_a_drop_while_held_raises_call_ended(self):
        threading.Timer(0.2, self.remote_hang_up).start()
        with self.assertRaisesRegex(tg.CallEnded, "during the hold"):
            self.link.watch_for(10.0, Classifier(None), proto.CMD_RESUME, NEVER_LEFT)
        (end,) = events_named(self.link, "hold_end")
        self.assertEqual(end["how"], "call_ended")
        self.assertFalse(self.link._watching)


# ---------------------------------------------------------------------------- the service

SETTINGS = {"SPEAK_QUIET_HOURS": "22:00-07:00", "quiet_window": (1320, 420), "tz": None,
            "phrases": {"hold on": proto.CMD_HOLD, "continue": proto.CMD_RESUME}}


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def monotonic(self) -> float:
        return self.t

    def time(self) -> float:
        return self.t


class FakeEngines:
    """One 0.5 s segment per sentence (split as Kokoro splits), tagged with its index; logged."""

    def __init__(self, log: list, split_long: int | None = None) -> None:
        self.log = log
        self.split_long = split_long   # this sentence index comes as two segments
        self.heard = "hi Angie"

    def synthesize_segments(self, text: str):
        for i, _ in enumerate(eng.split_sentences(text)):
            for _part in range(2 if i == self.split_long else 1):
                self.log.append(("synth", i))
                yield i, np.full(RATE // 2, 0.1, dtype=np.float32)

    def synthesize(self, text: str) -> np.ndarray:
        """Whole synthesis (the greeting, a switch announcement): 7 s, over the opener minimum."""
        self.log.append(("synthesize", text))
        return np.zeros(RATE * 7, dtype=np.float32)

    def transcribe(self, audio: np.ndarray) -> str:
        self.log.append(("transcribe",))
        return self.heard


class ScriptedLink:
    """play_watched plays segments until the script says to cut; watch_for answers from the script."""

    def __init__(self, log: list, clock: Clock) -> None:
        self.log = log
        self.clock = clock
        self._state = tg.CONNECTED
        self.cuts: list[tuple[int, bool]] = []   # per play_watched: (cut after this many segments, gap?)
        self.holds: list = []                    # per watch_for: (seconds held, result) or an exception
        self.hung_up = 0
        self.fail_hang_up: BaseException | None = None
        self.events: list[tuple[str, dict]] = []

    def record_event(self, event: str, **fields) -> None:
        self.events.append((event, fields))

    def info(self) -> dict:
        return {"state": self._state, "generation": 1, "end_reason": tg.END_HUNG_UP if self.hung_up else None}

    def state(self) -> str:
        return self._state

    def play_watched(self, segments, rate, classify, accept):
        tags = []
        cut = self.cuts.pop(0) if self.cuts else None
        it = iter(segments)
        for tag, audio in it:
            tags.append(tag)
            if cut is not None and len(tags) == cut[0]:
                break
        self.log.append(("play", tags, accept))
        if cut is None or len(tags) < cut[0]:
            return tg.PlayOutcome(None, len(tags), None, len(tags) * 0.5)
        if cut[1]:   # the cut fell after the last queued segment had all played
            return tg.PlayOutcome(proto.CMD_HOLD, len(tags), None, len(tags) * 0.5)
        return tg.PlayOutcome(proto.CMD_HOLD, len(tags), tags[-1], len(tags) * 0.5 - 0.25)

    def watch_for(self, timeout_s, classify, accept, client_left):
        self.log.append(("watch_for", round(timeout_s, 1), accept))
        step = self.holds.pop(0)
        if isinstance(step, BaseException):
            self._state = tg.NONE
            raise step
        seconds, result = step
        self.clock.t += min(seconds, timeout_s)
        return tg.TIMEOUT if seconds >= timeout_s else result

    def record(self, max_seconds, silence_seconds, start_timeout_seconds, cue, rate):
        self.log.append(("record",))
        return np.zeros(eng.STT_RATE, dtype=np.float32)

    def play(self, audio, rate) -> None:
        self.log.append(("cue",))

    def hang_up(self) -> str:
        self.log.append(("hang_up",))
        if self.fail_hang_up is not None:
            raise self.fail_hang_up
        self.hung_up += 1
        self._state = tg.NONE
        return tg.HUNG_UP


class ServiceLock(d.AudioLock):
    """The service tests' audio lock: plain FIFO, the floor never orders it. They run one op
    at a time, some as a session that would wait out another's floor in the daemon; the
    floor's ordering is AudioLock's own tests' (test_floor). The floor's state and events
    are the real ones."""

    def _first_eligible(self):
        return self._queue[0] if self._queue else None


class LockedService(d.SpeakService):
    """SpeakService whose tool ops run under the audio lock as their request's session, as
    the daemon's handler runs them (AudioLock.take_floor refuses a session without it)."""


def _locked(name: str):
    def op(self, sess, *args, **kwargs):
        self.lock.acquire(name, "test", sess.speaker)
        try:
            return getattr(d.SpeakService, name)(self, sess, *args, **kwargs)
        finally:
            self.lock.release()
    op.__name__ = name
    return op


for _name in ("speak", "listen", "converse", "call", "hang_up"):
    setattr(LockedService, _name, _locked(_name))


def locked_service(link, engines, settings: dict, record_event) -> LockedService:
    return LockedService(link, engines, settings, ServiceLock(record_event))


class HoldServiceCase(unittest.TestCase):
    TEXT = "One. Two. Three."

    def setUp(self):
        self.log: list = []
        self.clock = Clock()
        p = mock.patch.object(d, "time", self.clock)
        p.start()
        self.addCleanup(p.stop)
        self.engines = FakeEngines(self.log)
        self.link = ScriptedLink(self.log, self.clock)
        self.svc = locked_service(self.link, self.engines, SETTINGS, self.link.record_event)
        self.sess = d.Session(None, d.Speaker("s-test", "algolearn-speak"))

    def plays(self):
        return [e[1] for e in self.log if e[0] == "play"]

    def synths(self):
        return [e[1] for e in self.log if e[0] == "synth"]


class TestHoldLoop(HoldServiceCase):
    def test_never_held(self):
        self.assertEqual(self.svc.speak(self.sess, self.TEXT, NEVER_LEFT), {"seconds": 1.5, "held_seconds": 0.0})
        self.assertEqual(self.plays(), [[0, 1, 2]])
        self.assertEqual([e for e in self.log if e[0] == "play"][0][2], proto.CMD_HOLD)

    def test_continue_restarts_the_cut_sentence_without_resynthesizing(self):
        self.link.cuts = [(2, False)]   # cut while sentence 2 ("Two.") plays
        self.link.holds = [(42.0, proto.CMD_RESUME)]
        result = self.svc.speak(self.sess, self.TEXT, NEVER_LEFT)
        self.assertEqual(self.plays(), [[0, 1], [1, 2]])   # "Two." again from its start, then "Three."
        self.assertEqual(self.synths(), [0, 1, 2])          # each sentence synthesized once
        self.assertIn(("watch_for", 300.0, proto.CMD_RESUME), self.log)
        self.assertEqual(result, {"seconds": 0.75 + 1.0, "held_seconds": 42.0})

    def test_a_long_sentence_resumes_at_its_first_chunk(self):
        self.engines.split_long = 1   # "Two." comes as two Kokoro segments
        self.link.cuts = [(3, False)]   # cut in its second chunk
        self.link.holds = [(5.0, proto.CMD_RESUME)]
        self.svc.speak(self.sess, self.TEXT, NEVER_LEFT)
        self.assertEqual(self.plays(), [[0, 1, 1], [1, 1, 2]])

    def test_holds_of_one_speech_add_up_to_the_maximum(self):
        self.link.cuts = [(1, False), (1, False)]
        self.link.holds = [(200.0, proto.CMD_RESUME), (150.0, proto.CMD_RESUME)]
        result = self.svc.speak(self.sess, self.TEXT, NEVER_LEFT)
        self.assertEqual([e for e in self.log if e[0] == "watch_for"],
                         [("watch_for", 300.0, proto.CMD_RESUME), ("watch_for", 100.0, proto.CMD_RESUME)])
        self.assertEqual(result[proto.REPLY_INTERRUPTION]["cut_index"], 1)
        self.assertEqual(result[proto.REPLY_HELD_SECONDS], 300.0)

    def test_expiry_hangs_up_and_returns_the_unspoken_text(self):
        self.sess.expect("x")
        self.link.cuts = [(2, False)]
        self.link.holds = [(300.0, proto.CMD_RESUME)]
        result = self.svc.speak(self.sess, self.TEXT, NEVER_LEFT)
        self.assertEqual(result, {
            "seconds": 0.75, proto.REPLY_HELD_SECONDS: 300.0,
            proto.REPLY_INTERRUPTION: {"command": proto.INTERRUPTION_HOLD_EXPIRED, "cut_index": 2,
                                       "sentence_count": 3, "cut_sentence": "Two.", "unspoken": "Two. Three."}})
        self.assertEqual(self.link.hung_up, 1)
        self.assertTrue(self.sess.changed)
        self.assertIsNone(self.sess.generation)   # not a drop: no call-back expected

    def test_a_failed_hang_up_on_expiry_is_recorded_and_raised(self):
        self.sess.expect("x")
        self.link.cuts = [(2, False)]
        self.link.holds = [(300.0, proto.CMD_RESUME)]
        self.link.fail_hang_up = RuntimeError("leave_call failed")
        with self.assertRaisesRegex(RuntimeError, "leave_call failed"):
            self.svc.speak(self.sess, self.TEXT, NEVER_LEFT)
        events = [e for e in self.link.events if e[0] not in (d.FLOOR_TAKE_EVENT, d.FLOOR_RELEASE_EVENT)]
        self.assertEqual(events, [("hold_expire_hangup_failed",
                                   {"tool": "speak", "error": "RuntimeError: leave_call failed"})])
        self.assertTrue(self.sess.changed)
        self.assertIsNone(self.sess.generation)   # cleared even though the call may still be up

    def test_a_client_that_leaves_during_the_hold_ends_it_without_hanging_up(self):
        self.sess.expect("x")
        self.link.cuts = [(1, False)]
        self.link.holds = [(4.0, tg.CLIENT_LEFT)]
        with self.assertRaisesRegex(RuntimeError, "client left during a hold after 4s"):
            self.svc.converse(self.sess, self.TEXT, 120, 2.0, 45, NEVER_LEFT)
        self.assertEqual(self.link.hung_up, 0)   # owner decision 2026-10-16: the call stays up
        self.assertEqual(self.plays(), [[0]])     # nothing more is played
        self.assertNotIn(("record",), self.log)

    def test_the_client_left_predicate_reaches_the_hold(self):
        seen = []
        self.link.cuts = [(1, False)]
        self.link.holds = [(1.0, proto.CMD_RESUME)]
        orig = self.link.watch_for

        def watch_for(timeout_s, classify, accept, client_left):
            seen.append(client_left)
            return orig(timeout_s, classify, accept, client_left)
        self.link.watch_for = watch_for

        def predicate():
            return False
        self.svc.speak(self.sess, self.TEXT, predicate)
        self.assertEqual(seen, [predicate])

    def test_converse_does_not_listen_after_an_expired_hold(self):
        self.link.cuts = [(1, False)]
        self.link.holds = [(300.0, None)]
        result = self.svc.converse(self.sess, self.TEXT, 120, 2.0, 45, NEVER_LEFT)
        self.assertNotIn("text", result)
        self.assertEqual(result[proto.REPLY_INTERRUPTION]["unspoken"], self.TEXT)
        self.assertNotIn(("record",), self.log)
        self.assertNotIn(("transcribe",), self.log)
        self.assertEqual(self.link.hung_up, 1)

    def test_converse_listens_after_a_resumed_hold(self):
        self.link.cuts = [(1, False)]
        self.link.holds = [(3.0, proto.CMD_RESUME)]
        result = self.svc.converse(self.sess, self.TEXT, 120, 2.0, 45, NEVER_LEFT)
        self.assertEqual(result, {"text": "hi Angie", proto.REPLY_HELD_SECONDS: 3.0})
        names = [e[0] for e in self.log]
        self.assertLess(names.index("watch_for"), names.index("record"))

    def test_a_cut_between_segments_resumes_at_the_next_sentence(self):
        self.link.cuts = [(1, True)]   # "One." had all played when the hold landed
        self.link.holds = [(1.0, proto.CMD_RESUME)]
        self.svc.speak(self.sess, self.TEXT, NEVER_LEFT)
        self.assertEqual(self.plays(), [[0], [1, 2]])
        self.assertEqual(self.synths(), [0, 1, 2])

    def test_a_cut_after_everything_played_is_complete_and_does_not_hold(self):
        self.link.cuts = [(3, True)]
        result = self.svc.speak(self.sess, self.TEXT, NEVER_LEFT)
        self.assertEqual(result, {"seconds": 1.5, "held_seconds": 0.0})
        self.assertNotIn("watch_for", [e[0] for e in self.log])

    def test_a_drop_while_held_is_call_ended(self):
        self.sess.expect("x")
        self.link.cuts = [(1, False)]
        self.link.holds = [tg.CallEnded("the Telegram call ended during the hold")]
        with self.assertRaisesRegex(d.CallEndedError, "ended during converse.*during the hold"):
            self.svc.converse(self.sess, self.TEXT, 120, 2.0, 45, NEVER_LEFT)
        self.assertNotIn(("record",), self.log)
        self.assertEqual(self.link.hung_up, 0)


class TestClassify(HoldServiceCase):
    def heard(self, text: str) -> str | None:
        self.engines.heard = text
        return self.svc.classify(np.zeros(10, dtype=np.float32)).phrase

    def test_whole_utterance_match_after_normalizing(self):
        self.assertEqual(self.heard("Hold on."), proto.CMD_HOLD)
        self.assertEqual(self.heard(" hold   ON! "), proto.CMD_HOLD)
        self.assertEqual(self.heard("Hold-on"), proto.CMD_HOLD)
        self.assertEqual(self.heard("Continue."), proto.CMD_RESUME)

    def test_anything_more_or_less_is_not_a_command(self):
        for text in ("Hold on, what was that?", "hold", "please continue", "on hold", "", "Continued."):
            self.assertIsNone(self.heard(text), text)

    def test_the_transcript_is_kept_for_the_debug_event(self):
        self.engines.heard = "Hold on."
        self.assertEqual(self.svc.classify(np.zeros(10, dtype=np.float32)), tg.Classified(proto.CMD_HOLD, "Hold on."))


class TestPhraseSettings(unittest.TestCase):
    def test_both_phrases_are_required_settings(self):
        self.assertIn("SPEAK_HOLD_PHRASE", d.REQUIRED_SPEAK_ENV)
        self.assertIn("SPEAK_RESUME_PHRASE", d.REQUIRED_SPEAK_ENV)

    def test_parse(self):
        self.assertEqual(d.parse_phrases("Hold on", "continue."), {"hold on": proto.CMD_HOLD,
                                                                    "continue": proto.CMD_RESUME})

    def test_a_phrase_with_nothing_to_say_is_refused(self):
        with self.assertRaisesRegex(RuntimeError, "SPEAK_RESUME_PHRASE"):
            d.parse_phrases("hold on", "...")

    def test_the_same_phrase_twice_is_refused(self):
        with self.assertRaisesRegex(RuntimeError, "the same phrase"):
            d.parse_phrases("hold on", "Hold on!")

    def test_load_settings_reads_them(self):
        env = {"SPEAK_KOKORO_MODEL": "m", "SPEAK_VOICE": "af_heart", "SPEAK_SPEED": "1.0",
               "SPEAK_WHISPER_MODEL": "w", "SPEAK_LANGUAGE": "en", "SPEAK_QUIET_HOURS": "22:00-07:00",
               "SPEAK_HOLD_PHRASE": "hold on", "SPEAK_RESUME_PHRASE": "continue", "TZ": "America/Los_Angeles"}
        with mock.patch.dict(os.environ, env):
            self.assertEqual(d.load_settings()["phrases"], {"hold on": proto.CMD_HOLD, "continue": proto.CMD_RESUME})
        env.pop("SPEAK_HOLD_PHRASE")
        with mock.patch.dict(os.environ, env, clear=True):
            with self.assertRaisesRegex(RuntimeError, "SPEAK_HOLD_PHRASE"):
                d.load_settings()

    def test_compose_sets_the_owner_phrases(self):
        with open(os.path.join(ROOT, "docker-compose.yml")) as f:
            compose = f.read()
        self.assertIn('SPEAK_HOLD_PHRASE: "hold on"', compose)
        self.assertIn('SPEAK_RESUME_PHRASE: "continue"', compose)


# ---------------------------------------------------------------------------- engines

class FakeAudio:
    def __init__(self, a: np.ndarray) -> None:
        self.a = a

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class TestSegments(unittest.TestCase):
    def test_split_sentences_is_kokoros_split(self):
        self.assertEqual(eng.split_sentences("  One. Two!  Three?\nFour  "), ["One.", "Two!", "Three?", "Four"])
        self.assertEqual(eng.split_sentences("No split here"), ["No split here"])

    def test_synthesize_segments_tags_each_segment_with_its_sentence(self):
        e = eng.Engines("hexgrad/Kokoro-82M", "af_heart", 1.0, "large-v3-turbo", "en")

        def kokoro(text, voice, speed, split_pattern):
            for idx in (0, 1, 1, 3):   # sentence 1 in two chunks; sentence 2 had no phonemes
                yield types.SimpleNamespace(audio=FakeAudio(np.ones(2)), text_index=idx)

        e._kokoro = kokoro
        got = list(e.synthesize_segments("A. B. C. D."))
        self.assertEqual([i for i, _ in got], [0, 1, 1, 3])
        self.assertEqual(got[0][1].dtype, np.float32)

    def test_a_segment_without_its_index_raises(self):
        e = eng.Engines("hexgrad/Kokoro-82M", "af_heart", 1.0, "large-v3-turbo", "en")
        e._kokoro = lambda *a, **k: iter([types.SimpleNamespace(audio=FakeAudio(np.ones(2)), text_index=None)])
        with self.assertRaisesRegex(RuntimeError, "text_index"):
            list(e.synthesize_segments("A."))


# ---------------------------------------------------------------------------- the client

class TestClient(unittest.TestCase):
    EXPIRED = {"ok": True, "seconds": 3.2, proto.REPLY_HELD_SECONDS: 300.0,
               proto.REPLY_INTERRUPTION: {"command": proto.INTERRUPTION_HOLD_EXPIRED, "cut_index": 3,
                                          "sentence_count": 5, "cut_sentence": "Third.",
                                          "unspoken": "Third. Fourth. Fifth."}}

    def test_budgets_include_the_longest_hold(self):
        self.assertEqual(client._speak_budget("x" * 50), 60.0 + 10.0 + proto.HOLD_MAX_SECONDS)

    def test_never_held_results_are_unchanged(self):
        self.assertEqual(client._speak_result({"seconds": 2.04, "held_seconds": 0.0}),
                         "spoke for 2.0s" + proto.AUDIO_NOTE)
        self.assertEqual(client._converse_result({"text": "yes", "held_seconds": 0.0}), "yes" + proto.AUDIO_NOTE)

    def test_resumed_results_say_how_long_she_was_held(self):
        self.assertEqual(client._converse_result({"text": "yes", "held_seconds": 42.4}),
                         "[held 42s, then resumed]\nyes" + proto.AUDIO_NOTE)
        self.assertEqual(client._speak_result({"seconds": 5.0, "held_seconds": 7.0}),
                         "[held 7s, then resumed]\nspoke for 5.0s" + proto.AUDIO_NOTE)

    def test_expired_result(self):
        expected = ('[hold expired after 300s; the call was ended]\n'
                    'cut off in sentence 3 of 5: "Third."\n'
                    'not spoken (from the cut sentence): "Third. Fourth. Fifth."' + proto.AUDIO_NOTE)
        self.assertEqual(client._speak_result(self.EXPIRED), expected)
        converse = {k: v for k, v in self.EXPIRED.items() if k != "seconds"}
        self.assertEqual(client._converse_result(converse), expected)

    def test_an_unknown_interruption_raises(self):
        bad = {**self.EXPIRED, proto.REPLY_INTERRUPTION: {"command": "pause"}}
        with self.assertRaisesRegex(RuntimeError, "unknown interruption"):
            client._speak_result(bad)

    def test_the_instructions_and_docstrings_explain_holds(self):
        self.assertIn("hold expired", client._INSTRUCTIONS)
        self.assertIn('Never speak "hold on" or "continue"', client._INSTRUCTIONS)
        for tool in (client.speak, client.converse):
            self.assertIn("[hold expired after 300s; the call was ended]", tool.__doc__)

    def test_the_client_still_imports_no_numpy_torch_or_ntgcalls(self):
        code = ("import sys, speak_server; "
                "print(sorted(m for m in ('numpy', 'torch', 'ntgcalls', 'scipy') if m in sys.modules))")
        out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
        self.assertEqual(out.stdout.strip(), "[]")


if __name__ == "__main__":
    unittest.main()
