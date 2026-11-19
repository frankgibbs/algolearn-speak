"""Voice pulse: the case lights on proxmox01 pulse with the persona's voice
(lan/gpu/case-fan-loop/DESIGN-voice-pulse.md, owner approval 2026-11-18).

- the level of known PCM windows, the greeting envelope and the datagram format;
- SPEAK_PULSE_TARGET parsing and the real UDP sender;
- in the link: speech-only gating (cues send nothing, the switch cue is skipped), `end` on
  finish, cut, abort and call end, the greeting's envelope, and a failed send that never
  touches the speech.
"""

from __future__ import annotations

import os
import socket
import sys
import time
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import speak_protocol as proto  # noqa: E402
import speak_telegram as tg  # noqa: E402
from test_hold import Classifier, WatchCase, say  # noqa: E402
from test_speak_telegram import CALLER, LinkCase, events_named, sine  # noqa: E402

RATE = 24000


def parse(datagrams: list[bytes]) -> list[tuple[int, int, str]]:
    out = []
    for d in datagrams:
        tag, utt, seq, body = d.decode("ascii").split(" ")
        assert tag == "pulse1"
        out.append((int(utt), int(seq), body))
    return out


class TestLevel(unittest.TestCase):
    def test_full_scale_square_is_zero_dbfs(self):
        a = np.array([32767, -32768] * 720, dtype=np.int16)
        self.assertAlmostEqual(tg.pulse_level(a), 0.0, places=3)

    def test_half_scale_sine_is_minus_nine_dbfs(self):
        t = np.arange(1440) / 48000
        a = (0.5 * 32768 * np.sin(2 * np.pi * 1000 * t)).astype(np.int16)
        self.assertAlmostEqual(tg.pulse_level(a), 20 * np.log10(0.5 / np.sqrt(2)), places=2)   # -9.03

    def test_float_and_int_agree(self):
        f = sine(0.03)
        i = (f * 32768).astype(np.int16)
        self.assertAlmostEqual(tg.pulse_level(f), tg.pulse_level(i), places=2)

    def test_exact_silence_is_minus_infinity(self):
        self.assertEqual(tg.pulse_level(np.zeros(1440, dtype=np.int16)), float("-inf"))

    def test_no_samples_raises(self):
        with self.assertRaises(ValueError):
            tg.pulse_level(np.zeros(0, dtype=np.int16))

    def test_envelope_has_one_level_per_30_ms_window(self):
        audio = np.concatenate([sine(0.09), np.zeros(int(0.06 * RATE), dtype=np.float32), sine(0.01)])
        levels = tg.pulse_envelope(audio, RATE)
        self.assertEqual(len(levels), 6)   # 0.16 s: five whole windows and a 10 ms one
        self.assertAlmostEqual(levels[0], 20 * np.log10(0.3 / np.sqrt(2)), places=1)
        self.assertEqual(levels[3:5], [float("-inf")] * 2)
        self.assertGreater(levels[5], -20)

    def test_datagram_format(self):
        self.assertEqual(tg.pulse_datagram(3, 0, -23.456), b"pulse1 3 0 -23.5")
        self.assertEqual(tg.pulse_datagram(3, 1, float("-inf")), b"pulse1 3 1 -inf")
        self.assertEqual(tg.pulse_datagram(3, 2, None), b"pulse1 3 2 end")
        self.assertLess(len(tg.pulse_datagram(2 ** 31, 2 ** 31, -100.0)), 60)


class TestTarget(unittest.TestCase):
    def test_host_and_port(self):
        self.assertEqual(tg.parse_pulse_target("proxmox01.home.arpa:6743"), ("proxmox01.home.arpa", 6743))

    def test_bad_values_are_refused(self):
        for bad in ("proxmox01.home.arpa", ":6743", "host:", "host:abc", "host:0", "host:70000", ""):
            with self.subTest(bad=bad), self.assertRaisesRegex(RuntimeError, "SPEAK_PULSE_TARGET"):
                tg.parse_pulse_target(bad)

    def test_sender_reaches_a_udp_socket(self):
        rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        rx.bind(("127.0.0.1", 0))
        rx.settimeout(2)
        sender = tg.PulseSender("127.0.0.1", rx.getsockname()[1])
        try:
            sender.send(b"pulse1 1 0 -20.0")
            self.assertEqual(rx.recvfrom(100)[0], b"pulse1 1 0 -20.0")
        finally:
            sender.close()
            rx.close()

    def test_unresolvable_host_stops_the_start(self):
        with self.assertRaises(OSError):
            tg.PulseSender("no-such-host.invalid", 6743)


class PulseCase(LinkCase):
    def setUp(self):
        super().setUp()
        self.link.call(np.zeros(100, dtype=np.float32), RATE, CALLER)
        time.sleep(0.1)   # the greeting's envelope is done
        self.pulse.sent.clear()

    def got(self):
        return parse(self.pulse.sent)


class TestGating(PulseCase):
    def test_a_cue_sends_nothing(self):
        self.link.play(sine(0.3), RATE)
        time.sleep(0.1)
        self.assertEqual(self.pulse.sent, [])

    def test_a_stream_sends_levels_then_end(self):
        self.link.play_stream(iter([sine(0.3)]), RATE)
        got = self.got()
        utt = {u for u, _s, _b in got}
        self.assertEqual(len(utt), 1)
        self.assertEqual([s for _u, s, _b in got], list(range(len(got))))   # seq counts up
        self.assertEqual(got[-1][2], "end")
        levels = [float(b) for _u, _s, b in got[:-1]]
        self.assertGreaterEqual(len(levels), 10)   # 0.3 s of speech = 10 windows
        loud = 20 * np.log10(0.3 / np.sqrt(2))
        self.assertTrue(all(abs(v - loud) < 0.5 for v in levels[:10]), levels)

    def test_silence_before_the_first_speech_frame_is_not_sent(self):
        def produce():
            time.sleep(0.2)   # Kokoro still synthesizing: the pump sends idle silence
            yield sine(0.09)
        self.link.play_stream(produce(), RATE)
        got = self.got()
        self.assertNotEqual(got[0][2], "-inf")
        self.assertEqual(got[-1][2], "end")

    def test_leading_zero_padding_is_not_sent(self):
        audio = np.concatenate([np.zeros(int(0.09 * RATE), dtype=np.float32), sine(0.09)])
        self.link.play_stream(iter([audio]), RATE)
        bodies = [b for _u, _s, b in self.got()]
        self.assertNotIn("-inf", bodies)
        self.assertEqual(bodies[-1], "end")

    def test_an_id_is_the_utterances_wall_clock_start(self):
        # Regression (2026-11-18): ids were the daemon's start time + a counter, so after a daemon up
        # for hours they lagged the wall clock, and one hand-sent datagram with a current-time id
        # left the receiver dropping every real utterance as late. Each id is now its start in ms.
        before = time.time_ns() // 1_000_000
        self.link.play_stream(iter([sine(0.09)]), RATE)
        after = time.time_ns() // 1_000_000
        self.assertTrue(before <= self.got()[0][0] <= after, (before, self.got()[0][0], after))

    def test_ids_rise_when_the_wall_clock_does_not(self):
        real = tg.time.time_ns
        tg.time.time_ns = lambda: 1_000_000_000_000 * 1_000_000   # frozen, below the ids already given
        self.addCleanup(setattr, tg.time, "time_ns", real)
        self.link.play_stream(iter([sine(0.09)]), RATE)
        self.link.play_stream(iter([sine(0.09)]), RATE)
        ids = [u for u, _s, _b in self.got()]
        first, second = ids[0], ids[-1]
        self.assertEqual(second, first + 1)
        self.assertGreater(first, 1_000_000_000_000)   # above the greeting's id, not the frozen clock

    def test_a_gap_inside_the_utterance_is_sent_as_silence(self):
        def produce():
            yield sine(0.09)
            time.sleep(0.15)   # an underrun mid-utterance
            yield sine(0.09)
        self.link.play_stream(produce(), RATE)
        bodies = [b for _u, _s, b in self.got()]
        self.assertIn("-inf", bodies[1:-1])
        self.assertEqual(bodies[-1], "end")

    def test_each_utterance_has_its_own_id(self):
        self.link.play_stream(iter([sine(0.09)]), RATE)
        self.link.play_stream(iter([sine(0.09)]), RATE)
        ids = sorted({u for u, _s, _b in self.got()})
        self.assertEqual(len(ids), 2)
        self.assertLess(ids[0], ids[1])

    def test_the_switch_cue_is_skipped_and_the_speech_pulses(self):
        cue = np.zeros(int(0.3 * RATE), dtype=np.float32)   # silent stand-in for switch_pcm: any level is from speech
        cue[:240] = 0.3   # but not all-zero, so the pump sends it as audio
        self.link.play(np.concatenate([cue, sine(0.3)]), RATE, speech_from=len(cue))
        got = self.got()
        levels = [float(b) for _u, _s, b in got[:-1]]
        self.assertEqual(got[-1][2], "end")
        self.assertTrue(9 <= len(levels) <= 11, levels)   # the 0.3 s of speech, not the 0.3 s cue
        self.assertTrue(all(v > -20 for v in levels[:9]), levels)


class TestEnd(PulseCase):
    def test_call_end_mid_stream_sends_end(self):
        def produce():
            yield sine(0.3)
            time.sleep(0.2)   # some of the speech has played and pulsed
            self.remote_hang_up()
            yield sine(0.3)
        with self.assertRaises(tg.CallEnded):
            self.link.play_stream(produce(), RATE)
        got = self.got()
        self.assertEqual(got[-1][2], "end")
        self.assertEqual(sum(1 for g in got if g[2] == "end"), 1)

    def test_an_aborted_stream_sends_end(self):
        def produce():
            yield sine(0.3)
            time.sleep(0.2)
            raise RuntimeError("kokoro failed")
        with self.assertRaisesRegex(RuntimeError, "kokoro failed"):
            self.link.play_stream(produce(), RATE)
        self.assertEqual(self.got()[-1][2], "end")
        self.assertIsNone(self.link._pulse_utt)


class TestCut(WatchCase):
    def test_a_hold_cut_sends_end(self):
        self.pulse.sent.clear()
        t = self.owner_says(say(1.0, 0.5))
        out = self.link.play_watched(iter([("a", sine(2.0))]), RATE, Classifier("hold"), proto.CMD_HOLD)
        t.join(5)
        self.assertEqual(out.command, proto.CMD_HOLD)
        got = parse(self.pulse.sent)
        self.assertEqual(got[-1][2], "end")
        n = len(self.pulse.sent)
        time.sleep(0.2)
        self.assertEqual(len(self.pulse.sent), n)   # nothing after the cut while held


class TestGreeting(LinkCase):
    def test_the_greeting_envelope_is_sent_when_the_file_plays(self):
        greeting = np.concatenate([sine(0.3), np.zeros(int(0.09 * RATE), dtype=np.float32)])
        t0 = time.monotonic()
        self.link.call(greeting, RATE, CALLER)
        self.wait_until(lambda: self.pulse.sent and self.pulse.sent[-1].endswith(b"end"))
        elapsed = time.monotonic() - t0
        got = parse(self.pulse.sent)
        bodies = [b for _u, _s, b in got]
        self.assertEqual(len(bodies), 14)   # 13 windows of 30 ms, then end
        self.assertEqual(bodies[10:13], ["-inf"] * 3)
        self.assertGreaterEqual(elapsed, 0.38)   # paced on the 30 ms clock, not dumped

    def test_a_drop_during_the_greeting_ends_its_pulse(self):
        greeting = sine(2.0)
        self.calls_patch_long_opener()
        self.link.call(greeting, RATE, CALLER)
        self.wait_until(lambda: len(self.pulse.sent) >= 3)
        self.remote_hang_up()
        got = parse(self.pulse.sent)
        self.assertEqual(got[-1][2], "end")
        n = len(self.pulse.sent)
        time.sleep(0.15)
        self.assertEqual(len(self.pulse.sent), n)

    def calls_patch_long_opener(self):
        # The opener wait is opener_seconds + OPENER_TAIL_SECONDS; with a 2 s greeting the call
        # returns after 2 s, by which time the envelope is done. Call in a thread instead.
        orig = self.link.call

        def call_in_background(greeting, rate, caller):
            import threading
            threading.Thread(target=orig, args=(greeting, rate, caller), daemon=True).start()
        self.link.call = call_in_background


class TestGreetingOverlap(PulseCase):
    def test_the_first_stream_cuts_a_greeting_envelope_short(self):
        # The envelope task outlives call() only by a few windows at most; drive one long one
        # on the connected call to see a stream take over from it.
        import asyncio

        async def start():
            self.link._pulse_task = asyncio.create_task(self.link._pulse_greeting([-12.0] * 100))
        self.link._run(start(), 5, "starting a greeting envelope")
        self.wait_until(lambda: len(self.pulse.sent) >= 3)
        greeting_id = parse(self.pulse.sent)[0][0]
        self.link.play_stream(iter([sine(0.09)]), RATE)
        time.sleep(0.1)
        got = parse(self.pulse.sent)
        greeting = [b for u, _s, b in got if u == greeting_id]
        self.assertEqual(greeting[-1], "end")
        self.assertEqual(greeting.count("end"), 1)
        self.assertLess(len(greeting), 50)
        stream = [b for u, _s, b in got if u != greeting_id]
        self.assertEqual(stream[-1], "end")
        self.assertIsNone(self.link._pulse_task)


class TestSendFailure(PulseCase):
    def test_a_failed_send_never_touches_the_speech(self):
        self.pulse.fail = OSError(101, "Network is unreachable")
        n0 = len(self.calls.frames)
        played = self.link.play_stream(iter([sine(0.3)]), RATE)
        self.assertAlmostEqual(played, 0.3)
        loud = [f for f in self.calls.frames[n0:] if f != bytes(tg.FRAME_BYTES)]
        self.assertEqual(len(loud), 30)   # every speech frame went out
        self.link.play_stream(iter([sine(0.09)]), RATE)
        (failed,) = events_named(self.link, "pulse_send_failed")   # once per call
        self.assertIn("Network is unreachable", failed["error"])
        self.link.hang_up()
        (ended,) = events_named(self.link, "ended")
        self.assertGreater(ended["pulse_send_failures"], 10)

    def test_the_count_starts_again_on_the_next_call(self):
        self.pulse.fail = OSError("down")
        self.link.play_stream(iter([sine(0.09)]), RATE)
        self.link.hang_up()
        self.pulse.fail = None
        self.link.call(np.zeros(100, dtype=np.float32), RATE, CALLER)
        self.link.hang_up()
        self.assertEqual(events_named(self.link, "ended")[-1]["pulse_send_failures"], 0)


if __name__ == "__main__":
    unittest.main()
