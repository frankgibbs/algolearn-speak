"""Streamed speech in the daemon (docs/DESIGN_STREAMED_SPEECH.md, owner decision 2026-10-16):
SpeakService hands Kokoro's lazy per-segment output to TelegramLink.play_stream, the greeting
stays one whole synthesis, and Whisper runs only after the whole speech has played.

Fake engines and a fake link, as in test_whisper_http.py: no CUDA, no Telegram. The link's
own streaming (pump, underruns) is tested in test_speak_telegram.TestPlayStream."""

from __future__ import annotations

import types
import unittest

import numpy as np

import speak_engines as eng
import speak_telegram as tg
import speak_telegram_daemon as d

RATE = eng.TTS_RATE
SETTINGS = {"SPEAK_QUIET_HOURS": "22:00-07:00", "quiet_window": (1320, 420),
            "tz": None}


class FakeEngines:
    """Each text sentence is one segment of 0.5 s per sentence; every step is logged."""

    def __init__(self, log: list) -> None:
        self.log = log
        self.fail_after: int | None = None   # raise after this many segments

    def synthesize_stream(self, text: str):
        self.log.append(("stream_start", text))
        for i, sentence in enumerate(s for s in text.split(". ") if s):
            if self.fail_after is not None and i == self.fail_after:
                raise RuntimeError("kokoro failed")
            self.log.append(("synth", i + 1))
            yield np.full(RATE // 2, 0.1, dtype=np.float32)

    def synthesize(self, text: str) -> np.ndarray:
        self.log.append(("synthesize_whole", text))
        return np.zeros(RATE * 7, dtype=np.float32)

    def transcribe(self, audio: np.ndarray) -> str:
        self.log.append(("transcribe",))
        return "hi Angie"


class FakeLink:
    def __init__(self, log: list) -> None:
        self.log = log
        self._state = tg.CONNECTED
        self.drop_during_stream = False
        self.record_result = np.zeros(eng.STT_RATE, dtype=np.float32)
        self.drop_after_record = False

    def info(self) -> dict:
        return {"state": self._state, "generation": 1, "end_reason": None}

    def state(self) -> str:
        return self._state

    def play(self, audio, rate) -> None:
        self.log.append(("play", len(audio)))

    def play_stream(self, segments, rate) -> float:
        self.log.append(("stream_begin",))
        samples = 0
        for n, seg in enumerate(segments, 1):
            self.log.append(("queued", n))
            samples += len(seg)
            if self.drop_during_stream:
                self._state = tg.NONE
                raise tg.CallEnded("the Telegram call ended during playback")
        self.log.append(("stream_done",))
        return samples / rate

    def record(self, max_seconds, silence_seconds, start_timeout_seconds, cue, rate):
        self.log.append(("record", len(cue)))
        if self.drop_after_record:   # the owner said their piece and hung up
            self._state = tg.NONE
        return self.record_result

    def call(self, audio, rate) -> str:
        self.log.append(("dial", len(audio)))
        self._state = tg.CONNECTED
        return tg.ANSWERED


class ServiceCase(unittest.TestCase):
    def setUp(self):
        self.log: list = []
        self.engines = FakeEngines(self.log)
        self.link = FakeLink(self.log)
        self.svc = d.SpeakService(self.link, self.engines, SETTINGS)
        self.sess = d.Session(None)

    def names(self):
        return [e[0] for e in self.log]

    def test_speak_streams_and_interleaves_synthesis_with_queueing(self):
        result = self.svc.speak(self.sess, "One. Two. Three.")
        self.assertEqual(result, {"seconds": 1.5})   # the same seconds as one whole synthesis
        # Lazy: nothing is synthesized until the link has started the stream (its call check).
        self.assertEqual(self.log, [("stream_begin",), ("stream_start", "One. Two. Three."),
                                    ("synth", 1), ("queued", 1), ("synth", 2), ("queued", 2),
                                    ("synth", 3), ("queued", 3), ("stream_done",)])
        self.assertNotIn("synthesize_whole", self.names())

    def test_markdown_is_stripped_before_streaming(self):
        self.svc.speak(self.sess, "**Bold** `code`.")
        self.assertEqual(self.log[1], ("stream_start", "Bold code."))

    def test_empty_text_raises_before_any_audio(self):
        with self.assertRaisesRegex(ValueError, "empty text"):
            self.svc.speak(self.sess, "**`")
        self.assertEqual(self.log, [])

    def test_converse_listens_and_transcribes_only_after_the_whole_speech(self):
        result = self.svc.converse(self.sess, "One. Two.", 120, 2.0, 45)
        self.assertEqual(result, {"text": "hi Angie"})
        names = self.names()
        self.assertLess(names.index("stream_done"), names.index("record"))
        self.assertLess(names.index("record"), names.index("transcribe"))
        self.assertEqual(names.count("stream_done"), 1)   # the speech only: no spoken ack
        self.assertEqual(self.log[-1], ("transcribe",))   # nothing plays after the capture

    def test_greeting_is_one_whole_synthesis(self):
        self.link._state = tg.NONE
        result = self.svc.call(self.sess, "Hi Frank, it's Angie. Calling about the build.", True, 120, 2.0, 45)
        self.assertEqual(result["outcome"], tg.ANSWERED)
        names = self.names()
        self.assertEqual(names[:2], ["synthesize_whole", "dial"])
        self.assertEqual(self.log[1], ("dial", RATE * 7))
        self.assertNotIn(("stream_start", "Hi Frank, it's Angie. Calling about the build."), self.log)

    def test_a_drop_mid_stream_is_call_ended_and_no_listen_follows(self):
        self.link.drop_during_stream = True
        self.sess.expect("x")   # the session talked on this call
        with self.assertRaisesRegex(d.CallEndedError, "ended during converse"):
            self.svc.converse(self.sess, "One. Two.", 120, 2.0, 45)
        self.assertNotIn("record", self.names())
        self.assertNotIn("transcribe", self.names())

    def test_a_synthesis_error_mid_stream_propagates_while_connected(self):
        self.engines.fail_after = 1
        with self.assertRaisesRegex(RuntimeError, "kokoro failed"):
            self.svc.speak(self.sess, "One. Two.")
        self.assertNotIn("stream_done", self.names())


class FakeAudio:
    """Stands in for Kokoro's torch tensor: detach().cpu().numpy()."""

    def __init__(self, a: np.ndarray) -> None:
        self.a = a

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class TestEnginesStream(unittest.TestCase):
    def engines(self, chunks: list[np.ndarray], seen: list):
        e = eng.Engines("hexgrad/Kokoro-82M", "af_heart", 1.0, "large-v3-turbo", "en")

        def kokoro(text, voice, speed, split_pattern):
            seen.append((text, voice, speed, split_pattern))
            for c in chunks:
                seen.append("made")
                yield types.SimpleNamespace(audio=FakeAudio(c))

        e._kokoro = kokoro
        return e

    def test_stream_yields_each_segment_as_kokoro_makes_it(self):
        seen: list = []
        e = self.engines([np.ones((1, 3), dtype=np.float64), np.zeros(2, dtype=np.float32)], seen)
        it = e.synthesize_stream("A. B.")
        first = next(it)
        self.assertEqual(seen.count("made"), 1)   # lazy: the second segment is not made yet
        self.assertEqual(first.dtype, np.float32)
        self.assertEqual(first.shape, (3,))
        self.assertEqual(next(it).shape, (2,))
        self.assertEqual(seen[0], ("A. B.", "af_heart", 1.0, eng.SENTENCE_SPLIT))

    def test_whole_synthesis_concatenates_the_stream(self):
        e = self.engines([np.ones(3, dtype=np.float32), np.zeros(2, dtype=np.float32)], [])
        np.testing.assert_array_equal(e.synthesize("A. B."), np.array([1, 1, 1, 0, 0], dtype=np.float32))

    def test_no_audio_raises(self):
        e = self.engines([], [])
        with self.assertRaisesRegex(RuntimeError, "produced no audio"):
            list(e.synthesize_stream("..."))
        with self.assertRaisesRegex(RuntimeError, "produced no audio"):
            e.synthesize("...")


class ListenCues(ServiceCase):
    """Owner decision 2026-10-16: the ear-open beep is the only listen cue. No ear-closed
    beep, rising chime or spoken "Processing." ack follows the capture, and Whisper runs
    with nothing queued to play (the pump sends only silence)."""

    def test_listen_plays_only_the_ear_open_cue_then_transcribes(self):
        result = self.svc.listen(self.sess, 120, 2.0, 45)
        self.assertEqual(result, {"text": "hi Angie"})
        ear_open = len(d.cue_pcm(880.0, 0.5, 0.5, 0.2))
        self.assertEqual(self.log, [("record", ear_open), ("transcribe",)])

    def test_converse_plays_nothing_after_the_capture(self):
        self.svc.converse(self.sess, "One. Two.", 120, 2.0, 45)
        after = self.names()[self.names().index("record") + 1:]
        self.assertEqual(after, ["transcribe"])
        self.assertNotIn("play", self.names())

    def test_call_listen_plays_nothing_after_the_capture(self):
        self.link._state = tg.NONE
        result = self.svc.call(self.sess, "Hi Frank, it's Angie. Calling about the build.", True, 120, 2.0, 45)
        self.assertEqual(result, {"outcome": tg.ANSWERED, "text": "hi Angie"})
        self.assertEqual(self.names()[-2:], ["record", "transcribe"])
        self.assertNotIn("play", self.names())
        self.assertNotIn("stream_begin", self.names())

    def test_a_drop_right_after_the_capture_still_reports_the_words(self):
        self.link.drop_after_record = True
        with self.assertRaisesRegex(d.CallEndedError, "transcript before the drop: 'hi Angie'"):
            self.svc.listen(self.sess, 120, 2.0, 45)
        self.assertEqual(self.names(), ["record", "transcribe"])

    def test_no_speech_plays_the_timeout_beep_and_raises(self):
        self.link.record_result = tg.TIMEOUT
        with self.assertRaises(d.NoSpeech):
            self.svc.listen(self.sess, 120, 2.0, 45)
        self.assertEqual(self.names(), ["record", "play"])
        self.assertNotIn("transcribe", self.names())

    def test_the_daemon_does_not_require_an_ack_setting(self):
        self.assertNotIn("SPEAK_ACK_TEXT", d.REQUIRED_SPEAK_ENV)


if __name__ == "__main__":
    unittest.main()
