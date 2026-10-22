"""The switch of session on the shared call (owner decisions 2026-10-20).

A speak, listen or converse whose session differs from the previous audio op's on the same
call is preceded by the switch cue and this session's project name, spoken; the reply
carries `switched_from`. Service level against the scripted link and fake engines of
test_hold (no CUDA, no Telegram); the wire end to end is in test_daemon_wire.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import speak_protocol as proto  # noqa: E402
import speak_server as client  # noqa: E402
import speak_telegram as tg  # noqa: E402
import speak_telegram_daemon as d  # noqa: E402
import speak_tone  # noqa: E402
from test_hold import NEVER_LEFT, SETTINGS, HoldServiceCase, ScriptedLink, locked_service  # noqa: E402

A = d.Speaker("session-a", "algolearn-speak")
B = d.Speaker("session-b", "algolearn-hedge")


class CallLink(ScriptedLink):
    """ScriptedLink whose call generation can change (a new call)."""

    def __init__(self, log, clock) -> None:
        super().__init__(log, clock)
        self.generation = 1
        self.drop_on_play = False

    def info(self) -> dict:
        return {**super().info(), "generation": self.generation}

    def play(self, audio, rate) -> None:
        self.log.append(("cue", len(audio)))
        if self.drop_on_play:
            self._state = tg.NONE
            raise RuntimeError("call dropped while playing")


class SwitchCase(HoldServiceCase):
    TEXT = "One."

    def setUp(self):
        super().setUp()
        self.link = CallLink(self.log, self.clock)
        self.svc = locked_service(self.link, self.engines, SETTINGS, self.link.record_event)

    def placed_by(self, speaker: d.Speaker) -> None:
        """What call() does once answered: the caller's greeting is the call's first turn."""
        self.svc._turn = (self.svc.key(self.link.generation), speaker)

    def announcements(self) -> list[str]:
        return [e[1] for e in self.log if e[0] == "synthesize"]

    def switches(self) -> list[dict]:
        return [f for e, f in self.link.events if e == d.SESSION_SWITCH_EVENT]

    def req_sess(self, speaker: d.Speaker) -> d.Session:   # one Session per request, as run_locked
        return d.Session(None, speaker)


class TestSwitch(SwitchCase):
    def test_the_session_that_placed_the_call_talks_on_without_a_switch(self):
        self.placed_by(A)
        s = self.req_sess(A)
        self.svc.converse(s, self.TEXT, 120, 2.0, 45, NEVER_LEFT)
        self.assertIsNone(s.switched_from)
        self.assertEqual(self.announcements(), [])
        self.assertEqual(self.switches(), [])

    def test_another_session_is_announced_before_its_speech(self):
        self.placed_by(A)
        s = self.req_sess(B)
        self.svc.speak(s, self.TEXT, NEVER_LEFT)
        self.assertEqual(s.switched_from, "algolearn-speak")
        self.assertEqual(self.announcements(), ["Angie, algolearn hedge"])
        kinds = [e[0] for e in self.log]
        self.assertLess(kinds.index("cue"), kinds.index("play"))   # announcement, then the speech
        cue_len = [e[1] for e in self.log if e[0] == "cue"][0]
        self.assertEqual(cue_len, len(speak_tone.switch_pcm()) + d.TTS_RATE * 7)
        (event,) = self.switches()
        self.assertEqual(event, {"tool": "speak", "from_project": "algolearn-speak",
                                 "to_project": "algolearn-hedge", "to_persona": "Angie",
                                 "spoken": "Angie, algolearn hedge",
                                 "announce_s": round(cue_len / d.TTS_RATE, 2)})

    def test_the_same_session_again_is_not_a_switch(self):
        self.placed_by(A)
        self.svc.speak(self.req_sess(B), self.TEXT, NEVER_LEFT)
        s = self.req_sess(B)
        self.svc.converse(s, self.TEXT, 120, 2.0, 45, NEVER_LEFT)
        self.assertIsNone(s.switched_from)
        self.assertEqual(len(self.switches()), 1)

    def test_back_and_forth_announces_each_switch(self):
        self.placed_by(A)
        for speaker in (B, A, A, B):
            self.svc.speak(self.req_sess(speaker), self.TEXT, NEVER_LEFT)
        self.assertEqual(self.announcements(), ["Angie, algolearn hedge", "Angie, algolearn speak",
                                                 "Angie, algolearn hedge"])
        self.assertEqual([(e["from_project"], e["to_project"]) for e in self.switches()],
                         [("algolearn-speak", "algolearn-hedge"), ("algolearn-hedge", "algolearn-speak"),
                          ("algolearn-speak", "algolearn-hedge")])

    def test_two_sessions_in_the_same_folder_are_still_a_switch(self):
        self.placed_by(A)
        s = self.req_sess(d.Speaker("session-a2", "algolearn-speak"))
        self.svc.speak(s, self.TEXT, NEVER_LEFT)
        self.assertEqual(s.switched_from, "algolearn-speak")
        self.assertEqual(self.announcements(), ["Angie, algolearn speak"])

    def test_listen_alone_is_announced_before_the_ear_opens(self):
        self.placed_by(A)
        s = self.req_sess(B)
        self.assertEqual(self.svc.listen(s, 120, 2.0, 45), {"text": "hi Angie"})
        self.assertEqual(s.switched_from, "algolearn-speak")
        kinds = [e[0] for e in self.log]
        self.assertLess(kinds.index("cue"), kinds.index("record"))
        # owner decision 2026-10-20: a bare listen says it is still listening after the name
        self.assertEqual(self.announcements(), ["Angie, algolearn hedge, still listening"])
        (event,) = self.switches()
        self.assertEqual((event["tool"], event["spoken"]), ("listen", "Angie, algolearn hedge, still listening"))

    def test_speak_and_converse_switches_say_only_the_name(self):
        self.placed_by(A)
        self.svc.converse(self.req_sess(B), self.TEXT, 120, 2.0, 45, NEVER_LEFT)
        self.svc.speak(self.req_sess(A), self.TEXT, NEVER_LEFT)
        self.assertEqual(self.announcements(), ["Angie, algolearn hedge", "Angie, algolearn speak"])
        self.assertEqual([(e["tool"], e["spoken"]) for e in self.switches()],
                         [("converse", "Angie, algolearn hedge"), ("speak", "Angie, algolearn speak")])

    def test_a_new_call_starts_fresh(self):
        self.placed_by(A)
        self.svc.speak(self.req_sess(A), self.TEXT, NEVER_LEFT)
        self.link.generation = 2          # a new call, placed by B
        self.placed_by(B)
        s = self.req_sess(B)
        self.svc.speak(s, self.TEXT, NEVER_LEFT)
        self.assertIsNone(s.switched_from)

    def test_a_turn_from_an_earlier_call_never_counts(self):
        self.placed_by(A)
        self.svc.speak(self.req_sess(A), self.TEXT, NEVER_LEFT)
        self.link.generation = 2          # a new call with no recorded first turn
        s = self.req_sess(B)
        self.svc.speak(s, self.TEXT, NEVER_LEFT)
        self.assertIsNone(s.switched_from)
        s = self.req_sess(A)                  # but on this call, B spoke last
        self.svc.speak(s, self.TEXT, NEVER_LEFT)
        self.assertEqual(s.switched_from, "algolearn-hedge")

    def test_empty_text_raises_before_any_announcement_and_keeps_the_turn(self):
        self.placed_by(A)
        s = self.req_sess(B)
        with self.assertRaisesRegex(ValueError, "empty text"):
            self.svc.speak(s, "**", NEVER_LEFT)
        self.assertIsNone(s.switched_from)
        self.assertEqual(self.announcements(), [])
        self.assertEqual(self.svc._turn[1], A)

    def test_no_call_no_announcement(self):
        self.placed_by(A)
        self.link._state = tg.NONE
        s = self.req_sess(B)
        with self.assertRaisesRegex(RuntimeError, "needs a connected Telegram call"):
            self.svc.listen(s, 120, 2.0, 45)
        self.assertEqual(self.announcements(), [])
        self.assertEqual(self.svc._turn[1], A)

    def test_a_drop_during_the_announcement_is_call_ended_and_keeps_the_turn(self):
        self.placed_by(A)
        self.link.drop_on_play = True
        s = self.req_sess(B)
        with self.assertRaisesRegex(d.CallEndedError, "Telegram call ended during converse"):
            self.svc.converse(s, self.TEXT, 120, 2.0, 45, NEVER_LEFT)
        self.assertIsNone(s.switched_from)
        self.assertEqual(self.switches(), [])
        self.assertEqual(self.svc._turn[1], A)
        self.assertNotIn("play", [e[0] for e in self.log])


class TestRunLockedReply(SwitchCase):
    def setUp(self):
        super().setUp()
        self.daemon = d.SpeakDaemon.__new__(d.SpeakDaemon)
        self.daemon.service = self.svc
        self.daemon.personas = self.svc.personas

    def req(self, speaker: d.Speaker, **fields) -> dict:
        """As admitted: the project has its persona and the request has seen it."""
        self.svc.personas.ensure(speaker.project)
        return {proto.FIELD_SESSION_ID: speaker.session_id, proto.FIELD_PROJECT: speaker.project,
                "expect_generation": None, proto.FIELD_PERSONA_SEEN: self.svc.personas.key(speaker.project),
                **fields}

    def test_the_reply_names_the_session_it_took_over_from(self):
        self.placed_by(A)
        reply = self.daemon.run_locked(proto.OP_SPEAK, self.req(B, text="One."), NEVER_LEFT)
        self.assertEqual(reply[proto.REPLY_SWITCHED_FROM], "algolearn-speak")
        reply = self.daemon.run_locked(proto.OP_SPEAK, self.req(B, text="One."), NEVER_LEFT)
        self.assertNotIn(proto.REPLY_SWITCHED_FROM, reply)

    def test_a_failure_after_the_announcement_still_carries_it(self):
        self.placed_by(A)

        def nobody(*args):
            self.log.append(("record",))
            return tg.TIMEOUT
        self.link.record = nobody
        reply = self.daemon.run_locked(proto.OP_LISTEN, self.req(B, max_seconds=120, silence_seconds=2.0,
                                                                 start_timeout_seconds=45), NEVER_LEFT)
        self.assertIs(reply["ok"], False)
        self.assertEqual(reply["kind"], proto.KIND_TIMEOUT)
        self.assertEqual(reply[proto.REPLY_SWITCHED_FROM], "algolearn-speak")

    def test_the_reply_carries_the_persona_only_when_it_changed(self):
        self.placed_by(B)
        reply = self.daemon.run_locked(proto.OP_SPEAK, self.req(B, text="One."), NEVER_LEFT)
        self.assertNotIn(proto.REPLY_PERSONA, reply)
        req = self.req(B, text="One.")
        self.svc.personas.set("algolearn-hedge", speed=1.2)   # the sound changed while the op was queued
        reply = self.daemon.run_locked(proto.OP_SPEAK, req, NEVER_LEFT)
        self.assertIs(reply["ok"], True)
        self.assertNotIn(proto.REPLY_PERSONA, reply)   # design 2026-10-22: not the words, so no block
        req = self.req(B, text="One.")
        self.svc.personas.set("algolearn-hedge", personality=["Calm."])   # the words changed
        reply = self.daemon.run_locked(proto.OP_SPEAK, req, NEVER_LEFT)
        self.assertEqual(reply[proto.REPLY_PERSONA]["speed"], 1.2)
        self.assertEqual(reply[proto.REPLY_PERSONA]["words_revision"], 2)


class TestPersonaVoices(SwitchCase):
    """docs/DESIGN_PERSONAS.md 4.5 and 4.7: who speaks with which voice."""

    def setUp(self):
        super().setUp()
        self.personas = self.svc.personas
        self.personas.ensure("algolearn-hedge")
        self.personas.new("algolearn-hedge")
        self.personas.set("algolearn-hedge", voice={"bf_emma": 3, "am_adam": 1})
        self.personas.set("algolearn-hedge", speed=1.3)
        self.hedge = self.personas.persona_of("algolearn-hedge")

    def whole(self) -> list[tuple]:
        return [v[1:] for v in self.engines.voices if v[0] == "whole"]

    def segments(self) -> list[tuple]:
        return [v[1:] for v in self.engines.voices if v[0] == "segment"]

    def test_the_announcement_names_and_sounds_like_the_incoming_persona(self):
        self.placed_by(A)
        self.svc.speak(self.req_sess(B), self.TEXT, NEVER_LEFT)
        spoken = f"{self.hedge.name}, algolearn hedge"
        blend = (("am_adam", 0.25), ("bf_emma", 0.75))
        self.assertEqual(self.whole(), [(spoken, blend, 1.3)])
        self.assertEqual(self.segments(), [(0, blend, 1.3)])
        (event,) = self.switches()
        self.assertEqual((event["to_persona"], event["spoken"]), (self.hedge.name, spoken))

    def test_a_bare_listen_switch_keeps_the_suffix_after_the_persona(self):
        self.placed_by(A)
        self.svc.listen(self.req_sess(B), 120, 2.0, 45)
        self.assertEqual(self.announcements(), [f"{self.hedge.name}, algolearn hedge, still listening"])

    def test_the_greeting_is_in_the_callers_voice(self):
        self.link._state = tg.NONE

        def dial(audio, rate):   # the greeting is synthesized before the dial; stop there
            raise RuntimeError("not dialed in this test")
        self.link.call = dial
        with self.assertRaisesRegex(RuntimeError, "not dialed"):
            self.svc.call(self.req_sess(B), "Hi Frank, it's the hedge bot with news.", True, 120, 2.0, 45)
        self.assertEqual(self.whole(), [("Hi Frank, it's the hedge bot with news.",
                                         (("am_adam", 0.25), ("bf_emma", 0.75)), 1.3)])

    def test_a_change_between_sentences_lands_on_the_next_sentence(self):
        self.placed_by(B)
        real = self.engines.synthesize_segments

        def changing(text, voice_of):
            for i, audio in real(text, voice_of):
                if i == 0:   # the owner's change arrives while sentence 0 is out
                    self.personas.set("algolearn-hedge", speed=0.9)
                yield i, audio
        self.engines.synthesize_segments = changing
        self.svc.speak(self.req_sess(B), "One. Two. Three.", NEVER_LEFT)
        self.assertEqual([speed for _i, _blend, speed in self.segments()], [1.3, 0.9, 0.9])

    def test_a_continue_replays_the_cached_sentence_in_its_voice(self):
        self.placed_by(B)
        self.link.cuts = [(2, False)]
        self.link.holds = [(5.0, proto.CMD_RESUME)]
        real = self.engines.synthesize_segments

        def changing(text, voice_of):
            for i, audio in real(text, voice_of):
                yield i, audio
                if i == 1:   # changed while held, after sentence 1 was synthesized
                    self.personas.set("algolearn-hedge", speed=0.9)
        self.engines.synthesize_segments = changing
        self.svc.speak(self.req_sess(B), "One. Two. Three.", NEVER_LEFT)
        self.assertEqual(self.synths(), [0, 1, 2])   # "Two." replayed, not synthesized again
        self.assertEqual([speed for _i, _blend, speed in self.segments()], [1.3, 1.3, 0.9])


class TestPieces(unittest.TestCase):
    def test_speakable_name(self):
        for project, spoken in [("algolearn-speak", "algolearn speak"), ("my_project", "my project"),
                                ("a--b__c", "a b c"), ("  x.y  ", "x y"), ("hiro2", "hiro2"),
                                ("--", ""), ("", "")]:
            self.assertEqual(proto.speakable_name(project), spoken, project)

    def test_the_switch_cue_differs_from_every_other_cue(self):
        switch = speak_tone.switch_pcm()
        self.assertEqual(switch.dtype, np.float32)
        for other in (speak_tone.chime_pcm(), speak_tone.cue_pcm(880.0), speak_tone.cue_pcm(440.0)):
            self.assertFalse(len(other) == len(switch) and np.allclose(other, switch))

    def test_init_identity_takes_the_folder_name_and_a_fresh_session_id(self):
        first = client.init_identity("/Users/x/workspace/algolearn-hedge/")
        self.assertEqual(first[proto.FIELD_PROJECT], "algolearn-hedge")
        second = client.init_identity("/Users/x/workspace/algolearn-hedge")
        self.assertNotEqual(first[proto.FIELD_SESSION_ID], second[proto.FIELD_SESSION_ID])
        with self.assertRaisesRegex(RuntimeError, "has no name to announce"):
            client.init_identity("/")

    def test_the_switch_line_comes_first(self):
        line = ('[took over the call from "algolearn-speak"; the owner heard the switch cue and your persona\'s '
                'name and this project\'s name first]\n')
        reply = {proto.REPLY_SWITCHED_FROM: "algolearn-speak", proto.REPLY_HELD_SECONDS: 12.0, "seconds": 2.0,
                 "text": "yes"}
        self.assertEqual(client._speak_result(reply), line + "[held 12s, then resumed]\nspoke for 2.0s" + proto.AUDIO_NOTE)
        self.assertEqual(client._converse_result(reply), line + "[held 12s, then resumed]\nyes" + proto.AUDIO_NOTE)
        self.assertEqual(client._listen_result(reply), line + "yes")
        del reply[proto.REPLY_SWITCHED_FROM]
        self.assertEqual(client._listen_result(reply), "yes")
        expired = {proto.REPLY_SWITCHED_FROM: "algolearn-speak", "seconds": 1.0, proto.REPLY_HELD_SECONDS: 300.0,
                   proto.REPLY_INTERRUPTION: {"command": proto.INTERRUPTION_HOLD_EXPIRED, "cut_index": 1,
                                              "sentence_count": 1, "cut_sentence": "One.", "unspoken": "One."}}
        self.assertTrue(client._speak_result(expired).startswith(line + "[hold expired"))


if __name__ == "__main__":
    unittest.main()
