"""The persona registry (speak_personas.PersonaStore), the Engines voice list and the voice
blends (docs/DESIGN_PERSONAS.md section 8). No CUDA, no Telegram: the store is stdlib only,
and the engines are tested without loading Kokoro or Whisper (torch on the CPU only).
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import threading
import unittest
import zoneinfo
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import speak_personas as sp  # noqa: E402
import speak_protocol as proto  # noqa: E402
from speak_engines import BLEND_CACHE_SIZE, ENGLISH_VOICES, Engines  # noqa: E402

VOICES = ["af_bella", "af_heart", "am_adam", "bf_emma", "bm_george"]
TZ = zoneinfo.ZoneInfo("America/Los_Angeles")


class StoreCase(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="speak-personas-")
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.path = os.path.join(self.dir, "personas.json")
        self.events: list[tuple[str, dict]] = []

    def open(self) -> sp.PersonaStore:
        return sp.PersonaStore.open(self.path, VOICES, "af_heart", 1.0, TZ,
                                    lambda event, **fields: self.events.append((event, fields)))

    def on_disk(self) -> dict:
        with open(self.path, encoding="utf-8") as f:
            return json.load(f)

    def write(self, registry: dict) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(registry, f)

    def store(self, *projects: str) -> sp.PersonaStore:
        store = self.open()
        for project in projects:
            store.ensure(project)
        return store


class TestSeedAndLoad(StoreCase):
    def test_a_missing_file_seeds_angie_as_the_house_persona(self):
        self.open()
        reg = self.on_disk()
        (pid, angie), = reg["personas"].items()
        self.assertEqual((reg["schema"], reg["house"], reg["projects"]), (1, pid, {}))
        self.assertEqual((angie["name"], angie["voice"], angie["speed"], angie["personality"], angie["words_revision"]),
                         ("Angie", {"af_heart": 1.0}, 1.0, [], 1))

    def test_a_round_trip_keeps_everything(self):
        store = self.store("algolearn-speak")
        store.set("algolearn-speak", voice={"af_bella": 3, "bf_emma": 1})
        store.add("algolearn-speak", "Dry humor.")
        again = self.open()
        self.assertEqual(again.block("algolearn-speak"), store.block("algolearn-speak"))
        self.assertEqual(again._registry, store._registry)

    def test_a_corrupt_file_raises_and_is_never_replaced(self):
        with open(self.path, "w") as f:
            f.write("{not json")
        with self.assertRaisesRegex(RuntimeError, f"persona file {self.path} is not usable"):
            self.open()
        with open(self.path) as f:
            self.assertEqual(f.read(), "{not json")

    def test_an_invalid_file_raises(self):
        self.open()
        reg = self.on_disk()
        next(iter(reg["personas"].values()))["speed"] = 3.0
        self.write(reg)
        with self.assertRaisesRegex(RuntimeError, "speed 3.0 must be a number from 0.5 to 2.0"):
            self.open()

    def test_a_dangling_project_or_house_pointer_is_refused_on_load(self):
        self.open()
        reg = self.on_disk()
        reg["projects"]["algolearn-speak"] = "gone"
        self.write(reg)
        with self.assertRaisesRegex(RuntimeError, "points at persona 'gone', which does not exist"):
            self.open()
        reg["projects"] = {}
        reg["house"] = "gone"
        self.write(reg)
        with self.assertRaisesRegex(RuntimeError, "house points at persona 'gone'"):
            self.open()

    def test_a_pointer_that_is_not_a_string_is_refused_on_load(self):
        self.open()
        reg = self.on_disk()
        pid = reg["house"]
        reg["projects"]["algolearn-speak"] = [pid]
        self.write(reg)
        with self.assertRaisesRegex(RuntimeError, rf"persona file {self.path} is not usable: project "
                                                  r"'algolearn-speak' points at \['\w+'\], which is not a persona id"):
            self.open()
        reg["projects"] = {}
        reg["house"] = {"id": pid}
        self.write(reg)
        with self.assertRaisesRegex(RuntimeError, rf"persona file {self.path} is not usable: house is "
                                                  r"\{'id': '\w+'\}, which is not a persona id"):
            self.open()

    def test_a_default_voice_that_is_not_english_raises(self):
        with self.assertRaisesRegex(RuntimeError, "SPEAK_VOICE 'jf_alpha' is not an English voice"):
            sp.PersonaStore.open(self.path, VOICES, "jf_alpha", 1.0, TZ, lambda event, **fields: None)

    def test_a_missing_directory_is_refused_at_startup(self):
        with self.assertRaisesRegex(RuntimeError, "is the speak_state volume mounted"):
            sp.PersonaStore.check_dir(os.path.join(self.dir, "missing", "personas.json"))
        sp.PersonaStore.check_dir(self.path)

    def test_a_failed_write_keeps_the_old_file_and_the_registry(self):
        store = self.store("algolearn-speak")
        before = self.on_disk()
        with mock.patch.object(sp.os, "replace", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                store.add("algolearn-speak", "Dry humor.")
        self.assertEqual(self.on_disk(), before)
        self.assertEqual(store.block("algolearn-speak")["personality"], [])

    def test_a_failed_directory_fsync_raises_with_the_change_on_disk_and_in_memory(self):
        store = self.store("algolearn-speak")
        with mock.patch.object(sp.os, "open", side_effect=OSError("I/O error")):
            with self.assertRaisesRegex(sp.PersonaDirSyncError, "the change is on disk and in use, but the fsync "
                                                                f"of {self.dir} failed \\(I/O error\\)"):
                store.add("algolearn-speak", "Dry humor.")
        self.assertEqual(store.block("algolearn-speak")["personality"], ["Dry humor."])
        self.assertEqual(self.on_disk(), store._registry)
        self.assertEqual(self.open()._registry, store._registry)


class TestAssignment(StoreCase):
    def test_an_unknown_project_joins_the_house_persona(self):
        store = self.store()
        store.ensure("algolearn-hedge")
        self.assertEqual(store.block("algolearn-hedge")["name"], "Angie")
        self.assertEqual(self.events, [(sp.PERSONA_ASSIGN_EVENT, {"project": "algolearn-hedge", "persona": "Angie",
                                                                  "how": "auto", "from": None})])
        store.ensure("algolearn-hedge")   # known: no second assignment
        self.assertEqual(len(self.events), 1)
        self.assertEqual(self.on_disk()["projects"], {"algolearn-hedge": self.on_disk()["house"]})

    def test_the_house_persona_keeps_its_role_after_a_rename(self):
        store = self.store("algolearn-speak")
        store.set("algolearn-speak", name="Juno")
        store.ensure("algolearn-hedge")
        self.assertEqual(store.block("algolearn-hedge")["name"], "Juno")

    def test_a_new_name_renames_the_persona_for_every_project_sharing_it(self):
        store = self.store("algolearn-speak", "algolearn-hedge")
        block = store.set("algolearn-speak", name="Juno")
        self.assertEqual((block["name"], block["shared_with"]), ("Juno", ["algolearn-hedge"]))
        self.assertEqual(store.block("algolearn-hedge")["name"], "Juno")

    def test_moving_a_project_onto_another_persona_by_name(self):
        store = self.store("algolearn-speak", "algolearn-hedge")
        other = store.new("algolearn-hedge")["name"]
        moved = store.set("algolearn-speak", name=other.upper())   # case-insensitive
        self.assertEqual((moved["name"], moved["shared_with"]), (other, ["algolearn-hedge"]))
        self.assertEqual(self.events[-1], (sp.PERSONA_ASSIGN_EVENT, {"project": "algolearn-speak", "persona": other,
                                                                     "how": "moved", "from": "Angie"}))

    def test_names_are_unique_ignoring_case(self):
        store = self.store("algolearn-speak", "algolearn-hedge")
        pid = store.new("algolearn-hedge")["id"]
        broken = json.loads(json.dumps(store._registry))
        broken["personas"][pid]["name"] = "ANGIE"
        with self.assertRaisesRegex(ValueError, "persona name 'ANGIE' is used by"):
            sp.validate(broken, VOICES)

    def test_a_change_to_a_shared_persona_is_seen_by_every_project(self):
        store = self.store("algolearn-speak", "algolearn-hedge")
        store.set("algolearn-hedge", speed=1.2)
        self.assertEqual(store.persona_of("algolearn-speak").speed, 1.2)
        self.assertEqual(store.key("algolearn-speak"), store.key("algolearn-hedge"))

    def test_the_words_revision_rises_only_with_the_name_or_the_personality(self):
        """Design 2026-10-22: the key a client sends back changes only with what its words
        depend on; a voice or speed change leaves it as it is."""
        store = self.store("algolearn-speak")
        revisions = [store.block("algolearn-speak")["words_revision"]]
        for change in (lambda: store.set("algolearn-speak", voice={"af_bella": 1}),
                       lambda: store.set("algolearn-speak", speed=1.1),
                       lambda: store.set("algolearn-speak", personality=["Calm."]),
                       lambda: store.add("algolearn-speak", "Dry humor."),
                       lambda: store.set("algolearn-speak", name="Juno"),
                       lambda: store.set("algolearn-speak", voice={"bf_emma": 1}, speed=0.9)):
            change()
            revisions.append(store.block("algolearn-speak")["words_revision"])
        self.assertEqual(revisions, [1, 1, 1, 2, 3, 4, 4])

    def test_set_personality_raises_the_revision_only_when_the_texts_changed(self):
        store = self.store("algolearn-speak")
        store.set("algolearn-speak", personality=["Calm.", "Dry humor."])
        revision = store.block("algolearn-speak")["words_revision"]
        store.set("algolearn-speak", personality=["Calm.", "Dry humor."])   # the same list again
        self.assertEqual(store.block("algolearn-speak")["words_revision"], revision)
        store.set("algolearn-speak", personality=["Dry humor.", "Calm."])   # reordered: changed
        self.assertEqual(store.block("algolearn-speak")["words_revision"], revision + 1)

    def test_the_key_changes_on_a_move_and_on_words_but_not_on_the_sound(self):
        store = self.store("algolearn-speak", "algolearn-hedge")
        key = store.key("algolearn-speak")
        store.set("algolearn-hedge", voice={"af_bella": 2, "bf_emma": 1}, speed=1.3)
        self.assertEqual(store.key("algolearn-speak"), key)
        store.add("algolearn-hedge", "Dry humor.")
        self.assertNotEqual(store.key("algolearn-speak"), key)
        key = store.key("algolearn-speak")
        store.new("algolearn-hedge")
        moved = store.set("algolearn-speak", name=store.block("algolearn-hedge")["name"])
        self.assertNotEqual(store.key("algolearn-speak"), key)
        self.assertEqual(store.key("algolearn-speak"), proto.persona_key(moved))

    def test_the_block_says_whether_it_is_the_house_persona(self):
        """Owner decision 2026-10-22: the house persona never grows on its own, so the agent
        must know when it is on it. The flag follows the persona by id through a rename."""
        store = self.store("algolearn-speak", "algolearn-hedge")
        self.assertTrue(store.block("algolearn-speak")["house"])
        store.set("algolearn-speak", name="Juno")
        self.assertTrue(store.block("algolearn-hedge")["house"])
        self.assertFalse(store.new("algolearn-hedge")["house"])
        houses = {p["name"]: p["house"] for p in store.overview("algolearn-hedge")["personas"]}
        self.assertEqual(houses, {"Juno": True, store.block("algolearn-hedge")["name"]: False})

    def test_pool_exhaustion_raises(self):
        store = self.store("p")
        for _ in range(len(proto.PERSONA_POOL) - 1):
            store.new("p")
        with self.assertRaisesRegex(ValueError, "every pool name is in use; ask the owner for a name"):
            store.new("p")
        self.assertEqual(store.overview("p")["pool_left"], 0)


class TestValidation(StoreCase):
    def test_weights_are_normalized(self):
        self.assertEqual(sp.normalize_voice({"bf_emma": 1, "af_bella": 2}, VOICES),
                         {"af_bella": 0.6667, "bf_emma": 0.3333})
        self.assertEqual(sp.normalize_voice({"af_bella": 1, "am_adam": 1, "bf_emma": 1}, VOICES),
                         {"af_bella": 0.3334, "am_adam": 0.3333, "bf_emma": 0.3333})

    def test_more_than_four_voices_are_refused(self):
        with self.assertRaisesRegex(ValueError, "at most 4 voices, got 5"):
            sp.normalize_voice({v: 1 for v in VOICES}, VOICES)

    def test_a_non_english_voice_id_is_refused(self):
        with self.assertRaisesRegex(ValueError, "unknown voice id 'jf_alpha'; English voices only"):
            sp.normalize_voice({"jf_alpha": 1}, VOICES)

    def test_a_weight_that_rounds_to_zero_is_refused_naming_the_voice(self):
        with self.assertRaisesRegex(ValueError, "voice weight 1e-05 for 'am_adam' rounds to 0 at 4 decimals"):
            sp.normalize_voice({"af_bella": 1, "am_adam": 0.00001}, VOICES)
        store = self.store("algolearn-speak")
        before = self.on_disk()
        with self.assertRaisesRegex(ValueError, "'bf_emma' rounds to 0"):
            store.set("algolearn-speak", voice={"af_bella": 100000, "bf_emma": 1})
        self.assertEqual(self.on_disk(), before)

    def test_a_zero_or_negative_weight_is_refused(self):
        for weight in (0, -1, True, "1"):
            with self.assertRaisesRegex(ValueError, "must be a number above 0"):
                sp.normalize_voice({"af_bella": weight}, VOICES)

    def test_the_speed_range(self):
        self.assertEqual((sp.check_speed(0.5), sp.check_speed(2)), (0.5, 2.0))
        for speed in (0.49, 2.01, True, "1"):
            with self.assertRaisesRegex(ValueError, "must be a number from 0.5 to 2.0"):
                sp.check_speed(speed)

    def test_names(self):
        for name in ("Al", "Margo", "a" * proto.PERSONA_NAME_MAX):
            sp.check_name(name)
        for name in ("A", "a" * (proto.PERSONA_NAME_MAX + 1), "Mary Ann", "R2", "", None):
            with self.assertRaises(ValueError):
                sp.check_name(name)

    def test_the_personality_cap(self):
        store = self.store("algolearn-speak")
        store.add("algolearn-speak", "x" * (proto.PERSONALITY_MAX_CHARS - 1))
        with self.assertRaisesRegex(ValueError, "the personality is full; ask the owner for a persona review"):
            store.add("algolearn-speak", "yy")
        with self.assertRaisesRegex(ValueError, "the personality would be 4001 characters, over 4000"):
            store.set("algolearn-speak", personality=["x" * 4000, "y"])

    def test_a_replaced_personality_keeps_the_stamps_of_unchanged_lines(self):
        store = self.store("algolearn-speak", "algolearn-hedge")
        store.add("algolearn-hedge", "Dry humor.")
        stamped = store._registry["personas"][store.persona_of("algolearn-speak").id]["personality"][0]
        store.set("algolearn-speak", personality=["Calm.", "Dry humor."])
        lines = store._registry["personas"][store.persona_of("algolearn-speak").id]["personality"]
        self.assertEqual(lines[1], stamped)
        self.assertEqual(lines[0]["project"], "algolearn-speak")

    def test_one_change_per_call(self):
        store = self.store("algolearn-speak")
        with self.assertRaisesRegex(ValueError, "name cannot be combined"):
            store.set("algolearn-speak", name="Juno", speed=1.1)
        with self.assertRaisesRegex(ValueError, "nothing to change"):
            store.set("algolearn-speak")


class TestConcurrency(StoreCase):
    def test_concurrent_writes_and_persona_of_reads_stay_consistent(self):
        """Writers set voice, speed and personality together (af_bella at 1.0, bm_george at 1.5)
        while readers take snapshots: every snapshot is one of the two pairs, a reader's words
        revisions never go back, and at the end memory, disk and the revision count all agree."""
        store = self.store("algolearn-speak")
        pairs = {(("af_bella", 1.0),): 1.0, (("bm_george", 1.0),): 1.5, (("af_heart", 1.0),): 1.0}
        writers, writes = 3, 15
        done = threading.Event()
        errors: list[BaseException] = []

        def write(n: int) -> None:
            try:
                for i in range(writes):
                    voice, speed = ("af_bella", 1.0) if (n + i) % 2 else ("bm_george", 1.5)
                    store.set("algolearn-speak", voice={voice: 1}, speed=speed, personality=[f"Writer {n}, {i}."])
            except BaseException as e:
                errors.append(e)

        def read() -> None:
            try:
                last = 0
                while not done.is_set():
                    p = store.persona_of("algolearn-speak")
                    self.assertEqual(pairs[p.voice], p.speed)
                    self.assertGreaterEqual(p.words_revision, last)
                    last = p.words_revision
            except BaseException as e:
                errors.append(e)
        readers = [threading.Thread(target=read) for _ in range(3)]
        writing = [threading.Thread(target=write, args=(n,)) for n in range(writers)]
        for t in readers + writing:
            t.start()
        for t in writing:
            t.join(60)
        done.set()
        for t in readers:
            t.join(10)
        self.assertEqual(errors, [])
        self.assertEqual(store.persona_of("algolearn-speak").words_revision, 1 + writers * writes)
        self.assertEqual(self.on_disk(), store._registry)


class FakeHub:
    """snapshot_download into a temp dir holding a .pt pack for every id of `ids` (any
    language: the allow_patterns filter decides what the snapshot gets, as the hub does). Like
    the real one with the Hub unreachable, it serves whatever the cache has, even when that is
    not every file asked for."""

    def __init__(self, root: str, ids: list[str]) -> None:
        import torch
        self.root = root
        self.calls: list = []
        os.makedirs(os.path.join(root, "voices"))
        for i, vid in enumerate(ids):
            torch.save(torch.full((510, 1, 256), float(i), dtype=torch.float32),
                       os.path.join(root, "voices", f"{vid}.pt"))

    def __call__(self, repo_id: str, allow_patterns: list[str]) -> str:
        import fnmatch
        self.calls.append((repo_id, allow_patterns))
        out = os.path.join(self.root, "snapshot")
        os.makedirs(os.path.join(out, "voices"), exist_ok=True)
        for name in os.listdir(os.path.join(self.root, "voices")):
            if any(fnmatch.fnmatch(f"voices/{name}", p) for p in allow_patterns):
                shutil.copy(os.path.join(self.root, "voices", name), os.path.join(out, "voices", name))
        return out


class TestEnginesVoices(unittest.TestCase):
    """Owner decision 2026-10-22: the daemon loads exactly ENGLISH_VOICES and refuses to start,
    naming what is missing, when the snapshot (or the cache it falls back on) lacks any."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="speak-voices-")
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.engines = Engines("hexgrad/Kokoro-82M", "large-v3-turbo", "en")

    def load(self, default_voice: str, ids) -> None:
        self.hub = FakeHub(self.dir, list(ids))
        with mock.patch("huggingface_hub.snapshot_download", self.hub):
            self.engines.load_voices(default_voice)

    def test_the_expected_set_is_the_28_english_voices_listed_explicitly(self):
        self.assertEqual(len(ENGLISH_VOICES), 28)
        self.assertEqual(len(set(ENGLISH_VOICES)), 28)
        self.assertTrue(all(v.startswith(proto.PERSONA_VOICE_PREFIXES) for v in ENGLISH_VOICES))
        self.assertEqual({v[:3] for v in ENGLISH_VOICES}, set(proto.PERSONA_VOICE_PREFIXES))

    def test_exactly_the_expected_voices_are_fetched_and_loaded(self):
        self.load("af_heart", [*ENGLISH_VOICES, "jf_alpha", "zf_xiaobei", "ef_dora"])
        self.assertEqual(self.engines.voices, sorted(ENGLISH_VOICES))
        self.assertEqual(self.hub.calls, [("hexgrad/Kokoro-82M", [f"voices/{v}.pt" for v in ENGLISH_VOICES])])

    def test_a_missing_expected_voice_stops_the_load_naming_it(self):
        ids = [v for v in ENGLISH_VOICES if v not in ("am_santa", "bm_fable")]
        with self.assertRaisesRegex(RuntimeError, "2 of the 28 English voice packs of hexgrad/Kokoro-82M are "
                                                  r"missing from .*: am_santa, bm_fable$"):
            self.load("af_heart", ids)
        self.assertEqual(self.engines.packs, {})

    def test_an_empty_cache_stops_the_load(self):
        with self.assertRaisesRegex(RuntimeError, "28 of the 28 English voice packs"):
            self.load("af_heart", [])
        self.assertEqual(self.engines.packs, {})

    def test_a_non_english_default_voice_stops_the_load_before_any_fetch(self):
        with self.assertRaisesRegex(RuntimeError, "SPEAK_VOICE 'jf_alpha' is not an English voice"):
            self.load("jf_alpha", [*ENGLISH_VOICES, "jf_alpha"])
        self.assertEqual((self.engines.packs, self.hub.calls), ({}, []))

    def test_the_packs_are_cpu_float32(self):
        import torch
        self.load("af_heart", ENGLISH_VOICES)
        for pack in self.engines.packs.values():
            self.assertIsInstance(pack, torch.FloatTensor)
            self.assertEqual(tuple(pack.shape), (510, 1, 256))


class TestBlends(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        self.engines = Engines("hexgrad/Kokoro-82M", "large-v3-turbo", "en")
        self.engines.packs = {"af_bella": torch.full((510, 1, 256), 1.0), "bf_emma": torch.full((510, 1, 256), 3.0),
                              "am_adam": torch.full((510, 1, 256), 5.0)}

    def test_the_weighted_sum(self):
        pack = self.engines.voice_pack((("af_bella", 0.75), ("bf_emma", 0.25)))
        self.assertTrue(self.torch.allclose(pack, self.torch.full((510, 1, 256), 1.5)))
        self.assertIsInstance(pack, self.torch.FloatTensor)
        self.assertEqual(pack.device.type, "cpu")

    def test_a_single_voice_is_the_loaded_pack(self):
        self.assertIs(self.engines.voice_pack((("bf_emma", 1.0),)), self.engines.packs["bf_emma"])

    def test_the_cache_key_ignores_order(self):
        first = self.engines.voice_pack((("af_bella", 0.5), ("bf_emma", 0.5)))
        self.assertIs(self.engines.voice_pack((("bf_emma", 0.5), ("af_bella", 0.5))), first)

    def test_the_cache_keeps_the_most_recent_blends(self):
        first = self.engines.voice_pack((("af_bella", 0.5), ("bf_emma", 0.5)))
        for i in range(1, BLEND_CACHE_SIZE + 1):
            self.engines.voice_pack((("af_bella", round(1 - i / 1000, 3)), ("am_adam", i / 1000)))
        self.assertEqual(len(self.engines._blends), BLEND_CACHE_SIZE)
        self.assertIsNot(self.engines.voice_pack((("af_bella", 0.5), ("bf_emma", 0.5))), first)

    def test_an_empty_blend_raises(self):
        with self.assertRaisesRegex(ValueError, "at least one voice"):
            self.engines.voice_pack(())


if __name__ == "__main__":
    unittest.main()
