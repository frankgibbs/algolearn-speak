"""The persona registry: a name, a voice blend, a speed and a personality per project.

Design: docs/DESIGN_PERSONAS.md (owner decisions 2026-10-22). Stdlib only, no CUDA, no
Telegram and no audio, so it is tested directly like `speak_protocol`.

The registry is one JSON file (`SPEAK_PERSONAS_FILE`, on the `speak_state` volume), schema 1:

    {"schema": 1,
     "personas": {id: {"name", "voice": {voice id: weight}, "speed",
                       "personality": [{"text", "project", "added"}], "words_revision", "created"}},
     "projects": {project: id},
     "house": id}

- Every project that has spoken to the daemon points at exactly one persona; a project with
  no entry is assigned to the house persona (the one seeded as Angie, followed by id, so a
  rename keeps it the house persona) on its first identity-carrying request (`ensure`).
- `words_revision` rises only when something an agent's words depend on changes: the
  persona's name or personality. A voice or speed change leaves it as it is, since the daemon
  applies those per sentence (docs/DESIGN_PERSONAS.md 4.7, Design 2026-10-22). A project's
  key "<id>:<words_revision>" is what a client sends back as persona_seen, so moving the
  project to another persona changes the key too.
- The house persona never grows its personality on its own (owner decision 2026-10-22):
  the block's `house` flag tells the agent which rule applies. The store does not enforce
  it; the owner may still change the house persona's personality.
- Validation runs on load and on every change. A file that does not parse or validate
  stops the daemon (`PersonaStore.open` raises); a change that does not validate raises
  ValueError to the tool and nothing is written.
- A change is applied to a deep copy, validated, written atomically (`<file>.tmp`, fsync,
  os.replace), swapped in the moment os.replace succeeds, and then the directory is
  fsynced, so the registry in memory always matches the file on disk. A failed directory
  fsync raises PersonaDirSyncError, which says the change is on disk and in use.
- One threading.Lock guards read-modify-write. It is never held while waiting on the audio
  lock or while synthesizing: audio reads copy out an immutable `Persona` snapshot. A write
  holds it across its fsyncs, so `persona_of` waits for a write in progress (writes are
  rare, reads are per sentence; docs/DESIGN_PERSONAS.md 4.2).
"""

from __future__ import annotations

import copy
import datetime
import json
import logging
import os
import re
import secrets
import threading
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

import speak_protocol as proto

log = logging.getLogger("speak-personas")

SCHEMA = 1
NAME_RE = re.compile(rf"[A-Za-z]{{2,{proto.PERSONA_NAME_MAX}}}")
WEIGHT_DECIMALS = 4

PERSONA_ASSIGN_EVENT = "persona_assign"
PERSONA_CHANGE_EVENT = "persona_change"


class PersonaDirSyncError(RuntimeError):
    """The persona file was replaced (the change is on disk and in use), but the fsync of
    its directory failed, so the rename may not survive a power loss."""


@dataclass(frozen=True)
class Persona:
    """An immutable snapshot of one persona, for audio: the voice as (id, weight) pairs."""

    id: str
    name: str
    voice: tuple[tuple[str, float], ...]
    speed: float
    words_revision: int

    @property
    def key(self) -> str:
        """The "<id>:<words_revision>" a client sends back as persona_seen: it changes only
        with what the agent's words depend on (the persona, its name, its personality)."""
        return f"{self.id}:{self.words_revision}"


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def check_name(name) -> str:
    """A valid persona name (one word of 2 to 12 letters), as typed; raises ValueError."""
    if not isinstance(name, str) or not NAME_RE.fullmatch(name):
        raise ValueError(f"persona name {name!r} must be one word of 2 to {proto.PERSONA_NAME_MAX} letters")
    return name


def check_speed(speed) -> float:
    """A speed within PERSONA_SPEED_MIN..PERSONA_SPEED_MAX, as a float; raises ValueError."""
    if not _is_number(speed) or not proto.PERSONA_SPEED_MIN <= speed <= proto.PERSONA_SPEED_MAX:
        raise ValueError(f"speed {speed!r} must be a number from {proto.PERSONA_SPEED_MIN} "
                         f"to {proto.PERSONA_SPEED_MAX}")
    return float(speed)


def normalize_voice(voice, voices: Iterable[str]) -> dict[str, float]:
    """A blend {voice id: weight} normalized to sum 1: 1 to PERSONA_MAX_VOICES English ids from
    `voices`, each weight above 0. Weights are rounded to 4 decimals and the largest absorbs
    the rounding; the result is ordered by weight (largest first), then id. Raises ValueError."""
    if not isinstance(voice, Mapping) or not voice:
        raise ValueError("voice must be a non-empty mapping of voice id to weight")
    if len(voice) > proto.PERSONA_MAX_VOICES:
        raise ValueError(f"a voice blends at most {proto.PERSONA_MAX_VOICES} voices, got {len(voice)}")
    known = set(voices)
    for vid, weight in voice.items():
        if vid not in known:
            raise ValueError(f"unknown voice id {vid!r}; English voices only, see persona() for the list")
        if not _is_number(weight) or not weight > 0:
            raise ValueError(f"voice weight for {vid!r} must be a number above 0, got {weight!r}")
    total = float(sum(voice.values()))
    ordered = sorted(voice.items(), key=lambda kv: (-kv[1], kv[0]))
    rest = {vid: round(weight / total, WEIGHT_DECIMALS) for vid, weight in ordered[1:]}
    for vid, share in rest.items():
        if share == 0:
            raise ValueError(f"voice weight {voice[vid]!r} for {vid!r} rounds to 0 at {WEIGHT_DECIMALS} "
                             f"decimals of the total {total:g}; raise it or leave {vid!r} out")
    top = round(1.0 - sum(rest.values()), WEIGHT_DECIMALS)
    if not top > 0:
        raise ValueError(f"voice weights {dict(voice)!r} leave nothing for {ordered[0][0]!r}")
    return {ordered[0][0]: top, **rest}


def validate(registry, voices: Iterable[str]) -> None:
    """Raise ValueError naming the first thing wrong with `registry` (schema 1)."""
    voices = set(voices)
    if not isinstance(registry, dict):
        raise ValueError("the registry is not a JSON object")
    if registry.get("schema") != SCHEMA:
        raise ValueError(f"schema is {registry.get('schema')!r}, expected {SCHEMA}")
    if set(registry) != {"schema", "personas", "projects", "house"}:
        raise ValueError(f"top-level keys are {sorted(registry)}, expected schema, personas, projects, house")
    personas, projects = registry["personas"], registry["projects"]
    if not isinstance(personas, dict) or not personas:
        raise ValueError("personas must be a non-empty object")
    if not isinstance(projects, dict):
        raise ValueError("projects must be an object")
    names: dict[str, str] = {}
    for pid, p in personas.items():
        if not isinstance(p, dict):
            raise ValueError(f"persona {pid!r} is not an object")
        expected = {"name", "voice", "speed", "personality", "words_revision", "created"}
        if set(p) != expected:
            raise ValueError(f"persona {pid!r} has keys {sorted(p)}, expected {sorted(expected)}")
        check_name(p["name"])
        folded = p["name"].casefold()
        if folded in names:
            raise ValueError(f"persona name {p['name']!r} is used by {names[folded]!r} and {pid!r}")
        names[folded] = pid
        voice = p["voice"]
        if not isinstance(voice, dict) or not voice:
            raise ValueError(f"persona {pid!r}: voice must be a non-empty object")
        if len(voice) > proto.PERSONA_MAX_VOICES:
            raise ValueError(f"persona {pid!r}: at most {proto.PERSONA_MAX_VOICES} voices, got {len(voice)}")
        for vid, weight in voice.items():
            if vid not in voices:
                raise ValueError(f"persona {pid!r}: voice id {vid!r} is not an English voice of this daemon")
            if not _is_number(weight) or not weight > 0 or round(weight, WEIGHT_DECIMALS) != weight:
                raise ValueError(f"persona {pid!r}: weight {weight!r} for {vid!r} is not a normalized weight")
        if abs(sum(voice.values()) - 1.0) > 1e-6:
            raise ValueError(f"persona {pid!r}: voice weights sum to {sum(voice.values())}, not 1")
        try:
            check_speed(p["speed"])
        except ValueError as e:
            raise ValueError(f"persona {pid!r}: {e}") from None
        revision = p["words_revision"]
        if not isinstance(revision, int) or isinstance(revision, bool) or revision < 1:
            raise ValueError(f"persona {pid!r}: words_revision must be an integer of 1 or more")
        if not isinstance(p["created"], str):
            raise ValueError(f"persona {pid!r}: created must be a string")
        personality = p["personality"]
        if not isinstance(personality, list):
            raise ValueError(f"persona {pid!r}: personality must be a list")
        for entry in personality:
            if (not isinstance(entry, dict) or set(entry) != {"text", "project", "added"}
                    or not all(isinstance(entry[k], str) for k in entry) or not entry["text"].strip()):
                raise ValueError(f"persona {pid!r}: personality entry {entry!r} must be "
                                 "{text, project, added} strings with a non-blank text")
        chars = sum(len(e["text"]) for e in personality)
        if chars > proto.PERSONALITY_MAX_CHARS:
            raise ValueError(f"persona {pid!r}: personality is {chars} characters, "
                             f"over {proto.PERSONALITY_MAX_CHARS}")
    for project, pid in projects.items():
        if not isinstance(pid, str):
            raise ValueError(f"project {project!r} points at {pid!r}, which is not a persona id string")
        if pid not in personas:
            raise ValueError(f"project {project!r} points at persona {pid!r}, which does not exist")
    if not isinstance(registry["house"], str):
        raise ValueError(f"house is {registry['house']!r}, which is not a persona id string")
    if registry["house"] not in personas:
        raise ValueError(f"house points at persona {registry['house']!r}, which does not exist")


class PersonaStore:
    """The persona registry, backed by one JSON file. Build with `open`."""

    def __init__(self, path: str, registry: dict, voices: list[str], default_voice: str,
                 default_speed: float, tz: datetime.tzinfo, record_event: Callable[..., None]) -> None:
        self.path = path
        self.voices = voices
        self.default_voice = default_voice
        self.default_speed = default_speed
        self._tz = tz
        self._record_event = record_event
        self._registry = registry
        self._lock = threading.Lock()

    # ------------------------------------------------------------ startup

    @staticmethod
    def check_dir(path: str) -> None:
        """Startup, before the engines load: the file's directory must exist (the speak_state
        volume mount makes it). Raises RuntimeError otherwise."""
        directory = os.path.dirname(os.path.abspath(path))
        if not os.path.isdir(directory):
            raise RuntimeError(f"SPEAK_PERSONAS_FILE directory {directory} does not exist "
                               "(is the speak_state volume mounted?)")

    @classmethod
    def open(cls, path: str, voices: list[str], default_voice: str, default_speed: float,
             tz: datetime.tzinfo, record_event: Callable[..., None]) -> PersonaStore:
        """Load `path`, or seed it with the house persona Angie when it does not exist (a first
        start). A file that does not parse or validate raises RuntimeError naming the path;
        it is never replaced."""
        if default_voice not in voices:
            raise RuntimeError(f"SPEAK_VOICE {default_voice!r} is not an English voice of this daemon")
        check_speed(default_speed)
        if not os.path.exists(path):
            store = cls(path, {}, voices, default_voice, default_speed, tz, record_event)
            pid = secrets.token_hex(4)
            registry = {"schema": SCHEMA, "personas": {pid: store._fresh(proto.PERSONA_SEED_NAME)},
                        "projects": {}, "house": pid}
            validate(registry, voices)
            store._persist(registry)
            log.info("personas: no file at %s, seeded %s", path, proto.PERSONA_SEED_NAME)
            return store
        try:
            with open(path, encoding="utf-8") as f:
                registry = json.load(f)
            validate(registry, voices)
        except (OSError, ValueError) as e:   # json.JSONDecodeError is a ValueError
            raise RuntimeError(f"persona file {path} is not usable: {e}") from e
        log.info("personas: loaded %d personas, %d projects from %s",
                 len(registry["personas"]), len(registry["projects"]), path)
        return cls(path, registry, voices, default_voice, default_speed, tz, record_event)

    # ------------------------------------------------------------ reads

    def ensure(self, project: str) -> None:
        """Assign a project with no persona to the house persona (owner decision 12), write
        the file and record persona_assign (how auto). No-op for a known project."""
        with self._lock:
            if project in self._registry["projects"]:
                return
            house = self._registry["house"]
            self._commit(lambda r: r["projects"].__setitem__(project, house))
            name = self._registry["personas"][house]["name"]
        self._record_event(PERSONA_ASSIGN_EVENT, project=project, persona=name, how="auto", **{"from": None})

    def persona_of(self, project: str) -> Persona:
        """The project's persona as an immutable snapshot (the project must be known: `ensure`)."""
        with self._lock:
            pid = self._pid(project)
            p = self._registry["personas"][pid]
            return Persona(pid, p["name"], tuple(p["voice"].items()), p["speed"], p["words_revision"])

    def key(self, project: str) -> str:
        """The project's current "<id>:<words_revision>"."""
        return self.persona_of(project).key

    def block(self, project: str) -> dict:
        """The persona block a reply carries (proto.REPLY_PERSONA)."""
        with self._lock:
            return self._block(project)

    def overview(self, project: str) -> dict:
        """The persona op's reply: the project's block, every persona, the voice ids, the speed
        range and the pool names left."""
        with self._lock:
            reg = self._registry
            personas = [{"name": p["name"], "house": pid == reg["house"], "voice": dict(p["voice"]),
                         "speed": p["speed"],
                         "projects": sorted(pr for pr, pid2 in reg["projects"].items() if pid2 == pid)}
                        for pid, p in reg["personas"].items()]
            return {proto.REPLY_PERSONA: self._block(project), "personas": personas, "voices": list(self.voices),
                    "speed_range": [proto.PERSONA_SPEED_MIN, proto.PERSONA_SPEED_MAX],
                    "pool_left": len(self._pool_left())}

    # ------------------------------------------------------------ changes

    def new(self, project: str) -> dict:
        """A brand-new persona with a random unused pool name and the default voice and speed;
        the project moves onto it (persona_assign how new). Returns the block."""
        with self._lock:
            left = self._pool_left()
            if not left:
                raise ValueError("every pool name is in use; ask the owner for a name")
            name = secrets.choice(left)
            pid = secrets.token_hex(4)
            while pid in self._registry["personas"]:
                pid = secrets.token_hex(4)
            old = self._registry["personas"][self._pid(project)]["name"]

            def change(r: dict) -> None:
                r["personas"][pid] = self._fresh(name)
                r["projects"][project] = pid
            self._commit(change)
            block = self._block(project)
        self._record_event(PERSONA_ASSIGN_EVENT, project=project, persona=name, how="new", **{"from": old})
        return block

    def set(self, project: str, name=None, voice=None, speed=None, personality=None) -> dict:
        """Change the project's persona (docs/DESIGN_PERSONAS.md 4.9). `name` alone moves the
        project onto the persona of that name (case-insensitive), or renames its own persona
        for every project sharing it when no other persona has the name. `voice`, `speed` and
        `personality` (the whole note as a list of lines) change the persona for every project
        using it; only a change to the personality's texts raises words_revision (the same
        list again does not). Returns the block."""
        if name is not None and (voice is not None or speed is not None or personality is not None):
            raise ValueError("name cannot be combined with voice, speed or personality; one change per call")
        if name is None and voice is None and speed is None and personality is None:
            raise ValueError("nothing to change: pass name, voice, speed or personality")
        if name is not None:
            return self._set_name(project, check_name(name))
        with self._lock:
            pid = self._pid(project)
            p = self._registry["personas"][pid]
            new_voice = normalize_voice(voice, self.voices) if voice is not None else None
            new_speed = check_speed(speed) if speed is not None else None
            new_personality = self._replace_personality(p["personality"], personality, project) \
                if personality is not None else None
            fields = [f for f, v in (("voice", new_voice), ("speed", new_speed),
                                     ("personality", new_personality)) if v is not None]

            def change(r: dict) -> None:
                q = r["personas"][pid]
                if new_voice is not None:
                    q["voice"] = new_voice
                if new_speed is not None:
                    q["speed"] = new_speed
                if new_personality is not None:
                    if [e["text"] for e in new_personality] != [e["text"] for e in q["personality"]]:
                        q["words_revision"] += 1
                    q["personality"] = new_personality
            old = copy.deepcopy(p)
            self._commit(change)
            new = self._registry["personas"][pid]
            block = self._block(project)
        self._record_change(project, fields, old, new)
        return block

    def add(self, project: str, note) -> dict:
        """Append one line to the project's persona's personality. Returns the block."""
        if not isinstance(note, str) or not note.strip():
            raise ValueError("note must be a non-blank string")
        text = note.strip()
        with self._lock:
            pid = self._pid(project)
            p = self._registry["personas"][pid]
            if sum(len(e["text"]) for e in p["personality"]) + len(text) > proto.PERSONALITY_MAX_CHARS:
                raise ValueError("the personality is full; ask the owner for a persona review")

            def change(r: dict) -> None:
                q = r["personas"][pid]
                q["personality"].append({"text": text, "project": project, "added": self._now()})
                q["words_revision"] += 1
            old = copy.deepcopy(p)
            self._commit(change)
            new = self._registry["personas"][pid]
            block = self._block(project)
        self._record_change(project, ["personality"], old, new)
        return block

    # ------------------------------------------------------------ internals (lock held)

    def _set_name(self, project: str, name: str) -> dict:
        with self._lock:
            pid = self._pid(project)
            reg = self._registry
            other = next((q for q, p in reg["personas"].items()
                          if p["name"].casefold() == name.casefold() and q != pid), None)
            if other is not None:
                old_name = reg["personas"][pid]["name"]
                self._commit(lambda r: r["projects"].__setitem__(project, other))
                block = self._block(project)
                event = (PERSONA_ASSIGN_EVENT, dict(project=project, persona=reg["personas"][other]["name"],
                                                    how="moved", **{"from": old_name}))
            else:
                if reg["personas"][pid]["name"] == name:
                    raise ValueError(f"your persona is already named {name}")
                old = copy.deepcopy(reg["personas"][pid])

                def change(r: dict) -> None:
                    r["personas"][pid]["name"] = name
                    r["personas"][pid]["words_revision"] += 1
                self._commit(change)
                block = self._block(project)
                event = None
                new = self._registry["personas"][pid]
        if event is not None:
            self._record_event(event[0], **event[1])
        else:
            self._record_change(project, ["name"], old, new)
        return block

    def _replace_personality(self, current: list[dict], lines, project: str) -> list[dict]:
        """The note `lines` replaces `current`: an unchanged line keeps its project and added
        time (matched in order, once each); a new line is stamped with this project."""
        if not isinstance(lines, list) or not all(isinstance(t, str) and t.strip() for t in lines):
            raise ValueError("personality must be a list of non-blank strings")
        pool = list(current)
        now = self._now()
        result = []
        for text in (t.strip() for t in lines):
            match = next((e for e in pool if e["text"] == text), None)
            if match is not None:
                pool.remove(match)
                result.append(dict(match))
            else:
                result.append({"text": text, "project": project, "added": now})
        chars = sum(len(e["text"]) for e in result)
        if chars > proto.PERSONALITY_MAX_CHARS:
            raise ValueError(f"the personality would be {chars} characters, "
                             f"over {proto.PERSONALITY_MAX_CHARS}")
        return result

    def _pid(self, project: str) -> str:
        pid = self._registry["projects"].get(project)
        if pid is None:
            raise RuntimeError(f"project {project!r} has no persona (ensure was not called)")
        return pid

    def _block(self, project: str) -> dict:
        reg = self._registry
        pid = self._pid(project)
        p = reg["personas"][pid]
        return {"id": pid, "words_revision": p["words_revision"], "name": p["name"],
                "house": pid == reg["house"], "voice": dict(p["voice"]),
                "speed": p["speed"], "personality": [e["text"] for e in p["personality"]],
                "shared_with": sorted(pr for pr, q in reg["projects"].items() if q == pid and pr != project)}

    def _pool_left(self) -> list[str]:
        used = {p["name"].casefold() for p in self._registry["personas"].values()}
        return [n for n in proto.PERSONA_POOL if n.casefold() not in used]

    def _fresh(self, name: str) -> dict:
        return {"name": name, "voice": {self.default_voice: 1.0}, "speed": float(self.default_speed),
                "personality": [], "words_revision": 1, "created": self._now()}

    def _now(self) -> str:
        return datetime.datetime.now(self._tz).isoformat(timespec="seconds")

    def _commit(self, change: Callable[[dict], None]) -> None:
        """Apply `change` to a deep copy, validate it, then persist it (`_persist`)."""
        registry = copy.deepcopy(self._registry)
        change(registry)
        validate(registry, self.voices)
        self._persist(registry)

    def _persist(self, registry: dict) -> None:
        """Atomic write: <file>.tmp, flush + fsync, os.replace; `registry` is swapped in the
        moment os.replace succeeds, so memory matches the disk whatever happens next; then
        the directory is fsynced. A failure before os.replace raises as itself and leaves the
        registry as it was; a failed directory fsync raises PersonaDirSyncError."""
        tmp = f"{self.path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)
        self._registry = registry
        directory = os.path.dirname(os.path.abspath(self.path))
        try:
            fd = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError as e:
            raise PersonaDirSyncError(f"persona file {self.path} was replaced and the change is on disk and in "
                                      f"use, but the fsync of {directory} failed ({e}); the change may not "
                                      "survive a power loss") from e

    def _record_change(self, project: str, fields: list[str], old: dict, new: dict) -> None:
        self._record_event(PERSONA_CHANGE_EVENT, persona=new["name"], fields=fields, project=project,
                           old_name=old["name"], new_name=new["name"], old_voice=old["voice"],
                           new_voice=new["voice"], old_speed=old["speed"], new_speed=new["speed"])
