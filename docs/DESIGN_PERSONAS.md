# Design: personas (a name, a voice and a personality per project)

Status: deployed 2026-10-22 10:08 PDT in commit 6a9fcbe; tested live the same day.
Build note: `Engines.load_voices` loads each pack with `torch.load(weights_only=True,
map_location="cpu").float()` from the snapshot, as `KPipeline.load_single_voice` does for a
`.pt` path, instead of calling `load_single_voice` (which makes one hub call per voice).
Changes the wire (protocol 4).

Every statement below carries one of these labels:

- **Owner decision (date)**: the owner chose it. It is a decision, not a property of the
  system.
- **Verified (file:line)**: read in the code at HEAD 0058457 (or in the installed
  `kokoro` 0.9.4 package, path given).
- **Unverified, must read before build**: believed but not checked; the builder checks it
  first.
- **Design (2026-10-22)**: a technical choice made in this document that does not change
  what the owner hears or decides.

## 1. Problem

Today every session on the call is "Angie" and speaks in one voice. The voice and speed
are daemon settings (`SPEAK_VOICE`, `SPEAK_SPEED`) fixed at startup (Verified
`speak_telegram_daemon.py:1305-1307`, `speak_engines.py:45-52`), and the name is a line
in the client's server instructions (Verified `speak_server.py:359`). When several
projects share the call, the owner tells them apart only by the switch announcement,
which says the project name (Verified `speak_telegram_daemon.py:544-568`). The owner
wants each project to have its own persona: a name, a voice, a speed and a personality
that grows with the role, so a project is recognisable by ear and behaves consistently.

## 2. Owner decisions (2026-10-22, made on a call)

1. A persona is a name, a voice blend (one or more Kokoro voices with weights, for
   example 70% `af_heart` + 30% `af_bella`), a speed (default today's `SPEAK_SPEED`,
   1.0), and a personality note (tone, attributes, accumulated role context).
2. Each project (the client's project name) maps to exactly one persona. Several
   projects may share a persona; a shared name IS a shared persona. Setting a project to
   an existing persona's name moves it onto that persona, never copies it. A change to a
   persona's voice, speed or personality applies to every project using it.
3. The daemon stores the personas and the project-to-persona map in a local JSON file on
   a docker volume that survives rebuilds.
4. A brand-new persona gets a random unused name from a pool and the default voice.
   (The first-contact flow first decided here, in which the agent asked the owner which
   persona should manage a new project, is replaced by decision 12.)
5. The name pool: short, easy-to-hear female names, clearly distinct from each other on
   a phone line. Angie stays in the pool.
6. The persona's name replaces "Angie" for that project. The agent is told its name and
   personality. The switch announcement says the name and the project together ("Nova,
   algolearn kalshi").
7. The persona tools act on the calling session's own project only (no project
   parameter). They show the persona and the list of available Kokoro voices; change the
   name (which may move the project onto an existing persona); change the voice blend or
   the speed; add to the personality. Changes take effect on the next sentence spoken.
8. Agents add to their persona's personality note on their own as they learn their
   role. The owner can start a "persona review" at any time: the agent reads its persona
   back and the owner trims or changes it.
9. Real-time voice changes are for the owner to experiment with voice combinations. An
   agent changes the voice or speed only when the owner asks.

Owner decisions (2026-10-22, second call, answering the first draft):

10. Setting a name that no persona has renames the project's current persona, so the
    rename applies to every project sharing it. Owner's words: "if I decide I want to
    rename that persona, I'm renaming it for everyone that uses that persona."
11. On first start the daemon seeds a persona named "Angie" with the default voice and
    speed (`SPEAK_VOICE`, `SPEAK_SPEED`).
12. A new project has no first-contact question: it is assigned to Angie automatically,
    so a new session is not burdened. The owner moves a project to another persona
    whenever he wants, by asking. `new_persona` stays, for when the owner asks for a
    brand-new persona.
13. The switch announcement is spoken in the incoming persona's voice. (A separate
    switchboard persona was considered and dropped, to keep it simple.)
14. English voices only: ids starting `af_`, `am_`, `bf_` or `bm_`.
15. Speed from 0.5 to 2.0. The personality is capped at 4000 characters, after which
    the agent asks for a persona review. At most 4 voices in a blend. No delete tool:
    unused personas are kept. `new_persona` raises once the pool is used up. The pool
    is these 12 names: Angie, Nova, Ruby, Hazel, Paige, Stella, Gwen, Willow, Margo,
    Juno, Tess, Fiona.

Owner decisions (2026-10-22, third round, answering the build review):

16. The house persona (Angie, the persona new projects get) never adapts on its own.
    Agents on it do not call `add_to_personality` of their own accord; this narrows
    decision 8 to personas the owner assigned. The owner can still change the house
    persona in a persona review or on request. It is stated once, in the server
    instructions (section 6.1), and the persona block carries `house: true/false` so the
    agent knows. The daemon does not enforce it: it is etiquette, like the floor rules.
17. The voice cache: `snapshot_download` may serve the voices from the HF cache when the
    Hub is unreachable. After the fetch the daemon checks that every voice of an explicit
    constant list, the 28 English voices of `hexgrad/Kokoro-82M` (section 3.1), is
    present, and refuses to start, naming the missing ids, when any is not. The list is
    a constant in the code, not whatever the repo or the cache happens to hold.
18. British voices (`bf_`, `bm_`) keep American pronunciation: the one `KPipeline` runs
    with `lang_code="a"` for every voice. A British pronunciation path may come later.

Design (2026-10-22), with decision 16: the key the client tracks is
`"<persona id>:<words_revision>"`, where `words_revision` rises only when the persona's
name or personality changes. `speak`, `converse` and `call` are refused for a stale
`persona_seen` only when the name or the personality changed or the project moved to
another persona; a voice or speed change refuses nothing (the words the agent wrote do
not depend on how they sound). The client still only moves its key forward (4.7).

## 3. The system today (what this design builds on)

### 3.1 Kokoro and the engines

- `Engines(kokoro_model, voice, speed, whisper_model, language)` holds one voice and one
  speed for the life of the daemon (Verified `speak_engines.py:45-52`); `main` passes
  `SPEAK_VOICE` and `SPEAK_SPEED` (Verified `speak_telegram_daemon.py:1305-1307`).
- `load()` builds `KPipeline(lang_code="a", repo_id=..., device="cuda")` (Verified
  `speak_engines.py:62`), checks the espeak fallback and that the weights are fp32
  (Verified `speak_engines.py:65-70`), and warms up with `synthesize("Ready.")`
  (Verified `speak_engines.py:74`).
- `synthesize_stream` and `synthesize_segments` both call
  `self._kokoro(text, voice=self.voice, speed=self.speed, split_pattern=SENTENCE_SPLIT)`
  once for the whole text (Verified `speak_engines.py:87`, `:101`);
  `synthesize_segments` yields `(result.text_index, audio)` (Verified
  `speak_engines.py:93-107`). `synthesize` concatenates the stream and serves the
  greeting and the warm-up (Verified `speak_engines.py:109-113`). `split_sentences`
  mirrors KPipeline's split (Verified `speak_engines.py:31-38`, `SENTENCE_SPLIT` at
  `:26`).
- `KPipeline.__call__(text, voice: str | None, speed: float | Callable[[int], float],
  split_pattern, model)` (Verified, signature of the installed kokoro 0.9.4). It binds
  the voice pack once per call, `pack = self.load_voice(voice).to(model.device)`
  (Verified `kokoro/pipeline.py:361`), splits the text (Verified
  `kokoro/pipeline.py:366`) and runs inference per sentence independently (Verified
  `kokoro/pipeline.py:369`, `infer` at `:224-232`). Calling it once per sentence
  therefore gives the same segments as one call over the whole text.
- `load_voice(voice)` returns the argument unchanged when
  `isinstance(voice, torch.FloatTensor)`; a comma-separated string of voice ids is
  averaged with `torch.mean` (equal weights only), and results are cached by the string
  in `self.voices` (Verified `kokoro/pipeline.py:157-167`, `self.voices = {}` at `:105`).
  There is no weighted blend in Kokoro.
- `load_single_voice(voice)` downloads `voices/<id>.pt` with `hf_hub_download` into the
  HF cache, warns "Language mismatch" when the voice's first letter differs from the
  pipeline's `lang_code`, and loads it with `torch.load(weights_only=True)` (Verified
  `kokoro/pipeline.py:136-149`).
- Speed divides the predicted durations, `torch.sigmoid(duration).sum(axis=-1)/speed`,
  with no range check (Verified `kokoro/model.py:108`).
- A voice pack is a CPU float32 tensor of shape `[510, 1, 256]`, and a scaled copy still
  passes `isinstance(t, torch.FloatTensor)` (Verified on the Mac venv, 2026-10-22). A
  CUDA tensor does not pass that check, so a blended pack must be handed to Kokoro on
  the CPU. That the same holds under the container's torch cu126 build is Unverified,
  must read before build.
- The `hexgrad/Kokoro-82M` repo holds 54 voice packs under `voices/`, 28 of them
  English: `af_alloy, af_aoede, af_bella, af_heart, af_jessica, af_kore, af_nicole,
  af_nova, af_river, af_sarah, af_sky, am_adam, am_echo, am_eric, am_fenrir, am_liam,
  am_michael, am_onyx, am_puck, am_santa, bf_alice, bf_emma, bf_isabella, bf_lily,
  bm_daniel, bm_fable, bm_george, bm_lewis` (Verified, `huggingface_hub.list_repo_files`
  on the Hub, 2026-10-22). These 28 are `ENGLISH_VOICES` in `speak_engines.py` (Owner
  decision 17). The Mac's HF cache held only 21 of them that day (Verified, directory
  listing 2026-10-22).
- When the Hub cannot be reached, `snapshot_download` returns the cached snapshot of the
  revision if its folder exists, after checking it against the cached tree listing
  (Verified, `huggingface_hub/_snapshot_download.py:292-331`, huggingface-hub 1.29.0 in
  the Mac venv and `uv.lock`). The daemon does not rely on that check: it checks the
  files of `ENGLISH_VOICES` itself (4.5).

### 3.2 The daemon

- `PROTOCOL_VERSION = 3` (Verified `speak_protocol.py:54`); identity fields
  `FIELD_SESSION_ID`, `FIELD_PROJECT` (Verified `speak_protocol.py:62`);
  `REPLY_SWITCHED_FROM` (Verified `speak_protocol.py:63`); ops, `QUICK_OPS`,
  `VERSIONLESS_OPS`, `LOCKED_OPS` (Verified `speak_protocol.py:66-70`); error kinds
  (Verified `speak_protocol.py:75`); `speakable_name` (Verified
  `speak_protocol.py:120-124`).
- `_Handler.handle` checks the protocol version and the identity before the lock
  (Verified `speak_telegram_daemon.py:877-885`), serves quick ops without the lock
  (Verified `:886-892`) and queues tool ops in `lock.acquire` (Verified `:897`).
  `SpeakDaemon.quick` serves `ping`, `status`, `debug`, `set_debug`; quick ops carry no
  identity today (Verified `speak_telegram_daemon.py:1191-1209`). `run_locked` builds
  the reply with `expect` and `switched_from` (Verified `:1211-1242`).
- `Speaker(session_id, project)` (Verified `speak_telegram_daemon.py:389-395`), validated
  by `_speaker` (Verified `:817-831`).
- `_take_turn` plays `switch_pcm()`, synthesizes `speakable_name(project)` (plus
  `SWITCH_LISTEN_SUFFIX = ", still listening"` for a bare listen, Verified `:91`) with
  `self.engines.synthesize(spoken)`, records `session_switch` (tool, from_project,
  to_project, spoken, announce_s), then calls `take_floor` (Verified
  `speak_telegram_daemon.py:544-568`).
- `_speak` streams `self.engines.synthesize_segments(text)` (Verified
  `speak_telegram_daemon.py:602`) through the `play_watched` hold loop (Verified
  `:589-624`); `_Segments` caches synthesized segments so that "continue" replays the
  cut sentence without re-synthesizing (Verified `:426-464`).
- `call` checks quiet hours, then already connected, then `plain(greeting)`, then
  synthesizes it whole with `self.engines.synthesize(text)` (Verified
  `speak_telegram_daemon.py:750`) and checks the 6 s floor (Verified `:752`); an
  answered call sets the turn and takes the floor (Verified `:767-768`).
- The floor (`AudioLock.take_floor`, `yield_floor`; Verified
  `speak_telegram_daemon.py:322`, `:340`) and `FLOOR_IDLE_SECONDS = 45.0` (Verified
  `speak_protocol.py:112`).
- `transcribe_pcm` serves the HTTP whisper API with no session and no floor (Verified
  `speak_telegram_daemon.py:1168-1189`).
- `link.record_event` records an event into `CallEventLog`, a lock-protected ring buffer
  that also logs at INFO (Verified `speak_telegram.py:452-454`, `:261-282`).
- `load_settings` requires every `REQUIRED_SPEAK_ENV` variable and checks `SPEAK_SPEED`
  is a positive float (Verified `speak_telegram_daemon.py:78-79`, `:166-182`).

### 3.3 The client

- `init_identity(cwd)` sets the session id (`secrets.token_hex(8)`) and the project
  (basename of cwd) (Verified `speak_server.py:122-132`).
- `_request` adds identity only to `LOCKED_OPS`, applies the reply's `expect`, and puts
  the switch line in front of an error (Verified `speak_server.py:153-201`).
- Budgets: speak `60 + len/5 + 300` (Verified `speak_server.py:206-209`), listen
  (Verified `:212`), the announcement `10 + len(project)/5` (Verified `:216-217`,
  `ANNOUNCE_BUDGET_SECONDS` at `:63`).
- Result builders: `_switch_line`, `_held_prefix`, `_speak_result`, `_converse_result`,
  `_listen_result` (Verified `speak_server.py:227-266`).
- `_INSTRUCTIONS` (Verified `speak_server.py:356-416`) names the agent "Angie" (Verified
  `:359`). `mcp = MCPServer("algolearn-speak", instructions=_INSTRUCTIONS)` runs at
  import (Verified `speak_server.py:418`), so the instructions are fixed text, the same
  for every project, and cannot carry a per-project persona. The `call` docstring's
  example greeting is "Hi Frank, it's Angie..." (Verified `speak_server.py:561`).

### 3.4 Deployment

- `docker-compose.yml` sets `SPEAK_VOICE: af_heart` and `SPEAK_SPEED: "1.0"` and mounts
  one named volume, `speak_models:/root/.cache/huggingface` (Verified
  `docker-compose.yml`).
- The Dockerfile copies an explicit file list (`speak_server.py speak_protocol.py
  speak_engines.py speak_telegram.py speak_telegram_daemon.py speak_tone.py README.md
  LICENSE`) (Verified `Dockerfile`), so a new module must be added to it.

## 4. Mechanism

### 4.1 Nouns

The design reuses the daemon's nouns: settings in `load_settings`, a field on the tool
ops, a reply field, quick ops for bookkeeping, `SpeakService` for audio, `record_event`
for diagnostics. It adds one noun:

- **`PersonaStore`**, in a new stdlib-only module `speak_personas.py`. Design
  (2026-10-22). Justification: the registry, its validation and its atomic file writes
  are pure bookkeeping with no CUDA, Telegram or audio; keeping them out of
  `speak_telegram_daemon.py` (about 1300 lines) lets them be tested directly, the same
  way `speak_protocol.py` stands alone.

### 4.2 Storage

- File: the new required setting `SPEAK_PERSONAS_FILE`, set in compose to
  `/var/lib/speak/personas.json`. Design (2026-10-22).
- Volume: a new named volume **`speak_state`**, mounted at `/var/lib/speak`, declared at
  the top level of `docker-compose.yml` next to `speak_models`. It survives `up --build
  -d` like `speak_models` does (Owner decision 3 names a volume; the name and the mount
  are Design (2026-10-22)). It is kept apart from `speak_models` so a model-cache wipe
  never takes the personas with it.
- At startup, before the engines load: the directory must exist (the volume mount makes
  it) or startup stops. After the engines load (the voice list is needed to validate),
  the store loads the file. A missing file is a first start: the store seeds the
  "Angie" persona with `{SPEAK_VOICE: 1.0}` and `SPEAK_SPEED`, marks it as the house
  persona, writes the file and logs `personas: no file at <path>, seeded Angie` (Owner
  decision 11). A file that does not parse or does not validate stops the daemon with
  the reason; it is never replaced or ignored.
- Every change is written atomically: serialize the new registry, write it to
  `<file>.tmp` in the same directory, `flush` and `os.fsync`, `os.replace` onto the file,
  then `fsync` the directory. A crash leaves either the old file or the new one.
- The store applies a change to a deep copy of the registry, validates the copy, writes
  it, and swaps it in the moment `os.replace` succeeds, before the directory `fsync`, so
  the registry in memory always matches the file on disk. A failure before `os.replace`
  raises to the tool and leaves the in-memory registry as it was on disk. A failed
  directory `fsync` raises its own error (`PersonaDirSyncError`) saying the change is on
  disk and in use but may not survive a power loss. Design (2026-10-22).
- Concurrency: one `threading.Lock` inside the store guards read-modify-write. Reads
  for audio take the lock only to copy out an immutable snapshot of one persona. The
  store lock is never held while waiting on the audio lock or while synthesizing, so the
  two locks cannot deadlock. The daemon is the only writer (one daemon per login,
  Verified by the single-instance guard described in CLAUDE.md and the bind on 8765).
- A write holds the store lock across its `fsync` calls (the temporary file's and the
  directory's), and `persona_of`, which the audio path calls for every sentence, waits on
  that lock, so a sentence can wait for a write in progress. This is accepted: writes
  are rare (an owner-requested change, or a project's first request), while the reads
  are per sentence and each holds the lock only to copy out a snapshot. Holding the lock
  across the write keeps one writer at a time and the registry in memory equal to the
  file. Design (2026-10-22).

### 4.3 Schema (version 1)

```json
{
  "schema": 1,
  "personas": {
    "3f9a1c02": {
      "name": "Nova",
      "voice": {"af_heart": 0.7, "af_bella": 0.3},
      "speed": 1.0,
      "personality": [
        {"text": "Calm, terse, numbers first.", "project": "algolearn-kalshi",
         "added": "2026-10-22T14:03:11-07:00"}
      ],
      "words_revision": 3,
      "created": "2026-10-22T13:58:40-07:00"
    }
  },
  "projects": {"algolearn-kalshi": "3f9a1c02", "algolearn-speak": "a07c55e1"},
  "house": "a07c55e1"
}
```

Design (2026-10-22):

- The persona id is `secrets.token_hex(4)`, stable for the persona's life. Projects
  point at ids, so moving a project or renaming never rewrites other projects.
- `name`: one word of 2 to 12 letters, stored as typed, unique across personas compared
  case-insensitively. A rename may use any valid name, not only pool names.
- `voice`: one or more English voice ids from the daemon's voice list, each with a
  weight above 0, at most 4 voices (Owner decisions 14, 15). Stored normalized so the
  weights sum to 1 (rounded to 4 decimals; the largest weight absorbs the rounding).
- `speed`: a number from 0.5 to 2.0 (Owner decision 15).
- `personality`: an ordered list of entries `{text, project, added}`. The project that
  added the entry is kept so a persona review can say where each line came from. The
  total text is capped at 4000 characters (Owner decision 15).
- `house`: the id of the persona new projects are assigned to: the persona seeded as
  Angie. It follows the persona by id, so a rename of Angie keeps it the house persona
  under its new name. Design (2026-10-22).
- `words_revision`: an integer that rises by one when the persona's name or personality
  changes (a rename, `add_to_personality`, `set_persona(personality=...)`). A voice or
  speed change leaves it as it is. With the id it is the key the client tracks (section
  4.7). Design (2026-10-22).
- `projects`: project name (exactly the string the client sends as `project`) to
  persona id.
- Validation, on load and on every change: `schema == 1`; every voice id in the daemon's
  voice list; weights positive and normalized; at most 4 voices; speed in range; names
  valid and unique; every project pointer and `house` name an existing persona;
  personality within the cap. Any
  failure on load stops the daemon; on a change it raises `ValueError` to the tool and
  nothing is written.
- A persona that no project points at is kept; there is no delete tool (Owner decision
  15). It still appears in `persona()`'s list and can be chosen again by name.

### 4.4 What a project's persona is

- Every project that has spoken to the daemon has exactly one persona:
  `projects[project]`. It speaks in that persona's blend and speed and is announced as
  "<Name>, <speakable project>".
- **Automatic assignment** (Owner decision 12): the first identity-carrying request
  from a project with no entry (a tool op or a persona op) assigns it to the house
  persona, writes the file and records `persona_assign` with `how: auto`. This happens
  before the lock, in the same place the identity is validated, so there is no state in
  which a project has no persona while audio plays. Design (2026-10-22) for the
  placement.
- **Default values**: `SPEAK_VOICE` and `SPEAK_SPEED` keep their names; they are what
  the seeded Angie and every new persona start with. A persona stores a copy at
  creation, so changing `SPEAK_VOICE` later does not change existing personas. Design
  (2026-10-22). `SPEAK_VOICE` must be an English voice or the daemon stops (Owner
  decision 14).

### 4.5 Voices, blends and speed in the engines

Load (`Engines.load`), Design (2026-10-22):

1. Check `SPEAK_VOICE` is one of `ENGLISH_VOICES` (Owner decision 14), or stop before
   any fetch.
2. Fetch exactly the `ENGLISH_VOICES` packs (Owner decision 17):
   `huggingface_hub.snapshot_download(repo_id, allow_patterns=["voices/<id>.pt", ...])`,
   one pattern per id, into the `speak_models` cache. With the Hub unreachable it may
   serve the cached snapshot (3.1); a fetch that raises stops the daemon.
3. Check that every `voices/<id>.pt` of `ENGLISH_VOICES` is in the snapshot, or stop
   with "<n> of the 28 English voice packs of <repo> are missing from <dir> (the Hub
   unreachable and the HF cache incomplete?): <ids>" (Owner decision 17). Then load each
   pack once with `torch.load(weights_only=True, map_location="cpu").float()` (see the
   build note at the top), and keep them in `Engines.packs: dict[str, Tensor]`. No
   partial list.
4. Every pack is synthesized by the one `KPipeline(lang_code="a")`, so the British voices
   speak with American pronunciation (Owner decision 18).
5. Check every voice id in the loaded persona file is in the list, or stop (the store
   loads after the engines for this reason; section 4.2's directory check runs first).

A blend is resolved to a pack with `Engines.voice_pack(blend)`:

- `pack = sum(weight * packs[id])`, computed on the CPU, float32, and checked to be a
  `torch.FloatTensor` before use (it is handed to `KPipeline.__call__` as `voice=`,
  which returns it unchanged; Verified `kokoro/pipeline.py:157-167`).
- Cost: a weighted sum of up to four `[510, 1, 256]` tensors, about 130 k floats each,
  a fraction of a millisecond on the CPU. Each call then copies the pack to the GPU
  (Verified `kokoro/pipeline.py:361`), about 0.5 MB, the same copy the single-voice
  path already does today.
- Cache: an LRU of 64 packs keyed by the canonical blend (sorted `(id, weight)` pairs).
  A single-voice blend is the loaded pack itself. Design (2026-10-22).

Synthesis takes the voice per call instead of from `Engines`:

- `Engines.__init__` drops `voice` and `speed`; they become arguments.
- `synthesize(text, voice: Voice)` where `Voice` is a frozen `(pack, speed)`: the
  greeting, the switch announcement and the warm-up (warm-up uses the default voice).
- `synthesize_segments(text, voice_of: Callable[[], Voice])`: splits with
  `split_sentences` and calls Kokoro **once per sentence**, asking `voice_of()` before
  each one. Because Kokoro already infers each sentence independently (Verified
  `kokoro/pipeline.py:369-383`), the audio is the same as today's single call, and a
  persona change lands on the next sentence synthesized. The yielded index stays the
  sentence index `_Segments` uses. `synthesize_stream` goes away if nothing else calls
  it (Unverified, must read before build: its callers).

How each op picks its voice (all Design (2026-10-22), satisfying Owner decision 7's
"next sentence"):

| Audio | Voice | Resolved when |
|---|---|---|
| `call` greeting | the calling project's persona | once, before synthesis |
| `speak`/`converse` speech | the speaking project's persona | before each sentence is synthesized |
| Switch announcement | the incoming project's persona (Owner decision 13) | once, before synthesis |
| Cues (beeps, chime) | unchanged, no voice | n/a |
| HTTP whisper API | STT only, no voice | n/a |

Two consequences of streaming, stated for the owner's ear: a change made while a
sentence is already synthesized and queued is heard from the sentence after it; a
sentence replayed after "continue" comes from the `_Segments` cache and keeps the voice
it was made in (Verified `speak_telegram_daemon.py:426-464` replays cached audio).

### 4.6 The switch announcement

- "<Name>, <speakable project>" ("Nova, algolearn kalshi"), Owner decision 6. A bare
  listen keeps its suffix: "Nova, algolearn kalshi, still listening". Since every
  project has a persona (4.4), there is no project-name-only form any more.
- Spoken in the incoming persona's voice (Owner decision 13).
- The `session_switch` event gains `to_persona` (the name).
- The client's announcement budget grows by the longest valid name: `10 + (12 + 2 +
  len(project))/5` seconds. Design (2026-10-22): a fixed bound, so the budget never
  depends on whether the client has seen its persona yet.

### 4.7 How the agent learns its persona before it speaks

The server instructions are fixed at import (Verified `speak_server.py:418`), so the
persona cannot live there. It travels on the wire instead, and the daemon refuses to
voice words written for a persona the agent has not yet seen.

- The client keeps `_persona_seen`, a string, next to `_call_generation`: `null` at
  start, then `"<persona id>:<words_revision>"` as the daemon reports. The key changes
  when the project moves to another persona (the id) or the persona's name or
  personality changes (the words revision); a voice or speed change does not change it.
  Design (2026-10-22).
- Every tool op (`speak`, `listen`, `converse`, `call`, `hang_up`) carries
  `persona_seen`.
- Every reply to a tool op carries `persona`, the project's persona block, whenever the
  project's current key differs from `persona_seen`; a persona op's reply always
  carries it. Absent means the key is unchanged. A voice or speed change made by
  another session therefore reaches a session only through `persona()` or its own
  persona op replies, never through a tool reply. The block: `{"id", "words_revision",
  "name", "house", "voice", "speed", "personality": [text...], "shared_with":
  [projects]}`. `house` is true when the persona is the house persona (Owner decision
  16).
- **Refusal before audio.** Before the lock, the daemon compares `persona_seen` with
  the project's current key. A voice or speed change never refuses anything (Design
  (2026-10-22), section 2). For `speak`, `converse` and `call`, a mismatch is
  refused with kind `value` and the block in the reply: nothing is played and nothing
  is dialed. The client raises `ValueError` whose message is the persona block in text
  plus "Nothing was spoken (or dialed). Rewrite your words as <Name> and send again."
  This is how a fresh session learns its persona before its first `call` or `speak`
  (for a new project, the automatic assignment of 4.4 runs first, so the block names
  Angie or whatever the house persona is called): its first attempt, with
  `persona_seen = null`, is refused with the block, and its second attempt carries the
  right name. `listen` and `hang_up` are never refused for
  this; they carry no words.
- The cheap path: the instructions tell the agent to call `persona()` first thing,
  before its first `call` or `converse`, which also sets `_persona_seen`; the refusal is
  the guarantee, `persona()` the habit.
- **Change while queued.** A tool op that passed the check and then waits in the lock
  while its persona changes is not refused (it may already hold words the owner is
  waiting for); its voice follows the change sentence by sentence (4.5), and its reply
  carries the new block.
- The client puts the block in the tool result as one line,
  `[persona: you are <Name>; voice ...; speed ...; personality: ...]` (on the house
  persona, "you are <Name>, the house persona; ..."), after the switch line and before
  the held line.
- **Forward only.** On an ordinary reply the client replaces `_persona_seen` with a
  block's key only when the id differs or its `words_revision` is higher: it never moves
  back within one persona. Across a move to another persona, a late reply can set the key
  to the old persona's; that costs at most one extra refusal, which corrects it. A
  refusal (an error reply of kind value that carries a block) states the daemon's current
  key, so the client applies it unconditionally, with no forward-only check. Design
  (2026-10-22).

### 4.8 New projects and new personas

- A new project is assigned to the house persona automatically (4.4, Owner decision
  12). The agent asks nothing; it introduces itself by its persona's name like any
  other. The owner moves the project later by asking, which the agent carries out with
  `set_persona(name=...)` or `new_persona()`.
- `new_persona()` is for when the owner asks for a brand-new persona. It picks a random
  unused name from the pool with `secrets.choice` (a pool name is unused when no
  persona has it, compared case-insensitively) and copies the default voice and speed.
  When every pool name is in use it raises
  `ValueError("every pool name is in use; ask the owner for a name")` (Owner decision
  15); there is no generated name.
- Pool (Owner decisions 5 and 15): Angie, Nova, Ruby, Hazel, Paige, Stella, Gwen,
  Willow, Margo, Juno, Tess, Fiona.

### 4.9 The persona ops (daemon) and tools (client)

New ops in `speak_protocol.py`, served by `SpeakDaemon.quick` without the audio lock
(they never touch audio, so they never wait behind a call). Unlike today's quick ops,
they are versioned (refused on a protocol mismatch) and carry `session_id` and
`project`, validated by `_speaker`. Design (2026-10-22).

| Op | Tool | Effect |
|---|---|---|
| `persona` | `persona()` | Returns the project's block, every persona (name, voice, speed, projects), the voice ids, the speed range and the number of pool names left. Changes nothing. |
| `persona_new` | `new_persona()` | Creates a persona from the pool with the defaults, points this project at it (moving it off any persona it had), records `persona_assign`. |
| `persona_set` | `set_persona(name=None, voice=None, speed=None, personality=None)` | See below. |
| `persona_add` | `add_to_personality(note)` | Appends `{note, project, now}` to the persona's personality; raises past the cap with "the personality is full; ask the owner for a persona review". |

`set_persona` rules:

- `name` alone, matching another existing persona (case-insensitive): points this
  project at that persona (Owner decision 2: a move, not a copy). Records
  `persona_assign` with `how: moved`.
- `name` alone, matching no persona (or only this project's own persona, differing in
  case): renames this project's persona, for every project sharing it (Owner decision
  10). The name must be valid (4.3). Records `persona_change`.
- `name` cannot be combined with the other fields (one meaning per call).
- `voice` is a mapping of voice id to weight (any positive weights; the store
  normalizes). `speed` is a number in range. Both are for the owner's experiments (Owner
  decision 9).
- `personality` is the whole note as a list of strings and replaces it; it exists for
  the persona review (Owner decision 8). Each kept line keeps its original `project` and
  `added` when its text is unchanged; new lines are stamped with this project.
- A change of name or personality raises `words_revision`; a voice or speed change does
  not (Design (2026-10-22)). Every change writes the file, records `persona_change` (`persona`,
  `fields`, `project`, and the old and new values for name, voice and speed), and
  returns the new block.

### 4.10 Interaction with the floor, the switch and the whisper API

- **The floor**: unchanged. Persona ops are quick ops, so they never queue behind the
  floor and never take or yield it. A persona refusal happens before the lock, so it
  neither takes nor extends the floor.
- **The switch of session**: what counts as a switch is unchanged (it keys on the
  session, Verified `speak_telegram_daemon.py:544-568`). Only the spoken text and its
  voice change (4.6). Two sessions of one project share a persona and are still two
  sessions: the switch is announced with the same name.
- **Moving a project onto a persona another session is speaking with**: the two share
  the voice from the next sentence; each still gets its own switch announcement.
- **The HTTP whisper API**: unaffected (STT only, no session, Verified
  `speak_telegram_daemon.py:1168-1189`).
- **status**: unchanged.

### 4.11 Events (`call_debug`)

- `session_switch` gains `to_persona`.
- `persona_assign`: `project`, `persona` (name), `how` (`auto`, `new` or `moved`),
  `from` (previous persona name, or null for `auto`).
- `persona_change`: as in 4.9.
- `persona_refused`: `tool`, `project`, `seen`, `current`, recorded when a stale
  `persona_seen` is refused.

All via `link.record_event` (Verified `speak_telegram.py:452-454`), which is
thread-safe (Verified `speak_telegram.py:261-282`).

## 5. Wire and protocol

- `PROTOCOL_VERSION` 3 to **4**: tool ops gain a required `persona_seen` (null or a
  string; a missing field is refused like a missing protocol version), replies gain
  `persona`, and four ops are new. A protocol-3 client would read none of it and speak
  as "Angie" in a persona's voice, so the bump is required.
- New constants in `speak_protocol.py`: `FIELD_PERSONA_SEEN`, `REPLY_PERSONA`,
  `OP_PERSONA`, `OP_PERSONA_NEW`, `OP_PERSONA_SET`, `OP_PERSONA_ADD`, a new
  `PERSONA_OPS` set (versioned, identity-carrying, lock-free), `PERSONA_SPEED_MIN = 0.5`,
  `PERSONA_SPEED_MAX = 2.0`, `PERSONA_POOL`, `PERSONALITY_MAX_CHARS = 4000`,
  `PERSONA_MAX_VOICES = 4`, `PERSONA_VOICE_PREFIXES = ("af_", "am_", "bf_", "bm_")`,
  `PERSONA_SEED_NAME = "Angie"`.
- Deploy, in order:
  1. On the owner's word: `docker --context patrick compose up --build -d` from this
     repo. The same command creates the `speak_state` volume and passes
     `SPEAK_PERSONAS_FILE`. The first start finds no file and seeds Angie; from then on
     every project speaks as Angie, in today's voice, until the owner moves it.
  2. On every computer, in every session: `git pull` this repo, then `/mcp` reconnect
     (or a new session).
  3. Until a session does step 2 its tool ops fail with the protocol-mismatch error;
     `status`, `call_debug` and `set_call_debug` still work.
- Reversal: redeploying the protocol-3 daemon leaves the `speak_state` volume and file
  in place, unused. Reversal is an owner action.

## 6. Client text

### 6.1 Server instructions

Replace the "Angie" bullet (Verified `speak_server.py:359`) with the block below, and
change the switch bullet (Verified `speak_server.py:409-415`) to say the daemon speaks
"your persona's name and your project's name". Everything else in `_INSTRUCTIONS` stays.

```
- You have a persona: a name, a voice and a personality, shared by every project that
  uses it. Call persona() before your first call or converse. On calls and in anything
  spoken you are that name, never Claude, and you keep to its personality.
- A result line "[persona: ...]" means your persona is new or changed: from now on you
  are what it says. A ValueError carrying a persona means nothing was spoken or dialed;
  rewrite your words as that persona and send them again.
- When the owner asks to move this project to another persona, call
  set_persona(name=...); to rename your persona (for every project sharing it), the
  same call with the new name; for a brand-new persona, new_persona().
- As you learn your role, add short lines to your personality with
  add_to_personality(): tone, what the owner wants from you, standing context. Keep each
  line one sentence. This is only for a persona the owner assigned: the house persona
  (the persona line says "the house persona"; in persona() its "house" is true) never
  grows on its own, so add to it only when the owner asks. He can still change it in a
  persona review or on request.
- "Persona review": read your whole persona back, one item at a time, and apply the
  owner's trims and changes with set_persona(personality=[...]).
- Change your voice or speed only when the owner asks. The persona tools act on your
  own project only.
```

### 6.2 Tool docstrings (one line each)

- `persona()`: "Your project's persona (name, voice blend, speed, personality, the
  projects sharing it), every persona, and the voice ids and speed range you may use."
- `new_persona()`: "Give your project a brand-new persona with an unused name from the
  pool and the default voice; returns it."
- `set_persona(name, voice, speed, personality)`: "Change your persona: name alone moves
  your project onto the persona of that name, or renames yours (for every project
  sharing it) when no persona has it; voice {id: weight} and speed change how it
  sounds for every project using it from the next sentence; personality [lines]
  replaces the note (persona review)."
- `add_to_personality(note)`: "Add one line to your persona's personality."
- `call` docstring: the example greeting "Hi Frank, it's Angie..." (Verified
  `speak_server.py:561`) becomes "Hi Frank, it's <your persona's name>...".

## 7. Code changes

| File | Change |
|---|---|
| `speak_personas.py` (new) | `PersonaStore`: load, validate, atomic write, the four operations, the block, the pool. Stdlib only. |
| `speak_protocol.py` | Version 4, the constants of section 5. |
| `speak_engines.py` | Voice list and packs at load; `voice_pack` with the LRU; `synthesize(text, voice)`; per-sentence `synthesize_segments(text, voice_of)`; no voice or speed in `__init__`. |
| `speak_telegram_daemon.py` | `SPEAK_PERSONAS_FILE` in `REQUIRED_SPEAK_ENV`; store built in `main` after the engines; `persona_seen` check before the lock; persona ops in `quick`; voices passed to greeting, announcement and speech; announcement text; `persona` in `run_locked` replies; events. |
| `speak_server.py` | `_persona_seen`; `persona_seen` on tool ops; the four tools; the `[persona: ...]` line; the refusal message; budgets; instructions and docstrings of section 6. |
| `docker-compose.yml` | `SPEAK_PERSONAS_FILE: /var/lib/speak/personas.json`; `speak_state:/var/lib/speak`; top-level `speak_state:`. |
| `Dockerfile` | Add `speak_personas.py` to the COPY list. |
| `scripts/drive_daemon.py` | `persona`, `persona_new`, `persona_set`, `persona_add` subcommands. |

## 8. Tests

- `tests/test_personas.py` (new, no CUDA): a missing file seeds Angie with the default
  voice and speed and marks it `house`; a corrupt or invalid file raises; round trip;
  atomic write leaves the old file when the write fails (patched `os.replace`);
  case-insensitive unique names; an unknown project is assigned to the house persona;
  the house persona keeps its role after a rename; moving a project onto another
  persona by name; a new name renames the persona for every project sharing it; a
  change to a shared persona seen by every project; weight normalization; more than 4
  voices refused; a non-English voice id refused; speed range; personality cap; pool
  exhaustion raises; dangling project or `house` pointer refused on load; a project or
  `house` pointer that is not a string refused on load; a weight that rounds to 0
  refused, naming the voice; a failed directory fsync raises `PersonaDirSyncError`
  with the registry in memory matching the new file; concurrent writes and
  `persona_of` reads always see a consistent persona; `words_revision` rises on a
  rename or a personality change and not on a voice or speed change, so the key
  changes on words and on a move only; the block's `house` flag is true for the house
  persona, follows it through a rename, and is false for a new persona.
- Engines voice list (fake hub): `ENGLISH_VOICES` is 28 unique ids with the four English
  prefixes; exactly those are requested and loaded, other languages ignored; a
  snapshot missing any of them (an incomplete cache) stops the load naming the
  missing ids, with no packs kept; a non-English `SPEAK_VOICE` stops the load before
  any fetch.
- Blend math (`tests/test_streamed_speech.py` or a new file, fake pipeline): the
  weighted sum; single voice returns the loaded pack; the LRU key ignores input order;
  the result is a CPU float32 tensor; `synthesize_segments` asks `voice_of()` once per
  sentence and keeps sentence indices.
- Service (`tests/test_hold.py`, `tests/test_session_switch.py`): `FakeEngines` gains
  the voice arguments (Verified `tests/test_hold.py:357-371`); the greeting uses the
  caller's persona; the announcement text ("<Name>, <project>") and its voice (the
  incoming persona's), with and without the listen suffix; a change between sentences changes the
  next sentence's voice; a "continue" replay keeps the cached voice.
- Wire (`tests/test_daemon_wire.py`): protocol 4 required; missing `persona_seen`
  refused; a new project's first request assigns it to Angie and its refusal block
  names Angie; stale `persona_seen` refuses `speak`, `converse` and `call` with the block
  and plays nothing (the fake link records no audio, no dial); a voice or speed change by
  another session refuses none of them and attaches no block; `listen` and `hang_up`
  not refused; persona ops without the lock while another op holds it; persona ops
  refused on a protocol mismatch and without identity; the reply carries `persona` only
  when it changed.
- Client: the `[persona: ...]` line's position (after the switch line, before the held
  line); `_persona_seen` updated from replies and errors; the refusal message.

## 9. CLAUDE.md on build (described, not edited here)

- **Intro / Tools table**: "Angie" becomes "the project's persona"; four rows for the
  persona tools; the `call` row's greeting rule names the persona.
- **New section "Personas"**: the owner decisions of section 2, the storage file and
  volume, the seeded Angie and automatic assignment of new projects, moving and
  renaming, English voices only, the pool, the refusal and the
  `[persona: ...]` line, and "takes effect on the next sentence" with the two streaming
  consequences; the house persona never adapts on its own (`house` in the block); the
  words revision and that voice or speed changes refuse nothing; `ENGLISH_VOICES` and
  the startup check; American pronunciation for the British voices.
- **Switch of session**: the announcement is "<Name>, <project>" in the incoming
  persona's voice; `session_switch` gains `to_persona`; the budget.
- **Protocol**: version 4, `persona_seen`, `persona`, the persona ops as lock-free,
  versioned, identity-carrying ops.
- **Files**: `speak_personas.py`; `speak_engines.py` now takes the voice per call.
- **Daemon configuration**: `SPEAK_PERSONAS_FILE`; `SPEAK_VOICE` and `SPEAK_SPEED`
  described as the voice and speed the seeded Angie and every new persona start with
  (English voices only).
- **Deployment on patrick**: the `speak_state` volume and that the personas live there.
- **Development**: `tests/test_personas.py`.
