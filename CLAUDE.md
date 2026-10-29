# algolearn-speak

An ear and voice for Claude Code over a Telegram voice call. Claude drives an
MCP server (stdio) directly, so a spoken back-and-forth happens inside one
turn: Claude calls `call` (or joins a connected call), then `converse` for
each turn, reads the transcript as a tool result, and continues.

**Two parts (docs/DESIGN_REMOTE_DAEMON.md):**
- **The speak daemon**, one per LAN, in the `speak-telegram` container on
  patrick (192.168.86.31), on GPU0. It owns the Telegram login and the call,
  Kokoro (TTS) and faster-whisper (STT) on CUDA at full precision (fp32),
  Silero VAD on the CPU, the one audio lock, and all of the tool logic. It
  listens on TCP `0.0.0.0:8765`.
- **The thin MCP client** (`speak_server.py`), one per Claude Code session on
  any LAN computer. It holds no models and no audio; each tool is one request
  to the daemon named by `SPEAK_DAEMON`. It imports only `mcp`, `anyio`, the
  standard library and `speak_protocol`.
- **The HTTP whisper API** (owner decision 2026-10-16), a second listener in
  the daemon on `0.0.0.0:9000`, replacing the retired `whisper-service` with
  its API (see "HTTP whisper API" below).

**Telegram only (owner decision 2026-10-15).** No audio device is opened
anywhere. `speak`, `listen` and `converse` work only while a Telegram call is
connected and raise otherwise; there is no fallback. Only text crosses the
LAN; the audio travels only over the Telegram call.

**Personas (owner decisions 2026-10-22, docs/DESIGN_PERSONAS.md).** Every
project speaks as its persona: a name, a voice blend, a speed and a
personality, kept by the daemon (see "Personas" below). "Angie" is the
persona the daemon seeds and the one every new project joins; it is no
longer the only name. Deployed in commit 6a9fcbe on 2026-10-22.

**Open to the LAN by design (owner decision).** Port 8765 has no
authentication and no TLS: any LAN computer with the thin client can call the
owner and hold a spoken conversation.

## Tools

| Tool | What it does |
|---|---|
| `speak(text)` | The daemon synthesizes `text` with Kokoro-82M one segment (a sentence) at a time and plays each into the connected call as soon as it is made, synthesizing the next while earlier ones play (owner decision 2026-10-16). Returns once all of it has played, with the spoken duration. While it plays the owner can hold it (see "Hold on / continue" below): `[held Ns, then resumed]` on its own line comes first when it was held; an expired hold returns the expired form instead. A `[took over the call from "<project>"; ...]` line comes before all of that when this session took the call over (see "Switch of session" below), then a `[persona: ...]` line when the reply carries the project's persona (see "Personas" below). Speaks in the project's persona's voice, read again before each sentence. When it finishes it gives up the floor (see "The floor" below), except when the call ended under it. Agents use it only for the sign-off, once the owner has said they may go do the work (owner decision 2026-10-26), and do nothing more on the call after it. When it gives up the floor and no other tool op is queued, the daemon ends the call at once (owner decision 2026-10-28, see "The floor" below): the result then has `[call ended: nobody else was waiting]` on its own line after `spoke for Ns` (after the error's message on a failure), before the audio note, and the session's call expectation is cleared. |
| `listen(max_seconds=120, silence_seconds=2.0, start_timeout_seconds=45)` | High beep (the ear is open), then records the owner's call audio until Silero VAD sees `silence_seconds` of quiet after speech (or `max_seconds` of speech), then Whisper large-v3-turbo transcribes. Once text is in hand, a rising two-note chime tells the owner their words were captured (owner decision 2026-10-16). No ear-closed beep and no spoken acknowledgement. Raises `TimeoutError` if nobody speaks within `start_timeout_seconds`. After a switch of session the switch cue and "<persona name>, <project name>, still listening" come before the ear-open beep, and the result (or the `TimeoutError` message) starts with the `[took over the call from ...]` line. Keeps the floor, a no-speech `TimeoutError` included. |
| `converse(text, ...)` | `speak` then `listen` under one lock hold. Returns the user's words, after a `[held Ns, then resumed]` line when the speech was held. After an expired hold it does not listen and returns the expired form. The `[took over the call from ...]` line comes first after a switch of session, then the `[persona: ...]` line when there is one. Keeps the floor. An agent coming back with a report sends one `converse` and leaves it waiting in the queue, however long (owner decision 2026-10-26). |
| `call(greeting, override_quiet_hours=False, max_seconds=120, silence_seconds=2.0, start_timeout_seconds=45)` | Telegram voice call from the algolearn.ai account to the owner (@frankgibbs35); `greeting` is the whole opening, spoken the moment they answer in the calling project's persona's voice: who is calling (by the persona's name), why, and the context, ending with a question or invitation. It must be at least 6 s when spoken (measured on the synthesized audio); a shorter one raises `ValueError` BEFORE dialing (phone never rings; owner decision 2026-10-12) telling the model to add the context of why it is calling, or, on a fresh conversation with nothing to report, to open with something playful (a joke, something funny or flirty). Never padded. Once answered, `call` immediately does the `listen` (ear-open beep, VAD, Whisper, rising chime) under the same lock hold and returns the owner's words; listen parameters and defaults match `converse`, and a listen timeout or a drop raises as in `listen` (the call stays connected on a timeout). Returns `answered\n<their words>\n(audio: telegram)`, `already connected` (greeting not spoken, no listen), `not answered (declined or no answer)`, `busy (the owner's Telegram reports a call in progress)` (Telegram discarded it as busy; the phone did not ring), `answered, then hung up during the greeting` (the owner picked up and ended it before the greeting finished; one call back right away is allowed, then the one-hour rule), `hung up` (another session's `hang_up` ended it during the greeting), `answered but audio failed to connect: <error>` (the owner picked up but media failed; the daemon container restarts itself) or `not called: quiet hours (...)`. Check order (owner decision 2026-10-23): already connected (whatever the time: the owner is on the line), quiet hours, then the greeting length, then the wait until 10 s have passed since the last call ended, then the dial. Quiet hours, an empty greeting and a too-short greeting each record the event `call_refused` (`project`, `reason`: `quiet_hours`, `empty_greeting` or `greeting_too_short`, the last with `greeting_s`); `already connected` is not a refusal and records nothing. `call_refused` belongs to no call: it carries `call_id` null and no `elapsed_s` (`TelegramLink.record_unbound_event`), and the `quiet_hours` one carries `local_time` (`HH:MM`, daemon clock). Daemon only, no protocol change. Deployed in commits d503492 and fce8f93 on 2026-10-23. |
| `hang_up()` | End that call, only when the owner asks (the daemon itself hangs up after a sign-off `speak` with nobody queued, owner decision 2026-10-28). Always clears the session's call expectation. |
| `persona()` | This project's persona (the block: name, whether it is the house persona, voice blend, speed, personality, the projects sharing it), every persona with its projects, the English voice ids, the speed range and how many pool names are left. No lock. Agents call it first, before their first `call` or `converse`. |
| `new_persona()` | Moves this project onto a brand-new persona: a random unused pool name, the default voice and speed. Only when the owner asks. Raises `ValueError` once the pool is used up. No lock. |
| `set_persona(name=None, voice=None, speed=None, personality=None)` | One change per call to this project's persona. `name` alone moves the project onto the persona of that name (case-insensitive), or renames its own persona for every project sharing it when no persona has the name. `voice` (`{voice id: weight}`, 1 to 4 English voices, normalized to sum 1), `speed` (0.5 to 2.0) and `personality` (the whole note as a list of lines, for a persona review) change the persona for every project using it. Voice and speed only when the owner asks. No lock. |
| `add_to_personality(note)` | Appends one line to the persona's personality (agents do this on their own as they learn their role, except on the house persona, which grows only when the owner asks). Raises once the note would pass 4000 characters: ask the owner for a persona review. No lock. |
| `status()` | `{busy, current_tool, waiting, call}`: what holds the daemon's lock (`"<tool> (<client ip>)"`; `transcribe` is an HTTP whisper API request), how many requests from every session wait behind it, and the call state (`none`/`ringing`/`connected`). When the daemon is not reached, `call` is `daemon unreachable: <error>` and the other three are null. When the daemon speaks another protocol version, `call` is `protocol mismatch: ...` (update the client or deploy the daemon) and the other three are null. Never raises, never waits on the lock. While `call` is `connected`, a session coming back sends `converse` even when `busy` is true or `waiting` is above 0; this `busy` is not `call`'s `busy` outcome. |
| `call_debug(lines=50)` / `set_call_debug(enabled)` | The daemon's recent call events and log tail; DEBUG logging on or off at runtime. No lock. `{"daemon": "unreachable: <error>"}` when the daemon is not reached. |

Every `speak`/`listen`/`converse` result, and an answered `call`, ends with
`\n(audio: telegram)` (kept for the permission rule in
`~/.claude/settings.json` that keys on it).

## Hold on / continue (Release 2 of docs/DESIGN_INTERRUPT.md)

Owner decisions 2026-10-16. Deployed (committed in 2186c40 on 2026-10-16;
in the images built from 84c2740 on 2026-10-20 and every later deploy); the
echo gate (design section 8.1) and the live checks (8.2 to 8.4) have not
been run.

- While `speak`/`converse` plays, the daemon watches the owner's audio
  (Silero VAD; Whisper only on candidates). "hold on" (`SPEAK_HOLD_PHRASE`)
  cuts the speech at once and holds in silence; "continue"
  (`SPEAK_RESUME_PHRASE`) restarts the sentence she was cut off in, from its
  start, without re-synthesizing. No names in the phrases.
- A command counts only as the **whole** utterance (exact match after
  lowercasing and turning punctuation into spaces), 0.3 to 2.0 s long, after
  at least 0.8 s of non-speech, ending with 0.4 s of silence. Longer or
  run-on speech is dropped without Whisper. "hold on" acts only while
  playing, "continue" only while held.
- The holds of one op add up to at most `HOLD_MAX_SECONDS` (300 s,
  `speak_protocol.py`). Then the hold has expired: the daemon **hangs up**
  and clears the session's call expectation. The result is
  `[hold expired after 300s; the call was ended]`, then `cut off in sentence
  i of n: "<sentence>"`, then `not spoken (from the cut sentence): "<rest>"`,
  then `(audio: telegram)`. `converse` does not listen. The agent keeps the
  unspoken text and does not call back right away. If that hang-up fails,
  its error is the tool's result, the event `hold_expire_hangup_failed`
  (`tool`, `error`) is recorded, the call may still be up, and the
  expectation is cleared either way.
- While held, the op keeps the audio lock: other sessions wait up to 300 s.
- **Client leaves during a hold (owner decision 2026-10-16):** when the
  holding request's client goes away (its tool call was cancelled or its
  session closed), the hold ends at once, the lock is released to the next
  request, and the call is **not** hung up: it stays connected. The client
  closes its socket on cancellation (`_run`); the daemon polls for EOF while
  held (`watch_for(client_left=...)`).
- The greeting, the cues and `listen` cannot be held.
- Wire: the `speak`/`converse` replies gain `held_seconds` (always) and
  `interruption` (`{command: "hold_expired", cut_index, sentence_count,
  cut_sentence, unspoken}`, only after an expiry; the `converse` reply then
  has no `text`).
- Events (`call_debug`): `watch_candidate` (`mode`, `length_ms`,
  `leading_silence_ms`, `result`, `reason` or `transcribe_ms`, `matched`,
  `acted`; `text` only with debug on), `playback_cut` (`command`,
  `played_ms`, `dropped_ms`, `command_latency_ms`), `hold_start`, `hold_end`
  (`how`: `resume`/`expired`/`call_ended`/`client_left`, or `error` with
  `error` when classifying a candidate raised; `held_s`),
  `hold_expire_hangup_failed`, `first_incoming_frame` (`during`: `listen` or
  `watch`, whichever saw the first frame of the call), and `cut` in
  `speech_stream_done`.

## Switch of session (owner decisions 2026-10-20)

Deployed in commit 84c2740 and tested live on 2026-10-20 (the bare-listen
finding below came from that test). The bare-listen "still listening"
announcement is deployed (committed in 2fa4a20 on 2026-10-20; in the image
built from b5cd149 on 2026-10-20 and every later deploy).

- **The daemon announces the switch, not the agents.** Every client sends a
  session id (`secrets.token_hex(8)`, generated once per client process,
  `init_identity` in `speak_server.py`) and a project name (the basename of
  `os.getcwd()` when the client starts). The name and voice the owner hears
  are the project's persona's (see "Personas" below), not per session: two
  sessions of one project, or two projects sharing a persona, sound alike.
- A `speak`, `listen` or `converse` whose session differs from the session of
  the previous audio op on the same call is preceded by the switch cue (two
  short 523 Hz beeps, `switch_pcm` in `speak_tone.py`) and then "<persona
  name>, <project name made speakable>" (`speakable_name`: runs of
  non-alphanumerics become one space, so `algolearn-speak` with the persona
  Nova is spoken "Nova, algolearn speak"; owner decision 2026-10-22), in the
  incoming project's persona's voice (owner decision 2026-10-22). For a `speak` or
  `converse` nothing else is said. The announcement is synthesized whole and
  played as one plain play: like the greeting and the cues, it cannot be held.
- **A bare `listen` (owner decision 2026-10-20):** when the op that causes
  the switch is `listen`, the announcement after the cue is "<persona name>,
  <speakable project name>, still listening" (`SWITCH_LISTEN_SUFFIX` in
  `speak_telegram_daemon.py`), still one clip synthesized whole, then the ear
  opens as usual. The live test that led to it: a session waiting in a bare
  `listen` got the turn after another session's converse, and the owner heard
  a name with no context before the ear-open beep. `speak` and `converse`
  switches are unchanged. Daemon only: no protocol change.
- **What "previous" means (the turn):** the daemon keeps one turn, the call's
  key (`"<boot id>:<generation>"`) and the last session that played audio on
  it. An answered `call` makes the calling session the call's first turn (its
  greeting is that session's audio), so the caller's first `speak`, `listen`
  or `converse` after `call` is not a switch. A turn from an earlier call
  never counts: when a call ends, the next call starts from the session that
  placed it. `call` returning `already connected`, and `hang_up`, leave the
  turn as it is. Two sessions in the same folder are still two sessions: the
  switch is announced with the same name.
- The switch cue and the names are also what tells the owner the floor
  changed hands (see "The floor" below); there is no separate floor cue.
- The turn moves only once the announcement has played. An op refused before
  it plays (no call, empty text) or a call that drops during it leaves the
  turn unchanged (the drop raises `Telegram call ended during <tool>`).
- **The reply carries `switched_from`** (the previous session's project)
  whenever the switch was announced, on a failure too (the owner heard it).
  The client puts `[took over the call from "<project>"; the owner heard the
  switch cue and your persona's name and this project's name first]` on the
  result's own first line, before a persona, held or expired line, and in
  front of an error's message.
- The server instructions tell agents: after that line, the owner has just
  heard another project; open the next turn with one sentence recapping where
  you left off and your last question, unless his reply already answers it.
- Event (`call_debug`): `session_switch` (`tool`, `from_project`,
  `to_project`, `to_persona` (the incoming persona's name), `spoken` (the full text spoken, the "still listening" suffix
  included), `announce_s`).
- The client adds `10 + (12 + 2 + len(project))/5` s to the speak, listen
  and converse budgets for the announcement (12 is the longest persona name,
  a fixed bound so the budget never depends on whether the client has seen
  its persona); that covers the longer listen announcement too, on top of
  the listen budget's own 60 s margin.

## Personas (owner decisions 2026-10-22)

Design: `docs/DESIGN_PERSONAS.md`. Deployed in commit 6a9fcbe on 2026-10-22 and tested live.
Protocol 4.

- **What a persona is (owner decisions):** a name (one word of 2 to 12
  letters, unique ignoring case), a voice blend (1 to 4 Kokoro voices with
  weights, normalized to sum 1, e.g. 70% `af_heart` + 30% `af_bella`), a
  speed (0.5 to 2.0) and a personality note (lines of tone, attributes and
  role context, at most 4000 characters in all). Each project (the client's
  project name) points at exactly one persona; several projects may share
  one, and a shared name IS a shared persona. A change to a persona's voice,
  speed or personality applies to every project using it. No delete tool:
  unused personas are kept.
- **Storage:** one JSON file (schema 1), `SPEAK_PERSONAS_FILE`
  (`/var/lib/speak/personas.json`) on the `speak_state` docker volume, so it
  survives rebuilds. `speak_personas.py` (`PersonaStore`) validates it on
  load and on every change, applies a change to a copy, writes it
  atomically (`<file>.tmp`, fsync, `os.replace`, fsync of the directory)
  and swaps it in the moment `os.replace` succeeds, so memory matches the
  disk; a failed directory fsync raises `PersonaDirSyncError`, saying the
  change is on disk. A write holds the store lock across its fsyncs
  (accepted: writes are rare, the `persona_of` reads are per sentence). A file that does not parse or validate stops
  the daemon at startup and is never replaced; a change that does not
  validate raises `ValueError` to the tool and writes nothing.
- **Seeded Angie and new projects (owner decisions):** on a first start (no
  file) the daemon seeds one persona, Angie, on `SPEAK_VOICE` and
  `SPEAK_SPEED`, and marks it the house persona (by id, so a rename keeps
  it the house persona). A project with no persona joins the house persona
  on its first identity-carrying request (any tool op or persona op), with
  no question to the owner. The owner moves a project whenever he wants, by
  asking.
- **Moving and renaming (owner decisions):** `set_persona(name=X)` moves the
  project onto the persona named X when one exists (never a copy); when no
  persona has the name, it renames the project's current persona for every
  project sharing it. `new_persona()` (only when the owner asks) moves the
  project onto a fresh persona with a random unused pool name and the
  default voice and speed. The pool: Angie, Nova, Ruby, Hazel, Paige,
  Stella, Gwen, Willow, Margo, Juno, Tess, Fiona; `new_persona` raises once
  every pool name is in use.
- **English voices only (owner decision):** ids starting `af_`, `am_`, `bf_`
  or `bm_`. The expected set is the constant `ENGLISH_VOICES` in
  `speak_engines.py`: the 28 English voices of `hexgrad/Kokoro-82M`
  (verified from the Hub listing 2026-10-22; the repo has 54 voices).
  At startup `Engines.load_voices` refuses a `SPEAK_VOICE` that is not one
  of them before any fetch, then fetches exactly those packs
  (`snapshot_download` with one `allow_patterns` entry per id, into the HF
  cache on `speak_models`). With the Hub unreachable `snapshot_download` may
  serve the cached snapshot (owner decision 2026-10-22); the daemon then
  checks every one of the 28 is there and refuses to start, naming the
  missing ids, when any is not. It loads each once on the CPU as float32;
  the persona file is then validated against that list.
- **American pronunciation for every voice (owner decision 2026-10-22):**
  the one `KPipeline` runs with `lang_code="a"`, so the British voices
  (`bf_`, `bm_`) speak with American pronunciation. A British path may come
  later. A blend is the weighted sum of the packs, cached (LRU of
  64) and checked to be a CPU `torch.FloatTensor`.
- **Agents change voice or speed only when the owner asks** (for him to
  experiment); they add to the personality on their own as they learn their
  role, and the owner may start a "persona review" at any time (the agent
  reads the persona back and he trims it, `set_persona(personality=[...])`).
- **The house persona never adapts on its own (owner decision 2026-10-22):**
  agents on it do not call `add_to_personality` of their own accord; the
  owner can still change it in a persona review or on request. Stated once
  in the server instructions; the block carries `house` (true/false) and the
  persona line says "you are <Name>, the house persona". The daemon does not
  enforce it.
- **How the agent learns its persona:** the client keeps `_persona_seen`
  (`"<persona id>:<words_revision>"`, null at start; design 2026-10-22: the
  words revision rises only when the name or the personality changes, never
  on a voice or speed change) and sends it on every tool op, moving it only
  forward on an ordinary reply (a new id, or a higher words revision; a late
  reply across a move to another persona costs at most one extra refusal, which
  corrects it), but a refusal for a stale persona states the daemon's current
  key and is applied unconditionally, even when lower. A tool reply carries
  `persona` (the block: `id`, `words_revision`, `name`, `house`, `voice`,
  `speed`, `personality`, `shared_with`) when the project's current key
  differs from it, and every persona op reply carries it, and the client
  puts it in the result as one line, `[persona: you are <Name>; voice <id>
  <weight>, ...; speed <s>; personality: <line> / <line> | (none yet);
  shared with <projects>]`, after the switch line and before the held line
  (call and hang_up results start with it). Before the lock, a `speak`,
  `converse` or `call` whose `persona_seen` is stale (the project moved, or
  its persona's name or personality changed) is refused (nothing is
  spoken or dialed) with the block; a voice or speed change refuses nothing
  and reaches other sessions only through `persona()`; the client raises `ValueError`: the
  persona line, then "Nothing was spoken (or dialed). Rewrite your words as
  <Name> and send again." `listen` and `hang_up` carry no words and are
  never refused for it. The server instructions tell agents to call
  `persona()` first; the refusal is the guarantee.
- **Changes take effect on the next sentence (owner decision):** the
  speaking voice is read before each sentence is synthesized, so an op
  already queued or playing follows a change sentence by sentence (and its
  reply carries the new block when the name or personality changed). Two consequences of streaming: a sentence
  already synthesized and queued keeps its voice, and a sentence replayed
  after "continue" comes from the cache in the voice it was made in. The
  call greeting is in the calling persona's voice, the switch announcement
  in the incoming persona's.
- Events (`call_debug`): `persona_assign` (`project`, `persona`, `how`:
  `auto`/`new`/`moved`, `from`), `persona_change` (`persona`, `fields`,
  `project`, old and new name, voice and speed), `persona_refused` (`tool`,
  `project`, `seen`, `current`).

## The floor (owner decisions 2026-10-20)

Design: `docs/DESIGN_FLOOR.md`. Deployed in commit b5cd149 on 2026-10-20.
No protocol change: the protocol stays 3.

- **The floor:** the session that last played audio on the call holds it
  (the turn of "Switch of session": `_take_turn` and an answered `call` call
  `AudioLock.take_floor`). While the floor is live and the lock frees, only
  the holder's ops (and HTTP transcriptions) may take it, in arrival order;
  every other session's op stays queued, never rejected. If the holder has
  nothing queued, the lock stays free and the others keep waiting. Without
  a live floor: plain FIFO.
- **Idle lapse:** `FLOOR_IDLE_SECONDS` (45 s, `speak_protocol.py`) after the
  release of the holder's last op that held the lock (a queued op skipped
  because its client left counts), the floor lapses and the head of the
  queue goes. The deadline is cleared from the moment the holder's next op
  arrives (queued, not only granted), so an op that holds the lock past the
  deadline (a long transcription) cannot let the floor lapse under the
  holder's queued op. No timer thread: a waiter that is not eligible waits on the lock's condition until
  the deadline; the first thread that finds it past (a waiter, or a later
  `acquire`/`take_floor` when nobody waits) records the lapse once and wakes
  every waiter. The holder coming back after a lapse is just another queued
  session. Only the op holding the lock may take the floor (`take_floor`
  raises otherwise), so `from_project` is null in practice; an `acquire`
  whose wait raises removes its ticket (and, for the holder's with no other
  op of its queued or running, restarts the idle count).
- **Restart or stop:** right after `closing` is set (`SpeakDaemon.stop`, or
  the handler before an audio-failure restart), `AudioLock.close()` makes
  every queued op eligible in arrival order, so ops behind a floor reach the
  `speak daemon is restarting` refusal at once instead of after the idle
  time. No floor event.
- **Releases** (`AudioLock.yield_floor`, then the lock's release wakes the
  queue in FIFO order):

  | When | Reason |
  |---|---|
  | A `speak` that took the turn ends, however it ends (held and resumed, or failed), except when the call ended under it; with no other tool op queued the daemon then ends the call (see "Sign-off hang-up" below) | `speak` |
  | The holder sends nothing for 45 s after its last op ended | `idle` |
  | `hang_up` by the holder (even if the hang-up raised) | `hang_up` |
  | A hold expires and the daemon hangs up, whether or not that hang-up failed | `hold_expired` |
  | The holder's client leaves during a hold of `speak` or `converse` (the call stays up) | `client_left` |

  `converse` and `listen` keep the floor, including a `listen` that ends in
  `TimeoutError` (no speech): an agent that keeps reopening the ear keeps the
  floor (owner decision; watch for it on live calls). An unplanned drop (the
  owner hangs up, a network failure), also during a `speak`, keeps the floor
  for the idle window, so the interrupted session's `call` goes first; its
  answered call takes the floor again. `call` outcomes other than answered
  leave the floor as it was. A `hang_up` or a `call` from a session that does
  not hold the floor does not move it.
- **HTTP transcriptions** (port 9000) carry no session and skip the floor:
  they wait only for the op holding the lock, and their release leaves the
  idle deadline as it was.
- **Sign-off hang-up (owner decisions 2026-10-28; supersedes the agent-side
  rule of 2026-10-27, rule 4 below):** the daemon hangs up, not the agent.
  When a `speak` gives up the floor (release reason `speak`) and no other
  tool op is queued for the lock at that moment, the daemon ends the call at
  once, with no grace period; with any tool op queued (another session's,
  or this session's own) the call stays up and the floor passes as before.
  It is a deliberate end, like `hang_up`: the session's call expectation is
  cleared and nobody calls back. Events: `floor_release` as before, then
  `signoff_hangup` (`project`, `waiting` 0, `outcome`); a hang-up that
  raises records `signoff_hangup_failed` (`project`, `error`), its error is
  the speak's result, and the expectation is cleared either way. The reply
  (success or failure) carries `call_ended: "signoff"` and `expect` null;
  the client adds `[call ended: nobody else was waiting]`.
  Design choices (2026-10-28, made in the build, within the owner's
  decisions): an HTTP transcription queued at that moment does not keep the
  call up; it stays queued and runs after the hang-up, once the lock is
  released. Exactly which cases hang up: a `speak` that played to the end,
  one held and resumed, and one that raised while the call stayed connected
  (e.g. a synthesis error), each only when nothing else is queued. Never: a
  `speak` under which the call ended (it yields nothing), an expired hold
  (the daemon has already hung up, reason `hold_expired`), a client that
  left during a hold (reason `client_left`, the call stays up), a `speak`
  refused before its turn (empty text, no call), `converse`, `listen`.
  No protocol bump: a protocol 4 client that predates the field ignores it
  and still clears its expectation from `expect`.
  Verified from the code (2026-10-28): `AudioLock.yield_floor`
  (`speak_telegram_daemon.py:369`) clears the floor and counts the queued
  tickets that carry a session (`:381`; a transcription's ticket has none)
  under one hold of the lock's condition, while the speak still holds the
  audio lock; `SpeakService.speak` calls `_sign_off_hang_up` (`:764`) from
  its `finally` (`:761`) when that count is 0; the handler releases the
  audio lock only after `run_locked` returns (`:1008`, released at `:1014`).
  So no queued op can run between the count and the hang-up; an op that
  arrives after the count runs after it and gets "needs a connected
  Telegram call" (`require_call`, `:534`; the hang-up's `END_HUNG_UP`
  clears a matching expectation, so it is not "ended during"). Not
  deployed: the deploy is the owner's action.
- **Visible only in `call_debug`:** events `floor_take` (`project`,
  `from_project`, the previous holder's or null) and `floor_release`
  (`project`, `reason`, `held_s` since its `floor_take`, `waiting`: tickets
  queued then). `status` is unchanged, so `busy` can be false while
  `waiting` is non-zero (the lock is free and the waiters are not eligible).
  No tool reply tells the holder that others wait (owner decision).
- **Etiquette in the client (owner decisions 2026-10-22 and 2026-10-23,
  replaced by owner decisions 2026-10-26, agreed on a call, amended by
  the owner decision of 2026-10-27 on hanging up after the sign-off, which
  the owner decisions of 2026-10-28 superseded):**
  stated once,
  in the server instructions (`_INSTRUCTIONS`), written to the agent on the
  line in the second person ("you ..."), not as a description of the daemon
  (owner decision 2026-10-26). Verified 2026-10-26: Claude Code cuts MCP
  server instructions off at 4096 characters, and the 2026-10-23 text was
  6,891 characters, so every floor and coming-back rule sat past the cut and
  agents never saw them. The text now puts the floor and coming back first,
  then calls, the persona and speaking, and stays under the limit (3,796
  characters since the 2026-10-28 sign-off text, with shorter wording for
  points the docstrings carry: the 120 s backgrounding, the took-over recap,
  the persona review, `TimeoutError`, and the hold phrases, which
  `_HOLD_NOTE` explains; "never speak 'hold on' or 'continue' as a sentence
  by itself" is kept); `speak_server.py` raises `RuntimeError` at import
  (`check_instructions_length`) when `len(_INSTRUCTIONS)` reaches
  `INSTRUCTIONS_MAX_CHARS` (4096), and tests check it (4095 passes, 4096
  is refused). Detail lives in the tool docstrings, which are not cut. The
  rules (owner decisions 2026-10-26, and 2026-10-28 for rule 4):
  1. Stay in `converse` while you work the problem out with the owner
     (understanding it, deciding what to do). Every turn is one `converse`.
  2. When you are about to go do the work, whatever its length (even a
     2-minute job), ask the owner whether he is ready to let you go; then
     sign off with ONE `speak` that promises to come back on the call, never
     "by message". This replaces the rules of 2026-10-22 (stay in `converse`
     while an agent researches; sign off only when the topic has closed, he
     has nothing else and the remaining work is around 15 minutes or more).
  3. Coming back is owed, by voice: when the work is done, report it on the
     call before anything else, even after your context was summarized, a
     text only in addition, and never ask whether he wants a call back.
     `status()` first. On "connected", send ONE `converse` with the report
     and leave it waiting in the queue, however long it takes, even behind
     another session's conversation. While it waits, do not cancel it, do
     not resend it, do not text instead, do not `call`. On "ringing", `status()` again. On
     "none", `call`, and the call etiquette applies (`not answered`: wait an
     hour by background timer, and so on).
  4. (Owner decisions 2026-10-28; they supersede the owner decision of
     2026-10-27, under which the agent called `status()` right after its
     sign-off and `hang_up()` when `busy` was false and `waiting` 0.) After
     your one `speak` sign-off, do nothing more on the call: with nobody
     else waiting the daemon hangs up ("[call ended: nobody else was
     waiting]"), otherwise the next session goes. `hang_up()` is again
     "only when he asks". The instructions read "After your sign-off, do
     nothing more on the call: ..." in the floor section and "hang_up()
     only when he asks." in the call etiquette. The daemon side is
     "Sign-off hang-up" above.
  Verified 2026-10-26 in a live call from this repo: a queued converse that
  waited behind another session's conversation was moved to the background
  by Claude Code after 120 s, kept running (the request was not cancelled),
  played when the floor was released, and its result arrived as a task
  notification with the owner's words. Design (code review 2026-10-26): the
  no-cancel, no-resend rule holds only while the `converse` waits; when it
  returns "Telegram call ended during ..." or "needs a connected Telegram
  call", the agent calls back with `call` as the call etiquette says.
  The rules kept from 2026-10-22 and 2026-10-23, and earlier: when the
  owner asks for a call, `override_quiet_hours=True` without asking; when a
  call seems off, `set_call_debug(True)` and a test call without asking
  first; real work goes to agents, now once the owner has let you go; the
  session that last played audio holds the floor (`converse` and `listen` keep it, `speak`
  gives it up); `listen` alone only to reopen the ear after a
  `TimeoutError` or with nothing to say first; whenever the call etiquette
  says to wait (quiet hours, `not answered`, `busy` twice, hung up during
  the greeting twice, an expired hold), say in the terminal when you will
  try again and start a background timer (a background `sleep`); an expired
  hold's unspoken text comes back after the hour's wait, by timer; calls you
  start on your own respect quiet hours, which never apply while a call is
  connected; the recap sentence after a `[took over the call from ...]`
  line. Docstrings (2026-10-28): `speak` says it is only the sign-off once
  the owner has said you may go do the work, then nothing more on the call:
  the daemon hangs up when no other request is queued and the result says
  `[call ended: nobody else was waiting]` (on an error too), otherwise the
  next session takes the floor, and not to call `hang_up()` or call back for
  it; `hang_up` that it is used only when the owner asks, never for the
  sign-off; `call` that `hang_up()` is only when the owner asks and that
  the daemon hangs up after a sign-off with nobody queued (a test checks
  that no agent-facing text still tells the agent to call `status()` then
  `hang_up()` after the sign-off); `converse` that it keeps the floor and,
  coming back, may wait minutes in the queue, moved to the background, so
  you leave it waiting while it waits, and call back with `call` if it
  returns "Telegram call ended during ..." or "needs a connected Telegram
  call"; `listen` that it keeps the floor; `status` that
  `busy` true or `waiting` above 0 while "connected" still means `converse`,
  and that this `busy` is not `call`'s `busy` outcome; `call` that while the
  call is connected you use `converse`, not `call`; `_QUEUE_NOTE` that
  `busy` false with `waiting` above 0 means another session holds the floor
  and your request still queues behind it. No "one moment" before work, and
  the idle timeout is not mentioned to agents at all. The 2026-10-28 change
  adds the daemon's sign-off hang-up and the `call_ended` reply field, with
  no protocol bump. Running sessions get the new text after `/mcp` reconnect (a
  `git pull` on each computer first).

Without a connected call, `speak`/`listen`/`converse` raise `<tool> needs a
connected Telegram call (call state is ...); place one with the call tool`.
When the daemon cannot be reached, every tool except `status`, `call_debug`
and `set_call_debug` raises `speak daemon at <addr> is not reachable: ...`.

## Files

- `speak_server.py`: the thin MCP client. Tool definitions and docstrings,
  the server `instructions` (owner conventions sent to every client: speak
  as the project's persona, spoken style, one converse per turn, timeout
  handling, call etiquette, debugging), `_run`, the session's
  call expectation (`_call_generation`) and the persona it last saw
  (`_persona_seen`). `main()`
  reads `SPEAK_DAEMON` (`host:port`, required, no default; unset or malformed
  fails at startup).
- `speak_protocol.py`: the wire (stdlib only). Framing: a 4-byte big-endian
  header length, a JSON header, `payload_len` bytes (empty in practice). One
  request per TCP connection, TCP keepalive both ways. Op names, error kinds,
  and the outcome and state strings (`ANSWERED`, `CONNECTED`, `END_HUNG_UP`,
  `OPENER_MIN_SECONDS`, `HOLD_MAX_SECONDS`, `FLOOR_IDLE_SECONDS`, the hold
  command and reply field names, the persona limits and pool, ...), each
  defined once.
- `speak_personas.py`: the persona registry (stdlib only): `PersonaStore`
  (open or seed, `ensure`, `persona_of`, `block`, `overview`, `new`, `set`,
  `add`, the atomic write) and the validation.
- `speak_telegram_daemon.py`: the daemon (`speak-telegram` entry point). TCP
  server, the audio lock with the floor (`AudioLock`: FIFO tickets with the
  session id, `take_floor`, `yield_floor`, `close`, the idle lapse), the tool logic (`SpeakService`:
  call checks, quiet hours, markdown stripping, cues, the hold phrases,
  `classify` and the `_speak` hold loop, the voice per persona), the
  persona refusal before the lock (`SpeakDaemon.admit`) and the persona
  ops, the debug ops, the
  HTTP whisper API (`_WhisperHandler`, `SpeakDaemon.transcribe_pcm`), and
  `main()` (settings, engine load, Telegram, binds).
- `speak_engines.py`: `Engines` (`synthesize_segments` per segment with its
  sentence index for speak, one Kokoro call per sentence asking the voice
  before each; `synthesize` whole in one voice for the greeting and the
  announcement; `load_voices` and `voice_pack` for the English packs and
  blends; every synthesis takes a `Voice`, pack and speed), and
  `split_sentences`, the same sentence split Kokoro indexes. Kokoro `KPipeline` (hexgrad/Kokoro-82M) and
  faster-whisper `WhisperModel` (compute_type `float32`) on `cuda`. The load
  raises if CUDA is missing, Kokoro's weights are not all fp32, or Kokoro's
  espeak-ng fallback did not load.
- `speak_telegram.py`: `TelegramLink`, the call itself (Telethon + py-tgcalls +
  ntgcalls): dial, opener file, 10 ms raw-frame pump, Silero VAD on the
  owner's audio, and the hold watch (`play_watched`, `watch_for`,
  `_Watcher`). Instantiated only by the daemon.
- `speak_tone.py`: the cue PCM: the ear-open beep, the rising "captured" chime, the no-speech timeout beep and the switch-of-session cue (played into the call by the daemon).
- `Dockerfile`, `docker-compose.yml`, `.dockerignore`: the daemon image and
  service.
- `scripts/drive_daemon.py`: drives the daemon over TCP with the client's own
  request code (status, call, speak, listen, converse, hang_up, debug, the
  persona ops), no MCP.
- `pyproject.toml`: base dependencies are only `mcp` (the client). The
  `daemon` extra holds Telegram, torch (CUDA 12.6 build on Linux), kokoro,
  misaki, faster-whisper, silero-vad, numpy, scipy.

## Protocol

- **Protocol version:** every request carries `protocol` =
  `PROTOCOL_VERSION` (`speak_protocol.py`, now 4; the unversioned protocol
  before Release 2 counts as 1, Release 2 is 2, the switch of session made
  it 3 (owner decision 2026-10-20), personas made it 4 (2026-10-22)). The daemon refuses `ping` and every tool op
  whose version is missing or differs, before the lock, with a `value` error
  telling the session to update its client (git pull) and reconnect /mcp.
  `status`, `debug` and `set_debug` are served to any client (diagnostics
  that never touch the lock or the call) and their replies carry the
  daemon's version; the client's `status` reports a mismatch in `call`.
  Bump the version whenever a request or reply changes in a way an older
  peer would misread.
- **Identity (protocol 3):** every tool op (`speak`, `listen`, `converse`,
  `call`, `hang_up`) carries `session_id` and `project`, non-empty strings.
  The daemon refuses one that is missing, not a string, empty or blank with
  a `value` error before the lock (`speak client sent no <field> ...; update
  the client (git pull) and reconnect /mcp`), as it refuses a missing
  protocol version, and refuses a project with no letters or digits
  (nothing to speak). Quick ops other than the persona ops carry no
  identity. Replies to `speak`, `listen` and `converse` may carry
  `switched_from` (absent: no switch).
- **Sign-off hang-up (2026-10-28, still protocol 4):** a `speak` reply,
  success or failure, carries `call_ended: "signoff"` (`REPLY_CALL_ENDED`,
  `CALL_ENDED_SIGNOFF` in `speak_protocol.py`) when the daemon ended the
  call after it, with `expect` null; absent otherwise. No version bump
  (design choice 2026-10-28): an older protocol 4 client ignores the key
  and still clears its expectation from `expect`; the current client
  raises on an unknown value.
- **Personas (protocol 4):** every tool op also carries `persona_seen`
  (null or `"<persona id>:<words_revision>"`; missing is refused like a missing
  version). Before the lock the daemon assigns a new project to the house
  persona and refuses a `speak`, `converse` or `call` with a stale
  `persona_seen` (kind `value`, the block in `persona`); a voice or speed
  change does not make it stale. A reply to a tool op carries `persona`
  (`{id, words_revision, name, house, voice, speed, personality,
  shared_with}`) when the current key differs from `persona_seen`; a
  persona op's reply always carries it. The persona ops (`persona`,
  `persona_new`, `persona_set`, `persona_add`) are quick ops that never
  touch the lock, but unlike the other quick ops they are versioned and
  carry `session_id` and `project`.
- Tool ops (`speak`, `listen`, `converse`, `call`, `hang_up`) queue in the
  daemon's lock and are never rejected: FIFO, except that the floor
  holder's ops go first (see "The floor"; no protocol change, the session id
  every tool op already carries orders the queue). The daemon sends
  `{"ok": true, "phase": "started"}` the moment it holds the lock, then the
  final reply. The client waits for `started` with no timeout, then applies a
  budget (speak `60 + len/5 + 300` s, the 300 being `HOLD_MAX_SECONDS`;
  listen `start_timeout + max + 60`; converse the sum; call `60 + 30 + len/5 + listen`; hang_up 20 s). An overrun raises
  `RuntimeError` "did not reply", never `TimeoutError`.
- Quick ops (`ping`, `status`, `debug`, `set_debug` and the persona ops) get
  one reply and never touch the lock.
- Errors: `{"ok": false, "error", "kind"}`. The client raises `timeout` (no
  speech only) as `TimeoutError`, `value` as `ValueError`, `call_ended` and
  `runtime` as `RuntimeError`.
- **Call expectation, per session:** every tool op carries
  `expect_generation` (the key of the call this session talked on,
  `"<daemon boot id>:<call generation>"`, or null; the boot id keeps a key from
  before a daemon restart from matching a new call). The daemon
  applies the rules under the lock: connected, so proceed and expect this
  generation; ended by `hang_up` with the same generation, so clear and raise
  the no-call error; not connected while expected, so `Telegram call ended
  during <tool>; call back ...`. A reply may carry `"expect"` (a key, or null
  to clear); absent means unchanged.
- A client that disconnects mid-request does not cancel the daemon's work; the
  reply is discarded. A client that leaves while still queued is skipped (the
  daemon checks the socket for EOF when the op reaches the lock). The one
  exception is a hold: a cancelled tool call closes its socket (`_run`), and
  a hold whose client has left ends at once and releases the lock without
  hanging up (owner decision 2026-10-16; `hold_end` `how: client_left`).
- While the daemon is restarting or stopping, an op that reaches the lock gets
  `speak daemon is restarting` and never touches the call.

## HTTP whisper API (port 9000)

The retired `whisper-service` (`scratch/whisper-service/server.py`, FastAPI)
reproduced in the daemon (`_WhisperHandler` in `speak_telegram_daemon.py`,
stdlib `http.server`). Known client: `scratch/voice-claude/livekit_agent.py`
(`WhisperServiceSTT`, `WHISPER_URL=http://patrick:9000/`).

- `POST /`: raw body of 16 kHz mono int16 PCM, no header. `200
  application/json` `{"text": "..."}` (an empty text is returned, not
  raised). An odd byte count or an engine failure is `500 text/plain`
  `Internal Server Error`.
- `GET /health`: `{"status":"ok","model":"large-v3-turbo","device":"cuda"}`.
- As FastAPI: a wrong method is `405` `{"detail":"Method Not Allowed"}` with
  `Allow`, an unknown path `404` `{"detail":"Not Found"}`, a trailing slash
  `307` to the other spelling. HTTP/1.1 keep-alive and chunked bodies.
- It reuses the loaded Whisper engine (`Engines.transcribe`: large-v3-turbo
  fp32, `language=SPEAK_LANGUAGE` (en), `beam_size=1`, `vad_filter=False`).
  No new model, no new container, no new setting.
- Every transcription waits in the same FIFO audio lock as the tool ops
  (status shows `transcribe (<ip>)`), so inference never overlaps call audio;
  a request can wait behind a whole call or converse turn, including up to
  `HOLD_MAX_SECONDS` (300 s) behind a held speak or converse. The PCM is read
  before the request queues. Transcriptions skip the floor: they wait only
  for the op holding the lock, never for a conversation.
- Hardening (departures from the old service): a body over `MAX_BODY_BYTES`
  (15 minutes of 16 kHz mono int16 PCM, 28,800,000 bytes; owner decision
  2026-10-16) is `413` `{"detail": ...}` and the connection closes, checked
  on Content-Length before the read and on the running total of a chunked
  body. Once a request holds the lock, a client that has left is skipped (no
  inference, logged) and a restart or stop under way is `503`
  `{"detail":"speak daemon is restarting"}`. A `30 s` socket timeout
  (`HTTP_TIMEOUT_SECONDS`, above httpx's 5 s keep-alive expiry) closes idle
  and slow connections; a body that stalls is logged and answered `408`.
  TCP keepalive as on 8765.
- Port 9000 binds right after 8765, both only after the engines and Telegram
  are up; a failed bind on either stops the daemon. Open to the LAN like 8765:
  no authentication, no TLS.

## Telegram calls

Design and verified facts: `docs/DESIGN_TELEGRAM_CALL.md`. `call` phones the
owner on Telegram from their second account (algolearn.ai); it works wherever
the phone has signal.

- **One daemon, one login:** the container owns the Telegram login (one
  connection per login). The TCP bind on 8765 is the single-instance guard (a
  second daemon gets `EADDRINUSE`) and compose's fixed `container_name`
  refuses a second container. Never run a second daemon with the same
  `TELEGRAM_SESSION` anywhere (the retired Mac launchd daemon included).
- **Credentials:** `TELEGRAM_API_ID`, `TELEGRAM_API_HASH`, `TELEGRAM_SESSION`
  (a Telethon StringSession: a login, treat as a password),
  `TELEGRAM_CALL_TARGET` in a git-ignored `.env` next to `docker-compose.yml`
  in the repo on the computer that deploys; compose passes it with
  `env_file`, and `.dockerignore` keeps it out of the image. Never commit it:
  this repo is public. `env_file` puts the values in the container config, so
  `docker inspect speak-telegram` shows them to anyone in patrick's docker
  group.
- **Verified quirks (macOS, 2026-10-09), all handled in `speak_telegram.py`:**
  the call must open on an audio FILE of at least 6 s (the greeting itself; 2 s
  gave all-zero audio both ways). **Owner decision (2026-10-12):** no padding;
  `call` rejects a greeting under 6 s before dialing and `write_opener` raises
  on one (`OPENER_MIN_SECONDS`), then the link switches to raw frames; the
  outgoing stream is fed every 10 ms (silence when idle) and never bursts
  (bursts sped audio up); ntgcalls aborts in its C++ destructors at exit, so
  the daemon always leaves via `os._exit`. These are not yet re-verified on
  Linux.
- **Streamed speech (owner decision 2026-10-16, `docs/DESIGN_STREAMED_SPEECH.md`):**
  speak plays Kokoro's first segment as soon as it is synthesized and
  synthesizes the rest while earlier segments play (`Engines.synthesize_segments`
  feeding `TelegramLink.play_watched`, which is `play_stream` plus the hold
  watch). This supersedes the earlier rule that
  inference never overlaps outgoing speech, for Kokoro only. If a segment is
  not ready when the previous one finishes, the pump sends silence (never a
  burst) and records `speak_underrun` with the gap; every streamed speech ends
  with a `speech_stream_done` summary (segments, first-audio latency,
  underruns, max pump lag, pump stalls). The call greeting is still
  synthesized whole (the 6 s opener file). In `listen`, Whisper still never
  overlaps outgoing audio: `record` returns only after the ear-open beep has
  played and the capture is done, nothing is queued until Whisper has
  finished, and only then is the chime queued. **Amended for the hold watch
  (owner decision 2026-10-16):** during speak, Silero runs on the owner's
  audio and Whisper runs on short hold/continue candidates (2 s or less,
  after a pause) while the persona's audio plays. CTranslate2 releases the GIL
  (measured on the Mac CPU only); `pump_stall` events on patrick are the
  live check.
- **Listen cues (owner decisions 2026-10-16):** the ear-open beep opens the
  ear; the rising two-note chime, played after Whisper has the transcript,
  says "you're done, I heard you" (it is distinct from the ear-open beep).
  The ear-closed low beep and the spoken "Processing." ack are removed
  (`SPEAK_ACK_TEXT` is not a setting). If the call has just dropped when the
  chime plays (the owner said their piece and hung up), the chime's failure
  is skipped (`_after_capture`) so `guard_call` raises "call ended" with the
  transcript; any chime failure on a connected call raises. Sessions answer
  first and hand real work to agents rather than going silent. (A no-speech
  timeout still plays its short low beep before `TimeoutError`.) The server
  instructions changed with this, so running sessions need `/mcp` reconnect
  (or a new session) to get them.
- **Call required:** a call that ends mid-operation raises `Telegram call
  ended during ...`; the rule of engagement is to call back. The call is
  shared by every session on every computer (they take turns through the
  daemon's lock).
- **Audio failure and diagnostics:** if the owner answers but the media
  connect fails (`ntgcalls.BaseRTCException`, e.g. `TelegramServerError`),
  `call` returns `answered but audio failed to connect: <error>`, the call is
  discarded, and the daemon exits 1 right after replying so compose's
  `restart: unless-stopped` restarts it with fresh ntgcalls state (no automatic
  retry; clients get "not reachable" until the models reload). Every call
  writes structured events to a 500-entry ring buffer and the log;
  `call_debug` / `set_call_debug` read it and toggle DEBUG logging.
  **Debugging is reactive (owner decision 2026-10-12):** `set_call_debug(True)`
  also turns on the audio-level events (`opener_written`, `audio_in`,
  `audio_out`, `audio_opener_out`); with debug off they are not recorded.
- **Native crashes:** the daemon enables Python's `faulthandler` (all threads) first
  thing in `main()`, so a SIGSEGV/SIGABRT in ntgcalls, torch or ctranslate2 writes
  every Python thread's stack to `docker logs speak-telegram` before the process
  dies by the signal (exit 139 for SIGSEGV; compose restarts it). The kernel log on
  patrick (`journalctl -k | grep segfault`) names the library, the offset and the
  native thread. Verified 2026-10-16 09:17: a SIGSEGV on ntgcalls' `ntg-work`
  thread, a null read in `IncomingAudioChannel`'s constructor (ntgcalls 3.0.2,
  offset 0x39d642), about 9 s into the 16 s greeting.
- **Quiet hours:** `call` refuses 22:00-07:00 (`SPEAK_QUIET_HOURS`, judged on
  the daemon's clock, `TZ=America/Los_Angeles`) unless
  `override_quiet_hours=True`, used only when the owner said they are up.
  A connected call is checked first and returns `already connected` at any
  hour (owner decision 2026-10-23); a refusal records `call_refused`.
  After `not answered`, wait an hour before calling again. `busy (the owner's
  Telegram reports a call in progress)` is its own outcome (Telegram discarded
  the call as busy; the phone did not ring), not `not answered`. After `busy`,
  call once more; if that one is busy or not answered too, wait an hour (owner
  decision 2026-10-15). Owner decision 2026-10-15: `call` waits until
  `REDIAL_SETTLE_SECONDS` (10 s, `speak_protocol.py`) have passed since the
  previous call ended, whatever the reason, then dials; the wait is recorded as
  the event `redial_settle_wait`. Verified 2026-10-15: a redial 1.4 s after `hang_up` got
  `CallBusy`, after which the owner's Telegram app rang every new call for
  half a second and dropped it without telling Telegram (the daemon saw a
  60 s ring-out) until the app was force-quit; a redial 10 s after
  `hang_up` worked. If the owner hangs
  up at any point, call back once right away; if that call is not answered,
  wait an hour (owner decision 2026-10-12).
- **Pins:** `telethon==1.45.0`, `py-tgcalls==3.0.0`, `ntgcalls==3.0.2`; the
  private-call API is undocumented, so upgrade only deliberately, with a
  re-test.

## Daemon configuration (compose `environment`, all required, no defaults)

| Variable | Value | Meaning |
|---|---|---|
| `SPEAK_KOKORO_MODEL` | `hexgrad/Kokoro-82M` | HF repo of the Kokoro model (`KPipeline` repo_id) |
| `SPEAK_VOICE` | `af_heart` | The voice the seeded Angie and every new persona start with. English voices only (`af_*` American female, `am_*` male, `bf_*`/`bm_*` British, spoken with American pronunciation), one of `ENGLISH_VOICES`; any other stops the daemon |
| `SPEAK_SPEED` | `1.0` | The speed the seeded Angie and every new persona start with, 0.5 to 2.0 |
| `SPEAK_PERSONAS_FILE` | `/var/lib/speak/personas.json` | The persona registry, on the `speak_state` volume. Its directory must exist at startup; a missing file is seeded with Angie |
| `SPEAK_WHISPER_MODEL` | `large-v3-turbo` | faster-whisper model name |
| `SPEAK_LANGUAGE` | `en` | Whisper language |
| `SPEAK_QUIET_HOURS` | `22:00-07:00` | No-call window, may wrap midnight. Validated at startup |
| `SPEAK_HOLD_PHRASE` | `hold on` | Holds the speech mid-sentence. Matched as the whole normalized utterance |
| `SPEAK_RESUME_PHRASE` | `continue` | Resumes a hold. Startup refuses a phrase with no letters or digits, or two phrases that normalize alike |
| `TZ` | `America/Los_Angeles` | The clock quiet hours use. Resolved with `zoneinfo` at startup (the image has tzdata); an unknown zone stops the daemon |

Models are cached in the `speak_models` volume
(`/root/.cache/huggingface`). Startup order: the personas directory is
checked, engines load (with the English voice list) and warm, the persona
file is opened (or seeded) and validated against the voices, then Silero
and Telegram connect, then ports 8765 and 9000 bind; so a refused connection
means "not ready or not running", never "running without models".

## Deployment on patrick

Deployed from this repo with the `patrick` docker context (owner decision
2026-10-15); there is no deploy directory on patrick. Compose runs here,
streams the build context to patrick's Docker, and reads the git-ignored
`.env` from this directory (`env_file`; `.dockerignore` keeps it out of the
image).

```
docker --context patrick compose up --build -d
```

Workspace rules apply: `up --build -d`, never `down` first, never copy files
into the container, never run code inside it. Every build, start, restart
and stop is an owner action, approved per change. Logs: `docker logs
speak-telegram` (Alloy also ships them to Loki and Grafana).

The personas live in the `speak_state` volume (`/var/lib/speak`), which
survives rebuilds like `speak_models`.

**Deploying a protocol change** (`PROTOCOL_VERSION` bumped):
1. Deploy the daemon (the command above, on the owner's word).
2. On every computer, in every session: `git pull` this repo (the client
   runs from it), then `/mcp` reconnect (or start a new session).
3. Until a session does step 2, its tool ops and `ping` fail with `speak
   client protocol N does not match daemon protocol M ...` (or `sent no
   protocol version` from a pre-Release-2 client). `status`, `call_debug`
   and `set_call_debug` still work; the new client's `status` shows
   `protocol mismatch: ...` in `call` when it is the daemon that is behind.

## Registration with Claude Code (every computer)

```
uv sync     # the client needs only mcp
claude mcp add --scope user speak -e SPEAK_DAEMON=192.168.86.31:8765 -- "$(pwd)/.venv/bin/algolearn-speak"
```

`SPEAK_DAEMON` goes in the `speak` entry's `env` block in `~/.claude.json`.
Check with `claude mcp list`. A new session, or `/mcp` then reconnect, picks
up a change; env is read once at process start.

## Development

```
uv sync --extra daemon                          # client + daemon dependencies (torch, Telegram)
.venv/bin/python -m unittest discover -s tests  # run all tests
.venv/bin/python scripts/drive_daemon.py --daemon 192.168.86.31:8765 status
```

`tests/test_speak_telegram.py` tests `TelegramLink` against a fake
py-tgcalls; `tests/test_hold.py` tests the hold watch, the hold loop and the
client's hold results the same way; `tests/test_daemon_wire.py` runs a real
daemon on a loopback port with a fake link (protocol version, identity, the
switch of session end to end, a hold that outlives its client);
`tests/test_session_switch.py` tests the switch rules in `SpeakService`;
`tests/test_floor.py` tests the floor in `AudioLock` (real threads, a short
idle time), in `SpeakService`, and over the wire;
`tests/test_call_checks.py` tests `call`'s pre-dial check order and the
`call_refused` event;
`tests/test_personas.py` tests the persona store (in a temporary
directory), the English voice list and the blends (torch on the CPU); the
persona voices and refusals are tested in `tests/test_session_switch.py`
and over the wire in `tests/test_daemon_wire.py`. The service-level tests
build the service with `locked_service` (`tests/test_hold.py`): each tool op
runs under the audio lock as its session, as the handler runs it. No test
contacts Telegram. The daemon itself needs CUDA and runs only in its
container.

## Rules

- No fallback engines. One STT, one TTS, one VAD. Failures raise.
- Log to stderr only. Stdout is the client's MCP transport.
