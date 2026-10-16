# Design: holding and interrupting Angie by spoken command

Status: proposal, revised 2026-10-16 after a call with the owner. Not built.
Owner approval is needed before any code (System Modification Protocol,
rule 5). Every build, restart and stop of `speak-telegram` remains an owner
action, approved per change.

How this document labels things:
- **Owner decision (date)**: something the owner decided. It is a choice, not
  a claim about how the system behaves.
- **Verified (file:line)**: read in the code at commit `8b598e8` or observed
  live, with the evidence named.
- **Unverified, must read before build**: a claim the build must check first.
  Until then it is a question, not part of the design.
- **Proposal**: this document's recommendation. The owner confirms it or
  changes it before the build.

"Angie" below is the voice this document describes, whatever name the
calling session gives her. The command phrases contain no name (section 2).

## 1. Releases

**Owner decision (2026-10-16):** the work ships in three releases, one at a
time. The owner uses each release for a couple of days before the next one is
built.

| Release | What it adds | Where it is designed |
|---|---|---|
| 1 | Streamed speech: Kokoro synthesizes sentence by sentence during playback, so the first sentence plays sooner | `docs/DESIGN_STREAMED_SPEECH.md` (separate document; a prerequisite of this one) |
| 2 | "hold on" pauses Angie; "continue" resumes her | This document, section 5 |
| 3 | "pause" stops Angie and opens the ear | This document, section 6 |

Release 2 is built on the playback path as Release 1 leaves it, and Release 3
on Release 2. Section 3 describes the code at `8b598e8`, before Release 1.
Every file:line reference in this document is **unverified, must read before
build** against the code as the previous release leaves it.

## 2. Requirements

Owner decisions, from the calls of 2026-10-16:
1. While Angie speaks, the owner often talks to someone in the room (for
   example his wife) and answers them. That must not interrupt Angie.
2. Voice detection alone cannot tell who the owner is talking to, so
   interruption works by command phrases only.
3. **The command phrases contain no name.** The owner gives each calling
   agent a different female name and does not want to say those names around
   his wife.
4. **Release 2:** "hold on" pauses Angie mid-speech; "continue" resumes her
   from where she stopped.
5. **Release 3:** "pause" stops Angie's speech and opens the ear; the tool
   result tells Claude where she was cut off and what the owner said. The
   owner rejected "stop" as the word.
6. The standing owner rules apply: no fallback logic, no open questions in
   this document, decisions kept apart from verified facts.

Requirement 3 removes what protected requirement 1 in the previous draft of
this document (a name prefix). Section 4.3 states what protects it now, and
what risk remains.

## 3. The system today (verified at `8b598e8`)

### 3.1 How a reply is played

- `SpeakService._speak` turns markdown into plain text, synthesizes all of
  it with `Engines.synthesize`, then calls `link.play`
  (`speak_telegram_daemon.py:340-348`).
- `Engines.synthesize` runs Kokoro with `split_pattern=SENTENCE_SPLIT`
  (`speak_engines.py:24`) and joins the chunks into one array, dropping the
  sentence boundaries (`speak_engines.py:68-75`). Each Kokoro `Result` has a
  `graphemes` field holding that chunk's text (kokoro 0.9.4,
  `kokoro/pipeline.py:323-328` in the project venv).
- `TelegramLink.play` converts the audio to 48 kHz stereo s16le, appends it
  to `_outgoing`, and blocks the calling thread until the pump has drained it
  (`speak_telegram.py:646-668`).
- The 10 ms pump on the link's asyncio loop sends one frame from `_outgoing`
  every 10 ms, or a silence frame when `_outgoing` is empty. When it falls
  more than `PUMP_MAX_LAG` = 50 ms behind, it resets its clock instead of
  catching up, and records a `pump_stall` event (`speak_telegram.py:59`,
  `588-625`). Cutting playback short therefore only means clearing
  `_outgoing`; the pump keeps the stream fed with silence. `_end_call`
  already clears it this way (`speak_telegram.py:376-377`).
- `converse` is `_speak` then `_listen` inside one `guard_call`, under one
  hold of the audio lock (`speak_telegram_daemon.py:376-384`).
- The `call` greeting is not played by the pump. It is the opener **file**
  that ntgcalls plays itself (`speak_telegram.py:431`, `469`), so it cannot
  be cut by clearing `_outgoing`.

### 3.2 Why the owner is not heard while Angie speaks

- py-tgcalls hands every incoming frame to `on_update` on the loop thread.
  Frames are always counted for the amplitude stats while connected, but are
  put on the `_incoming` queue only while `_listening` is true
  (`speak_telegram.py:284-295`).
- `_listening` is set only inside `record`, which first plays the ear-open
  cue and empties any stale frames from the queue (`speak_telegram.py:672-687`).
- Silero VAD (`VADIterator`, threshold 0.5, 32 ms blocks at 16 kHz) runs on
  the request thread that called `record`, not on the loop
  (`speak_telegram.py:198-207`, `689-719`).

So nothing listens during playback. Incoming frames still arrive and are
measured (`audio_in` events, `speak_telegram.py:529-542`), which the echo
gate in section 8.1 uses.

### 3.3 Why inference does not overlap outgoing speech today

The rule is **Decision 4.6 of `docs/DESIGN_REMOTE_DAEMON.md`**: Kokoro and
Whisper run in the same process as the pump's asyncio loop, and the Python
parts of their work (misaki and spaCy G2P, faster-whisper's decode loop)
compete with the pump for the GIL. The sequence keeps them apart, so while
models run the pump sends only silence. The code follows it: `_speak`
synthesizes before playing (`speak_telegram_daemon.py:340-348`), and
`_listen` runs Whisper only after the ear-closed cue has played
(`speak_telegram_daemon.py:356-358`).

The harm the rule guards against is **quirk 2** of
`docs/DESIGN_TELEGRAM_CALL.md`, observed on the Mac on 2026-10-09: "Gaps, or
a stalled event loop followed by a burst of catch-up frames, played the next
audio sped up." The never-burst pump fixed the burst half. Two facts limit
what is known:
- The current pump cannot burst; a stall becomes a gap (verified,
  `speak_telegram.py:619-625`). But quirk 2 names **gaps alone** as a cause
  of sped-up audio too, so a stall during speech is still a risk.
- The Linux re-test of the pump on patrick was skipped before the build
  (`docs/DESIGN_REMOTE_DAEMON.md:737`). How often the pump stalls on patrick
  while a model runs has not been measured.

Release 1 changes this rule for Kokoro; its document states the amended
rule. Section 7 of this document amends it again, for Silero VAD and Whisper
only.

## 4. The shared mechanism: watching while Angie speaks

Release 2 builds this mechanism; Release 3 reuses it unchanged and adds one
phrase.

### 4.1 Overview

While a reply plays, the daemon **watches** the owner's audio:
1. Incoming frames go to the `_incoming` queue during playback as well.
2. Silero VAD cuts them into utterances, with a short end-of-speech silence.
3. Only a short utterance, preceded by a pause, is a candidate (section 4.3).
   Anything else is conversation, and it is dropped without being
   transcribed.
4. Whisper transcribes each candidate with the loaded engine. The text is
   normalized and compared for an **exact** match with the command phrases
   of the current release.
5. A match cuts playback and returns the command to `SpeakService`.

All of this runs inside the tool op that already holds the audio lock, on
the request thread that already runs `record`'s VAD. It adds no engine, no
thread, no lock and no module.

### 4.2 Command phrases

| Command | Phrase | Normalized form matched | Release | Status |
|---|---|---|---|---|
| hold | "hold on" | `hold on` | 2 | Owner decision (2026-10-16) |
| resume | "continue" | `continue` | 2 | Owner decision (2026-10-16) |
| pause | "pause" | `pause` | 3 | Owner decision (2026-10-16) |

How they are matched (Proposal):
- **An exact match, not "contains".** The normalized transcript of the whole
  candidate utterance must equal the phrase. "Hold on, honey", "continue
  with the dishes" or "pause it" never match. Normalizing means: lowercase,
  remove every character that is not a letter, digit, apostrophe or space,
  and collapse spaces. A variant from Whisper ("hold up", "holdin'") does not
  match. The owner says the phrase again. There are no alias lists and no
  fuzzy matching (owner rule: no fallbacks). Section 8.3 measures how often a
  phrase is missed.
- **No Whisper biasing.** Watch transcriptions do not use `initial_prompt` or
  `hotwords` (both exist in faster-whisper 1.2.1, `transcribe.py:279`,
  `296`). Whisper is known to hallucinate prompt text on noise, and here
  that would cause false triggers. They use `Engines.transcribe` unchanged
  (`speak_engines.py:77-80`).
- **Each command acts only in the state where it means something.** "hold
  on" only while Angie is playing; "continue" only while she is held;
  "pause" (Release 3) while playing or held. A phrase heard in any other
  state is dropped.

The phrases are daemon settings, set in compose's `environment` block and
added to `REQUIRED_SPEAK_ENV` (`speak_telegram_daemon.py:77-78`), with the
same rule as `SPEAK_ACK_TEXT`: required, no defaults. Release 2 adds
`SPEAK_HOLD_PHRASE` and `SPEAK_RESUME_PHRASE`; Release 3 adds
`SPEAK_PAUSE_PHRASE`. At startup the daemon refuses an empty phrase, or two
phrases that normalize to the same text.

### 4.3 False triggers without a name

Requirement 1 says talking to someone in the room must not interrupt Angie.
"Hold on" and "continue" are things the owner may well say to his wife;
"pause" less so. With no name in the phrase, the protection is the shape of
the utterance and the state it arrives in.

What protects requirement 1 (Proposal):
1. **The phrase must be the whole utterance.** The exact match of section
   4.2 means the phrase followed by anything ("hold on, let me get that"),
   or preceded by anything ("honey, hold on"), in one breath does not match.
   Most room talk is in sentences, and sentences never match.
2. **The utterance must be short.** Over `WATCH_MAX_UTTERANCE_SECONDS`
   (section 4.4) it is dropped untranscribed.
3. **The utterance must follow a pause.** A candidate counts only when at
   least `WATCH_LEADING_SILENCE_SECONDS` (section 4.4) of non-speech, as
   Silero sees it, precede it. In a back-and-forth with someone in the room,
   a "hold on" that comes right after the owner's own previous words is not
   a candidate. Combined with the end-of-speech silence, the phrase must
   stand alone, with a pause on both sides.
4. **State gating** (section 4.2). "continue" does nothing unless Angie is
   held; "hold on" does nothing while she is already held.
5. **Every false trigger is recoverable without losing words.** A false
   hold leaves the reply intact; "continue" resumes it from the cut
   sentence, and if he never says it, the hold expiry returns the unspoken
   text to Claude (section 5.2, section 5.3). A false pause (Release 3) returns the cut
   sentence and the unspoken rest to Claude with whatever the owner said
   next.

**Residual risk, stated plainly.** If the owner, while Angie is speaking,
pauses and then says just "hold on" to someone in the room, Angie stops; the
design cannot tell that apart from a command. Likewise, a bare "continue"
said to the room while Angie is held resumes her, and (Release 3) a bare
"pause" said to the room while she speaks cuts her off and opens the ear, so
Claude receives what he says next to the room as his answer. The cost is an
unwanted stop or resume, never lost content. Section 8.3 measures the false
trigger rate on a live call before the owner keeps a release in use.

### 4.4 Watch parameters (Proposal; measured in section 8)

Defined once, in `speak_telegram.py` next to `PRE_ROLL_SECONDS`:

| Constant | Value | Why |
|---|---|---|
| `WATCH_SILENCE_SECONDS` | 0.4 | End of a command utterance. `listen` uses 2.0 s, which would add 1.6 s to every command |
| `WATCH_LEADING_SILENCE_SECONDS` | 0.8 | Non-speech required before a candidate (section 4.3, item 3) |
| `WATCH_MIN_UTTERANCE_SECONDS` | 0.3 | Shorter is a click or a cough |
| `WATCH_MAX_UTTERANCE_SECONDS` | 2.0 | Longer is conversation: dropped untranscribed. Each phrase takes under 1 s |
| `HOLD_MAX_SECONDS` | 300 | Total hold time allowed for one tool op; when it runs out the hold expires (section 5.2). Lives in `speak_protocol.py`, because the client's budget uses it |

The VAD threshold (0.5) and pre-roll (0.5 s) are the same as in `record`.

What this means for the owner: pause briefly, say the phrase **by itself**,
then pause briefly. "Hold on, I need to check something" in one breath is a
single long utterance, and it does not match.

### 4.5 TelegramLink changes (`speak_telegram.py`)

- **`_watching` flag.** `on_update` puts frames on `_incoming` when
  `_listening or _watching` (today only `_listening`,
  `speak_telegram.py:290-295`).
- **`play_watched(audio, rate, segments, classify) -> PlayOutcome`**. It runs
  on the request thread, as `record` does.
  - It queues the PCM on the loop without blocking (today's `_play` split
    into "queue" and "wait"; its exact shape after Release 1 is unverified,
    must read before build).
  - It sets `_watching` and drains the stale queue, as `record` does at
    `speak_telegram.py:681-682`.
  - It loops: it reads frames, runs the VAD and builds candidate utterances
    using the section 4.4 bounds. For each candidate it calls
    `classify(audio_16k)`, which returns a command or `None`.
  - It returns when the pump has drained the queue (`completed`), or when
    `classify` returns a command. It first cuts playback **on the loop
    thread**: it clears `_outgoing` and sets `_drained`, as `_end_call`
    does. The pump goes on sending silence, which honours quirk 2.
  - `PlayOutcome` holds the command, or `completed`, and `played_bytes`:
    the bytes queued minus `len(_outgoing)` at the moment of the cut.
  - It raises `CallEnded` exactly as `_play` and `_capture` do.
- **`watch_for(commands, timeout_s, classify) -> str | TIMEOUT`**. The same
  loop with nothing playing: it returns the first command in `commands`, or
  `TIMEOUT`. It is used while a reply is held.
- **`play` and `record` are unchanged.** The cues, the "Processing." ack and
  the `listen` capture still use them, and none of them can be interrupted.

The VAD-and-utterance loop is shared with `_capture`
(`speak_telegram.py:689-719`) as one helper that yields utterances. The
listen path keeps its own end conditions (`start_timeout_seconds`,
`max_seconds`).

### 4.6 Sentences

A cut must map to a sentence, so `_speak` needs the text and the audio of
each Kokoro result. Release 1 streams Kokoro sentence by sentence; how it
exposes each sentence's text and audio is in its document. This design uses
that per-sentence form and adds nothing to `speak_engines.py` of its own.
Unverified, must read before build: that the per-sentence texts Release 1
keeps, joined, give back the plain text with nothing lost or reordered.
`KPipeline` can split a long sentence into several results; each counts as
its own "sentence" here.

### 4.7 Events and privacy

These go into `CallEventLog` (`speak_telegram.py:81-101`):
- `watch_candidate`: the utterance's length, the leading silence, whether it
  was transcribed or dropped (too long, too short, no leading pause), and
  whether it matched.
- `playback_cut`: the command and the milliseconds played.
- `hold_start`, and `hold_end` with how it ended and `held_s`.
- `command_latency_ms`: the time from the end of the candidate utterance to
  the cut.

Watching transcribes some of what the owner says to other people in the room
(short utterances after a pause). **Proposal:** the transcript text goes into
`watch_candidate` only while `set_call_debug(True)` is on, the same switch
as the audio-level events. With debug off, only the length and the match
flag are logged. Logs go to Loki through Alloy (CLAUDE.md). Unmatched text is
never returned to Claude.

## 5. Release 2: "hold on" and "continue"

### 5.1 What changes for the owner

| Owner says | What Angie does | What Claude gets back |
|---|---|---|
| "hold on" (while she speaks) | Stops talking at once. Waits for "continue". | Nothing yet; the tool is still running |
| "continue" (while held) | Starts again at the beginning of the sentence she was cut off in, and finishes the reply | The normal result, with one line saying how long she was held |
| (nothing, for 300 s of hold) | The hold expires: the daemon hangs up | The tool result says the hold expired and the call was ended, and holds the text that was never spoken |

Anything else the owner says while Angie speaks or is held changes nothing.

### 5.2 SpeakService changes (`speak_telegram_daemon.py`)

`_speak` becomes a loop over `play_watched`:

```
sentences = per-sentence form from Release 1
start = 0; held = 0.0
loop:
  outcome = link.play_watched(audio of sentences[start:], ..., classify)
  completed -> return SpokenResult(held_seconds=held)
  hold      -> cut_index = sentence at played_bytes
               wait = link.watch_for({resume}, HOLD_MAX_SECONDS - held, classify)
               held += time waited
               resume  -> start = cut_index; continue loop
               TIMEOUT -> return HoldExpired(cut=sentences[cut_index], unspoken=rest)
                          # the op ends here: the daemon hangs up
```

How the loop meets Release 1's streaming (synthesis of later sentences while
earlier ones play) is unverified, must read before build: the build reads
Release 1's `_speak` and fits the hold into it, keeping Release 1's rule for
when Kokoro may run.

- **Resume point (Proposal).** Angie starts again at the beginning of the
  sentence she was cut off in, not at the exact sample. By the time a command
  is recognized she has spoken past the point where the owner began it
  (section 8.4 measures by how much), and a restart mid-word sounds broken.
- **`classify`** is a `SpeakService` method. It calls
  `engines.transcribe(candidate)`, normalizes the text, compares it with the
  phrases, and applies the state gating of section 4.2.
- **Hold expiry.** **Owner decision (2026-10-16):** when a hold reaches its
  maximum (`HOLD_MAX_SECONDS`, 300 s, the holds in one op added up), the
  daemon **hangs up**: it ends the call. It does not give up the turn on an
  open line. Reason: the owner does not want another queued agent to pick up
  the call mid-hold; after the hang-up he ignores incoming calls, so other
  agents' messages stay queued. The held `speak`/`converse` ends with
  `HoldExpired`. Its tool result says the hold expired and the call was ended,
  and includes the text that was never spoken, starting from the beginning of
  the cut sentence, so the agent can pick it up later. `converse` does not
  open the ear after an expired hold: the listen half does not run. While a
  hold lasts, the op holds the audio lock, so other sessions wait behind it
  for up to 300 s; `status` already shows the holder as `current_tool`
  (`speak_telegram_daemon.py:803-805`, verified at `8b598e8`).
- **The `guard_call` and call-expectation logic is unchanged**
  (`speak_telegram_daemon.py:276-306`). A drop during playback or a hold
  raises "Telegram call ended during ...", as today.
- **Not holdable:** the `call` greeting (the opener file, section 3.1), the
  cues, the ack, and anything inside a `listen`.

### 5.3 Wire and client (`speak_protocol.py`, `speak_server.py`)

There are no new ops. The `speak` and `converse` replies gain:

| Field | When | Content |
|---|---|---|
| `held_seconds` | always | Total time held (0.0 when never held) |
| `interruption` | after a hold expiry | `{"command": "hold_expired", "cut_index", "sentence_count", "cut_sentence", "unspoken"}`; `unspoken` starts at the cut sentence |

The client's budgets for `speak` and `converse` gain `HOLD_MAX_SECONDS`
(`speak_server.py:125-130`). Tool results:

```
converse/speak, resumed:  [held 42s, then resumed]\n<normal result>
converse/speak, expired:  [hold expired after 300s; the call was ended]
                          cut off in sentence 3 of 5: "<sentence>"
                          not spoken (from the cut sentence): "<rest>"
                          (audio: telegram)
```

Every result still ends with `AUDIO_NOTE`, which a permission rule in
`~/.claude/settings.json` depends on. The server instructions
(`speak_server.py:221-249`) gain two rules:
- After a hold expiry, the daemon has ended the call and the reply was not
  finished. Do not assume the owner heard it, and do not call back right away:
  he ignores incoming calls after the hang-up. Pick up the unspoken text
  later; `converse` did not listen.
- Never speak "hold on" or "continue" as a sentence by itself. If the
  owner's phone does not cancel echo, Angie would hear herself (section 8.1).

### 5.4 Release 2 files touched

| File | Change |
|---|---|
| `speak_telegram.py` | `_watching`, `play_watched`, `watch_for`, a shared utterance helper, watch constants, events |
| `speak_telegram_daemon.py` | `_speak` hold loop, `classify`, two phrase settings, reply fields |
| `speak_protocol.py` | `HOLD_MAX_SECONDS`, interruption field names |
| `speak_server.py` | Budgets, result formatting, docstrings, two instruction rules |
| `docker-compose.yml` | `SPEAK_HOLD_PHRASE`, `SPEAK_RESUME_PHRASE` |
| `CLAUDE.md`, `docs/DESIGN_REMOTE_DAEMON.md` 4.6 | Amended overlap rule (section 7); configuration table; tool descriptions |
| `tests/` | Section 8.5, Release 2 items |

## 6. Release 3: "pause"

### 6.1 What changes for the owner

| Owner says | What Angie does | What Claude gets back |
|---|---|---|
| "pause" (while she speaks or is held) | Stops talking, plays the ear-open beep and records the owner's next words, as `listen` does | Where she was cut off, what she never said, and the owner's words |

"hold on" and "continue" work as in Release 2.

### 6.2 SpeakService changes

The Release 2 loop gains one branch in each state:

```
  pause (while playing) -> return Interrupted(cut=sentence at played_bytes, unspoken=rest)
  hold:  wait = link.watch_for({resume, pause}, ...)
         pause (while held) -> return Interrupted(cut=sentences[cut_index], unspoken=rest)
```

- **Pause then listens.** On `Interrupted`, `speak` and `converse` both call
  `_listen` with the listen parameters (ear-open beep, capture, ear-closed
  cue, Whisper, chime and "Processing."). `converse` does **not** run a
  second listen after that one. For `speak`, which has no listen parameters
  today, the client sends `listen`'s defaults (120 s, 2.0 s, 45 s). Those
  defaults move into `speak_protocol.py` as constants, so they are defined
  once.
- A drop during the capture after a pause raises "Telegram call ended during
  ...", as today.

### 6.3 Wire and client

`interruption.command` gains the value `"pause"`. The `text` field (the
owner's words) is present for `converse` always except after a hold expiry,
and for `speak` after a pause. `speak`'s client budget gains a listen budget
(`speak_server.py:125-130`). Tool result:

```
converse/speak, pause:    [interrupted: the owner said pause]
                          cut off in sentence 3 of 5: "<sentence>"
                          not spoken: "<rest>"
                          owner said: <owner's words>
                          (audio: telegram)
```

The server instructions gain:
- After a pause, answer what the owner said. Do not replay the unspoken part
  unless he asks.
- The "never speak a command phrase by itself" rule covers "pause" too.

### 6.4 Release 3 files touched

| File | Change |
|---|---|
| `speak_telegram_daemon.py` | `pause` branches, `SPEAK_PAUSE_PHRASE`, listen after a pause |
| `speak_protocol.py` | Listen default constants, the `pause` command value |
| `speak_server.py` | `speak` listen parameters and budget, result format, instructions |
| `docker-compose.yml` | `SPEAK_PAUSE_PHRASE` |
| `CLAUDE.md` | Tool descriptions, configuration table |
| `tests/` | Section 8.5, Release 3 items |

## 7. Overlap with outgoing speech: the rule changes

Release 2 amends the overlap rule (Decision 4.6 of
`docs/DESIGN_REMOTE_DAEMON.md`, as Release 1 leaves it). The amendment is
part of what the owner approves:
- **Kokoro**: Release 1's rule, unchanged by this document.
- **Silero VAD** runs during playback (32 ms blocks, on the CPU, on the
  request thread).
- **Whisper** runs during playback only on watch candidates: short
  utterances after a pause, never long speech.

What keeps the risk down:
- Long utterances, and short ones without a leading pause, are never
  transcribed, so Whisper runs rarely, on clips of 2 s or less.
- Unverified, must read before build: that CTranslate2 4.8.3 releases the GIL
  during `generate`/`encode` (in its compiled extension). If it does, the
  pump competes only with faster-whisper's Python feature extraction and
  decode loop.
- The pump cannot burst (section 3.3). A stall becomes a gap, and every gap
  is recorded as a `pump_stall` event, so section 8.2 measures any harm
  instead of guessing it.

There is no fallback. If section 8.2 fails, the build stops and the result
goes back to the owner. Release 3 changes nothing here.

## 8. Test plan

All code tests run in the local venv with the fake py-tgcalls
(`tests/test_speak_telegram.py:52-104`, `FakeCalls`; frames are injected as
at line 165). No test runs in the container (workspace rule).

### 8.1 Gate before Release 2's build: echo, with no code change

Whether the owner's phone and Telegram app remove Angie's voice from his
microphone (acoustic echo cancellation) is **unverified**. It matters two
ways: if echo comes through, the VAD hears Angie as speech, the leading
pause of section 4.3 never occurs, and commands are missed; and echo of a
command phrase spoken by itself would trigger it (the instruction rules in
sections 5.3 and 6.3 cover that).

This gate uses what exists today: `set_call_debug(True)` turns on the
`audio_in` and `audio_out` events (1 s windows for the first 10 s after
answer, then 10 s; `speak_telegram.py:64-66`, `529-542`).
1. **Greeting window.** Place a call with a greeting of about 12 s. The owner
   answers with the phone at his ear and **stays silent** until the beep. The
   1 s `audio_in` windows in phase `opener` cover the greeting. Then take a
   baseline: `listen` with the owner silent, and read the `audio_in` windows
   after the ear-open beep.
2. **Long speech.** `speak` about 40 s of text with the owner silent. Compare
   the `audio_in` windows that fall entirely inside `audio_out` windows with
   `audio_frames` > 0 against the baseline.
3. Repeat steps 1 and 2 on **speakerphone**, and with any headset the owner
   uses on calls.
4. **Pass:** on every setup the owner uses, `audio_in` `rms_dbfs` while Angie
   speaks is within 3 dB of the silent baseline. A failing setup is reported
   to the owner, and Release 2 is not built as written; section 9's envelope
   technique is the documented alternative for that report.

### 8.2 After each release's build: pump health while watching

With debug on, run five `converse` turns, each with a reply of about 60 s.
During each the owner talks over Angie in short remarks with pauses, which
creates many watch candidates and Whisper runs. **Pass:** zero `pump_stall`
events during watched playback, and the owner hears no choppy or sped-up
audio. Any stall fails the test (section 7).

### 8.3 After each release's build: recognition and false triggers

On a call, while Angie reads a long text:
- The owner says each phrase of the release 10 times, alone, after a pause.
  **Pass:** at least 9 of 10 match for each phrase (`watch_candidate` and
  `playback_cut` events).
- The owner says 30 room remarks: "hold on, honey", "okay, continue with
  that", "can you pause the TV", "hold on" right after another sentence with
  no pause, "continue" while Angie is not held, and sentences longer than
  2 s. **Pass:** zero matches.
- The residual risk of section 4.3 is measured, not judged pass/fail: the
  owner says a bare "hold on" (and in Release 3 a bare "pause") after a
  pause, as he would to the room, 5 times; each is expected to trigger. The
  result goes to the owner with the release.
- With the owner silent for 5 minutes of Angie's speech: zero
  `watch_candidate` events. This is the echo check at the VAD level, on
  patrick.

### 8.4 After each release's build: latency and the result

- **Latency:** `command_latency_ms` (end of utterance to cut), plus
  `WATCH_SILENCE_SECONDS`. **Pass** if end-of-phrase to silence is 1.5 s or
  less, median over the 8.3 runs.
- **Release 2, resume:** hold, wait 30 s, continue. Angie restarts at the cut
  sentence and finishes, and the result says `[held 30s, then resumed]`.
- **Release 2, hold expiry:** with `HOLD_MAX_SECONDS` = 20 in the local venv
  against the fake link only; on the real daemon the constant is not changed.
  The link is hung up, the result says the hold expired and the call was
  ended and carries the unspoken text from the cut sentence, and `converse`
  ran no listen.
- **Release 2, a drop while held:** the owner hangs up while held. The op
  raises "Telegram call ended during converse".
- **Release 3, pause:** while playing and while held, the result names the
  right sentence and the unspoken rest, and holds the owner's words.
  `converse` ran exactly one listen.

### 8.5 Unit tests (local venv)

Release 2:
- Normalization and exact matching, including near-misses and punctuation.
- Candidate bounds: dropped untranscribed when too short, too long, or
  without the leading pause.
- State gating: "continue" ignored while playing, "hold on" ignored while
  held.
- `play_watched`: cut position, `_outgoing` cleared, pump still sending
  silence, `CallEnded` on a drop.
- Mapping `played_bytes` to a sentence across segment boundaries.
- Hold, resume and expiry transitions in `SpeakService` with a stub
  `classify`; on expiry the link is hung up, the result says so and
  carries the unspoken text from the cut sentence, and `converse` does not
  listen.
- Startup refuses missing, empty or duplicate phrases.
- The client's result formats and budgets, and `import speak_server` still
  imports no numpy, torch or ntgcalls.

Release 3:
- Pause from playing and from held, then exactly one listen.
- Startup refuses a pause phrase that duplicates another.
- `speak`'s listen defaults come from `speak_protocol.py`.

## 9. Reference: foovox barge-in

Read in `redfoo22/foovox` (MIT), commit `3cd8bf9`, files `public/echo.js`,
`public/app.js:355-430`, `public/app.js:510-535` and `README.md` ("Hearing
you over itself").

- foovox interrupts on **any** speech, using energy. It has no command words.
  Its phrase-based intents (`public/intent.js`) are for status and dispatch,
  not for barge-in.
- Its echo guard keeps the loudness envelope of the exact samples it plays,
  measures the speaker-to-microphone delay by correlation, and learns the
  echo coupling as a running **minimum** ratio (a person can only add
  energy). It demands that the microphone exceed the predicted echo, and in
  the gaps between words it uses full sensitivity, because "energy heard in
  a gap cannot be echo" (`echo.js:28-31`, `149-156`, `252-263`).
- The first 700 ms of every reply is a measuring period with barge-in turned
  off (`HOLD_OFF`, `echo.js:43`). It stops playback locally first and only
  then tells the server (`app.js:510-535`).
- Its README says transcript matching for echo rejection costs about a
  second per decision.

Not adopted: it answers "is someone talking", and requirement 1 rules that
question out. Its envelope technique is the documented alternative if the
section 8.1 gate finds echo, because the daemon also holds the exact PCM it
plays (`_outgoing`).

## 10. What does not change

- One STT, one TTS, one VAD (CLAUDE.md rule). No keyword-spotting engine is
  added. A wake-word engine (openWakeWord, Porcupine) would be a second
  recognizer and would need custom training for these phrases.
- The FIFO audio lock, the port-9000 whisper API, call etiquette, quiet
  hours, the opener, the cues and the call expectation.
- `listen`, and the listen half of `converse` after a normal end of speech.

## 11. Order of work

1. Release 1 ships (`docs/DESIGN_STREAMED_SPEECH.md`) and the owner uses it
   for a couple of days.
2. The owner approves Release 2 of this design: the matching and
   false-trigger rules (4.2, 4.3), the parameters (4.4), the resume point
   (5.2), the transcript logging (4.7) and the amended overlap rule (7).
3. Section 8.1 echo gate, on a live call, with no code change.
4. Read every "unverified" item against the code as Release 1 left it.
5. Build Release 2; section 8.5 Release 2 tests green in the venv.
6. Independent code review (code-reviewer).
7. The owner approves the deploy (`docker --context patrick compose up
   --build -d`).
8. Sections 8.2 to 8.4 for Release 2 on a live call. The owner uses Release 2
   for a couple of days.
9. The owner approves Release 3; build, Release 3 tests, review, owner-
   approved deploy, sections 8.2 to 8.4 for Release 3.
