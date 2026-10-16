# Design: holding and interrupting Angie by spoken command

Status: Release 2 approved by the owner on a call (2026-10-16) and built in
the working tree on 2026-10-16; not committed, not deployed. The section 8.1
echo gate and sections 8.2 to 8.4 have not been run. Release 3 is not built.
Every build, restart and stop of `speak-telegram` remains an owner action,
approved per change.

How this document labels things:
- **Owner decision (date)**: something the owner decided. It is a choice, not
  a claim about how the system behaves.
- **Verified (file:line)**: read in the code (section 1 names which tree) or
  observed live, with the evidence named.
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
on Release 2. Sections 3 to 5 and 7 were re-read on 2026-10-16 against the
working tree that holds Release 2 (on top of commit `9083a59`: Release 1's
streamed speech and the cue changes); their file:line references point at
that tree. Release 3's references (section 6) are **unverified, must read
before build** against the code as Release 2 leaves it.

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

## 3. The system before Release 2 (re-verified 2026-10-16)

Section 3 was first written against commit `8b598e8`. Release 1 (streamed
speech, `d2a772c`) and the cue changes (`92fd79e`, `9083a59`) moved the code
since; this section now describes the code Release 2 was built on, with
line numbers in the working tree that holds Release 2.

### 3.1 How a reply is played

- Before Release 2, `SpeakService._speak` turned markdown into plain text and
  handed `Engines.synthesize_stream(text)` to `link.play_stream`, so each
  Kokoro segment was queued as soon as it was synthesized (Release 1). Release
  2 replaced that call with the hold loop over `link.play_watched` (section
  5.2; now `speak_telegram_daemon.py:412-446`). `play_stream` is still in the
  link (`speak_telegram.py:879-911`) and is tested, but `_speak` no longer
  calls it.
- Kokoro runs with `split_pattern=SENTENCE_SPLIT` (`speak_engines.py:26`).
  Each Kokoro `Result` carries `text_index`, the index of its piece in
  `re.split(split_pattern, text.strip())` (kokoro 0.9.4,
  `kokoro/pipeline.py:366` in the project venv). A long sentence gives several
  results with the same index; that is what section 4.6 uses.
- `TelegramLink.play` (`speak_telegram.py:845-859`) converts audio to 48 kHz
  stereo s16le, appends it to `_outgoing`, and blocks until the pump has
  drained it. The cues and the chime still use it.
- The 10 ms pump (`speak_telegram.py:773-824`) sends one frame from
  `_outgoing` every 10 ms, or a silence frame when `_outgoing` is empty. When
  it falls more than `PUMP_MAX_LAG` = 50 ms behind (`speak_telegram.py:67`) it
  resets its clock instead of catching up and records a `pump_stall` event
  (`speak_telegram.py:818-823`, at most `MAX_LAG_EVENTS` per call). Cutting
  playback short therefore only means clearing `_outgoing` and setting
  `_drained`, which `_end_call` already does (`speak_telegram.py:561-562`).
- `converse` is `_speak` then `_listen` inside one `guard_call`, under one
  hold of the audio lock (`speak_telegram_daemon.py:499-515`).
- The `call` greeting is not played by the pump. It is the opener **file**
  that ntgcalls plays itself (`speak_telegram.py:594-616`), so it cannot be
  cut by clearing `_outgoing`.

### 3.2 Why the owner was not heard while Angie spoke

- py-tgcalls hands every incoming frame to `on_update` on the loop thread.
  Frames are always counted for the amplitude stats while connected; before
  Release 2 they were queued on `_incoming` only while `_listening` was true.
  Release 2 queues them while `_listening or _watching`
  (`speak_telegram.py:473`), each with its arrival time.
- `_listening` is set only inside `record`, after it has played the ear-open
  cue and emptied stale frames from the queue
  (`speak_telegram.py:1075-1090`).
- Silero VAD (`VADIterator`, threshold 0.5, 32 ms blocks at 16 kHz) runs on
  the request thread that called `record`, not on the loop (`_capture`,
  `speak_telegram.py:1092` on).

So before Release 2 nothing listened during playback. Incoming frames still
arrived and were measured (`audio_in` events, `_emit_audio_stats`,
`speak_telegram.py:714-727`), which the echo gate in section 8.1 uses.

### 3.3 The overlap rule before Release 2

The rule is **Decision 4.6 of `docs/DESIGN_REMOTE_DAEMON.md`**: Kokoro and
Whisper run in the same process as the pump's asyncio loop, and the Python
parts of their work (misaki and spaCy G2P, faster-whisper's decode loop)
compete with the pump for the GIL. Release 1 amended it for Kokoro (owner
decision 2026-10-16): speak synthesizes each segment while earlier ones play.
For Whisper the rule held: `_listen` (`speak_telegram_daemon.py:460-482`)
runs Whisper right after the capture with nothing queued to play, and queues
the rising chime only after Whisper has the transcript (owner decisions
2026-10-16: the ear-closed cue and the "Processing." ack are removed, the
chime stays). Section 7 amends the rule again, for Silero and Whisper during
watched playback.

The harm the rule guards against is **quirk 2** of
`docs/DESIGN_TELEGRAM_CALL.md`, observed on the Mac on 2026-10-09: "Gaps, or
a stalled event loop followed by a burst of catch-up frames, played the next
audio sped up." The never-burst pump fixed the burst half. Two facts limit
what is known:
- The current pump cannot burst; a stall becomes a gap (verified,
  `speak_telegram.py:818-823`). But quirk 2 names **gaps alone** as a cause
  of sped-up audio too, so a stall during speech is still a risk.
- The Linux re-test of the pump on patrick was skipped before the build
  (`docs/DESIGN_REMOTE_DAEMON.md:737`). How often the pump stalls on patrick
  while a model runs has not been measured.


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

How they are matched (owner decision 2026-10-16, built in Release 2):
- **An exact match, not "contains".** The normalized transcript of the whole
  candidate utterance must equal the phrase. "Hold on, honey", "continue
  with the dishes" or "pause it" never match. Normalizing means: lowercase,
  turn every run of characters that are not a letter, digit or apostrophe
  into one space, and trim (`normalize_phrase`,
  `speak_telegram_daemon.py:129-133`). "Hold on." and "hold-on" both become
  `hold on`. A variant from Whisper ("hold up", "holdin'") does not
  match. The owner says the phrase again. There are no alias lists and no
  fuzzy matching (owner rule: no fallbacks). Section 8.3 measures how often a
  phrase is missed.
- **No Whisper biasing.** Watch transcriptions do not use `initial_prompt` or
  `hotwords` (both exist in faster-whisper 1.2.1, `transcribe.py:279`,
  `296`). Whisper is known to hallucinate prompt text on noise, and here
  that would cause false triggers. They use `Engines.transcribe` unchanged
  (`speak_engines.py:115-118`), through `SpeakService.classify`
  (`speak_telegram_daemon.py:406-410`).
- **Each command acts only in the state where it means something.** "hold
  on" only while Angie is playing; "continue" only while she is held;
  "pause" (Release 3) while playing or held. A phrase heard in any other
  state is dropped. As built, the link's watcher takes the one command it
  `accept`s in its state (`hold` in `play_watched`, `resume` in `watch_for`);
  `classify` only names the phrase it heard.

The phrases are daemon settings, set in compose's `environment` block and
added to `REQUIRED_SPEAK_ENV` (`speak_telegram_daemon.py:78-79`), with the
same rule as the other `SPEAK_*` settings: required, no defaults. Release 2 adds
`SPEAK_HOLD_PHRASE` and `SPEAK_RESUME_PHRASE`; Release 3 adds
`SPEAK_PAUSE_PHRASE`. At startup the daemon refuses a missing phrase, a phrase
with no letters or digits once normalized, or two phrases that normalize to
the same text (`parse_phrases`, `speak_telegram_daemon.py:136-146`).

### 4.3 False triggers without a name

Requirement 1 says talking to someone in the room must not interrupt Angie.
"Hold on" and "continue" are things the owner may well say to his wife;
"pause" less so. With no name in the phrase, the protection is the shape of
the utterance and the state it arrives in.

What protects requirement 1 (owner decision 2026-10-16, built in Release 2):
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

### 4.4 Watch parameters (owner decision 2026-10-16; measured in section 8)

Defined once, in `speak_telegram.py:71-76` next to `PRE_ROLL_SECONDS`
(`WATCH_POLL_SECONDS`, 0.05 s, is how often the watch looks at the queue
while nothing else runs):

| Constant | Value | Why |
|---|---|---|
| `WATCH_SILENCE_SECONDS` | 0.4 | End of a command utterance. `listen` uses 2.0 s, which would add 1.6 s to every command |
| `WATCH_LEADING_SILENCE_SECONDS` | 0.8 | Non-speech required before a candidate (section 4.3, item 3) |
| `WATCH_MIN_UTTERANCE_SECONDS` | 0.3 | Shorter is a click or a cough |
| `WATCH_MAX_UTTERANCE_SECONDS` | 2.0 | Longer is conversation: dropped untranscribed. Each phrase takes under 1 s |
| `HOLD_MAX_SECONDS` | 300 | Total hold time allowed for one tool op; when it runs out the hold expires (section 5.2). Lives in `speak_protocol.py`, because the client's budget uses it |

The VAD threshold (0.5) and pre-roll (0.5 s) are the same as in `record`.

How the leading silence is counted, as built: the VAD blocks Silero calls
non-speech before the candidate, times 32 ms. After a previous utterance the
count also includes that utterance's end-of-speech silence (0.4 s), since
Silero reports the end only after it. The count starts at zero each time a
watch starts (each `play_watched` and each `watch_for`), so a phrase said in
the first 0.8 s after Angie starts or resumes, or after a hold starts, is
dropped as `no_leading_pause`.

What this means for the owner: pause briefly, say the phrase **by itself**,
then pause briefly. "Hold on, I need to check something" in one breath is a
single long utterance, and it does not match.

### 4.5 TelegramLink changes (`speak_telegram.py`, as built)

- **`_watching` flag.** `on_update` queues frames on `_incoming` when
  `_listening or _watching` (`speak_telegram.py:473`). Each item is now
  `(arrival time, frame)`; the arrival time of the last frame of a candidate
  is its end, which `command_latency_ms` is measured from. `_end_call` clears
  `_watching` as it clears `_listening`.
- **`_VadFeed`** (`speak_telegram.py:131-146`) turns raw frames into 16 kHz
  VAD blocks with Silero's start/end events. It is the helper shared by the
  listen capture (`_capture`, which keeps its own end conditions) and the
  watch.
- **`_Watcher`** (`speak_telegram.py:148-251`) holds one watch: a fresh
  Silero `VADIterator` (silence 0.4 s), the pre-roll, the leading-silence
  count, and the candidate bounds of section 4.4. A candidate that is too
  short, too long or without the leading pause is dropped without Whisper;
  otherwise it calls `classify(audio_16k) -> Classified(phrase, text)` and
  reports a hit only when `phrase` is the command it `accept`s. Audio past
  the maximum length is not kept.
- **`play_watched(segments, rate, classify, accept) -> PlayOutcome`**
  (`speak_telegram.py:964-1020`). It runs on the request thread.
  - `segments` yields `(tag, audio)`; the tag is the sentence index (section
    4.6). It opens the stream as `play_stream` does (`_stream_begin`), drains
    the stale queue, sets `_watching`, and queues each segment with
    `_stream_add` (non-blocking), remembering where each tag's bytes end.
    Between segments, and every 50 ms after the last one, it runs the
    watcher.
  - It returns when the pump has drained everything (`command` None), or on a
    hit. A hit cuts playback **on the loop thread** (`_stream_cut`,
    `speak_telegram.py:1029-1042`): it clears `_outgoing`, sets `_drained`
    and closes the stream, as `_end_call` does. The pump goes on sending
    silence, which honours quirk 2.
  - `PlayOutcome` holds `command`, `played_bytes` (bytes queued minus what
    was left in `_outgoing` at the cut), `cut_tag` (the tag whose bytes the
    cut fell in; None if everything queued so far had played) and `seconds`
    played.
  - It raises `CallEnded` on a drop, `TimeoutError` if the queue has not
    drained 30 s after its expected end, and re-raises a synthesis error
    after aborting the stream, as `play_stream` does.
- **`watch_for(timeout_s, classify, accept) -> command | TIMEOUT`**
  (`speak_telegram.py:1044-1073`). The same watcher with nothing playing.
  It raises `CallEnded` ("the Telegram call ended during the hold") on a drop.
- **`play`, `play_stream` and `record` are unchanged in behaviour.** The
  cues, the chime and the `listen` capture use `play` and `record`, and none
  of them can be interrupted.

### 4.6 Sentences

Verified 2026-10-16: kokoro 0.9.4 gives each `Result` a `text_index`, the
index of its piece in `re.split(split_pattern, text.strip())`, skipping
empty pieces (`kokoro/pipeline.py:366` in the project venv). With
`SENTENCE_SPLIT` = `(?<=[.!?])\s+` no piece is empty.

As built:
- `split_sentences(text)` (`speak_engines.py:31-38`) repeats that exact
  split, so the daemon has the text of sentence *i*; it raises on an empty
  piece instead of guessing an index shift.
- `Engines.synthesize_segments(text)` (`speak_engines.py:93-107`) yields
  `(text_index, audio)` per Kokoro result, and raises if a result has no
  `text_index`.
- A long sentence that Kokoro splits into several results is still **one**
  sentence: all its results carry the same index, and a cut in any of them
  resumes at the sentence's start. (This replaces the earlier draft's "each
  result counts as its own sentence".)
- The earlier check "the per-sentence texts, joined, give back the plain
  text" is not needed: the daemon never joins Kokoro's graphemes. The cut
  sentence and the unspoken text are taken from `split_sentences`, the same
  split Kokoro indexes into.
- A sentence with no phonemes yields no result. That is harmless: the cut
  maps to the index of the result that was playing.

### 4.7 Events and privacy

These go into `CallEventLog` (`speak_telegram.py:258-279`), as built:
- `watch_candidate`: `mode` (`playing` or `held`), `length_ms`,
  `leading_silence_ms`, `result` (`dropped` with `reason` `too_short`,
  `too_long` or `no_leading_pause`; or `transcribed` with `transcribe_ms`),
  `matched` (the phrase's command or null), `acted` (whether it was the
  command accepted in that state), and `text` only while debug is on.
- `playback_cut`: `command`, `played_ms`, `dropped_ms`, and
  `command_latency_ms`, the time from the end of the candidate utterance
  (the arrival of its last frame) to the cut.
- `speech_stream_done` gains `cut` (the command, or null).
- `hold_start` (`max_s`), and `hold_end` with `how` (`resume`, `expired`,
  `call_ended`, `client_left`, or `error` with `error` when classifying a
  candidate raised), `held_s`, and `command_latency_ms` on a resume.
- `hold_expire_hangup_failed` (`tool`, `error`): the hang-up at a hold
  expiry raised; the call may still be up.
- `first_incoming_frame` gains `during` (`listen` or `watch`): the first
  frame of the call may now arrive while a speech is watched.

Watching transcribes some of what the owner says to other people in the room
(short utterances after a pause). **Owner decision (2026-10-16), built:** the transcript text goes into
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

`_speak` is a loop over `play_watched` (`speak_telegram_daemon.py:412-446`,
as built):

```
pieces   = split_sentences(text)                       # what text_index indexes
segments = _Segments(engines.synthesize_segments(text))
start = 0; held = 0.0
loop:
  outcome = link.play_watched(segments.from_sentence(start), ..., classify, accept=hold)
  no command -> return Spoken(seconds, held)
  hold       -> cut = outcome.cut_tag, or the next sentence not yet queued
                (none left -> everything was spoken: return as completed)
                got = link.watch_for(HOLD_MAX_SECONDS - held, classify, accept=resume)
                held += time waited
                resume  -> start = cut; continue loop
                TIMEOUT -> return Spoken(..., interruption={hold_expired, cut, pieces[cut:]})
                           # speak/converse then hang up (below)
```

How the hold fits Release 1's streaming (resolved 2026-10-16 by reading the
code): synthesis stays lazy and interleaved with playback, as Release 1
made it. `_Segments` (`speak_telegram_daemon.py:284-322`) caches every
`(index, audio)` Kokoro has produced; `from_sentence(start)` first replays
the cached segments from that sentence on, then pulls new ones from the same
Kokoro generator. So a resume never re-synthesizes, and Kokoro is never
started twice for one reply. While held, Kokoro does not run (the generator
is pulled only by `play_watched`).

- **Resume point (owner decision 2026-10-16, built).** Angie starts again at the beginning of the
  sentence she was cut off in, not at the exact sample. By the time a command
  is recognized she has spoken past the point where the owner began it
  (section 8.4 measures by how much), and a restart mid-word sounds broken.
- **`classify`** is a `SpeakService` method. It calls
  `engines.transcribe(candidate)`, normalizes the text and looks it up in the
  phrases; the link's watcher applies the state gating of section 4.2.
- **Hold expiry.** **Owner decision (2026-10-16):** when a hold reaches its
  maximum (`HOLD_MAX_SECONDS`, 300 s, the holds in one op added up), the
  daemon **hangs up**: it ends the call. It does not give up the turn on an
  open line. Reason: the owner does not want another queued agent to pick up
  the call mid-hold; after the hang-up he ignores incoming calls, so other
  agents' messages stay queued. The held `speak`/`converse` returns an ok
  reply carrying `interruption: {command: "hold_expired", ...}` (no
  exception). Its tool result says the hold expired and the call was ended,
  and includes the text that was never spoken, starting from the beginning of
  the cut sentence, so the agent can pick it up later. `converse` does not
  open the ear after an expired hold: the listen half does not run. While a
  hold lasts, the op holds the audio lock, so other sessions wait behind it
  for up to 300 s; `status` already shows the holder as `current_tool`
  (`speak_telegram_daemon.py:935-936`). As built, `_hold_expired`
  (`speak_telegram_daemon.py:448-458`) calls `link.hang_up()`, clears the
  session's call expectation (so the next tool reports "no call", not "call
  ended during"), and returns the reply with `held_seconds` and
  `interruption`. If the hang-up itself fails, that error is the tool's
  result (no fallback), the event `hold_expire_hangup_failed` is recorded,
  and the expectation is cleared either way.
- **Client leaves during a hold.** **Owner decision (2026-10-16):** when the
  holding request's client goes away (its tool call was cancelled or its
  session closed), the hold ends at once, the lock is released to the next
  request, and the call is **not** hung up: it stays connected. As built,
  the client's `_run` shuts down the request's socket when anyio cancels the
  tool call; `TelegramLink.watch_for` polls a `client_left` predicate (the
  daemon handler's EOF check) and returns `CLIENT_LEFT` with `hold_end how:
  client_left`; `SpeakService._client_left_during_hold` raises, so the
  handler releases the lock. Nobody reads the reply.
- **The `guard_call` and call-expectation logic is unchanged**
  (`speak_telegram_daemon.py:348-378`). A drop during playback or a hold
  raises "Telegram call ended during ...", as today.
- **Not holdable:** the `call` greeting (the opener file, section 3.1), the
  cues, and anything inside a `listen`.

### 5.3 Wire and client (`speak_protocol.py`, `speak_server.py`)

There are no new ops. The `speak` and `converse` replies gain:

| Field | When | Content |
|---|---|---|
| `held_seconds` | always | Total time held (0.0 when never held) |
| `interruption` | after a hold expiry | `{"command": "hold_expired", "cut_index", "sentence_count", "cut_sentence", "unspoken"}`; `unspoken` starts at the cut sentence |

The client's budgets for `speak` and `converse` gain `HOLD_MAX_SECONDS`
(`_speak_budget`, `speak_server.py:127-130`). As built, `held_seconds` is
rounded to 0.1 s; the expired reply of `converse` has no `text`, and that of
`speak` keeps `seconds`. Tool results (`speak_server.py:144-169`):

```
converse/speak, resumed:  [held 42s, then resumed]\n<normal result>   (no line when held_seconds is 0)
converse/speak, expired:  [hold expired after 300s; the call was ended]
                          cut off in sentence 3 of 5: "<sentence>"
                          not spoken (from the cut sentence): "<rest>"
                          (audio: telegram)
```

Every result still ends with `AUDIO_NOTE`, which a permission rule in
`~/.claude/settings.json` depends on. The server instructions
(`_INSTRUCTIONS`, `speak_server.py:253-292`) gain two rules, and the `speak`
and `converse` docstrings gain a note on the two result forms:
- After a hold expiry, the daemon has ended the call and the reply was not
  finished. Do not assume the owner heard it, and do not call back right away:
  he ignores incoming calls after the hang-up. Pick up the unspoken text
  later; `converse` did not listen.
- Never speak "hold on" or "continue" as a sentence by itself. If the
  owner's phone does not cancel echo, Angie would hear herself (section 8.1).

### 5.4 Release 2 files touched

| File | Change |
|---|---|
| `speak_telegram.py` | `_watching`, timestamped `_incoming`, `_VadFeed`, `_Watcher`, `play_watched`, `_stream_cut`, `watch_for`, watch constants, events |
| `speak_engines.py` | `split_sentences`, `synthesize_segments` (section 4.6) |
| `speak_telegram_daemon.py` | `normalize_phrase`, `parse_phrases`, two phrase settings, `_Segments`, `classify`, `_speak` hold loop, `_hold_expired`, reply fields |
| `speak_protocol.py` | `HOLD_MAX_SECONDS`, command names, interruption field names |
| `speak_server.py` | Budgets, result formatting, docstrings, two instruction rules |
| `docker-compose.yml` | `SPEAK_HOLD_PHRASE`, `SPEAK_RESUME_PHRASE` |
| `CLAUDE.md`, `README.md`, `docs/DESIGN_REMOTE_DAEMON.md` 4.6 | Amended overlap rule (section 7); configuration table; tool descriptions |
| `tests/test_hold.py` (new), `tests/test_streamed_speech.py`, `tests/test_whisper_http.py` | Section 8.5, Release 2 items; the existing service tests gained the phrase settings and `held_seconds` |

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
  `_listen` with the listen parameters (ear-open beep, capture, Whisper; owner
  decisions 2026-10-16 removed the ear-closed cue and the ack, and kept the rising chime after Whisper). `converse` does **not** run a
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
`docs/DESIGN_REMOTE_DAEMON.md`, as Release 1 leaves it). The owner approved
the amendment with Release 2 (2026-10-16):
- **Kokoro**: Release 1's rule, unchanged by this document.
- **Silero VAD** runs during playback and while held (32 ms blocks, on the
  CPU, on the request thread).
- **Whisper** runs during playback and while held only on watch candidates:
  short utterances after a pause, never long speech.

As built, Kokoro, Silero and Whisper all run on the one request thread, so
they never run at the same moment as each other. One consequence: while
Kokoro synthesizes the next segment, the watch does not read the queue; the
frames wait (nothing is lost) and are judged once the segment is queued,
which adds up to one segment's synthesis time to `command_latency_ms`.

What keeps the risk down:
- Long utterances, and short ones without a leading pause, are never
  transcribed, so Whisper runs rarely, on clips of 2 s or less.
- **Verified locally (2026-10-16), not on patrick:** CTranslate2 4.8.3
  releases the GIL while it runs. Its compiled extension imports
  `PyEval_SaveThread`, and a probe on the Mac measured a 10 ms ticker thread
  (like the pump) while faster-whisper transcribed a 2 s clip three times on
  the CPU (fp32, `beam_size=1`, `vad_filter=False`, as `Engines.transcribe`):
  tick lag p50 2.0 ms, p99 2.9 ms, max 3.2 ms, the same as idle (p50 2.0,
  p99 2.5, max 4.1 ms). A pure-Python busy loop, the control, pushed it to
  p50 7.0, max 8.8 ms. So the pump competes only with faster-whisper's Python
  feature extraction and decode loop. Whether that holds on patrick's CUDA
  build is measured live by section 8.2.
- The pump cannot burst (section 3.3). A stall becomes a gap, and every gap
  is recorded as a `pump_stall` event, so section 8.2 measures any harm
  instead of guessing it.

There is no fallback. If section 8.2 fails, the build stops and the result
goes back to the owner. Release 3 changes nothing here.

## 8. Test plan

All code tests run in the local venv with the fake py-tgcalls
(`tests/test_speak_telegram.py`, `FakeCalls` from line 52). No test runs in the container (workspace rule).

### 8.1 Gate before Release 2's build: echo, with no code change

Whether the owner's phone and Telegram app remove Angie's voice from his
microphone (acoustic echo cancellation) is **unverified**. It matters two
ways: if echo comes through, the VAD hears Angie as speech, the leading
pause of section 4.3 never occurs, and commands are missed; and echo of a
command phrase spoken by itself would trigger it (the instruction rules in
sections 5.3 and 6.3 cover that).

Status 2026-10-16: **not run.** The owner approved building Release 2
before it; it must pass before the owner keeps Release 2 in use.

This gate uses what exists today: `set_call_debug(True)` turns on the
`audio_in` and `audio_out` events (1 s windows for the first 10 s after
answer, then 10 s; `speak_telegram.py:80-82`, `714-727`).
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
events during watched playback (each `speech_stream_done` also reports
`pump_stalls` and `max_pump_lag_ms` for its speech), and the owner hears no
choppy or sped-up audio. Any stall fails the test (section 7).

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
- **Release 2, hold expiry:** in the local venv against a scripted link and
  a fake clock (`tests/test_hold.py`, `TestHoldLoop`); on the real daemon the
  constant is not changed. The link is hung up, the result says the hold
  expired and the call was ended and carries the unspoken text from the cut
  sentence, and `converse` ran no listen. On a live call it takes the full
  300 s.
- **Release 2, a drop while held:** the owner hangs up while held. The op
  raises "Telegram call ended during converse".
- **Release 3, pause:** while playing and while held, the result names the
  right sentence and the unspoken rest, and holds the owner's words.
  `converse` ran exactly one listen.

### 8.5 Unit tests (local venv)

Release 2 (built: `tests/test_hold.py`, 45 tests; the whole suite, 177 tests,
passes):
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
3. Section 8.1 echo gate, on a live call, with no code change. **Not run;**
   the owner approved building before it (2026-10-16).
4. Read every "unverified" item against the code as Release 1 left it.
   **Done 2026-10-16** (sections 3 to 7).
5. Build Release 2; section 8.5 Release 2 tests green in the venv. **Done
   2026-10-16, in the working tree, not committed.**
6. Independent code review (code-reviewer).
7. The owner approves the deploy (`docker --context patrick compose up
   --build -d`).
8. Sections 8.2 to 8.4 for Release 2 on a live call. The owner uses Release 2
   for a couple of days.
9. The owner approves Release 3; build, Release 3 tests, review, owner-
   approved deploy, sections 8.2 to 8.4 for Release 3.
