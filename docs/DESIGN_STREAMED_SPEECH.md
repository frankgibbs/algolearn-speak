# Design: streamed speech (Release 1, faster speech start)

Status: built in the working tree, 2026-10-16. Not committed and not deployed.
Deploying `speak-telegram` is an owner action, approved per change.

How this document labels things:
- **Owner decision (date)**: something the owner decided. It is a choice, not
  a claim about how the system behaves.
- **Verified (file:line)**: read in the code, with the evidence named. Line
  numbers marked "at 8b598e8" refer to the code before this change; the rest
  refer to the working tree with this change.
- **Design choice**: a choice this document makes inside the owner's
  decision.
- **Unverified**: not yet observed; the live test in section 8 checks it.

## 1. What changes for the owner

**Owner decision (2026-10-16, on a call):** Release 1 is a faster start of
Angie's speech. Today `speak` and `converse` synthesize the whole reply with
Kokoro before any of it plays. From now on the daemon synthesizes the first
sentence, starts playing it into the call, and synthesizes the remaining
sentences while the earlier ones play. The owner hears the reply sooner: after
the first sentence's synthesis, not after the whole reply's.

**Owner decision (2026-10-16):** this replaces the rule "inference never
overlaps outgoing speech" (Decision 4.6 of `docs/DESIGN_REMOTE_DAEMON.md`)
for Kokoro during playback of speech only. Whisper still never runs while
audio plays. The call greeting is unchanged (section 6).

**Owner decision (2026-10-16):** the MCP tools and the wire protocol stay as
they are; this is a daemon-only change. Section 7 confirms that it holds.

**Owner decision (2026-10-16), after streamed speech made replies start
fast:** `listen` (and so `converse` and `call`'s listen) keeps only the
ear-open beep. The ear-closed low beep, the rising chime and the spoken ack
("Processing.", `SPEAK_ACK_TEXT`) are removed, and `SPEAK_ACK_TEXT` is no
longer a setting. Whisper still never overlaps outgoing speech: nothing is
queued to play after the capture, so the pump sends only silence while it
runs. Sections 2 and 7 describe the cues as they were when streamed
speech was built.

## 2. Where synthesis and playback happen today (Verified, at 8b598e8)

- `SpeakService._speak` strips markdown, calls `Engines.synthesize(text)`,
  then `link.play(audio)`, and returns `len(audio) / TTS_RATE`
  (`speak_telegram_daemon.py:340-348`).
- `Engines.synthesize` iterates Kokoro's `KPipeline` over the text split at
  sentence ends (`SENTENCE_SPLIT = r"(?<=[.!?])\s+"`), collects every chunk
  and concatenates them (`speak_engines.py:24`, `68-75`).
- `KPipeline.__call__` is a lazy generator: for each sentence it runs G2P,
  splits the phonemes into chunks of at most 510, runs the model on one chunk
  and yields it before touching the next (kokoro 0.9.4,
  `.venv/lib/python3.12/site-packages/kokoro/pipeline.py:366-386`). Sentence
  by sentence synthesis therefore needs no change to Kokoro, only to who
  consumes the generator and when.
- `TelegramLink.play` converts the audio to 48 kHz stereo s16le, appends it to
  the outgoing buffer on the link's asyncio loop, and waits until the pump
  has drained it (`speak_telegram.py:646-668`).
- The pump sends one 10 ms frame every 10 ms for the whole call: queued audio
  when there is some, a silence frame when the buffer is empty. If it falls
  more than `PUMP_MAX_LAG` (50 ms) behind it resets its clock instead of
  bursting and records a `pump_stall` event, at most `MAX_LAG_EVENTS` (20)
  per call (`speak_telegram.py:59`, `67`, `588-625`).
- `converse` runs `_speak` then `_listen` under one lock hold
  (`speak_telegram_daemon.py:376-384`). `_listen` runs Whisper only after the
  ear-closed cue has played (superseded 2026-10-16: that cue is removed, only the ear-open beep remains) (`speak_telegram_daemon.py:356-358`).
- (Superseded 2026-10-16: removed, only the ear-open beep remains.) The ack after every listen is a chime plus `_speak(SPEAK_ACK_TEXT)`
  (`speak_telegram_daemon.py:326-330`).
- `call` synthesizes the greeting in full, measures it against
  `OPENER_MIN_SECONDS`, and hands it to `link.call`, which writes it as the
  opener WAV file (`speak_telegram_daemon.py:398-410`,
  `speak_telegram.py:169-183`, `409-412`).

## 3. The new flow: producer and consumer

No new thread and no new queue. The two ends already exist:
- **Producer:** the request thread, which runs the tool op under the audio
  lock. It iterates Kokoro's generator, so each `next()` is one segment's
  synthesis, on the same thread that ran synthesis before.
- **Queue:** the link's outgoing byte buffer, owned by the asyncio loop.
- **Consumer:** the 10 ms pump, unchanged in how it paces frames.

Code (working tree):
- `Engines.synthesize_stream(text)` yields one float32 array per Kokoro
  segment, as Kokoro produces it, and raises `RuntimeError` if Kokoro yields
  nothing. `Engines.synthesize(text)` is now the concatenation of that stream
  and stays in use for the greeting and the warm-up (`speak_engines.py`).
- `TelegramLink.play_stream(segments, rate)` (`speak_telegram.py`):
  1. opens a stream on the loop (raises `CallEnded` with no connected call);
  2. for each segment the iterator yields, converts it to call PCM and
     appends it to the outgoing buffer at once, recording `playback_queued`
     with the segment number and `produce_ms` (the time the iterator took to
     yield it, which is Kokoro's synthesis time for that segment);
  3. after the last segment, waits until the buffer drains, as `play` does,
     and records one `speech_stream_done` summary (section 4);
  4. returns the total seconds of audio.
- `SpeakService._speak` hands `engines.synthesize_stream(text)` to
  `link.play_stream` (`speak_telegram_daemon.py`).

**Design choice:** playback starts as soon as the first segment is queued.
There is no pre-buffer of two segments; the point of the release is the
earliest start.

**Design choice: an error stops the speech.** If synthesis raises after some
segments were queued, or the call ends, the stream clears what is still
queued (nothing plays after the error), records `speech_stream_done` with
`aborted: true` and the error, and the exception propagates. This matches
`play`, which clears the buffer when it times out
(`speak_telegram.py:663-665` at 8b598e8). There is no retry and no fallback
to whole-text synthesis.

Timeouts: each queue step is one short loop round-trip
(`STREAM_STEP_TIMEOUT`, 5 s). The final wait for the drain has the same budget
as `play`: the queued audio's length plus 30 s.

## 4. Steady output and underruns

The pump is unchanged and still never bursts. Between segments two cases
arise:
- **Synthesis keeps ahead of playback** (the expected case: Kokoro-82M on the
  GPU produces audio much faster than real time, an estimate from
  `docs/DESIGN_REMOTE_DAEMON.md` section 8, not measured on patrick). The next
  segment is already in the buffer when the previous one ends; the frames are
  contiguous.
- **Synthesis falls behind** (a long sentence after a short one, a slow G2P
  call). The buffer empties, and the pump sends silence frames, exactly as it
  does between tool calls (`speak_telegram.py:604-608` at 8b598e8). The owner
  hears a pause. When the next segment arrives, the pump resumes with it on
  its normal 10 ms clock, so nothing is sped up.

**Design choice:** the gap is silence from the pump and nothing else. Each one
is recorded as a call event `speak_underrun` with `gap_ms` (the silence
frames the pump sent between the end of one segment and the start of the next,
times 10 ms) and `before_segment` (the segment that resumed speech). A gap is
counted only after the first segment has started playing, so the wait for the
first segment is the start latency, not an underrun. The silence after the last
segment is not an underrun. A segment's last partial frame is padded with
zeros, as `play` does today (under 10 ms, not counted).

`speech_stream_done` summarizes every streamed utterance: `segments`,
`audio_ms`, `first_audio_ms` (from the start of the stream until the first
segment was queued: the owner-facing start latency on top of the wire),
`underruns`, `underrun_ms`, `max_pump_lag_ms`, `pump_stalls`, and `aborted`
(with `error` when true). Like every call event it is written to the daemon
log at INFO and kept in the `call_debug` ring, with debug on or off.

## 5. Threading and the GIL

The risk the old rule guarded against is quirk 2 of
`docs/DESIGN_TELEGRAM_CALL.md` (observed on macOS, 2026-10-09): gaps, or a
stalled loop followed by a burst of catch-up frames, played the next audio
sped up. With streaming, Kokoro's Python work (misaki G2P and the torch call
setup) runs on the request thread while the pump plays speech on the loop
thread, and both need the GIL. CUDA kernels and their waits release it; G2P
does not.

What is measured, so the live test can tell whether this matters:
- The existing `pump_stall` event records every lag over 50 ms, with
  `sending_audio` saying whether speech was playing (Verified,
  `speak_telegram.py:619-623` at 8b598e8). It is capped at 20 per call.
- New: the pump keeps the largest lag behind its 10 ms schedule, and the
  number of stalls, over each playback. `speech_stream_done` reports them as
  `max_pump_lag_ms` and `pump_stalls`, uncapped, for every utterance. The
  `playback_done` event of a plain `play` (the cues, during which no model
  runs) carries `max_pump_lag_ms` too, which gives the baseline to compare
  against.

**Design choice:** Release 1 does not change the interpreter's GIL switch
interval or move synthesis to another process. The measurements above decide
whether a later release needs to.

## 6. The call greeting is unchanged

The greeting must be one audio file of at least `OPENER_MIN_SECONDS` (6 s),
measured on the synthesized audio before dialing (Verified,
`speak_telegram_daemon.py:398-410` at 8b598e8; quirk 1 of
`docs/DESIGN_TELEGRAM_CALL.md`). It is still synthesized in full with
`Engines.synthesize` before the dial. Nothing streams during the opener.

## 7. What the tools see

- **Protocol and client: unchanged.** No op, field, error kind or reply
  changes; `speak_server.py` and `speak_protocol.py` are not touched. Running
  Claude Code sessions need no reconnect.
- **`speak` result:** the spoken duration is the sum of the segments' lengths,
  the same number as the length of the concatenated audio before. Underrun
  silence is not counted.
- **`speak` budget:** the client allows `60 + len(text) / 5` s after
  `started` (Verified, `speak_server.py:125-126`). Streaming finishes no
  later than before: synthesis now overlaps playback, and an underrun only
  replaces synthesis time that used to come first.
- **`converse` and the ack:** `_speak` returns only after every segment has
  been synthesized and played, so `_listen`, the ear-open cue and Whisper
  start exactly as before. Whisper still never overlaps outgoing speech. The
  ack ("Processing.") is one segment and plays as before (ack removed 2026-10-16: only the ear-open beep remains).
- **A call that drops mid-speech** raises `Telegram call ended during
  speak` (or `converse`) as before, through `guard_call`.

## 8. Test plan

Unit tests (no CUDA, no Telegram; fakes as in the existing tests):
- `tests/test_speak_telegram.py`, against the fake py-tgcalls:
  - the first segment is sent before the iterator yields the second (playback
    starts before synthesis ends);
  - segments that arrive in time play contiguously, with no `speak_underrun`;
  - a producer slower than playback gives silence frames between the
    segments, one `speak_underrun` with its `gap_ms`, and audio frames that
    match the input exactly (nothing sped up or dropped);
  - a producer error clears the queue, records an aborted summary, and
    propagates; nothing plays after it;
  - a remote hang-up mid-stream raises `CallEnded`;
  - no connected call raises `CallEnded` before the iterator is touched;
  - an iterator that yields nothing raises;
  - the summary event's fields, and `max_pump_lag_ms` on `playback_done`.
- `tests/test_streamed_speech.py`, `SpeakService` with fake engines and a fake
  link:
  - `speak` streams `synthesize_stream` and returns the summed seconds;
  - `converse` listens only after the stream has finished;
  - the greeting still uses the whole-text `synthesize`;
  - empty text raises `ValueError` before anything is synthesized;
  - a call dropping mid-stream is reported as "call ended";
  - `Engines.synthesize_stream` and `synthesize` with a fake Kokoro pipeline.
- The whole suite: `.venv/bin/python -m unittest discover -s tests`.

Live test on patrick, by the owner, after the owner approves the deploy:
1. Deploy with `docker --context patrick compose up --build -d`.
2. Place a call and run several `converse` turns, one with a reply of one
   short sentence followed by a long one (the underrun case), and one of five
   or more sentences.
3. The owner listens for: an earlier start than before, no sped-up or choppy
   speech, and pauses between sentences no longer than natural.
4. Read `call_debug` and the daemon log: `speech_stream_done` for each turn
   (`first_audio_ms`, `underruns`, `max_pump_lag_ms`, `pump_stalls`), any
   `speak_underrun` and `pump_stall` events, and the cues' `playback_done`
   `max_pump_lag_ms` as the baseline.
5. The release passes when the owner hears no artifact. Measured stalls or
   underruns without an audible artifact are recorded here as the Linux
   baseline.

**Unverified until the live test:** the start latency on patrick, how often
underruns happen, and whether Kokoro's Python work during playback stalls the
pump on Linux.

## 9. Related documents

- `docs/DESIGN_REMOTE_DAEMON.md` Decision 4.6 stated "inference never
  overlaps outgoing speech". For Kokoro during speech it is superseded by the
  owner decision in section 1; for Whisper it stands.
- `docs/DESIGN_INTERRUPT.md` (a proposal, not approved or built) says in its
  section 6 that every sentence is synthesized before the first is queued.
  That was written against the old rule; if that design goes ahead, its
  overlap section and its resume point must be rewritten against streamed
  playback.
