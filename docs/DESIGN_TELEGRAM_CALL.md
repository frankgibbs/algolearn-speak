# Design: the speak server calls the owner over Telegram

Written 2026-10-09. Status: approved by the owner 2026-10-09; build in progress.
**Owner decision** marks something the owner decided (2026-10-08/09).
**Decision** marks something I decided, with the reason. **Verified** marks
something observed in the 2026-10-09 spike (scratch scripts, not repo code).
Claims about repo code name the function or the commit they were read from.

## 1. What this is

The `call` tool places a Telegram voice call from the owner's second account
(**algolearn.ai**, id 8745217897) to their main account (**@frankgibbs35**).
Once the owner answers, `speak`, `listen` and `converse` run over the call
until it ends. It works wherever the phone has signal, and it works with the
Mac locked. Networking is outbound only, through Telegram's servers. No LAN
port, relay or VPN is involved.

**Owner decision:** this replaces FaceTime call mode
(`docs/DESIGN_FACETIME_CALL.md`, shipped 2026-10-08). FaceTime works only
while the Mac is unlocked, because a screen lock ends a FaceTime call and
blocks dialing. The FaceTime code is removed in the same change (section 11).

## 2. What the spike proved (Verified, 2026-10-09)

The spike used py-tgcalls 3.0.0 and ntgcalls 3.0.2 (Python 3.12, macOS 15.7.7,
Apple Silicon) with Telethon 1.45.0:

| Check | Result |
|---|---|
| algolearn.ai rings @frankgibbs35 | Answered in 4–11 s on every call |
| Answer detected | `play(..., CallConfig)` returns when the owner picks up |
| Owner hang-up detected | A `ChatUpdate` with status `DISCARDED_CALL` arrives |
| Mac → owner audio | Raw 48 kHz stereo s16le frames via `send_frame`, heard clearly, including the speak server's beeps |
| Owner → Mac audio | Raw frames via `record(RecordStream(audio=True))` and the `stream_frame` handler (`Direction.INCOMING`, `Device.MICROPHONE`). Whisper transcribed every turn word for word |
| Turn-taking | Speak, ear-open beep, Silero end-of-speech, low beep, Whisper, reply: three turns with no errors |
| Mac locked | `IOConsoleLocked` was true at dial time and at hang-up, and the call worked end to end |

Quirks found, each built into this design:

1. **The call must open on a real audio file.** Opening directly on raw
   frames (`ExternalMedia.AUDIO`), or on 1 s of silence switched after 1.5 s,
   gave all-zero audio in both directions, even though frames flowed at
   100/s. Opening on a spoken 5.8–6.2 s file, waiting for it to finish
   (+1 s), then switching to raw frames worked every time (six calls).
   Research found no public report of this. It is observed behavior, not
   explained.
2. **The outgoing stream must stay fed.** Gaps, or a stalled event loop
   followed by a burst of catch-up frames, played the next audio sped up. A
   10 ms pump that sends silence when idle and never bursts fixed it.
3. **ntgcalls aborts at interpreter exit.** Its C++ global destructors call
   `abort()` while its WebRTC thread is alive (macOS crash report,
   2026-10-09 15:08:55). The process must exit with `os._exit` after closing
   the call.
4. **Kokoro's eSpeak setup failed in a freshly created venv** with the same
   package versions (eSpeak looks for its data at a path baked in at build
   time). It works in the project venv. Not explained; the daemon below needs
   no Kokoro.

Not yet observed: what a **declined** call and an **unanswered** call look
like. Both are build-phase checks (section 12).

## 3. Architecture

**Decision: one always-on `speak-telegram` daemon per Mac, run by launchd.**
Reasons:
- A Telegram login (session) must be used by one connection at a time, but
  every Claude Code session runs its own speak server. Only one process can
  own the login and the call.
- The call is a long-lived object (a WebRTC connection, a 10 ms pump, the
  incoming stream). It has to outlive any single tool call.
- The repo already used this exact shape for the phone link: the
  `speak-phone` daemon (commit b719b7a, removed in 5c10120). It was one
  launchd process owning the connection, speak servers as clients over a
  Unix socket, VAD in the daemon, and "daemon not running = no call = Mac".
  This design reuses that pattern and its IPC framing rather than inventing
  a new one.

```
speak server (one per session)                 speak-telegram daemon (one per Mac)
  Kokoro TTS, Whisper, audio lock  ── Unix socket ──>  Telethon + py-tgcalls + ntgcalls
  routing, quiet hours, tools                          the call, 10 ms pump, Silero VAD
                                                         └── Telegram servers ──> owner's phone
```

- **Socket:** `~/.algolearn-speak/telegram.sock`, mode 0600.
- **Framing (from the phone daemon's `send_msg`/`recv_msg`):** a 4-byte
  big-endian header length, a UTF-8 JSON header, then `payload_len` raw bytes
  (float32 little-endian PCM). One request per connection, one reply:
  `{"ok": true, ...}` or `{"ok": false, "error": "..."}`. The client re-raises
  the error as `RuntimeError`.
- **Requests:**

| Request | Payload | Does |
|---|---|---|
| `ping` | none | liveness |
| `state` | none | returns `none`, `ringing`, `connected` or `ended` |
| `call` | the opening greeting, 24 kHz float32 | dials, plays the greeting file, switches to raw frames; replies `answered`, `not answered` or `busy` |
| `play` | PCM at a given rate | resamples to 48 kHz stereo, queues it on the pump, replies once it has gone out |
| `record` | an ear-open cue, plus VAD parameters | plays the cue, then runs Silero on incoming audio with the speak server's rules; replies with the 16 kHz capture, or `timeout` |
| `hang_up` | none | ends the call |

- **Where each piece runs:** the speak server keeps Kokoro, Whisper, the
  audio lock and all tool logic. The daemon only moves audio and runs VAD on
  the owner's incoming stream, as the phone daemon did. Kokoro stays out of
  the daemon (quirk 4).
- **The daemon's process:** an asyncio loop (Telethon and py-tgcalls are
  async) plus a socket server thread. Requests are handed to the loop with
  `run_coroutine_threadsafe`. VAD and resampling run in worker threads, so
  the pump's loop never stalls (quirk 2). On shutdown it closes the call,
  then `os._exit(0)` (quirk 3).
- **Credentials:** the daemon reads `TELEGRAM_API_ID`, `TELEGRAM_API_HASH`,
  `TELEGRAM_SESSION` and `TELEGRAM_CALL_TARGET` from the repo's git-ignored
  `.env` (**Owner decision**: credentials live in `.env`). A missing value
  stops the daemon at startup with the variable named. The speak server
  never reads them.

## 4. Call flow inside the daemon

All steps are the spike's working sequence:

1. `play(target, MediaStream(<greeting.wav>, AudioParameters(48000, 2)), CallConfig(timeout=60))`.
   It returns on answer, or raises after 60 s of ringing.
2. `record(target, RecordStream(audio=True, audio_parameters=AudioParameters(48000, 2)))`.
3. Wait for the greeting to finish, plus 1 s.
4. `play(target, MediaStream(ExternalMedia.AUDIO, AudioParameters(48000, 2)))`,
   then start the 10 ms pump. The pump sends one 1920-byte frame every 10 ms
   for the whole call: queued audio when there is some, silence otherwise. If
   it falls more than 50 ms behind, it resets its clock instead of bursting.
5. A `ChatUpdate` of `DISCARDED_CALL` (or `LEFT_CALL` / `BUSY_CALL`) marks
   the call `ended`. Any `play` or `record` in flight then fails with "call
   ended".

**Decision: `call` takes the opening sentence.** The model passes what it is
calling about, for example "Hi Frank, the backtest finished. Got a minute?".
That sentence becomes the greeting file (quirk 1), so the owner hears why
they were called the moment they answer. The spike's greeting was about 6 s.
Whether a shorter one also works is the first build check (section 12). If
it doesn't, the daemon pads the greeting file with trailing silence up to
the length that works.

## 5. The tools

- `call(greeting: str, override_quiet_hours: bool = False) -> str`, under the
  audio lock. It returns one of:
  - `answered`
  - `already connected`
  - `not answered (declined or no answer)`
  - `not called: quiet hours (22:00-07:00)`

  It raises if the daemon isn't running. No "Mac is locked" outcome is
  needed any more.
- `hang_up() -> str`: `hung up` or `no active call`.
- `speak` / `listen` / `converse`: unchanged signatures. On the `telegram`
  route, audio goes to the daemon instead of the audio worker. Every result
  ends with `(audio: telegram)` or `(audio: mac)`.
- `status()`: `{busy, current_tool, waiting, audio: "telegram" | "mac" | "error", call: <daemon state or "daemon not running">}`.

## 6. Routing

These carry over from FaceTime call mode unchanged, all **Owner decisions**
(2026-10-08) or decisions already made there (`_begin_call`, `_guard_call`,
`_call_expected` in `speak_server.py`):

- At the start of each `speak` / `listen` / `converse` (under the lock): the
  daemon reports `connected` → route `telegram`. A call is expected but gone
  → raise "call ended; call back". Otherwise → route `mac`.
- **Daemon not running = no call = Mac**, unless a call is expected; then it
  raises. This is the phone daemon's rule.
- A call that ends mid-operation raises, with any captured transcript in the
  message. It never switches to the Mac's speakers. **Owner decision (rule
  of engagement):** the session calls back.
- **Owner decision:** the call is shared by every session. They take turns
  through the cross-process audio lock.
- No voice-clone archiving on the call route: it is codec audio.

## 7. When to call

Unchanged **Owner decisions** from 2026-10-08:
- No calls 22:00–07:00 local, every day (`SPEAK_QUIET_HOURS`). Override only
  when the owner has said in the conversation that they are up.
- After `not answered`, the session may retry after one hour.
- The Mac does not read Focus state. During the day the phone's Focus
  decides whether it rings.
- A Telegram text channel for "anything waiting?" is a separate piece,
  outside this server.

## 8. Keeping the Mac available

**Decision: while a call is connected, the daemon holds
`caffeinate -i -w <daemon pid>`** (no idle system sleep), released when the
call ends. The display may sleep and the screen may lock, which Telegram
survives (Verified). System sleep would stop the process. This Mac has
system sleep disabled today (`pmset` shows `sleep 0`). The assertion keeps
calls safe if that setting ever changes. Unlike FaceTime, no display
assertion and no user-activity pulses are needed.

## 9. Dependencies

Added to `pyproject.toml`, pinned exactly because the private-call API is
undocumented and changed between releases (py-tgcalls 2.0.0 was pulled as
broken for private calls):
- `telethon==1.45.0`
- `py-tgcalls==3.0.0`
- `ntgcalls==3.0.2`

A console script `speak-telegram` and a launchd plist
`launchd/com.algolearn.speak-telegram.plist` are added, modeled on the
removed `speak-phone` ones. `ffmpeg` (Homebrew) is required for the greeting
file: py-tgcalls' `MediaStream` plays a file through an `ffmpeg` shell
command (read in `pytgcalls/types/stream/media_stream.py`).

## 10. Risk

- **Account:** Telegram watches accounts that use unofficial clients
  (`core.telegram.org/api/obtaining_api_id`). Calling the owner's own
  account, with consent, is low risk. A ban would cost only the algolearn.ai
  account, never the owner's main one.
- **Library:** private calls are undocumented in py-tgcalls, and the
  open-on-a-file quirk is unexplained. A future library version may change
  either, so the pins stay fixed until a deliberate upgrade with a re-test.

## 11. What is removed

The FaceTime path:
- `ftcall/` (Swift helper and build script)
- `speak_facetime.py`, `tests/test_speak_facetime.py`, `tests/fake_ftcall`,
  `tests/fake_caffeinate`
- `docs/DESIGN_FACETIME_CALL.md`
- the `SPEAK_CALL_NUMBER`, `SPEAK_CALL_OUTPUT_DEVICE` and
  `SPEAK_CALL_INPUT_DEVICE` settings, and the BlackHole routing in
  `speak_server.py`

The owner may then also, optionally:
- uninstall BlackHole (`brew uninstall --cask blackhole-2ch blackhole-16ch`)
- sign the Mac's FaceTime back into their own Apple ID
- remove `SPEAK_CALL_NUMBER` from the `speak` entry in `~/.claude.json`

`SPEAK_QUIET_HOURS`, the call-expected flag, the drop guard and the tool
descriptions' rule of engagement carry over.

## 12. Build plan and checks

1. **Short-greeting check (first, before other code):** with the spike
   scripts, open calls on 2 s and 4 s spoken greetings. Record the shortest
   that carries audio both ways; that sets the padding in section 4.
2. **Decline and no-answer check:** decline one call, and let one ring out.
   Record what py-tgcalls reports for each, then map both to
   `not answered`.
3. Build the daemon and client: socket server and framing (from the phone
   daemon), the call flow, pump, VAD, `caffeinate`, and clean shutdown.
4. Wire `speak_server.py`: telegram route, tools, status. Remove the
   FaceTime path.
5. Tests:
   - daemon and client against a fake py-tgcalls (scripted answer, decline,
     hang-up, incoming PCM), over a temp socket
   - routing, drop guard, quiet hours and lock tests carried over from the
     FaceTime suite
   - no test ever contacts Telegram
6. Independent code review, then owner approval to install the launchd
   daemon.
7. End-to-end on real calls: answer, three-turn converse, owner hang-up,
   decline, Mac locked, and two sessions sharing the call.
