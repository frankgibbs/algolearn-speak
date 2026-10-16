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

**Open to the LAN by design (owner decision).** Port 8765 has no
authentication and no TLS: any LAN computer with the thin client can call the
owner and hold a spoken conversation.

## Tools

| Tool | What it does |
|---|---|
| `speak(text)` | The daemon synthesizes `text` with Kokoro-82M one segment (a sentence) at a time and plays each into the connected call as soon as it is made, synthesizing the next while earlier ones play (owner decision 2026-10-16). Returns once all of it has played, with the spoken duration. |
| `listen(max_seconds=120, silence_seconds=2.0, start_timeout_seconds=45)` | High beep (the ear is open), then records the owner's call audio until Silero VAD sees `silence_seconds` of quiet after speech (or `max_seconds` of speech), then Whisper large-v3-turbo transcribes. Once text is in hand, a rising two-note chime tells the owner their words were captured (owner decision 2026-10-16). No ear-closed beep and no spoken acknowledgement. Raises `TimeoutError` if nobody speaks within `start_timeout_seconds`. |
| `converse(text, ...)` | `speak` then `listen` under one lock hold. Returns the user's words. |
| `call(greeting, override_quiet_hours=False, max_seconds=120, silence_seconds=2.0, start_timeout_seconds=45)` | Telegram voice call from the algolearn.ai account to the owner (@frankgibbs35); `greeting` is the whole opening, spoken the moment they answer: who is calling, why, and the context, ending with a question or invitation. It must be at least 6 s when spoken (measured on the synthesized audio); a shorter one raises `ValueError` BEFORE dialing (phone never rings; owner decision 2026-10-12) telling the model to add the context of why it is calling, or, on a fresh conversation with nothing to report, to open with something playful (a joke, something funny or flirty). Never padded. Once answered, `call` immediately does the `listen` (ear-open beep, VAD, Whisper, rising chime) under the same lock hold and returns the owner's words; listen parameters and defaults match `converse`, and a listen timeout or a drop raises as in `listen` (the call stays connected on a timeout). Returns `answered\n<their words>\n(audio: telegram)`, `already connected` (greeting not spoken, no listen), `not answered (declined or no answer)`, `busy (the owner's Telegram reports a call in progress)` (Telegram discarded it as busy; the phone did not ring), `answered, then hung up during the greeting` (the owner picked up and ended it before the greeting finished; one call back right away is allowed, then the one-hour rule), `hung up` (another session's `hang_up` ended it during the greeting), `answered but audio failed to connect: <error>` (the owner picked up but media failed; the daemon container restarts itself) or `not called: quiet hours (...)`. Check order: quiet hours, already connected, then the greeting length, then the wait until 10 s have passed since the last call ended, then the dial. |
| `hang_up()` | End that call (only when the owner asks). Always clears the session's call expectation. |
| `status()` | `{busy, current_tool, waiting, call}`: what holds the daemon's lock (`"<tool> (<client ip>)"`; `transcribe` is an HTTP whisper API request), how many requests from every session wait behind it, and the call state (`none`/`ringing`/`connected`). When the daemon is not reached, `call` is `daemon unreachable: <error>` and the other three are null. Never raises, never waits on the lock. |
| `call_debug(lines=50)` / `set_call_debug(enabled)` | The daemon's recent call events and log tail; DEBUG logging on or off at runtime. No lock. `{"daemon": "unreachable: <error>"}` when the daemon is not reached. |

Every `speak`/`listen`/`converse` result, and an answered `call`, ends with
`\n(audio: telegram)` (kept for the permission rule in
`~/.claude/settings.json` that keys on it).

Without a connected call, `speak`/`listen`/`converse` raise `<tool> needs a
connected Telegram call (call state is ...); place one with the call tool`.
When the daemon cannot be reached, every tool except `status`, `call_debug`
and `set_call_debug` raises `speak daemon at <addr> is not reachable: ...`.

## Files

- `speak_server.py`: the thin MCP client. Tool definitions and docstrings,
  the server `instructions` (owner conventions sent to every client: the name
  Angie, spoken style, one converse per turn, timeout handling, call
  etiquette, debugging), `_run`, and the session's
  call expectation (`_call_generation`). `main()`
  reads `SPEAK_DAEMON` (`host:port`, required, no default; unset or malformed
  fails at startup).
- `speak_protocol.py`: the wire (stdlib only). Framing: a 4-byte big-endian
  header length, a JSON header, `payload_len` bytes (empty in practice). One
  request per TCP connection, TCP keepalive both ways. Op names, error kinds,
  and the outcome and state strings (`ANSWERED`, `CONNECTED`, `END_HUNG_UP`,
  `OPENER_MIN_SECONDS`, ...), each defined once.
- `speak_telegram_daemon.py`: the daemon (`speak-telegram` entry point). TCP
  server, the FIFO audio lock (`AudioLock`), the tool logic (`SpeakService`:
  call checks, quiet hours, markdown stripping, cues), the debug ops, the
  HTTP whisper API (`_WhisperHandler`, `SpeakDaemon.transcribe_pcm`), and
  `main()` (settings, engine load, Telegram, binds).
- `speak_engines.py`: `Engines` (`synthesize_stream` per segment for speak,
  `synthesize` whole for the greeting). Kokoro `KPipeline` (hexgrad/Kokoro-82M) and
  faster-whisper `WhisperModel` (compute_type `float32`) on `cuda`. The load
  raises if CUDA is missing, Kokoro's weights are not all fp32, or Kokoro's
  espeak-ng fallback did not load.
- `speak_telegram.py`: `TelegramLink`, the call itself (Telethon + py-tgcalls +
  ntgcalls): dial, opener file, 10 ms raw-frame pump, Silero VAD on the
  owner's audio. Instantiated only by the daemon.
- `speak_tone.py`: the cue PCM: the ear-open beep, the rising "captured" chime and the no-speech timeout beep (played into the call by the daemon).
- `Dockerfile`, `docker-compose.yml`, `.dockerignore`: the daemon image and
  service.
- `scripts/drive_daemon.py`: drives the daemon over TCP with the client's own
  request code (status, call, speak, listen, converse, hang_up, debug), no MCP.
- `pyproject.toml`: base dependencies are only `mcp` (the client). The
  `daemon` extra holds Telegram, torch (CUDA 12.6 build on Linux), kokoro,
  misaki, faster-whisper, silero-vad, numpy, scipy.

## Protocol

- Tool ops (`speak`, `listen`, `converse`, `call`, `hang_up`) queue in the
  daemon's FIFO lock and are never rejected. The daemon sends
  `{"ok": true, "phase": "started"}` the moment it holds the lock, then the
  final reply. The client waits for `started` with no timeout, then applies a
  budget (speak `60 + len/5` s; listen `start_timeout + max + 60`; converse the
  sum; call `60 + 30 + len/5 + listen`; hang_up 20 s). An overrun raises
  `RuntimeError` "did not reply", never `TimeoutError`.
- Quick ops (`ping`, `status`, `debug`, `set_debug`) get one reply and never
  touch the lock.
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
  daemon checks the socket for EOF when the op reaches the lock).
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
  a request can wait behind a whole call or converse turn. The PCM is read
  before the request queues.
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
  synthesizes the rest while earlier segments play (`Engines.synthesize_stream`
  feeding `TelegramLink.play_stream`). This supersedes the earlier rule that
  inference never overlaps outgoing speech, for Kokoro only. If a segment is
  not ready when the previous one finishes, the pump sends silence (never a
  burst) and records `speak_underrun` with the gap; every streamed speech ends
  with a `speech_stream_done` summary (segments, first-audio latency,
  underruns, max pump lag, pump stalls). The call greeting is still
  synthesized whole (the 6 s opener file). Whisper still never overlaps
  outgoing audio: `record` returns only after the ear-open beep has played
  and the capture is done, nothing is queued until Whisper has finished, and
  only then is the chime queued, so the pump sends only silence while
  Whisper runs.
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
| `SPEAK_VOICE` | `af_heart` | Kokoro voice (`af_*` American female, `am_*` male, `bf_*`/`bm_*` British) |
| `SPEAK_SPEED` | `1.0` | Speech rate multiplier |
| `SPEAK_WHISPER_MODEL` | `large-v3-turbo` | faster-whisper model name |
| `SPEAK_LANGUAGE` | `en` | Whisper language |
| `SPEAK_QUIET_HOURS` | `22:00-07:00` | No-call window, may wrap midnight. Validated at startup |
| `TZ` | `America/Los_Angeles` | The clock quiet hours use. Resolved with `zoneinfo` at startup (the image has tzdata); an unknown zone stops the daemon |

Models are cached in the `speak_models` volume
(`/root/.cache/huggingface`). Startup order: engines load and warm, then
Silero and Telegram connect, then ports 8765 and 9000 bind; so a refused connection
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
py-tgcalls; no test contacts Telegram. The daemon itself needs CUDA and runs
only in its container.

## Rules

- No fallback engines. One STT, one TTS, one VAD. Failures raise.
- Log to stderr only. Stdout is the client's MCP transport.
