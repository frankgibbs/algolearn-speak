# algolearn-speak

An ear and voice for Claude Code over a Telegram voice call. An MCP server
(stdio) that Claude drives directly, so a spoken back-and-forth happens inside
one turn: Claude calls `call` (or joins a connected call), then `speak` and
`listen`, reads the transcript as a tool result, and continues.

**Telegram only (owner decision 2026-10-15).** There is no local microphone or
speaker path: no audio device is opened anywhere in this repo. `speak`,
`listen` and `converse` work only while a Telegram call is connected and raise
otherwise; there is no fallback. Speech synthesis and transcription run on
this host; the audio travels only over the Telegram call.

## Tools

| Tool | What it does |
|---|---|
| `speak(text)` | Kokoro-82M on MLX synthesises `text` and plays it into the connected call. Returns spoken duration. |
| `listen(max_seconds=120, silence_seconds=2.0, start_timeout_seconds=45)` | High beep, then records the owner's call audio until Silero VAD sees `silence_seconds` of quiet after speech (or `max_seconds` of speech). Low beep, then Whisper large-v3-turbo on MLX transcribes. Once text is in hand, a rising two-note chime and the spoken word "Processing" tell the user their words were captured. Raises `TimeoutError` if nobody speaks within `start_timeout_seconds`. |
| `converse(text, ...)` | `speak` then `listen` under one lock. Returns the user's words. |
| `call(greeting, override_quiet_hours=False, max_seconds=120, silence_seconds=2.0, start_timeout_seconds=45)` | Telegram voice call from the algolearn.ai account to the owner (@frankgibbs35); `greeting` is the whole opening, spoken the moment they answer: who is calling, why, and the context, ending with a question or invitation. It must be at least 6 s when spoken (measured on the synthesized audio); a shorter one raises `ValueError` BEFORE dialing (phone never rings; owner decision 2026-10-12) telling the model to add the context of why it is calling, or, on a fresh conversation with nothing to report, to open with something playful (a joke, something funny or flirty). Never padded. Once answered, `call` immediately does the `listen` (ear-open beep, VAD, Whisper, chime + ack) under the same lock hold and returns the owner's words; listen parameters and defaults match `converse`, and a listen timeout or a drop raises as in `listen` (the call stays connected on a timeout). Returns `answered\n<their words>\n(audio: telegram)`, `already connected` (greeting not spoken, no listen), `not answered (declined or no answer)`, `answered, then hung up during the greeting` (the owner picked up and ended it before the greeting finished; one call back right away is allowed, then the one-hour rule), `answered but audio failed to connect: <error>` (the owner picked up but media failed; daemon restarts itself) or `not called: quiet hours (...)`. Check order: quiet hours, daemon state / already connected, then the greeting length, then the dial. While connected, `speak`/`listen`/`converse` run over the call. Works with the Mac locked. See "Telegram calls". |
| `hang_up()` | End that call (only when the owner asks). |
| `status()` | `{busy, current_tool, waiting, call}`: what holds the lock and the daemon's call state (`none`/`ringing`/`connected`/`daemon not running`/`error: ...`). Never blocks. Every `speak`/`listen`/`converse` result ends with `(audio: telegram)` on its own line (kept for the permission rule in `~/.claude/settings.json` that keys on it). |

Without a connected call, `speak`/`listen`/`converse` raise `<tool> needs a
connected Telegram call (call state is ...); place one with the call tool`.
`speak` and `listen` never overlap (a cross-process lock at
`~/.algolearn-speak/audio.lock`, shared by every session on this host). Tool
functions are async and run the work in a thread so the MCP event loop stays
responsive during a long `listen`.

## Files

- `speak_server.py` — the MCP server. `main()` starts a background thread that
  loads Kokoro and Whisper (about 9 s on the M1 Pro), then runs the MCP stdio
  loop. Tool calls block on the loading event; a load failure is raised into
  every tool call rather than hidden. It opens no audio device.
- `speak_tone.py` — the beep/chime cue PCM (played into the call).
- `speak_telegram.py` — `TelegramLink`: the Telegram call itself (Telethon + py-tgcalls + ntgcalls): dial, opener file, 10 ms raw-frame pump, Silero VAD on the owner's audio, `caffeinate` during a call. Instantiated only by the daemon.
- `speak_telegram_daemon.py` — the `speak-telegram` daemon (one per Mac, launchd: `launchd/com.algolearn.speak-telegram.plist`; holds the Telegram login and the call) and `TelegramClient`, which `speak_server` uses over a Unix socket. `speak_server` never imports Telethon.
- `pyproject.toml` — uv project, Python 3.12 (mlx-whisper pulls torch; Kokoro
  needs `misaki[en]`; Silero adds only torchaudio on top of that).

## Telegram calls

Design and verified facts: `docs/DESIGN_TELEGRAM_CALL.md`. `call` phones the
owner on Telegram from their second account (algolearn.ai); works wherever the
phone has signal, with the Mac locked or unlocked, outbound-only networking.

- **One daemon:** `speak-telegram` (launchd) owns the Telegram login (one
  connection per login) and the call; it holds `~/.algolearn-speak/telegram.sock.lock`
  so a second daemon refuses to start. Speak servers are clients over
  `~/.algolearn-speak/telegram.sock` (0600). **Daemon not running = no call**:
  `speak`/`listen`/`converse` raise (with "call back" if a call was expected),
  and `call` raises.
- **Credentials:** `TELEGRAM_API_ID`, `TELEGRAM_API_HASH`, `TELEGRAM_SESSION`
  (a Telethon StringSession: a login, treat as a password), `TELEGRAM_CALL_TARGET`
  in the repo's git-ignored `.env`. Only the daemon reads them.
- **Install:** `uv sync`, then `cp launchd/com.algolearn.speak-telegram.plist
  ~/Library/LaunchAgents/ && launchctl bootstrap gui/$(id -u)
  ~/Library/LaunchAgents/com.algolearn.speak-telegram.plist`. Logs:
  `~/Library/Logs/speak-telegram.log`. Uninstall: `launchctl bootout
  gui/$(id -u)/com.algolearn.speak-telegram && rm
  ~/Library/LaunchAgents/com.algolearn.speak-telegram.plist`. The plist sets
  `PATH` to include `/opt/homebrew/bin` (py-tgcalls plays the opener via ffmpeg).
- **Verified quirks (2026-10-09), all handled in `speak_telegram.py`:** the call
  must open on an audio FILE of at least 6 s (the greeting itself; 2 s gave
  all-zero audio both ways). **Owner decision (2026-10-12):** no padding;
  `call` rejects a greeting under 6 s before dialing and the daemon's
  `write_opener` raises on one (`OPENER_MIN_SECONDS`, the single definition), then switch to raw frames; the
  outgoing stream is fed every 10 ms (silence when idle) and never bursts
  (bursts sped audio up); ntgcalls aborts in its C++ destructors at exit, so
  the daemon always leaves via `os._exit`.
- **Call required:** `speak`/`listen`/`converse` run only on a connected call
  (results end `(audio: telegram)`). A call that ends mid-operation raises
  `Telegram call ended during ...`; the rule of engagement is to call back.
  Captures are not archived. The call is shared by every session (they take
  turns through the audio lock).
- **Audio failure and diagnostics:** if the owner answers but the media connect
  fails (`ntgcalls.BaseRTCException`, e.g. `TelegramServerError`), `call` returns
  `answered but audio failed to connect: <error>`, the call is discarded, and the
  daemon exits 1 right after replying so launchd restarts it with fresh ntgcalls
  state (no automatic retry). Every call writes structured events (call id,
  timestamps, state transitions, errors, duration) to a 500-entry ring buffer and
  the log; `call_debug` / `set_call_debug` read it and toggle DEBUG logging over
  the daemon socket (ops `debug`, `set_debug`), without the audio lock.
  **Debugging is reactive (owner decision 2026-10-12):** `set_call_debug(True)`
  also turns on the audio-level events (`opener_written`, `audio_in`,
  `audio_out`, `audio_opener_out`: per-window peak/RMS dBFS each way); with debug
  off they are not recorded. When a call seems off, turn debug on and call
  again; turn it off once calls work.
- **Quiet hours:** `call` refuses 22:00-07:00 local (`SPEAK_QUIET_HOURS`)
  unless `override_quiet_hours=True`, used only when the owner said they are
  up. After `not answered`, wait an hour before calling again. If the owner hangs up
  at any point (during the greeting or mid-conversation), call back once right
  away; if that call is not answered, they are busy: wait an hour (owner
  decision 2026-10-12).
- **Pins:** `telethon==1.45.0`, `py-tgcalls==3.0.0`, `ntgcalls==3.0.2`; the
  private-call API is undocumented, so upgrade only deliberately, with a re-test.
- Tests: `tests/test_speak_telegram.py` (link vs fake py-tgcalls),
  `tests/test_speak_telegram_daemon.py` (daemon + client over a temp socket),
  `tests/test_speak_server_telegram.py` (tools vs a fake client).
  No test contacts Telegram.

## Configuration (environment variables, read at startup)

| Variable | Default | Meaning |
|---|---|---|
| `SPEAK_WHISPER_MODEL` | `mlx-community/whisper-large-v3-turbo` | HF repo of the MLX Whisper model |
| `SPEAK_KOKORO_MODEL` | `mlx-community/Kokoro-82M-bf16` | HF repo of the MLX Kokoro model |
| `SPEAK_VOICE` | `af_heart` | Kokoro voice preset (`af_*` American female, `am_*` male, `bf_*`/`bm_*` British) |
| `SPEAK_SPEED` | `1.0` | Speech rate multiplier |
| `SPEAK_LANGUAGE` | `en` | Whisper language hint |
| `SPEAK_ACK_TEXT` | `Processing.` | Spoken after every successful `listen`, right after the rising chime |
| `SPEAK_QUIET_HOURS` | `22:00-07:00` | No-call window, local time, may wrap midnight. Validated at startup. |

Models are cached under `~/.cache/huggingface`. Environment variables go in
the `speak` MCP server's `env` block in `~/.claude.json`; a new session (or
`/mcp` then reconnect) picks up a change, since env is read once at process
start.

## Registration with Claude Code

Registered at user scope (in `~/.claude.json`) so it is available in every project:

```
claude mcp add --scope user speak -- /Users/frankg/workspace/algolearn/live/algolearn-speak/.venv/bin/algolearn-speak
```

Check with `claude mcp list`. A new session, or `/mcp` then reconnect, picks it up.

## Development

```
uv sync                                   # install
uv run algolearn-speak                    # run the server on stdio (for manual MCP testing)
.venv/bin/python -m unittest discover -s tests   # run all tests
```

Run the tests with `HOME` pointed at a scratch directory while a live call may
be in progress: the lock is resolved under `$HOME/.algolearn-speak` at import
time, and the tests must never contend with the live one.

Smoke test without MCP: start the daemon, place a call, then import
`speak_server`, call `engines.load()`, and `_speak_impl("hello")` /
`_listen_impl(30, 1.2, 10)`.

## Rules

- No fallback engines. One STT, one TTS, one VAD. Failures raise.
- Log to stderr only. Stdout is the MCP transport.
