# algolearn-speak

Talk to Claude Code out loud over a Telegram voice call. Claude calls `call`
to phone you on Telegram, then `speak` to say something, `listen` to hear you,
or `converse` to do both in one round trip.

It has two parts:

- **A speak daemon**, one per network, in a Docker container on a Linux host
  with an NVIDIA GPU. It holds the Telegram login and places the call, runs
  Kokoro-82M (speech synthesis) and Whisper large-v3-turbo through
  faster-whisper (transcription) on the GPU at full precision, and Silero VAD
  (end of speech) on the CPU. It listens on TCP port 8765.
- **A thin MCP client** on every computer that runs Claude Code. It has no
  models and opens no audio device; each tool is one request to the daemon.

Nothing opens a local microphone or speaker. `speak`, `listen`, and
`converse` work only while a Telegram call is connected and raise otherwise.

**Port 8765 has no authentication and no TLS.** Anyone who can reach it can
make the daemon call you. Run it only on a network you trust.

## Requirements

Daemon host:
- Linux with an NVIDIA GPU (about 5-6 GB of free VRAM at fp32), the NVIDIA
  driver (CUDA 12.6 or newer), Docker with Compose and the NVIDIA Container
  Toolkit.
- A second Telegram account to call from, with its API keys and a Telethon
  StringSession (see [Telegram calls](#telegram-calls)).

Each client computer:
- [uv](https://docs.astral.sh/uv/) (it fetches Python 3.12 on its own).
- [Claude Code](https://claude.com/claude-code).

## Setup

1. **The daemon.** Copy the repo to the GPU host (without `.git`, `.venv`, or
   any `.env`), then create `.env` next to `docker-compose.yml` with
   `TELEGRAM_API_ID`, `TELEGRAM_API_HASH`, `TELEGRAM_SESSION` and
   `TELEGRAM_CALL_TARGET`, and `chmod 600` it. `.dockerignore` keeps it out of
   the image; compose passes it to the container with `env_file`. Never commit
   it. Then:

   ```bash
   docker compose up --build -d
   docker logs -f speak-telegram   # wait for "listening on 0.0.0.0:8765"
   ```

   The first start downloads the models into the `speak_models` volume.
   `docker-compose.yml` pins the container to GPU 0; change `device_ids` for
   another GPU.

2. **Each client.** Clone the repo, install the client (only `mcp`), and
   register it at user scope with the daemon's address:

   ```bash
   uv sync
   claude mcp add --scope user speak -e SPEAK_DAEMON=<daemon-host>:8765 -- "$(pwd)/.venv/bin/algolearn-speak"
   ```

   `SPEAK_DAEMON` is required. The client refuses to start without it.

## Verify

Without MCP, from a client checkout:

```bash
uv sync   # the script uses only the client code
.venv/bin/python scripts/drive_daemon.py --daemon <daemon-host>:8765 status
.venv/bin/python scripts/drive_daemon.py --daemon <daemon-host>:8765 \
  call "Hi, it's Claude, testing the speak daemon. Can you hear me clearly?" \
  speak "This came from the GPU." listen hang_up
```

From Claude Code: start a new session (or `/mcp`, reconnect), then type `call
me`. Your phone rings on Telegram. When you answer, Claude speaks its
greeting, you hear a beep, and you can answer out loud. When you stop talking
you hear a rising chime and the word "Processing", which means your words
were transcribed and handed to Claude.

## Tools

| Tool | What it does |
|---|---|
| `call(greeting, override_quiet_hours=False, max_seconds=120, silence_seconds=2.0, start_timeout_seconds=45)` | Telegram voice call to you (from a second Telegram account). `greeting` (at least 6 s when spoken) is played when you answer; then `call` listens and returns your words. No calls 22:00-07:00 unless overridden. |
| `speak(text)` | Says `text` into the connected call. Returns the spoken duration. |
| `listen(max_seconds=120, silence_seconds=2.0, start_timeout_seconds=45)` | High beep, records your call audio until you have been quiet for `silence_seconds` (or spoke for `max_seconds`). Low beep, transcribes, then a rising chime and "Processing". Raises if nobody speaks within `start_timeout_seconds`. |
| `converse(text, ...)` | `speak` then `listen`. Returns your words. |
| `hang_up()` | Ends the call. |
| `status()` | `{"busy", "current_tool", "waiting", "call"}`: what is running in the daemon now (and from which computer), how many requests wait behind it, and the call state. Never blocks. |
| `call_debug(lines=50)` | The daemon's recent call events, log tail, call state, pid, uptime, debug flag. |
| `set_call_debug(enabled)` | Turns the daemon's DEBUG logging on or off at runtime. |

Every `speak`/`listen`/`converse` result ends with `(audio: telegram)` on its
own line. The tools that use the call are serialized through one first-come,
first-served lock in the daemon, shared by every Claude Code session on every
computer: a second caller waits its turn and is never rejected.

## Telegram calls

The daemon places a Telegram voice call from a second account to yours, and
the voice tools run through the call until it ends. Design and verified
behaviour: [docs/DESIGN_TELEGRAM_CALL.md](docs/DESIGN_TELEGRAM_CALL.md); the
move to a networked daemon:
[docs/DESIGN_REMOTE_DAEMON.md](docs/DESIGN_REMOTE_DAEMON.md). One Telegram
login must be used by one daemon only: never run two daemons with the same
`TELEGRAM_SESSION`.

## Audio cues

All cues are played into the call.

| Cue | Meaning |
|---|---|
| Long high beep | The ear is open. Start talking. |
| Short low beep | The ear closed. Recording stopped. |
| Rising two-note chime, then "Processing" | Your words were transcribed and sent to Claude. |
| Short lower beep | Nobody spoke before the timeout. `listen` raises a timeout error. |

## Configuration

Client: `SPEAK_DAEMON` (`host:port`), required.

Daemon: set in `docker-compose.yml`'s `environment`, all required:
`SPEAK_KOKORO_MODEL` (`hexgrad/Kokoro-82M`), `SPEAK_VOICE` (`af_heart`;
`af_*`/`am_*` American, `bf_*`/`bm_*` British), `SPEAK_SPEED` (`1.0`),
`SPEAK_WHISPER_MODEL` (`large-v3-turbo`), `SPEAK_LANGUAGE` (`en`),
`SPEAK_ACK_TEXT` (`Processing.`), `SPEAK_QUIET_HOURS` (`22:00-07:00`), and
`TZ`, the clock the quiet hours use.

## Troubleshooting

- **"speak daemon at ... is not reachable".** The container is down, still
  loading models, or restarting after a failed call. Check `docker logs
  speak-telegram` on the daemon host.
- **"needs a connected Telegram call".** There is no call. Place one with
  `call`.
- **Claude says the tool is unavailable.** Run `claude mcp list`; the path
  passed to `claude mcp add` must be absolute and `SPEAK_DAEMON` must be set.

## Development

```bash
uv sync --extra daemon                          # client + daemon dependencies
.venv/bin/python -m unittest discover -s tests  # run all tests
```

Logs go to stderr. The client's stdout is the MCP transport, so never print
to it.

## License

MIT. See `LICENSE`.
