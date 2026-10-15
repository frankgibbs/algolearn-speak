# algolearn-speak

Talk to Claude Code out loud over a Telegram voice call. This is a small MCP
server that gives Claude an ear and a voice: Claude calls `call` to phone you
on Telegram, then `speak` to say something, `listen` to hear you, or `converse`
to do both in one round trip. Speech synthesis is Kokoro-82M, transcription is
Whisper large-v3-turbo, and end-of-speech detection is Silero VAD, all running
on Apple Silicon via MLX.

The server is Telegram only. It never opens a local microphone or speaker:
`speak`, `listen`, and `converse` work only while a Telegram call is connected
and raise otherwise.

If you are an agent setting this up for someone: follow "Setup" top to bottom,
then run "Verify". The whole thing takes a few minutes, most of it model
downloads.

## Requirements

- A Mac with Apple Silicon (M1 or later). MLX does not run on Intel Macs.
- macOS 14 or later.
- [uv](https://docs.astral.sh/uv/) installed (`brew install uv`). It will fetch
  Python 3.12 on its own if needed.
- [Claude Code](https://claude.com/claude-code) installed.
- `ffmpeg` (`brew install ffmpeg`): py-tgcalls plays the call's opening
  greeting through it.
- A second Telegram account to call from, with its API keys and session (see
  [Telegram calls](#telegram-calls)). About 3 GB of disk for the models.

## Setup

1. Clone the repo and install the dependencies:

   ```bash
   git clone git@github.com:frankgibbs/algolearn-speak.git
   cd algolearn-speak
   uv sync
   ```

2. Download the models and warm the engines once. The first run pulls Kokoro,
   Whisper, and Silero from Hugging Face into `~/.cache/huggingface` and can
   take a few minutes. Later starts take about 10 seconds.

   ```bash
   uv run python -c "import speak_server as s; s.engines.load()"
   ```

3. Set up the Telegram credentials in `.env` and install the `speak-telegram`
   daemon (see [Telegram calls](#telegram-calls)).

4. Register the server with Claude Code at user scope so it is available in
   every project. Use the absolute path to the venv entrypoint:

   ```bash
   claude mcp add --scope user speak -- "$(pwd)/.venv/bin/algolearn-speak"
   ```

## Verify

Start a new Claude Code session, or run `/mcp` in an existing one and
reconnect. Then type:

```
call me
```

Your phone rings on Telegram. When you answer, Claude speaks its greeting, you
hear a beep, and you can answer out loud. When you stop talking you hear a
rising chime and the word "Processing", which means your words were
transcribed and handed to Claude.

## Tools

| Tool | What it does |
|---|---|
| `call(greeting, override_quiet_hours=False, max_seconds=120, silence_seconds=2.0, start_timeout_seconds=45)` | Telegram voice call to you (from a second Telegram account). `greeting` is spoken when you answer; then `call` listens and returns your words. While connected, `speak`/`listen`/`converse` run over the call. Works with the Mac locked; no calls 22:00-07:00. See [Telegram calls](#telegram-calls). |
| `speak(text)` | Says `text` into the connected call. Returns the spoken duration. |
| `listen(max_seconds=120, silence_seconds=2.0, start_timeout_seconds=45)` | High beep, records your call audio until you have been quiet for `silence_seconds` (or spoke for `max_seconds`). Low beep, transcribes, then a rising chime and "Processing". Raises if nobody speaks within `start_timeout_seconds`. |
| `converse(text, ...)` | `speak` then `listen`. Returns your words. |
| `hang_up()` | Ends the call. |
| `call_debug(lines=50)` | The daemon's recent structured call events, log tail, call state, pid, uptime, debug flag. Never takes the audio lock; `{"daemon": "daemon not running"}` if down. |
| `set_call_debug(enabled)` | Raises/restores the daemon's DEBUG logging (speak, py-tgcalls, Telethon) at runtime. |
| `status()` | Returns `{"busy": bool, "current_tool": str \| None, "waiting": int, "call": str}` -- what's running now, how many calls are queued behind it, and the call state. Never blocks. |

Without a connected call, `speak`, `listen`, and `converse` raise
`<tool> needs a connected Telegram call (call state is ...); place one with the
call tool`. Every `speak`/`listen`/`converse` result ends with
`(audio: telegram)` on its own line.

`speak`, `listen`, and `converse` are serialized through one lock that is both
process-wide (a `threading.Lock`, for two calls in the same server process)
and cross-process (an `fcntl.flock` on `~/.algolearn-speak/audio.lock`, for
two separate `speak_server.py` processes -- e.g. two Claude Code sessions
each running their own copy of this server): the call is shared, so if two
callers use it at once, the second one queues and waits for the first to
finish, then proceeds -- it is never rejected or allowed to talk over the
first. `status()` reports the queue depth and which tool currently holds the
lock, including `"<tool> (pid <n>, other process)"` when another process holds
it.

## Telegram calls

`call` lets a session talk to you wherever you are: an always-on
`speak-telegram` daemon places a Telegram voice call from a second account to
yours, and the voice tools run through the call until it ends. It works with
the Mac locked, and the network traffic is outbound only. Design, verified
behaviour and setup (a second Telegram account, its API keys and session in
`.env`, the launchd daemon): [docs/DESIGN_TELEGRAM_CALL.md](docs/DESIGN_TELEGRAM_CALL.md)
and the "Telegram calls" section of CLAUDE.md.

## Audio cues

All cues are played into the call.

| Cue | Meaning |
|---|---|
| Long high beep | The ear is open. Start talking. |
| Short low beep | The ear closed. Recording stopped. |
| Rising two-note chime, then "Processing" | Your words were transcribed and sent to Claude. |
| Short lower beep | Nobody spoke before the timeout. `listen` raises a timeout error. |

## Configuration

All optional, read from the environment at startup. To set one for Claude
Code, pass `-e NAME=value` to `claude mcp add`, or edit the `speak` entry's
`env` block in `~/.claude.json`.

| Variable | Default | Meaning |
|---|---|---|
| `SPEAK_WHISPER_MODEL` | `mlx-community/whisper-large-v3-turbo` | Hugging Face repo of the MLX Whisper model |
| `SPEAK_KOKORO_MODEL` | `mlx-community/Kokoro-82M-bf16` | Hugging Face repo of the MLX Kokoro model |
| `SPEAK_VOICE` | `af_heart` | Kokoro voice. `af_*` American female, `am_*` American male, `bf_*` and `bm_*` British |
| `SPEAK_SPEED` | `1.0` | Speech rate multiplier |
| `SPEAK_LANGUAGE` | `en` | Whisper language hint |
| `SPEAK_ACK_TEXT` | `Processing.` | Word spoken after each successful `listen` |
| `SPEAK_QUIET_HOURS` | `22:00-07:00` | No-call window, local time, may wrap midnight |

Then start a new Claude Code session, or run `/mcp` and reconnect `speak`, for
the new environment to take effect -- the server reads its environment once at
startup.

## Troubleshooting

- **Claude says the tool is unavailable.** Run `claude mcp list` and confirm
  `speak` is listed and connected. The path passed to `claude mcp add` must be
  absolute and must point into this repo's `.venv`.
- **First tool call hangs for 10 seconds.** Normal. The models load in the
  background at startup and the first call waits for them.
- **`speak`/`listen` raise "needs a connected Telegram call".** There is no
  call. Place one with `call`. If `status()` shows `daemon not running`, start
  the `speak-telegram` daemon (see CLAUDE.md, "Telegram calls").
- **You changed `speak_server.py`.** The running server keeps the old code.
  Run `/mcp` in Claude Code and reconnect `speak`, or start a new session.

## Development

```bash
uv sync                  # install
uv run algolearn-speak   # run the server on stdio for manual MCP testing
.venv/bin/python -m unittest discover -s tests   # run all tests
```

`speak_server.py` is the whole server. Logs go to stderr. Stdout is the MCP
transport, so never print to it.

## License

MIT. See `LICENSE`.
