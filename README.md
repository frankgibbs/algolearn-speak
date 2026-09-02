# algolearn-speak

Talk to Claude Code out loud. This is a small MCP server that gives Claude an
ear and a voice on your Mac: Claude calls `speak` to say something, `listen` to
hear you, or `converse` to do both in one round trip. Nothing leaves the
machine. Speech synthesis is Kokoro-82M, transcription is Whisper
large-v3-turbo, and end-of-speech detection is Silero VAD, all running on Apple
Silicon via MLX.

If you are an agent setting this up for someone: follow "Setup" top to bottom,
then run "Verify". The whole thing takes a few minutes, most of it model
downloads.

## Requirements

- A Mac with Apple Silicon (M1 or later). MLX does not run on Intel Macs.
- macOS 14 or later.
- [uv](https://docs.astral.sh/uv/) installed (`brew install uv`). It will fetch
  Python 3.12 on its own if needed.
- [Claude Code](https://claude.com/claude-code) installed.
- A microphone and speakers. The built-in ones work when the lid is open.
  AirPods work too. About 3 GB of disk for the models.

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
   uv run python -c "import speak_server as s; s.engines.load(); s._speak_impl('Setup complete.')"
   ```

   You should hear "Setup complete" from your speakers. If you hear nothing,
   check the output device in System Settings, Sound.

3. Register the server with Claude Code at user scope so it is available in
   every project. Use the absolute path to the venv entrypoint:

   ```bash
   claude mcp add --scope user speak -- "$(pwd)/.venv/bin/algolearn-speak"
   ```

4. Grant microphone access. macOS asks the first time audio is recorded, and
   the permission belongs to the app that launched Claude Code (Terminal,
   iTerm, VS Code, or the desktop app). If nothing is ever heard, check System
   Settings, Privacy and Security, Microphone, and enable that app.

## Verify

Start a new Claude Code session, or run `/mcp` in an existing one and
reconnect. Then type:

```
talk to me
```

Claude will speak, you will hear a beep, and you can answer out loud. When you
stop talking you hear a rising chime and the word "Processing", which means
your words were transcribed and handed to Claude.

To test the microphone without Claude in the loop:

```bash
uv run python -c "import speak_server as s; s.engines.load(); print(s._listen_impl(30, 1.2, 10))"
```

Say something after the beep. Your words should be printed.

## Tools

| Tool | What it does |
|---|---|
| `speak(text)` | Says `text` through the default output. Streams sentence by sentence. Returns the spoken duration. |
| `listen(max_seconds=120, silence_seconds=1.2, start_timeout_seconds=45)` | High beep, records the default input until you have been quiet for `silence_seconds` (or spoke for `max_seconds`). Low beep, transcribes, then a rising chime and "Processing". Raises if nobody speaks within `start_timeout_seconds`. |
| `converse(text, ...)` | `speak` then `listen`. Returns your words. |

`speak` and `listen` never overlap.

## Audio cues

| Cue | Meaning |
|---|---|
| Long high beep | The ear is open. Start talking. |
| Short low beep | The ear closed. Recording stopped. |
| Rising two-note chime, then "Processing" | Your words were transcribed and sent to Claude. |
| Short lower beep | Nobody spoke before the timeout. `listen` raises a timeout error. |

## Configuration

All optional, read from the environment at startup. To set one for Claude
Code, pass `-e NAME=value` to `claude mcp add`.

| Variable | Default | Meaning |
|---|---|---|
| `SPEAK_WHISPER_MODEL` | `mlx-community/whisper-large-v3-turbo` | Hugging Face repo of the MLX Whisper model |
| `SPEAK_KOKORO_MODEL` | `mlx-community/Kokoro-82M-bf16` | Hugging Face repo of the MLX Kokoro model |
| `SPEAK_VOICE` | `af_heart` | Kokoro voice. `af_*` American female, `am_*` American male, `bf_*` and `bm_*` British |
| `SPEAK_SPEED` | `1.0` | Speech rate multiplier |
| `SPEAK_LANGUAGE` | `en` | Whisper language hint |
| `SPEAK_ACK_TEXT` | `Processing.` | Word spoken after each successful `listen` |

Audio devices are whatever macOS has as default input and output. Change them
in System Settings, Sound.

## Troubleshooting

- **Claude says the tool is unavailable.** Run `claude mcp list` and confirm
  `speak` is listed and connected. The path passed to `claude mcp add` must be
  absolute and must point into this repo's `.venv`.
- **First tool call hangs for 10 seconds.** Normal. The models load in the
  background at startup and the first call waits for them.
- **`listen` times out even though you spoke.** The mic is returning silence.
  With the lid closed the built-in MacBook mic returns exact zeros. A USB audio
  adapter with nothing plugged in does the same. Pick a real input device in
  System Settings, Sound, and check the microphone permission for your
  terminal app.
- **You changed `speak_server.py`.** The running server keeps the old code.
  Run `/mcp` in Claude Code and reconnect `speak`, or start a new session.

## Development

```bash
uv sync                  # install
uv run algolearn-speak   # run the server on stdio for manual MCP testing
```

`speak_server.py` is the whole server. Logs go to stderr. Stdout is the MCP
transport, so never print to it.

## License

MIT. See `LICENSE`.
