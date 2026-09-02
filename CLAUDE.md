# algolearn-speak

A local ear and voice for Claude Code. An MCP server (stdio) that Claude drives
directly, so a spoken back-and-forth happens inside one turn: Claude calls
`speak`, then `listen`, reads the transcript as a tool result, and continues.

Everything runs on this Mac. No audio or text leaves the machine.

## Tools

| Tool | What it does |
|---|---|
| `speak(text)` | Kokoro-82M on MLX synthesises `text`, streamed sentence by sentence to the default output device. Returns spoken duration. |
| `listen(max_seconds=120, silence_seconds=1.2, start_timeout_seconds=45)` | High beep, then records the default input until Silero VAD sees `silence_seconds` of quiet after speech (or `max_seconds` of speech). Low beep, then Whisper large-v3-turbo on MLX transcribes. Once text is in hand, a rising two-note chime and the spoken word "Processing" tell the user their words were captured. Raises `TimeoutError` if nobody speaks within `start_timeout_seconds`. |
| `converse(text, ...)` | `speak` then `listen` under one lock. Returns the user's words. |

`speak` and `listen` never overlap (a process-wide lock). Tool functions are
async and run the audio work in a worker thread so the MCP event loop stays
responsive during a long `listen`.

## Files

- `speak_server.py` — the whole server. `main()` starts a background thread that
  loads Kokoro, Silero VAD, and Whisper (about 9 s on the M1 Pro), then runs the
  MCP stdio loop. Tool calls block on the loading event; a load failure is raised
  into every tool call rather than hidden.
- `pyproject.toml` — uv project, Python 3.12 (mlx-whisper pulls torch; Kokoro
  needs `misaki[en]`; Silero adds only torchaudio on top of that).

## Configuration (environment variables, read at startup)

| Variable | Default | Meaning |
|---|---|---|
| `SPEAK_WHISPER_MODEL` | `mlx-community/whisper-large-v3-turbo` | HF repo of the MLX Whisper model |
| `SPEAK_KOKORO_MODEL` | `mlx-community/Kokoro-82M-bf16` | HF repo of the MLX Kokoro model |
| `SPEAK_VOICE` | `af_heart` | Kokoro voice preset (`af_*` American female, `am_*` male, `bf_*`/`bm_*` British) |
| `SPEAK_SPEED` | `1.0` | Speech rate multiplier |
| `SPEAK_LANGUAGE` | `en` | Whisper language hint |
| `SPEAK_ACK_TEXT` | `Processing.` | Spoken after every successful `listen`, right after the rising chime |

Models are cached under `~/.cache/huggingface`. Audio devices are whatever
macOS has as default input and output. Set them in System Settings, Sound.

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
```

Smoke test without MCP: import `speak_server`, call `engines.load()`, then
`_speak_impl("hello")` and `_listen_impl(30, 1.2, 10)`.

## Rules

- No fallback engines. One STT, one TTS, one VAD. Failures raise.
- Log to stderr only. Stdout is the MCP transport.
- The microphone permission belongs to the terminal app that launched Claude
  Code. If `listen` returns silence forever, check System Settings, Privacy and
  Security, Microphone.

## Microphone notes (verified 2026-09-02)

- The server records from whatever macOS has as the default input. With the
  lid closed (clamshell mode) the built-in MacBook mic returns exact zeros.
  The C-Media "USB Advanced Audio Device" adapter delivers only noise floor
  unless a mic is plugged into it. AirPods Pro work as soon as they connect
  and become the default input; first live round trip was transcribed exactly.
- Terminal already has the microphone permission. If a different app launches
  Claude Code (iTerm, VS Code, the desktop app), that app needs its own grant.
- A mic that returns exact digital silence is a permission or hardware issue,
  not a VAD threshold issue. Check levels with `sounddevice.rec` before touching
  the server.
