# Audio lifecycle audit — PortAudio -9986 after idle

Date: 2026-09-06
Scope: static analysis of `speak_server.py` + review of MCP client logs
(`~/Library/Caches/claude-cli-nodejs/*/mcp-logs-speak/*.jsonl`). No audio
played or recorded during this audit (device list queried only via
`sd.query_devices()`).

## 1. Stream site audit

| Site | file:line | API | Closed? | Held across calls? | `sd.stop()`/`sd.wait()` after? |
|---|---|---|---|---|---|
| `_cue` | speak_server.py:98-99 | `sd.play` + `sd.wait` | Yes — `sd.wait()` blocks until the tone finishes; `sd.play`/`sd.wait` use PortAudio's implicit global stream, which PortAudio closes itself when playback completes | No — no stream object is stored anywhere | Yes, `sd.wait()` called immediately after `sd.play()` |
| `_ack` | speak_server.py:105-107 | `sd.play` + `sd.wait`, then `_speak_impl` | Yes, same as above | No | Yes |
| `_speak_impl` (TTS playback) | speak_server.py:179-182 | `sd.OutputStream` via `_audio_stream` | Yes — `with _audio_stream(sd.OutputStream, ...) as out:` — context manager guarantees `.close()`/`.stop()` on exit, including on exception, since `with` always unwinds | No — `out` is a local variable inside `_speak_impl`, not stored on `Engines`, not module-global | N/A (streaming write loop; the `with` block's `__exit__` calls `stop()`+`close()`) |
| `_listen_impl` (mic capture) | speak_server.py:202-221 | `sd.InputStream` via `_audio_stream` | Yes — same `with _audio_stream(sd.InputStream, ...) as mic:` pattern | No — `mic` is local | N/A (blocking `.read()` loop; `with` handles teardown) |

Additional checks:
- No module-level `sd.OutputStream`/`sd.InputStream`/`sd.Stream` instance anywhere (`grep -n "sd\."` shows only the four call sites above, all inside function bodies).
- No thread other than the daemon `kokoro` producer thread (speak_server.py:177) and the one-shot `engine-load` thread (speak_server.py:299) is created. Neither holds a stream, a `sd.Stream` object, or a PortAudio handle — the kokoro thread only pushes numpy arrays into a `queue.Queue`.
- `engines` (the `Engines` singleton, speak_server.py:83) holds `self.kokoro` (MLX TTS model) and `self.vad` (Silero VAD model) — both are just in-memory model objects, no audio device handles.
- `audio_lock` (speak_server.py:84) is a plain `threading.Lock` used via `with audio_lock:` in `_speak_sync`/`_listen_sync`/`_converse_sync` (lines 233-246) — always released on the `with` block exit, including on exception. No path leaves it held.
- No `sd.default.device`/`sd.default.samplerate` is ever mutated, cached, or read at import/module-load time — every stream call passes `samplerate=` explicitly and lets sounddevice resolve the *current* default device at open time (it does not pin a stale device index anywhere).
- TTS side (Kokoro): `engines.kokoro.generate(...)` (speak_server.py:66, 170) is called fresh each time; no file handles, sockets, or subprocesses — it yields numpy arrays in-process.
- Whisper side: `mlx_whisper.transcribe(...)` (speak_server.py:69, 226) is called fresh each time on an in-memory numpy array; no file is written, no subprocess spawned.
- `VADIterator` (speak_server.py:194) is constructed fresh on every `_listen_impl` call from the persistent `engines.vad` model; it is a stateless-between-calls wrapper (holds only VAD probabilities in Python objects), not an audio handle.

**Verdict on every site: closed correctly, nothing held across calls.** There is no leaked stream, no dangling PortAudio handle, no thread outliving its call, no cached stale device index in the code.

## 2. Log timing evidence

MCP client logs (`Calling MCP tool` / `Tool '<name>' failed/completed` entries) give an exact timeline of every tool call and its outcome, across this project's log and five other projects' `mcp-logs-speak` files (the server is registered at user scope, so it's the *same* long-lived process across all of them). Grepping for `9986` across all `mcp-logs-speak/*.jsonl` finds it only in this project's two log files, at these 8 occurrences (idle gap = time since the *previous* tool call of any kind on that same server process):

| Failure timestamp | Idle gap before it | Time to fail |
|---|---|---|
| 2026-09-03T21:25:17Z | 2h 49m 30s | 1.4s |
| 2026-09-03T23:50:48Z | 2h 07m 38s | 1.3s |
| 2026-09-04T14:56:19Z | 14h 47m 08s | 0.1s |
| 2026-09-04T18:51:44Z | 3h 37m 45s | 0.1s |
| 2026-09-05T00:52:05Z | 5h 57m 30s | 0.1s |
| 2026-09-05T15:03:11Z | 13h 53m 48s | 0.2s |
| 2026-09-06T04:24:49Z | 11h 06m 55s | 1.2s |
| 2026-09-06T13:06:56Z | 8h 34m 20s | 1.2s |

Every one of the dozens of *other* tool calls in these logs (gaps ranging from a few seconds up to ~1h20m) succeeded or failed only with an unrelated, expected error (`no speech detected within Ns`). No -9986 ever occurs after a gap under ~2 hours; every gap over ~2 hours in the data produces one. The failure is always on the **first stream opened after the idle gap** (`OutputStream`, since `converse`/`speak` open output first) and fails **near-instantly** (0.1-1.4s) — consistent with `_audio_stream`'s immediate first attempt failing, not a slow timeout.

No device-change events are logged anywhere (there is no code path that logs a CoreAudio device-change notification — sounddevice/PortAudio does not surface one to this server), so the log cannot show *why* the device list went stale, only *that* the failure tracks wall-clock idle time, not call volume or call type.

## 3. Device context

`sd.query_devices()` in the repo `.venv` (read-only, no audio played) shows the current default input **and** output device is `Frank's AirPods Pro` (Bluetooth), alongside `MacBook Pro Speakers`/`MacBook Pro Microphone` and a `USB Advanced Audio Device`. Bluetooth (A2DP/HFP) CoreAudio devices are the textbook case for PortAudio -9986 after a device has been idle long enough for macOS's Bluetooth audio power management to suspend the link: the *first* attempt to open a stream against the stale cached device description fails, and the failure is at the CoreAudio/PortAudio layer, before any of this server's Python code runs (`_audio_stream`'s `kind(**kw)` call, speak_server.py:126, is the very first line executed — there is nothing upstream of it in this codebase that could leak state).

## 4. Verdict: leak hypothesis NOT supported

- Every stream is opened via a `with` context manager and closed deterministically, including on exception (speak_server.py:179, 202).
- No stream object, PortAudio handle, or device index is stored on `Engines`, at module scope, or on any thread that outlives its call.
- The lock (`audio_lock`) is always released via `with`.
- The failure correlates with **wall-clock idle time since the last tool call**, not with the number or type of prior calls, not with whether the prior call succeeded or errored, and not with which tool (`speak`/`listen`/`converse`) ran last.
- The failure hits on the very first stream-open attempt after idle, before any of this server's application logic could have "held" anything — it's the OS/PortAudio device snapshot that's stale, not an application-level resource.
- The default device is Bluetooth (AirPods), a device class known to suspend/reconnect on idle, which is a plausible independent mechanism fully explaining the -9986-after-idle timing without invoking a leak.

**This is not a resource-cleanup bug in speak_server.py.** The two prior fix attempts in git history (`127119b` retry-once, `e8f39fa` re-exec) were already attacking the right layer (a process-wide stale PortAudio state that only a fresh PortAudio init clears) — they just have the acceptable-workaround problem stated in the task: re-exec breaks the stdio pipe.

## 5. Recommended fix (not implemented, per instructions)

Move audio I/O into a **short-lived helper subprocess per call**, so the long-lived MCP process (which owns the stdio pipe to Claude Code and must never restart) never touches PortAudio directly:

- The MCP process keeps its stdio pipe, model-loading (`Engines`), and MCP tool surface exactly as-is.
- `_audio_stream`'s two responsibilities — opening `sd.OutputStream`/`sd.InputStream` and pumping audio through them — move into a small script/entry point invoked via `subprocess.run`/`multiprocessing.Process` for each `speak`/`listen` call. Audio data crosses the process boundary as raw bytes/numpy buffers (stdin/stdout pipes or a temp file), text/transcription results as return values.
- Each helper process does a fresh PortAudio init on launch and exits after the one stream closes, so a stale Bluetooth device snapshot never survives past a single call — there is nothing to go stale *between* calls because there is no long-lived PortAudio state to begin with.
- If the helper's `sd.OutputStream`/`sd.InputStream` open fails with -9986, the parent MCP process can retry by launching a *new* helper subprocess (cheap, no re-exec, no pipe disruption) instead of retrying in a process whose PortAudio state is already known-stale.
- Kokoro/Whisper/Silero model loading stays in the parent (it's the expensive, one-time cost this server was built to warm) — only the actual `sd.*` calls and their immediate read/write loops move to the helper.

Not implemented or committed, per task instructions.
