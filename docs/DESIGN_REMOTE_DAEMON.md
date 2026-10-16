# Design: the speak daemon moves to patrick; every computer runs a thin client

Written 2026-10-15. Status: approved and built on 2026-10-15 (branch
`telegram-only`). The image is built on patrick; the container has not been
started. Owner decisions made at build time (full precision, CUDA 12.6, the
patrick deploy directory) are folded into the sections below and listed in
section 10.

Labels used throughout:
- **Owner decision** (2026-10-15 unless dated otherwise): decided by the owner.
  It is a decision, not a claim about how anything behaves today.
- **Decision**: decided in this design, with the reason.
- **Verified**: observed directly on 2026-10-15, by reading code (file:line)
  or by a read-only command on patrick or this Mac.
- **Measured**: a timing taken on 2026-10-15, with how it was taken.
- **Estimate**: not measured. The reasoning is given.
- **Unverified, read before build**: a claim about a library that has not been
  read yet. Per the workspace's System Modification Protocol it is a build
  check, not a design element.

Companion documents, referenced for provenance, not repeated here:
`docs/DESIGN_TELEGRAM_CALL.md` (the outbound call and its spike quirks) and
`docs/DESIGN_INBOUND_CALL.md` (the proposed inbound call). The inbound design
was written for the current layout (a Unix socket and a lock on one host). If
it is built, it has to be rebased onto the TCP protocol and the daemon-held
lock described here.

## 1. What changes

**Owner decisions:**
1. The `speak-telegram` daemon moves to patrick (192.168.86.31) and becomes
   the heavy side. It holds the Telegram login and the call, plus TTS
   (Kokoro) and STT (Whisper) on patrick's GPU. CUDA builds replace the MLX
   builds.
2. Audio is Telegram only. A client machine uses no local audio, no mic and
   no GPU. (The removal of the Mac audio path is already done in the working
   tree; section 2.)
3. Each computer runs only a thin MCP client that talks to the daemon on
   patrick over the LAN.
4. **No authentication, on purpose.** Any agent on any computer on the LAN
   can reach the daemon and hold a spoken conversation with the owner: place
   a call, speak, listen and receive the transcript. That is the capability
   the owner wants. The LAN is the trust boundary.
5. The daemon runs as a Docker container on patrick: docker compose,
   `up --build -d`, like the other services there. It reaches the GPU through
   the NVIDIA container runtime. The Telegram credentials stay in a
   git-ignored `.env` that is passed to the container.
6. **One container holds every dependency the speak service needs, and all
   of it is built from this repo.** The daemon loads its own Whisper
   large-v3-turbo. It does not call the existing `whisper-service` on
   patrick.
7. The daemon shares the RTX 3060 that runs `whisper-service` today (GPU0).
8. **Retire `whisper-service` on patrick once the new daemon is verified
   working.** This is the last cutover step (section 9). Superseded in part by
   the owner decision of 2026-10-16 (section 11): the daemon now serves
   `whisper-service`'s HTTP API on port 9000 itself.
9. **Silero VAD runs on the CPU**, not the GPU.

Before and after:

```
TODAY (one Mac; working tree, section 2)
  speak_server.py (one per Claude Code session)          speak-telegram (launchd, one per Mac)
    Kokoro (MLX), Whisper (MLX), audio lock (flock)  ─ Unix socket ─>  Telethon + py-tgcalls + ntgcalls
    tools, call expectation, quiet hours, cues                         the call, 10 ms pump, Silero VAD (CPU)

AFTER
  speak_server.py (thin; one per session, any LAN computer)   speak-telegram container (patrick, GPU0)
    MCP tools, per-session call expectation  ── TCP, LAN ──>  audio lock, tool logic, quiet hours, cues
    no numpy, no torch, no audio                               Kokoro + Whisper on CUDA, Silero on CPU
                                                               Telethon + py-tgcalls + ntgcalls, pump
                                                                 └── Telegram servers ──> owner's phone
```

## 2. The current split (Verified)

The baseline is the repo's **working tree** on 2026-10-15, on branch
`telegram-only`, not a commit. The last commit is 1347ee0, and Telegram-only
(Owner decision 2026-10-15) is uncommitted on top of it:
- Staged deletions: `speak_audio_worker.py`, `tests/test_speak_audio_worker.py`,
  `docs/AUDIO_LIFECYCLE_AUDIT.md` and `docs/PORTAUDIO_REINIT_NOTES.md`.
- Unstaged edits: `speak_server.py` (807 lines), `speak_tone.py`,
  `speak_telegram.py`, `speak_telegram_daemon.py`, `pyproject.toml`
  (`sounddevice` dropped), `CLAUDE.md` and two test files.

The line numbers below are for the working tree. If those edits change before
they are committed, the line numbers move with them.

### 2.1 speak_server.py: the MCP process, one per Claude Code session

- **Telegram only.** The module docstring states that all audio goes over the
  call and there is no local mic or speaker path (`speak_server.py:1-5`).
  Without a connected call, speak, listen and converse raise
  "`<tool>` needs a connected Telegram call (call state is ...); place one
  with the call tool" (`speak_server.py:92-107`).
- **Engines.** `Engines.load` loads Kokoro through `mlx_audio` and warms both
  models in a background thread: one Kokoro "Ready." and one second of
  silence through `mlx_whisper` (`speak_server.py:162-196`). Its settings are
  `SPEAK_WHISPER_MODEL`, `SPEAK_KOKORO_MODEL`, `SPEAK_VOICE`, `SPEAK_SPEED`,
  `SPEAK_LANGUAGE`, `SPEAK_ACK_TEXT` (removed 2026-10-16) and `SPEAK_QUIET_HOURS`
  (`speak_server.py:54-62`).
- **Audio lock.** There are two layers:
  - `_CrossProcessLock` takes `flock(LOCK_EX)` on `~/.algolearn-speak/audio.lock`
    and writes a JSON sidecar naming the holder (`speak_server.py:209-285`).
  - `_SerializingLock` puts a `threading.Lock` in front of it and counts this
    process's waiters (`speak_server.py:288-353`).

  `status()` documents that `waiting` counts only the caller's own process
  (`speak_server.py:786-789`). The lock spans processes on one host only.
- **The call expectation.** `_call_expected` and `_call_generation` are
  per process (`speak_server.py:76-80`).
  - `_require_call` runs under the lock at the start of each tool. If the
    daemon reports `connected`, the session sets its expectation and goes
    ahead.
  - A call ended by `hang_up` (end reason `hung_up`, same generation) clears
    the expectation.
  - Otherwise an expected call that is gone raises "Telegram call ended
    during ...; call back", and no expectation raises "needs a connected
    Telegram call" (`speak_server.py:92-107`, `85-89`).
  - `_guard_call` re-checks the call state after the work and raises with
    any captured transcript (`speak_server.py:110-132`).
  - `_after_capture` (removed and then restored 2026-10-16, for the rising chime) keeps the words when the owner hangs up right after
    speaking (`speak_server.py:446-457`).
- **TTS.** `_speak_impl` collects every Kokoro sentence chunk and then plays
  them as one buffer through `telegram.play` (`speak_server.py:395-423`,
  `361-363`). Synthesis therefore finishes before any of it is heard.
- **STT.** `_listen_impl` works in this order (`speak_server.py:426-443`):
  1. It sends the ear-open cue with the daemon's `record` op, and the daemon
     returns 16 kHz float32 (`speak_server.py:385-392`).
  2. It plays the ear-closed cue.
  3. Only then does it transcribe with `mlx_whisper`, in this process.
  4. It plays the ack: a rising chime plus Kokoro saying "Processing."
     (Owner decisions 2026-10-16: step 2 is removed, and step 4 keeps only
     the rising chime; the spoken ack and `SPEAK_ACK_TEXT` are gone.)
     (`speak_server.py:370-374`).
- **Call.** `_call_sync` takes the lock and runs these checks in order:
  quiet hours, daemon not running, already connected, then the greeting
  length (≥ 6 s, measured on the synthesized audio). Only then does it call
  `telegram.call(audio, 24000)`. If the call is answered, it listens under
  the same lock hold (`speak_server.py:502-541`). `_hang_up_sync` always
  clears the expectation (`speak_server.py:544-553`).
- **Debug and status.** `call_debug` and `set_call_debug` go to the daemon
  without the lock (`speak_server.py:723-762`). `status()` returns
  `{busy, current_tool, waiting, call}` and never raises
  (`speak_server.py:774-796`).
- **Error mapping.** `_run` turns `TimeoutError`, `RuntimeError` and
  `ValueError` into `ToolError` with the message
  (`speak_server.py:565-578`).
- **Imports.** `speak_server` imports `numpy`, `speak_telegram`,
  `speak_telegram_daemon` and `speak_tone` (`speak_server.py:42-48`).
  `speak_telegram` imports `ntgcalls` at module load
  (`speak_telegram.py:42`), so the speak server loads ntgcalls although it
  never uses it.
- **Result suffix.** Every speak, listen and converse result ends with
  `\n(audio: telegram)` (`speak_server.py:82`). A permission rule in the
  owner's `~/.claude/settings.json` keys on that text: it treats approvals
  the owner speaks on a call as the owner's approval.

### 2.2 speak_telegram_daemon.py: the daemon and its client

Telegram-only changed only the module docstring here (`speak_telegram_daemon.py:24`).

- **Framing.** A 4-byte big-endian header length, a JSON header, then
  `payload_len` raw bytes. Each connection carries one request and one
  reply (`speak_telegram_daemon.py:78-92`, docstring `9-17`).
- **Ops.** `ping`, `state`, `debug`, `set_debug`, `call`, `play`, `record`,
  `hang_up` (`speak_telegram_daemon.py:174-206`). `state`, `debug` and
  `set_debug` never touch the audio lock or the call loop.
- **Single instance.** The daemon holds `flock` on `<socket>.lock`, and the
  socket is mode 0600 (`speak_telegram_daemon.py:226-239`).
- **Restart on audio failure.** After replying with
  `answered but audio failed to connect`, the daemon exits 1 so that
  launchd restarts it (`speak_telegram_daemon.py:171-172`, `246-250`,
  `381-392`). Every exit path goes through `os._exit`
  (`speak_telegram_daemon.py:396-400`).
- **Credentials.** `load_env` parses `TELEGRAM_*` from the repo's `.env`
  *file* (`speak_telegram_daemon.py:60-61`, `351-358`).
- **Client.** `TelegramClient` opens a fresh socket per request, each op
  with its own reply budget.
  - A stalled daemon becomes a `RuntimeError`, never a `TimeoutError`
    (`speak_telegram_daemon.py:269-346`).
  - A missing socket reads as `NOT_RUNNING` (`speak_telegram_daemon.py:294-300`).

### 2.3 speak_telegram.py: the call

- **Opener, then raw frames.** The greeting is written as a 48 kHz WAV and
  played as the opener file. The stream then switches to
  `ExternalMedia.AUDIO` (`speak_telegram.py:180-194`, `415-506`). A greeting
  shorter than `OPENER_MIN_SECONDS` = 6 s raises (`speak_telegram.py:52`,
  `187-188`).
- **Pump.** The 10 ms pump sends one frame every 10 ms, silence when idle,
  and runs on the link's asyncio loop thread (`speak_telegram.py:240-241`).
  When it falls more than 50 ms behind, it resets its clock instead of
  bursting, and records a `pump_stall` event (`speak_telegram.py:56`,
  `587-624`).
- **VAD.** Silero runs on the CPU and is preloaded at link start
  (`speak_telegram.py:197-218`, `282-284`). Capture uses threshold 0.5, a
  0.5 s pre-roll, and 32 ms blocks downsampled from 48 kHz to 16 kHz
  (`speak_telegram.py:57`, `671-718`). The VAD runs on the socket handler
  thread that called `record`, not on the asyncio loop.
- **caffeinate (macOS only).** While a call is connected, the link runs
  `caffeinate -i -w <pid>`. This is still present in the working tree
  (`speak_telegram.py:232`, `239`, `253`, `393-396`, `475`).

### 2.4 Already removed (Verified, working tree; not changed by this design)

- `speak_audio_worker.py` and its tests.
- `sounddevice`, along with the `SPEAK_INPUT_DEVICE`, `SPEAK_OUTPUT_DEVICE`
  and `SPEAK_SAVE_DIR` settings (`CLAUDE.md` configuration table,
  `pyproject.toml` diff).
- The Mac route.
- `docs/AUDIO_LIFECYCLE_AUDIT.md` and `docs/PORTAUDIO_REINIT_NOTES.md`.

## 3. patrick today (Verified, read-only commands, 2026-10-15 ~13:00 PDT)

| Fact | Value |
|---|---|
| CPU / RAM | Intel i5-9600K, 6 cores; 62 GB RAM. Load average 1.16 / 1.27 / 1.32 |
| OS / TZ | Ubuntu 24.04.4, kernel 7.0.0-38; `America/Los_Angeles` (`timedatectl`) |
| GPUs | 2× GeForce RTX 3060, 12288 MiB each, compute capability 8.6 |
| Driver | 580.178.04 (the loaded module matches), reports CUDA 13.0 |
| GPU0 (bus 01:00.0) | 3857 MiB used, 8051 MiB free, 0 % util. One process: `whisper-service` (uvicorn, pid 2871), 3848 MiB |
| GPU1 (bus 03:00.0) | 3037 MiB used, 8873 MiB free, **99 % util, 168 W / 170 W**. One process: `train_kronos.py` in container `kronos_f5_gbpusd_rg_ft_s42`. That container and `aimodels-trainer-p` both reserve `DeviceIDs ["1"]` |
| Docker | 29.4.2, Compose v5.1.3. Runtimes `nvidia` and `runc`; **the default runtime is `nvidia`** (`/etc/docker/daemon.json`). `nvidia-container-toolkit` 1.19.1 |
| GPU in containers | Already works: `whisper-service` (GPU0) and the Kronos trainer (GPU1) run CUDA processes in containers. No setup step is needed |
| Firewall | `ufw` inactive |
| Port 8765 | Nothing listening (`ss -ltnu`) |
| Suspend | No suspend entries in the 30-day journal. Default target `multi-user.target`; GNOME AC sleep is `'nothing'` |
| ffmpeg | Not installed on the host (the image installs its own, section 7) |

**The existing Whisper service on GPU0 (Verified):**
- Container `whisper-service`, compose project `whisper-service`, on port
  9000, with `restart: unless-stopped` and a reservation of `count: 1` GPU
  (it lands on GPU0).
- Source: `algolearn/scratch/whisper-service/`, outside this repo.
- Model: faster-whisper **`large-v3`** (not turbo), fp16, greedy decoding,
  English.
- API: `POST /` takes raw 16 kHz int16 PCM and returns `{"text"}`.
  `GET /health` also exists.
- VRAM: 3848 MiB.
- Use: its log shows **no `POST /` requests in the last 72 h**. Its only
  known caller is `scratch/voice-claude/livekit_agent.py`.

Owner decision 6 rules it out as a dependency, and Owner decision 8 retires it.

**Other GPU tenants (Verified, read in docs):**
- `lan/inventory.md:60-61` says that `llama-server` (disabled today) splits a
  model across both GPUs at ~9 GB each.
- `live/algolearn-ai-models/CLAUDE.md:55-58` records an owner decision of
  2026-10-12: GPU0 is retired from training and "goes to whisper-service and
  the new image-gallery manager". GPU1 is the patrick training GPU.

## 4. The new split

### 4.1 Modules (in the repo's existing nouns)

| Module | Today (working tree) | After |
|---|---|---|
| `speak_server.py` | MCP tools, engines, the two-layer lock, call expectation, tool logic | **Thin MCP client.** Keeps the tool definitions and docstrings, `_run`, and the per-session `_call_expected` / `_call_generation`. Imports only `mcp`, `anyio` and `speak_protocol`: no numpy, no torch, no ntgcalls |
| `speak_telegram_daemon.py` | Unix-socket server, `TelegramClient` | **The daemon:** TCP server and the audio lock. It also takes the tool logic moved from `speak_server`: `_speak_impl`, `_listen_impl`, `_synthesize`, `_call_sync`, `_hang_up_sync`, `_require_call`, `_guard_call`, `_after_capture`, quiet hours, `_plain`, `_ack`, `_cue`. That logic calls `TelegramLink` directly instead of over IPC. `TelegramClient` moves to `speak_server.py` and speaks TCP |
| `speak_engines.py` | (`Engines` is in `speak_server.py:162-196`) | `Engines`, moved: Kokoro and Whisper on CUDA (section 6) |
| `speak_telegram.py` | `TelegramLink` | `caffeinate` removed (section 5). Silero stays on the CPU, as today |
| `speak_protocol.py` | (inside `speak_telegram_daemon.py`) | **New module, justified:** the framing (`send_msg` / `recv_msg`), the op names, and the outcome and state strings (`ANSWERED`, `NOT_ANSWERED`, `CONNECTED`, `END_HUNG_UP`, ...). The client needs these strings without importing `speak_telegram`, which loads ntgcalls at import (`speak_telegram.py:42`). It uses the stdlib only. `speak_telegram.py` imports its constants from it, so each string is still defined once |
| `speak_tone.py` | cue synthesis | Unchanged; used by the daemon only |

### 4.2 Protocol over TCP

**Decision: keep the existing framing and one connection per request, over
a TCP stream.** The framing in `speak_telegram_daemon.py:78-92` does not
depend on the transport. It moves to `speak_protocol.py` unchanged, and
`MAX_HEADER_BYTES` and `MAX_PAYLOAD_BYTES` stay as bounds.

- **Address:** the daemon listens on `0.0.0.0:8765` on patrick (host
  networking, section 7). The client reads `SPEAK_DAEMON`
  (`192.168.86.31:8765`) and has no default. An unset or malformed value
  fails at startup, per the workspace's no-fallback rule.
- **Open to the LAN by design (Owner decision 4).** There is no
  authentication and no TLS. Any agent on any LAN computer that has the thin
  client can call the owner and hold a spoken conversation. This is the
  intended capability.
- **Two reply messages for the tool ops.** Tool ops wait for the audio lock
  inside the daemon. A request can wait in that queue for any length of
  time, as it can today on `flock` (`speak_server.py:240`).
  - The daemon sends `{"ok": true, "phase": "started"}` the moment it holds
    the lock, then the final reply.
  - The client waits for `started` with no timeout, as today's queue never
    rejects.
  - TCP keepalive is on in both directions, so a dead peer is detected.
  - After `started`, the client applies a budget (below). An overrun raises
    `RuntimeError` "daemon did not reply", never a `TimeoutError`, so it is
    never mistaken for listen's "no speech detected". This is the rule in
    `speak_telegram_daemon.py:283-287` today.
- **Ops:**

| Op | Request fields | Lock | Reply |
|---|---|---|---|
| `ping` | none | no | `{}` |
| `status` | none | no | `{busy, current_tool, waiting, call: {state, generation, end_reason}}` |
| `speak` | `text`, `expect_generation` | yes | `{seconds, expect?}` |
| `listen` | `max_seconds`, `silence_seconds`, `start_timeout_seconds`, `expect_generation` | yes | `{text, expect?}` |
| `converse` | `text` + listen fields + `expect_generation` | yes | `{text, expect?}` |
| `call` | `greeting`, `override_quiet_hours` + listen fields + `expect_generation` | yes | `{outcome, text?, expect?}` |
| `hang_up` | none | yes | `{outcome, expect: null}` |

  `expect` is present only when the session's expectation changed: a string
  key, or `null` to clear. An error reply carries it too when it changed. The
  key is `"<boot id>:<call generation>"`; the boot id is random per daemon
  process, because generations restart at 0 after a restart and a key from
  before it must never match a new call.
| `debug` | `lines` | no | as today (`speak_telegram_daemon.py:181-186`) |
| `set_debug` | `enabled` | no | as today (`speak_telegram_daemon.py:187-192`) |

  `play`, `record` and `state` are retired as wire ops. Audio never crosses
  the LAN; only text does.

  Errors come back as `{"ok": false, "error", "kind"}`, with `kind` one of
  `timeout`, `value`, `call_ended` or `runtime`. The client re-raises
  `timeout` as `TimeoutError` (no speech), `value` as `ValueError` (short
  greeting, empty text) and the others as `RuntimeError`. The exception
  types a tool sees stay as they are today (`speak_server.py:565-578`).
- **Client budgets after `started`.** Each one guards against a stalled
  daemon. None of them limits behavior.

  | Op | Budget |
  |---|---|
  | speak | `60 + len(text)/5` s. Speech runs about 15 characters per second, so this gives 3× slack |
  | listen | `start_timeout + max_seconds + 60` |
  | converse | the speak budget plus the listen budget |
  | call | `RING_TIMEOUT_SECONDS (60) + CONNECT_GRACE_SECONDS (30) + len(greeting)/5 + listen budget` |
  | hang_up | 20 s (`QUICK_REPLY_TIMEOUT_SECONDS`, `speak_telegram_daemon.py:52`) |
- **A client that disconnects mid-request** (a closed session, a cancelled
  tool) does not cancel the daemon's work. The op finishes and its reply is
  discarded. This is today's behavior too: the worker thread in `_run`
  cannot be cancelled (`speak_server.py:565-570`).

### 4.3 The audio lock lives in the daemon

**Decision:** one lock, in the daemon process, replaces `_SerializingLock`
and `_CrossProcessLock`. It is a FIFO ticket lock: a `threading.Condition`
with a deque of waiters, plus `current_tool` and the holder's peer address.
- **Why FIFO:** today's ordering comes from the kernel queueing on `flock`
  (`speak_server.py:240`), and a bare `threading.Lock` promises no
  fairness.
- **Waiter count:** every session on every computer queues in one place, so
  `status().waiting` counts every waiting caller. Today it counts only the
  caller's own process (`speak_server.py:786-789`).
- **Removed:** `~/.algolearn-speak/audio.lock`, its sidecar, and the CLAUDE.md
  advice to run tests with a scratch `HOME` because of that lock.

### 4.4 The per-session call expectation

**Decision: the expectation stays per session, in the client, and the check
runs in the daemon under the lock.** The rule that a later drop raises in
every session that talked on the call needs memory per session
(`speak_server.py:76-80`, `92-107`). The thin client keeps `_call_expected`
and `_call_generation`. With every tool op it sends `expect_generation`: its
key, or `null`.

Under the lock, the daemon applies `_require_call`'s rules to that value:

| Call state | What happens |
|---|---|
| `connected` | The op proceeds. The reply carries the `expect` key the session now holds |
| Ended by `hang_up` (`end_reason == hung_up`), key matches | The reply tells the client to clear its expectation. The op raises the no-call error below |
| Not connected, generation given | A `call_ended` error: "Telegram call ended during <tool>; call back ...". The client clears its expectation, as `_call_ended_error` does today (`speak_server.py:85-89`) |
| No call, no expectation | `RuntimeError` "<tool> needs a connected Telegram call (call state is ...); place one with the call tool". The message is unchanged from today |

`_guard_call` and `_after_capture` move into the daemon with their logic
unchanged. They read `link.state()` directly instead of over IPC.

**Decision:** every speak, listen and converse result still ends with
`\n(audio: telegram)`. The permission rule in the owner's settings matches
on that text (section 2.1).

### 4.5 Tool mapping

| Tool (signature unchanged) | Client does | Daemon does |
|---|---|---|
| `speak(text)` | `speak` op; returns `spoke for Ns\n(audio: telegram)` | lock → call check → `_plain` → Kokoro (all chunks) → `link.play` → guard |
| `listen(max, silence, start_timeout)` | `listen` op | lock → call check → ear-open cue + `link.record` → ear-closed cue → Whisper → ack (chime + "Processing.") → guard. Since owner decisions 2026-10-16: ear-open cue + `link.record` → Whisper → rising chime → guard (no ear-closed cue, no spoken ack) |
| `converse(text, ...)` | `converse` op | speak then listen, one lock hold |
| `call(greeting, override_quiet_hours, ...)` | `call` op; sets its expectation from the reply | lock → quiet hours (daemon clock, `TZ=America/Los_Angeles`) → already connected → Kokoro greeting and the ≥ 6 s check → `link.call` → on `answered`, the listen above in the same hold. Outcomes and check order are as in `speak_server.py:502-541` |
| `hang_up()` | `hang_up` op; always clears its expectation (`speak_server.py:552`) | lock → `link.hang_up()` |
| `status()` | `status` op; never raises | no lock; busy, current_tool, waiting, call |
| `call_debug(lines)` / `set_call_debug(enabled)` | as today | as today, no lock |

**`status()` shape (Decision):** `{busy, current_tool, waiting, call}`, as
today (`speak_server.py:780-796`).
- `call` is `none`, `ringing`, `connected`, or `error: <reason>`.
- When the client cannot reach the daemon, `call` is
  `daemon unreachable: <error>` (today it is `daemon not running`).
- `current_tool` becomes `"<tool> (<peer ip>)"`.

**Daemon not reachable (Decision):**
- Every tool except `status`, `call_debug` and `set_call_debug` raises
  `RuntimeError` "speak daemon at <addr> is not reachable: <error>".
- `call_debug` and `set_call_debug` return
  `{"daemon": "unreachable: <error>"}` (today `{"daemon": "daemon not running"}`,
  `speak_telegram_daemon.py:329-334`).

### 4.6 Opener and raw-frame pump

These are unchanged: the 6 s opener file, the 1 s tail, the switch to raw
frames, the 10 ms pump that never bursts, and teardown on every failure
(`speak_telegram.py:415-506`, `587-643`). On Linux these spike quirks are
**not yet verified**; section 9 re-tests them.

**Decision: inference never overlaps outgoing speech.** Today the daemon
process runs only VAD. After the move, Kokoro and Whisper run in the same
process as the pump's asyncio loop. The Python parts of their work (misaki
and spaCy G2P, faster-whisper's decode loop) compete with the pump for the
GIL. The current sequence already keeps them apart:
- speak synthesizes every chunk before playing any of it
  (`speak_server.py:414-422`);
- `link.play` blocks until the queue drains (`speak_telegram.py:650-667`);
- Whisper runs only after the ear-closed cue has played
  (`speak_server.py:435-438`). (Owner decision 2026-10-16 removed the
  ear-closed cue, the chime and the ack; Whisper now runs right after the
  capture, with nothing queued to play, so the pump still sends only
  silence. Owner decision later on 2026-10-16: the rising chime is back,
  queued only after Whisper has the transcript.)

So while models run, the pump sends only silence frames. The move keeps that
order as a rule.

**Superseded for Kokoro during speak (owner decision 2026-10-16):** speak now
synthesizes each segment while earlier ones play; the greeting is still
synthesized whole, and Whisper still never overlaps outgoing speech. See
`docs/DESIGN_STREAMED_SPEECH.md`.

Silero VAD stays on the CPU (Owner decision 9). It runs during capture, when
the pump is sending silence, as it does on the Mac today. Section 9 re-tests
on Linux whether a stall among silence frames is harmless. The pump's
`pump_stall` events (`speak_telegram.py:619-623`) record any stall on every
call.

## 5. What this design deletes

These are additions to what Telegram-only already removed (section 2.4).
- From `speak_server.py`:
  - `Engines` and the MLX imports;
  - `_CrossProcessLock`, `_SerializingLock` and the lock paths;
  - the tool logic, which moves to the daemon (section 4.1).
- `mlx-whisper` and `mlx-audio` from the dependencies.
- `caffeinate` in `speak_telegram.py` (the lines in section 2.3) and
  `tests/fake_caffeinate`. It is macOS only, and patrick does not suspend
  (section 3).
- `launchd/com.algolearn.speak-telegram.plist`, the Unix socket, its `.lock`
  file and `SPEAK_TELEGRAM_SOCKET`.

  **Decision:** on patrick, the TCP bind is the single-instance guard. A
  second daemon gets `EADDRINUSE` on 8765, and compose's fixed
  `container_name` refuses a second container.
- In `CLAUDE.md`: the launchd install steps and the Unix-socket and
  `audio.lock` text, plus the MLX model defaults. These are rewritten to
  match this design in the same change.

## 6. GPU, models and VRAM

### 6.1 Which GPU

**GPU0, pinned with `device_ids: ["0"]`** (Owner decision 7). The verified
state supports this:
- GPU1 is not free. A Kronos training run holds it at 99 % util and 168 W of
  170 W, and it is the patrick training GPU.
- GPU0 is idle (0 % util) with 8051 MiB free, and the owner has assigned it
  to services, not training.

### 6.2 Model packages

**TTS: `kokoro==0.9.4`** (hexgrad's reference PyTorch package).
- Today: `mlx-audio` with `mlx-community/Kokoro-82M-bf16`, voice `af_heart`.
- After: `KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", device="cuda")`,
  voice `af_heart`.
- Why:
  - The Kokoro-82M weights and voices are the same, in their reference
    implementation, so the voice stays the same.
  - G2P is the same: `misaki[en]` 0.9.4 is already locked in `uv.lock`.
  - It runs on torch, which Silero needs anyway, so there is one tensor
    stack.
  - `kokoro-onnx` with `onnxruntime-gpu` was rejected: it adds a second GPU
    runtime for an 82M model.
- **Unverified, read before build:** the `KPipeline` constructor arguments
  (`repo_id`, `device`) and `split_pattern` in 0.9.4.

**STT: `faster-whisper==1.2.1`** (CTranslate2).
- Today: `mlx-whisper` with `mlx-community/whisper-large-v3-turbo`.
- After: model `large-v3-turbo`, `device="cuda"`, `compute_type="float32"`
  (Owner decision 2026-10-15: both models at full precision),
  `beam_size=1`, `language="en"`, `vad_filter=False`.
- Why:
  - It is the same model as today (Owner decision 6).
  - faster-whisper is already proven on this card by `whisper-service`
    (Verified, section 3).
  - Greedy decoding matches `mlx_whisper.transcribe`'s default.
  - VAD has already run before Whisper sees the audio, so `vad_filter` is
    off.
- **Unverified, read before build:**
  - that 1.2.1 maps `large-v3-turbo` to a CTranslate2 conversion (its
    `utils._MODELS` table);
  - which CUDA and cuDNN major versions the matching `ctranslate2` wheel
    needs.

**VAD: `silero-vad==6.2.1` on the CPU** (Owner decision 9). It runs in the
daemon, unchanged from `speak_telegram.py:197-218`.
- Each 32 ms block is a tiny model, so moving it to the GPU would add a copy
  and a kernel launch per block for no gain.
- The code is unchanged, so there is nothing new to verify.

**Torch: `torch==2.14.0`** from the PyTorch CUDA 12.6 wheel index on Linux,
PyPI on macOS (`[tool.uv.sources]` with a platform marker).
- Driver 580.178.04 supports CUDA ≤ 13.0 (Verified).
- Verified at build: 2.14.0 has no cu128 build; the cu126 `manylinux_2_28`
  wheel is locked in `uv.lock`. One CUDA major (12) in the process, shared
  with CTranslate2.

**Settings.**
- New defaults: `SPEAK_KOKORO_MODEL` becomes `hexgrad/Kokoro-82M` and
  `SPEAK_WHISPER_MODEL` becomes `large-v3-turbo`.
- These two, plus `SPEAK_VOICE`, `SPEAK_SPEED`, `SPEAK_LANGUAGE`,
  `SPEAK_ACK_TEXT` (removed 2026-10-16) and `SPEAK_QUIET_HOURS`, move to the daemon's compose
  `environment`.
- The client reads only `SPEAK_DAEMON`.

**Daemon startup order (Decision):**
1. Load and warm the engines, as `speak_server.py:179-182` does today.
2. Preload Silero.
3. Connect Telegram.
4. Bind port 8765.

So a refused connection always means "not ready or not running", never
"running without models". A load failure exits non-zero, with no partial
service, and compose restarts the container.

### 6.3 VRAM budget on GPU0

| Component | VRAM | Basis |
|---|---|---|
| whisper-service (existing, large-v3 fp16), until it is retired | 3848 MiB | **Measured** (nvidia-smi, 2026-10-15) |
| CUDA context, daemon process (one per process, shared by torch and CTranslate2) | ~300–500 MiB | **Estimate**: typical for an Ampere card with torch's kernels loaded |
| Whisper large-v3-turbo, fp32 (809 M parameters) | ~3.8–4.5 GB | **Estimate**: ~3.2 GB of weights (809 M × 4 bytes) plus ~0.6–1.3 GB of CTranslate2 buffers for a 30 s window with greedy decoding |
| Kokoro-82M, fp32 | ~0.5–0.8 GB | **Estimate**: ~0.3 GB of weights (82 M × 4 bytes) plus torch's caching allocator for one sentence |
| Silero VAD | 0 | On the CPU (Owner decision 9) |
| **Daemon total** | **~4.6–5.8 GB** | Estimate (the sum of the rows above) |
| **GPU0 during cutover** (both running) | **~8.5–9.7 GB of 12.0 GB used** | 3.85 GB measured + the daemon estimate |
| **GPU0 after whisper-service is retired** | **~4.6–5.8 GB used; ~6.2–7.4 GB free** | Daemon estimate only. That free memory is for the planned image-gallery manager |

Section 9 replaces every estimate here with a measurement:
`nvidia-smi --query-compute-apps` after warm-up and after a 60 s listen.

## 7. Deployment on patrick

The daemon is a compose service, built and started on patrick from this repo
with the `patrick` docker context (owner decision 2026-10-15; no deploy
directory on patrick): `docker --context patrick compose up --build -d`.
Compose reads the git-ignored `.env` from the repo directory and passes it
via `env_file`; `.dockerignore` keeps it out of the image. The
workspace deployment rules apply: always rebuild, never copy files into a
container, never `down` first, and never run tests or interpreters inside
the running container. Every build, restart and stop is an owner action,
approved per change.

Files added to the repo (Owner decision 6: every dependency comes from this
repo):

**`Dockerfile`**
- Base image: `nvidia/cuda:12.6.3-runtime-ubuntu24.04`, the CUDA major of
  the torch cu126 wheel. CTranslate2 needs cuBLAS 12 and cuDNN 9; both come
  from the `nvidia-cublas-cu12` and `nvidia-cudnn-cu12` pip packages torch
  installs, put on `LD_LIBRARY_PATH` so the loader finds them whatever the
  import order.
- apt installs `ffmpeg`, `espeak-ng` and `tzdata`. py-tgcalls plays the
  opener file through ffmpeg (`docs/DESIGN_TELEGRAM_CALL.md`, section 9).
  Kokoro's out-of-vocabulary G2P needs espeak-ng, and the daemon refuses to
  start without it. Without tzdata, glibc silently runs `TZ` as UTC; the
  daemon also resolves `TZ` with `zoneinfo` at startup and refuses to start
  when it does not resolve.
- Python 3.12 comes from `uv` (the repo's tool), via
  `uv sync --locked --extra daemon`.
- The entrypoint is `speak-telegram`.

**`docker-compose.yml`**
- Service `speak-telegram`, with `container_name: speak-telegram` and
  `restart: unless-stopped`.
- `network_mode: host`. **Decision:** the call's WebRTC media is UDP, and
  host networking avoids a second NAT layer in front of ICE. With it, the
  daemon binds 8765 on patrick's LAN address directly.
- `env_file: .env` with `TELEGRAM_API_ID`, `TELEGRAM_API_HASH`,
  `TELEGRAM_SESSION` and `TELEGRAM_CALL_TARGET`.
  - `.env` is already git-ignored (Verified, `.gitignore`).
  - It is also listed in a new `.dockerignore`, so it is never baked into
    an image layer.
- `environment`:
  - `TZ=America/Los_Angeles`, set in compose as the IB bots do (workspace
    `CLAUDE.md`, "IB bot timezone");
  - the `SPEAK_*` model and quiet-hours settings from section 6.2.
- `deploy.resources.reservations.devices`: `driver: nvidia`,
  `device_ids: ["0"]`, `capabilities: [gpu]`.
- Volume `speak_models:/root/.cache/huggingface`, so the models download
  once, not on every rebuild.
- `stop_grace_period: 30s`. A hang-up (≤ 10 s) and a disconnect (≤ 5 s)
  finish before SIGKILL. This carries over the plist's `ExitTimeOut`.

**Code changes that support the deployment**
- **`load_env`** reads the four `TELEGRAM_*` values from the process
  environment, which compose fills from `.env`. It still fails at startup
  and names any value that is missing (`speak_telegram_daemon.py:351-358`).
- **Restart on audio failure:** the daemon still exits 1 after replying.
  `restart: unless-stopped` takes the place of launchd's `KeepAlive`. Until
  the models reload, clients get "not reachable" (section 8 estimates how
  long).
- **Logs** go to stdout and stderr. Alloy ships them to Loki and Grafana, as
  it does for every patrick container (`lan/inventory.md`, "Containers on
  patrick").

**pyproject split (Decision)**
- The base dependencies become only what the thin client needs: `mcp`, with
  `anyio` coming through mcp.
- An optional extra, `daemon`, holds `telethon==1.45.0`, `py-tgcalls==3.0.0`,
  `ntgcalls==3.0.2`, `torch==2.14.0`, `kokoro==0.9.4`, `misaki[en]==0.9.4`,
  the `en-core-web-sm` wheel, `faster-whisper==1.2.1`, `silero-vad==6.2.1`,
  `numpy` and `scipy`.
- A client computer runs `uv sync`, which installs no torch and no audio
  stack.
- The MCP registration command stays the same. The `speak` entry's `env` in
  `~/.claude.json` becomes `{"SPEAK_DAEMON": "192.168.86.31:8765"}`.

**Telegram pins on Linux**
- `ntgcalls` 3.0.2 publishes a `cp312-manylinux_2_28_x86_64` wheel
  (Verified, PyPI). `py-tgcalls` 3.0.0 and `telethon` 1.45.0 are pure Python
  (Verified, PyPI). Ubuntu 24.04's glibc is newer than 2.28.
- A wheel existing is not the same as private calls working on Linux. Every
  quirk in `docs/DESIGN_TELEGRAM_CALL.md` section 2 was seen on macOS only.
  The pins **need a re-test on Linux** before cutover (section 9).

**One login, one connection.** The Mac's launchd daemon and the patrick
container must never be connected at the same time, because both use the
same `TELEGRAM_SESSION`. Telegram is reported to invalidate an auth key that
is used from two places at once (`AuthKeyDuplicatedError` in Telethon).
**Unverified, read before build.** The cutover order treats it as true.

## 8. Expected latency effect

**Measured on this Mac** (M1 Pro, the repo venv, 2026-10-15). Each figure is
three warm runs on the 12.22 s greeting from the `call` tool's example, with
no audio device involved:
- Kokoro (MLX), full synthesis: **0.83–0.87 s** (1.15 s on the first run).
- Whisper large-v3-turbo (MLX), on the same 12.22 s at 16 kHz: **0.78–0.83 s**.
- Engine load and warm-up: **9.0 s**.

**Measured, LAN:** the ICMP round trip from this Mac to patrick, over 20
pings, averages **3.17 ms** (max 4.03 ms).

**Estimates for patrick.** These are not measured, because measuring would
have meant starting a GPU container.
- Kokoro-82M on an RTX 3060 (torch fp32): **~0.2–0.4 s** for the same 12 s
  of speech. An 82 M model is dominated by kernel launches. G2P runs on the
  CPU on both machines.
- faster-whisper large-v3-turbo fp32 on a 3060: **~0.5–1.0 s** for 12 s of
  audio: one encoder pass over a 30 s window plus 4-layer greedy decoding.
- Engine load at daemon start: **~15–40 s** (CUDA init plus loading models
  from the cache volume). The first start downloads the models and takes
  minutes.
- Wire cost per tool call: **~5–15 ms**, for one TCP connect plus a request
  and reply at about a 3 ms round trip. This estimate is built on the
  measured round trip. Audio no longer crosses any socket.

**Net per converse turn on a call (Estimate): about 0.6–1.2 s faster.**
- TTS saves about 0.85 − 0.3 s, and STT about 0.8 − 0.45 s.
- The wire adds about 10 ms.
- At the low end of the estimates it is no worse than today.
- The fixed parts are unchanged: the 6 s opener, the cues,
  `silence_seconds` of end-of-speech wait, and the "Processing." ack (ack removed 2026-10-16; the rising chime still plays after Whisper).
- After an audio-failure restart, the daemon is unreachable for the engine
  load time above. Today's launchd restart takes a few seconds.

## 9. Build plan, checks and cutover

1. **Read-before-build checks.** Each one is resolved by reading code or a
   package index before writing the code that depends on it:
   - kokoro 0.9.4 `KPipeline` arguments;
   - faster-whisper 1.2.1's `large-v3-turbo` mapping, and the CUDA and cuDNN
     major versions of its `ctranslate2`;
   - torch 2.14.0 CUDA wheels (resolved: cu126, section 10);
   - the CUDA base image tag (resolved: `nvidia/cuda:12.6.3-runtime-ubuntu24.04`, section 10);
   - Telethon's behavior when one session is used from two places at once.
2. **Code, built on the Telegram-only working tree once it is committed:**
   - `speak_protocol.py` and `speak_engines.py`;
   - the daemon: TCP server, FIFO lock and the moved tool logic;
   - the thin `speak_server.py`;
   - the section 5 deletions;
   - `Dockerfile`, `docker-compose.yml` and `.dockerignore`;
   - the pyproject split and the `CLAUDE.md` update.
3. **Tests, in a local venv.** No test contacts Telegram or touches a GPU.
   - framing and ops over a localhost TCP socket, against a fake link and
     fake engines (carried over from `tests/test_speak_telegram_daemon.py`);
   - the expectation rules with `expect_generation`: drop, hang-up on
     request, already connected, no call (carried over from
     `tests/test_speak_server_telegram.py`);
   - FIFO order and `waiting` across several concurrent clients;
   - the client's two-message reply, its budgets and its error-kind mapping;
   - a client import test: `import speak_server` imports no numpy, torch or
     ntgcalls.
4. **Independent code review** (code-reviewer), then the owner's approval to
   build the image.
5. **Linux re-test on patrick, before cutover**, after an owner-approved
   `up --build -d`. Boot out the Mac daemon first (section 7). Check that:
   - the 6 s opener to raw-frame switch gives audio both ways;
   - the pump produces no sped-up audio, and `pump_stall` counts stay low,
     including while Whisper and Kokoro run between turns;
   - decline, no answer and an owner hang-up map to today's outcomes;
   - a restart after `answered but audio failed` recovers under compose;
   - host networking carries the media;
   - the VRAM of section 6.3 and the TTS and STT timings of section 8 are
     measured on the same 12.22 s greeting;
   - a three-turn converse works, and two sessions on two LAN computers can
     share one call.
6. **Cutover (owner actions):**
   1. On the Mac: `launchctl bootout gui/$(id -u)/com.algolearn.speak-telegram`.
   2. Start the container on patrick.
   3. Point each computer's `speak` MCP entry at `SPEAK_DAEMON` and
      reconnect its sessions.

   **Rollback:** stop the container
   (`ssh frankg@192.168.86.31 docker stop speak-telegram`), re-bootstrap the plist
   on the Mac, and check out the pre-change commit of this repo. The Mac's
   MLX venv is left in place until the owner retires it.
7. **Retire `whisper-service` (Owner decision 8, an owner action).** Do this
   only after step 6 is verified working:
   - Take it offline from its compose directory:
     `docker --context patrick compose down`, run in
     `algolearn/scratch/whisper-service/`. Taking a service offline is the
     one use of `down` the workspace rules allow.
   - Confirm with `nvidia-smi` that its ~3.8 GB on GPU0 is free.
   - `scratch/voice-claude/livekit_agent.py`, its only known caller, stops
     working at that point, until the daemon's HTTP whisper API (section 11)
     is deployed on the same port.

## 10. Build record (2026-10-15)

Owner decisions given with the build instruction, superseding the earlier
text of this design where they differ (the sections above are updated to
match):
- Both models at full precision: faster-whisper `compute_type="float32"`,
  Kokoro fp32. `Engines.load` refuses a Kokoro load in any other dtype.
- Only the speak daemon runs on GPU0; `whisper-service` is retired later by
  the owner.
- The Linux Telegram re-test of section 9 step 5 is skipped before the build.
- No new tests: the obsolete Mac-daemon and old-client tests are deleted;
  `tests/test_speak_telegram.py` remains.
- The daemon is deployed from the repo with the `patrick` docker context; the
  `.env` stays in the repo directory on the deploying computer, and there is
  no deploy directory on patrick.

Verified while building:
- torch 2.14.0 has no cu128 build; the cu126 wheel is used (section 6.2).
- `nvidia/cuda:12.6.3-runtime-ubuntu24.04` exists, and the image builds on
  patrick with `uv sync --locked --extra daemon`.
- kokoro 0.9.4 `KPipeline` takes `repo_id`, `device` and `split_pattern`.
  It swallows an espeak-ng load failure, which the daemon turns into a load
  error.
- faster-whisper 1.2.1 maps `large-v3-turbo` to
  `mobiuslabsgmbh/faster-whisper-large-v3-turbo`.

Not verified (they need the container running, an owner action): the Linux
call quirks, Telethon's behaviour with one session in two places, the VRAM of
section 6.3, the timings of section 8, and that torch's cuDNN 9 satisfies
CTranslate2 at run time.

## 11. HTTP whisper API on port 9000 (owner decision 2026-10-16)

The owner retired `whisper-service` and decided that the daemon serves its
API, so its clients need no change:
- A second listener in the daemon, on HTTP `0.0.0.0:9000` (stdlib
  `ThreadingHTTPServer`): `POST /` takes raw 16 kHz mono int16 PCM and returns
  `{"text"}`; `GET /health` returns `{"status":"ok","model","device"}`. The
  status codes, bodies and content types are those of the old FastAPI service.
- It reuses the loaded Whisper engine: no new model, container or setting.
- Every transcription queues in the same FIFO audio lock as the tool ops, so
  inference never overlaps call audio; a request may wait a conversational
  turn.
- The port binds after Telegram connects, with 8765.

Differences from the old service (Decision): `/health` reports the model the
daemon actually runs (`large-v3-turbo` fp32, not `large-v3` fp16);
transcription uses `Engines.transcribe`, whose decoding settings match the
old service (`language="en"`, `beam_size=1`, `vad_filter=False`) but which
joins segment texts with no separator (the old service joined them with a
space, which doubled the space between segments); FastAPI's `/docs`,
`/redoc` and `/openapi.json` pages are not served.

Hardening, a deliberate departure from the old service, which had no limits
(Decision, owner 2026-10-16):
- A body larger than 15 minutes of 16 kHz mono int16 PCM (28,800,000 bytes)
  is refused with `413` `{"detail": ...}` before it is read: Content-Length is
  checked before the read, a chunked body's running total as each chunk size
  arrives. The connection then closes.
- A request that reaches the audio lock after its client has gone is skipped
  without inference, as tool ops are; one that reaches it while the daemon is
  restarting or stopping gets `503` `{"detail":"speak daemon is restarting"}`.
- A 30 s socket timeout closes idle keep-alive and slowloris connections
  (above httpx's 5 s keep-alive expiry, so the client closes an idle
  connection first); a body that stalls is logged and answered `408`. TCP
  keepalive is on, as on 8765.
