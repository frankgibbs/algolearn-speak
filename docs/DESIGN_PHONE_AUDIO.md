# Design: the iPhone as the speak server's microphone and speaker

Written 2026-10-04. Status: design for build. Owner request (voice/text,
2026-10-04): "a mobile app I can side load on my iPhone that mimics the speak
MCP but uses the mic and speakers on my phone over WiFi or cellular."
Decisions the owner made are marked **Owner decision**; decisions I made are
marked **Decision** with the reason; statements about existing code carry
file:line and were read, not assumed.

## 1. What changes and what does not

Today the MCP server (`speak_server.py`) runs Kokoro (TTS), Whisper (STT) and
Silero (VAD) on the Mac and does every audio I/O through a fresh
`speak_audio_worker.py` subprocess that owns `sounddevice`
(`speak_audio_worker.py:1-10`; `speak_server.py:298-371`). The MCP tools
`speak` / `listen` / `converse` and their lock semantics are unchanged by
this design.

**What changes:** the server gains a second audio device, *the phone*: a
WebSocket connection over which it streams speaker PCM out and receives
microphone PCM in. Synthesis, transcription and end-of-speech detection stay
on the Mac. The phone is a dumb microphone and speaker with a status screen.

**Owner decision (2026-10-04): WiFi only for v1.** The phone connects to the
Mac's LAN address directly. No public ingress, no relay. (Cellular was
offered via Tailscale or a relay on patrick; deferred.)

**Owner decision: a sideloaded native app**, not a web page. Reason stated
by the owner (sideload); technical reason recorded: a native app keeps the
mic and playback alive with the screen off (background-audio capability)
and follows Bluetooth route changes, which Safari web pages cannot.

## 2. Routing rule (Decision, with reason)

The server has at most one phone connected. **When a phone is connected,
it is the audio device; when none is, the Mac's devices are.** The active
device is reported by the `status` tool (`audio: phone` / `audio: mac`) and
appended to every `speak`/`listen`/`converse` tool result as a trailing
note, so Claude and the owner always know where the sound went.

Reason: the owner wants to pick the phone up and talk; forcing a mode switch
(env var, restart) would defeat that. This is a documented routing rule,
not a silent fallback: a phone that disconnects *mid-call* causes that call
to raise (`RuntimeError("phone disconnected during listen")`), never a
quiet switch to the Mac mic in the middle of an operation. The switch only
happens between calls.

If a phone connects while the Mac is mid-`speak`, that `speak` finishes on
the Mac; the next call uses the phone.

## 3. Transport and protocol

- Server listens on the Mac on **TCP 8772** (free as of 2026-10-04; 8770 is
  the whiteboard, 8765 the owner's inoviov2 dev server). Plain `ws://` on
  the LAN. (**Decision**: no TLS on LAN for v1; the audio never leaves the
  home network. TLS and auth come with any cellular path.)
- One connection at a time. A second phone connecting is refused with a
  close code and reason `busy`.
- Frames:
  - Text frames are JSON control messages.
  - Binary frames are raw PCM16 little-endian mono. Phone→server is always
    microphone audio at **16,000 Hz** (Whisper/Silero's contract,
    `speak_server.py:59`). Server→phone is always speaker audio at
    **24,000 Hz** (Kokoro's rate, `speak_server.py:61`; cues are generated
    at that rate too, `speak_server.py:278-285`). Float32 → PCM16 conversion
    happens on the server to halve the bytes on the wire.

| direction | message | meaning |
|---|---|---|
| phone→server | `{"type":"hello","app":"algolearn-speak-ios","version":"1","mic_rate":16000,"spk_rate":24000}` | first message; server replies `ready` or closes `busy`/`bad-rates` |
| server→phone | `{"type":"ready","server":"algolearn-speak","version":"0.1.0"}` | accepted |
| server→phone | `{"type":"state","value":"idle"|"speaking"|"listening"|"processing"}` | drives the phone's status screen |
| server→phone | `{"type":"play_start","rate":24000}` → binary chunks → `{"type":"play_end","id":n}` | a speaker segment (a cue, or one TTS sentence) |
| phone→server | `{"type":"played","id":n}` | the phone has finished *rendering* segment `n` (its player queue drained) — the server waits for this before opening the mic, preserving today's "cue finishes before the mic opens" rule (`speak_audio_worker.py:412-418`) |
| server→phone | `{"type":"mic_start"}` / `{"type":"mic_stop"}` | the phone sends mic PCM only between these; outside them it sends nothing (privacy: the mic is streamed only while Claude is listening) |
| phone→server | binary PCM16 @16k, 32 ms blocks (512 samples) while mic is started | the server runs Silero on these blocks exactly as the worker does (`speak_audio_worker.py:388-394`, threshold, pad, silence) |
| either | `{"type":"ping"}` / `{"type":"pong"}` | every 5 s; two missed pongs = disconnect |

## 4. Server-side changes (`algolearn-speak`)

New module **`speak_phone_audio.py`** (one new noun, justified: it is the
phone counterpart of `speak_audio_worker.py` — the only other code that
performs audio I/O; everything else stays where it is).

- `PhoneAudioServer` — a `websockets` server on a background thread with
  its own asyncio loop, started by `speak_server.main()` beside the engine
  loader. Holds the single connection, the state, and two queues: mic
  frames in, speaker segments out. Exposes synchronous methods for the
  server's worker thread: `connected() -> bool`, `play(pcm_f32, rate)`
  (blocks until `played`), `record(max_seconds, silence_seconds,
  start_timeout_seconds, cue) -> np.ndarray | TIMEOUT` (plays the cue,
  waits for `played`, sends `mic_start`, runs Silero on incoming blocks
  with the same VAD parameters and pre-roll as `cmd_record`, sends
  `mic_stop`, returns 16 kHz float32), `set_state(...)`.
- `speak_server.py`: `_play_pcm`, `_play_pcm_stream` and the record path
  (`_start_record_worker` / its handle) consult `phone.connected()` at the
  START of each tool call and route the whole call to the phone or the
  worker. The lock, cues, chime, "Processing" word, Whisper call and return
  values are untouched. `status()` gains `audio`. Tool results gain the
  trailing note.
- Silero import in the server process: today the server deliberately never
  imports `sounddevice` (`CLAUDE.md`, "Audio process isolation"); that rule
  is about PortAudio, not torch. `silero_vad` is already a dependency and
  is loaded fresh per record in the worker; in the phone path it is loaded
  once in `PhoneAudioServer` (no PortAudio involved, so the isolation
  reasons do not apply).
- Dependency: `websockets>=13` added to `pyproject.toml`.
- No fallbacks: a `hello` with other rates is refused; a disconnect mid-call
  raises; a `played` that does not arrive within segment duration + 10 s
  raises.
- Tests (`tests/test_speak_phone_audio.py`): protocol handshake, busy
  refusal, play waits for `played`, record runs VAD on synthetic speech
  blocks and returns the right slice with pre-roll, timeout path,
  mid-call disconnect raises, routing picks phone only when connected.
  Use a real `websockets` client in-process; no network beyond localhost.

## 5. The iPhone app (`ios/AlgolearnSpeak/`, SwiftUI, iOS 17+)

- **AudioEngine**: `AVAudioSession` category `.playAndRecord`, mode
  `.voiceChat`, options `[.allowBluetooth, .allowBluetoothA2DP,
  .defaultToSpeaker]`; `AVAudioEngine` with an input tap converted via
  `AVAudioConverter` to 16 kHz Int16 mono in 512-sample blocks; an
  `AVAudioPlayerNode` fed Float32 buffers at 24 kHz for playback; posts
  `played` when its scheduled buffers for a segment have completed.
- **PhoneLink**: `URLSessionWebSocketTask` to `ws://<host>:8772`; JSON
  control + binary PCM; ping/pong; reconnect with backoff while the app is
  foreground or playing.
- **UI** (one screen): server address field (default `192.168.86.188`),
  Connect / Disconnect, a large state badge (Idle / Speaking / Listening /
  Processing) mirroring `state`, a mic level meter while listening, a log
  of the last few events. Haptic tick on `mic_start`.
- **Capabilities**: Background Modes → Audio, AirPlay, and Picture in
  Picture (audio) so a conversation continues with the screen locked;
  `NSMicrophoneUsageDescription`; Local Network usage description.
- **Sideloading**: an Xcode project with a personal-team signing
  placeholder. The owner opens it in Xcode 26, selects his Apple ID team,
  and runs on the phone. Known constraint, stated: a free Apple ID signs
  for 7 days, after which the app must be re-run from Xcode; a paid
  developer account signs for a year. The project builds for the
  simulator from the command line (`xcodebuild -scheme AlgolearnSpeak
  -destination 'platform=iOS Simulator,name=iPhone 17 Pro'`) so the build
  is verified here before the owner's device step.

## 6. Latency budget (expected, to be measured)

Mic block 32 ms + WiFi ≈ 5–20 ms + VAD ≈ 1 ms; speaker: TTS sentence
streaming already chunked (`speak_server.py:386-392`), first-sound latency
≈ one chunk + network. Comparable to AirPods today. If measured round-trip
exceeds 150 ms on the LAN, the block size is the first knob.

## 7. Out of scope for v1

Cellular / any path off the LAN (owner decision); TLS and authentication
(come with that path); multiple phones; running STT/TTS on the phone;
replacing the Mac worker (it stays the default device).

## 8. Verification before the owner's device step

1. `uv run pytest` green in `algolearn-speak`.
2. Server started, iOS app in the simulator connected to the Mac over
   localhost: `speak("test")` is heard from the simulator; `listen()` on a
   synthetic mic feed returns a transcript; `status()` says `audio: phone`.
3. Disconnect the simulator; `status()` says `audio: mac`; `speak` plays on
   the Mac again.
4. Then the owner installs on the phone and the same three checks run over
   WiFi with his voice.
