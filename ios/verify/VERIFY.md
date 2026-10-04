# Phone-audio verification (design section 8, steps 2-3) -- 2026-10-03

Run on this Mac (MacBookPro, lid closed, 4K external display mirrored) with
the iOS Simulator "iPhone 17 Pro" (96CF3880-2DB4-4BB4-B637-5BB2F0842522,
iOS 26.3, Xcode 26.3) and a TEST instance of the new server code
(`ios/verify/test_driver.py`: loads the engines, starts `PhoneAudioServer` on
0.0.0.0:8772 exactly as `speak_server.main()` does, and exposes the same sync
functions the MCP tools call -- `_speak_sync`, `_listen_sync` -- over
`127.0.0.1:8773` for curl). The four running MCP `algolearn-speak` processes
were not touched (none of them binds 8772 -- they predate this code).
Nothing was committed; `ios/verify/` is untracked.

## Result in one paragraph

The server side of the design is verified end to end, but only half of it
with the real app. The real app in the simulator builds, launches, reads the
host from `UserDefaults`, connects, completes the `hello`/`ready` handshake,
and the server reports `audio: phone` within 34 ms -- and then the app is
killed by CoreAudio nine seconds later, every time, because **CoreAudio's
input path on this Mac is stalled machine-wide right now** (not the app, not
the simulator: a native macOS AVAudioEngine probe and PortAudio both hang on
the first touch of ANY input device, while output works). The app's
`AudioEngine.start()` needs `.playAndRecord`, so it cannot survive. The
speak / listen / disconnect checks were therefore run against a
protocol-faithful stand-in client (`phone_standin.py`) and all passed; they
need to be repeated with the real app once the Mac's audio input is unstuck
(reboot, or `sudo killall coreaudiod` -- I have no sudo).

## What passed / failed

| Check | Result | Evidence |
|---|---|---|
| 8.1 `uv run pytest` | PASS: 124 passed, 39 warnings, 17.3 s | pytest is not installed in `.venv` and is not a dev dependency (`uv run pytest` fails "Failed to spawn: pytest"); ran `uv run --no-sync --with pytest pytest -q tests` so the venv the live servers use was not modified |
| Build for simulator | PASS | `xcodebuild.log` ends `** BUILD SUCCEEDED **` |
| App launch, host from UserDefaults (`serverHost`) | PASS | `01_launched.png` shows `127.0.0.1` in the field |
| 8.2 (a) `audio: phone` once the REAL app connects | PASS | `server.log` 22:26:52.655 `phone connected`; `status_poll.log` line 18: 22:26:52.689 `"audio":"phone"` (34 ms later); stays `phone` for 9.1 s (18 polls) |
| Real app stays connected | FAIL (environment) | app SIGABRTs at +9.2 s: `Initialize: RPC timeout. Apparently deadlocked. Aborting now.` (`crash_01.ips`, stack: `AudioEngine.start()` -> `mainMixerNode` -> `AURemoteIO::Initialize`) |
| 8.2 (b) `speak("phone test one two three")` routed to phone | PASS with stand-in | `standin1.log`: `play_start` at 22:28:43.027, 102000 bytes = 2.12 s, `played` sent, server `/speak done in 2.482s` `audio: phone` |
| 8.2 (b) `played` round trip | PASS with stand-in | `played 0 sent` 22:28:45.159 -> server `state: idle` 22:28:45.162 (3 ms) |
| 8.2 (c) `listen` -> transcript | PASS with stand-in | mic fed the Kokoro phrase; server: `listen: 5.1s of audio transcribed in 0.7s: 'The quick brown fox jumps over the lazy dog.'` `audio: phone` |
| 8.2 (c) with the simulator mic + Mac speakers (`afplay`) | NOT RUN | blocked: app cannot start its audio engine (above); Mac input is stalled anyway |
| 8.3 disconnect -> `audio: mac` | PASS (real app AND stand-in) | real app: `phone disconnected` 22:27:01.825, poll shows `"audio":"mac"` from the next sample; stand-in: 22:28:54.285 disconnected, `/status` -> `mac` at 22:28:55.320 |
| 8.3 `speak` plays on the Mac again | PASS | `/speak?text=back+on+the+mac` -> `spoke for 1.6s`, `audio: mac` (worker path; Mac output works) |
| Mid-listen disconnect raises | PASS with stand-in | `/listen` -> `RuntimeError: phone disconnected during listen` after 2.217 s (socket closed 1.5 s after `mic_start`) |
| Mid-speak disconnect raises | PASS with stand-in | `/speak` -> `RuntimeError: phone disconnected during speak` after 0.846 s (socket closed 0.3 s after `play_start`) |

Design-level observations from the run:

- `speak` sends the WHOLE utterance before the first `play_start`
  (`_speak_impl` collects every Kokoro chunk, then `_play_pcm_stream`), so
  first-sound latency on the phone = synthesis time + one segment send:
  347 ms from the `/speak` call to `play_start` for a 2.1 s utterance; the
  2.12 s segment (102 kB) arrived in 3 ms on localhost. The design's section
  6 estimate ("one chunk + network") assumes streaming; today it is one
  segment per sentence, after all sentences are synthesized.
- Listen timeline (stand-in): ear-open cue 0.70 s -> `mic_start` 22:28:45.921
  (0.73 s after the call) -> speech from +0.5 s -> VAD end 2.0 s after speech
  ended -> `mic_stop` 22:28:51.303 -> ear-closed cue 0.12 s -> `processing`
  -> Whisper 0.7 s -> chime 0.24 s + "Processing." 1.62 s -> `idle`. Total
  9.05 s for a 3.25 s phrase. State messages reached the stand-in in the
  designed order (idle, speaking, idle, speaking, listening, speaking,
  processing, speaking, speaking, idle).
- In the real app `AudioEngine.start()` runs on the main actor (called from
  `PhoneLink.handle(.ready)`), so when CoreAudio stalls the UI freezes
  mid-transition (`02_connected.png`: badge "Connecting", button mid-fade).
  Not a bug under normal conditions, but worth knowing when the owner sees
  a frozen screen: it is CoreAudio, not the socket.

## The blocker, isolated

1. Real app, Connect tapped: handshake OK, then SIGABRT after ~9 s in
   `AURemoteIO::Initialize` (RPC timeout) -- three times out of three, with the
   Mac default input set to "USB Advanced Audio Device" (original) and also
   with it temporarily set to "MacBook Pro Microphone" (restored afterwards).
2. Probes inside the simulator (`UITapper/UITapperTests/AudioProbeTests.swift`,
   run in the XCUITest runner process): `.playback` only -> engine starts in
   0.2 s; `.record`, `.playAndRecord` default mode WITHOUT touching
   `inputNode`, and the app's exact `.playAndRecord/.voiceChat` config -> all
   abort with the same RPC timeout. Any session with input dies.
3. Mac side, outside the simulator: PortAudio `sd.rec` on device 1 (USB) and
   device 2 (built-in mic) both hang > 2 min (`timeout` guards fired);
   a native Swift `AVAudioEngine` probe (`macmicprobe.swift`) hangs on
   `inputNode.outputFormat(forBus:)` -- the first HAL input call -- and the
   10 s watchdog fired; `sd.play` on MacBook Pro Speakers works in 0.7 s.
   `coreaudiod` (pid 200, up since Oct 1) logged nothing during the probes;
   no `tccd` microphone prompt/denial logged. Lid is closed
   (`AppleClamshellState = Yes`); `usbaudiod` was (re)spawned at 22:11 on
   first enumeration.
4. Consequence for the LIVE servers: their Mac `listen` path (record worker)
   will also hang/raise until CoreAudio input recovers. Not caused by this
   work -- the first thing this session did on audio was enumerate devices.

## Exact commands

```bash
# test server (speak venv), phone server on 0.0.0.0:8772, control on 127.0.0.1:8773
cd live/algolearn-speak && nohup .venv/bin/python ios/verify/test_driver.py > ios/verify/server.log 2>&1 &
# engines ready in 8.1 s (22:10:47 -> 22:10:55)

# iOS app
SIM=96CF3880-2DB4-4BB4-B637-5BB2F0842522
xcodebuild -project ios/AlgolearnSpeak/AlgolearnSpeak.xcodeproj -scheme AlgolearnSpeak \
  -destination "platform=iOS Simulator,id=$SIM" -configuration Debug -derivedDataPath ios/verify/DerivedData build
xcrun simctl boot $SIM && open -a Simulator
xcrun simctl install $SIM ios/verify/DerivedData/Build/Products/Debug-iphonesimulator/AlgolearnSpeak.app
xcrun simctl privacy $SIM grant microphone ai.algolearn.speak
xcrun simctl spawn $SIM defaults write ai.algolearn.speak serverHost -string 127.0.0.1   # key: PhoneLink.hostKey
xcrun simctl launch $SIM ai.algolearn.speak
xcrun simctl io $SIM screenshot ios/verify/01_launched.png

# Tapping Connect: osascript/System Events is refused ("not allowed assistive access"),
# simctl has no tap. A throwaway XCUITest bundle (ios/verify/UITapper, xcodegen) attaches
# to the running app by bundle id and taps the button:
cd ios/verify/UITapper && xcodegen generate && \
xcodebuild -project UITapper.xcodeproj -scheme UITapper -destination "platform=iOS Simulator,id=$SIM" -derivedDataPath ../DerivedData build-for-testing && \
xcodebuild -project UITapper.xcodeproj -scheme UITapper -destination "platform=iOS Simulator,id=$SIM" -derivedDataPath ../DerivedData test-without-building -only-testing:UITapperTests/TapperTests/testTapConnect
# (each tap run costs ~30 s of runner startup)

# status while the real app is connected (poll every 0.5 s)
curl -s 127.0.0.1:8773/status

# stand-in phone (protocol-faithful; see phone_standin.py docstring)
.venv/bin/python ios/verify/phone_standin.py --mic-wav ios/verify/speech16k.wav &
curl -s "127.0.0.1:8773/speak?text=phone+test+one+two+three"
curl -s "127.0.0.1:8773/listen?max=20&sil=2&start=15"
.venv/bin/python ios/verify/phone_standin.py --mic-wav ios/verify/speech16k.wav --die-after-mic-start 1.5 &   # then /listen
.venv/bin/python ios/verify/phone_standin.py --mic-wav ios/verify/speech16k.wav --die-after-play-start 0.3 &  # then /speak

# Mac audio diagnostics
swiftc -O -o audiodev audiodev.swift && ./audiodev list|get-input|set-input "<name>"   # default input was restored to "USB Advanced Audio Device"
swiftc -O -o macmicprobe macmicprobe.swift && ./macmicprobe                            # -> TIMED OUT after 10s
```

`speech16k.wav` / `speech24k.wav`: Kokoro (af_heart) "The quick brown fox
jumps over the lazy dog." 3.25 s, generated in the speak venv; the 16 kHz
copy is linear-resampled from 24 kHz.

## Timings

| What | Value |
|---|---|
| Engines load (test server) | 8.1 s |
| Real app: `phone connected` -> `status.audio == phone` | 34 ms |
| Real app: connect -> CoreAudio abort | 9.2 s (3/3 runs) |
| `/speak` call -> `play_start` (2.1 s utterance, one sentence) | 347 ms (= Kokoro synthesis of the whole text) |
| 2.12 s segment (102,000 B) on the wire, localhost | 3 ms |
| `played` -> server `state: idle` | 3 ms |
| `/speak` total (incl. waiting for `played`) | 2.48 s for 2.1 s of audio |
| `/listen` call -> `mic_start` | 0.73 s (ear-open cue 0.70 s + `played`) |
| `/listen` total for a 3.25 s phrase, silence 2 s | 9.05 s (Whisper 0.7 s; the chime + "Processing." 1.9 s) |
| Mid-listen disconnect -> RuntimeError | 2.2 s after the call (socket closed 1.5 s after mic_start; raise within ~0.7 s) |
| Mid-speak disconnect -> RuntimeError | 0.85 s after the call (socket closed 0.3 s after play_start) |
| Disconnect -> `status.audio == mac` | next poll (< 0.5 s) |

## Cleanup state

Test server stopped (8772/8773 free); simulator shut down; Mac default input
restored to "USB Advanced Audio Device"; `~/.algolearn-speak/` untouched
(`audio.lock` only, no sidecar left behind); `DerivedData` (423 MB) and the
compiled probe binaries deleted, sources kept. The live MCP servers and
`~/.claude.json` were not touched.
