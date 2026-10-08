# Design: the speak server calls the owner's iPhone over FaceTime Audio

Written 2026-10-08. Status: built and under end-to-end test.
**Owner decision** marks something the owner decided (over voice, 2026-10-08).
**Decision** marks something I decided, with the reason. **Verified** marks
something observed on this Mac during the 2026-10-08 spike. Every claim
about existing code carries file:line.

## 1. What this is

A `call` tool makes the Mac place a FaceTime Audio call to the owner's
iPhone. Once the owner answers, `speak`, `listen` and `converse` run over
the call until it ends: Kokoro's voice goes into the call, and the owner's
voice comes back to Whisper. It works wherever the phone has signal. No
iOS app, relay, VPN or open port is involved; Apple's FaceTime servers
carry the call.

**Owner decision:** FaceTime Audio replaces the custom iPhone app for
remote use. The owner asked for no LAN port and no third party such as
Tailscale. Apple does not give third-party apps a server-carried audio
channel, so the alternatives all needed a relay.

**Owner decision:** the iPhone app and the `speak-phone` daemon will be
retired. That removal is a separate change made after this one passes its
end-to-end test. Until then the phone route keeps working unchanged
(`speak_server.py:81-91`).

## 2. The Mac must be unlocked (Verified 2026-10-08)

- When the display slept and the screen locked mid-call, the FaceTime call
  ended. On this Mac the screen locks immediately on display sleep, and the
  display sleeps and the screensaver starts after 20 idle minutes.
- Dialing while locked placed no call. The "Click to Call" banner did not
  appear on the lock screen. After unlocking it appeared, but only as an
  opaque `FaceTimeNotificationExtension` element with no buttons, so it
  could not be pressed.
- Research turned up no supported way around this. FaceTime skips the
  prompt through a private Apple entitlement (`com.apple.FaceTime.NoPrompt`).
  The calling service (TelephonyUtilities) is entitlement-gated, and
  FaceTime has no scripting dictionary.

**Owner decision:** `call` only works while the Mac is unlocked. From the
moment a call connects until it ends, the server keeps the Mac from
sleeping or locking (section 6, "Keep-awake").

## 3. How the pieces fit (all Verified in the spike)

| Piece | What it is |
|---|---|
| Caller identity | FaceTime on the Mac is signed into the owner's developer Apple ID. With the same Apple ID as the phone, the phone only offers Handoff ("transfer to phone") and never rings. |
| Dialing | `open "facetime-audio://<E.164 number>"`. macOS then shows a "Click to Call" banner that needs a click. |
| Mac → owner audio | Kokoro plays into the **BlackHole 2ch** virtual device. FaceTime's Microphone is set to BlackHole 2ch. |
| Owner → Mac audio | FaceTime's Output is set to **BlackHole 16ch**. The record worker reads BlackHole 16ch. |
| Call state | The call banner belongs to Notification Center, not FaceTime. Its accessibility group has identifier `FACETIME_NOTIFICATION`. |
| Control | A small Swift helper, `ftcall`, reads that banner and presses its buttons through the macOS Accessibility API. |

Banner states seen in the spike (static text of the group):

| Text | Buttons | Meaning |
|---|---|---|
| `Click to Call` | Call, Cancel | Dial requested, waiting for the click |
| `FaceTime Audio…` | Messages, FaceTime Video, Mute, Share, End | Ringing the phone |
| `FaceTime Audio - M:SS` | same | Connected; the timer starts at 0:00 the moment the owner answers |
| (group with no labels yet) | | Drawing; seen for about 1.1 s after dialing. Reported as `loading` |
| (no group) | | No call |

The "Click to Call" banner carries its labels in `AXDescription`; the in-call
banner carries them in `AXValue`. `ftcall` reads both (Verified).

Two separate BlackHole devices are required. With only one, FaceTime's
microphone would also hear FaceTime's own output and echo the owner's
voice back to them.

FaceTime's Video menu lists Microphone and Output devices and marks the
selected one with a check. `ftcall` reads those marks (Verified). The
`defaults` key `PreferredAudioInputDeviceUID` does not reflect the real
selection (Verified), so the menu is the source of truth.

## 4. The `ftcall` helper

Source: `ftcall/main.swift`. Built with `ftcall/build.sh` into
`ftcall/ftcall` (git-ignored). Commands:

- `ftcall state` prints one JSON line: `{"state": "locked"|"none"|"loading"|"click_to_call"|"ringing"|"connected"|"unknown", "text": "<banner text>"}`.
  `locked` comes from the IORegistry's `IOConsoleLocked`, read before any
  Accessibility call. While the screen is locked, Accessibility calls into
  Notification Center hang (Verified: `ftcall` timed out after 10 s).
  `loading` is a banner with no labels yet. Every wait keeps polling through
  it, bounded by that wait's own timeout.
  `unknown` means a FaceTime banner whose text matches none of the known
  states (for example an incoming call), or more than one FaceTime banner
  at once. The server never treats it as one of the known states (see
  section 5 for the short grace period).
- `ftcall press <button>` presses the banner button whose label is exactly
  `<button>` (`Call`, `Cancel`, `End`). With more than one banner up it
  refuses rather than guess.
- `ftcall devices` prints `{"microphone": "<checked item>", "output": "<checked item>"}`
  from FaceTime's Video menu. FaceTime must be running.

Exit 0 on success. Exit 2 if the process lacks Accessibility permission.
Exit 3 on any other failure, with the reason on stderr.

**Decision:** a compiled helper, not AppleScript. `System Events` cannot
address the banner's nested groups by identifier without walking the whole
tree, and the walk took several seconds in AppleScript. The Swift walk takes
about 35 ms (Verified).

**Permission:** Accessibility is granted to Terminal (Verified, 2026-10-08).
`ftcall` runs as a descendant of the Claude Code process that Terminal
launched, so macOS attributes it to Terminal. A Claude Code launched from
another app needs that app granted too.

## 5. The `call` tool

```
call(override_quiet_hours: bool = False) -> str
```

Runs under the audio lock (`speak_server.py:235-300`) as tool name `call`,
so no other session can speak or dial while it runs. Steps:

1. `SPEAK_CALL_NUMBER` unset → raise.
2. Quiet hours (section 7) and `override_quiet_hours` false → return
   `not called: quiet hours (22:00-07:00)`. Nothing is dialed.
3. `ftcall state` is `connected` → return `already connected`. `locked` →
   return `not called: the Mac is locked (FaceTime can only call while it is
   unlocked)`; nothing is dialed. Any other state but `none` → raise (a call
   is in an unknown or half-set-up state).
4. `open facetime-audio://<number>`.
5. Wait up to 15 s for `click_to_call`, else raise.
6. `ftcall devices`. If the checked Microphone does not contain
   `SPEAK_CALL_OUTPUT_DEVICE`, or the checked Output does not contain
   `SPEAK_CALL_INPUT_DEVICE`, press `Cancel` and raise, naming both the
   expected and the actual devices.
   **Decision:** raise rather than select the devices by pressing menu
   items. FaceTime's device choice is the owner's setting, and changing it
   silently would also change their personal FaceTime calls.
7. Press `Call`, then wait up to 10 s for `ringing` (or `connected`), else
   raise. Waiting for `ringing` first means a brief banner gap right after
   the press can never be misread as a declined call.
8. Poll `ftcall state` every 0.25 s for up to 90 s:
   - `connected` → return `answered`.
   - `none` that stays `none` for 1 s → return `not answered (declined or no answer)`.
     If the banner comes back within that second, keep waiting.
   - still `ringing` at 90 s → press `End`, return `not answered (declined or no answer)`.
   - anything else → raise with the banner text.

Throughout, an `unknown` reading is tolerated for up to 1 s in a row. A
banner caught mid-redraw has no text yet. Only an `unknown` that persists
is an error.

If any step after dialing fails while the "Click to Call" banner is still
up, the tool presses `Cancel` before raising, so a later click can never
dial. If the cleanup fails too, the error names both failures.

**Decision:** the spike saw no banner text for a declined or unanswered
call, so none is named. If FaceTime shows a final banner such as "Call
Failed" or "Unavailable", it reads as `unknown` and the tool raises with
that text rather than guessing an outcome. The end-to-end test records what
FaceTime actually shows.

Declined and no-answer are reported together. **Decision:** the banner
alone does not distinguish them, and both lead to the same action (wait an
hour), so a guess would add nothing.

## 6. Routing while a call is up

`_begin_call` (`speak_server.py:81-84`) picks the route once per tool call,
under the audio lock. The order becomes:

1. `ftcall state` is `connected` → route `facetime`.
2. A call is expected (below) but is not connected → raise
   `FaceTime call ended during <tool>`.
3. A banner in any other state (`ringing`, `click_to_call`, `unknown`) →
   raise. Audio is never routed while a call is half set up.
4. Phone connected → route `phone` (unchanged until retirement).
5. Otherwise → route `mac`.

**Keep-awake (Owner decision, section 2).** When `call` returns `answered`
or `already connected`, the server starts `caffeinate -d -i -w <server pid>`,
which blocks display sleep and idle sleep and exits if the server dies. It
also declares user activity every 60 s (`caffeinate -u -t 1`). That resets
the screensaver's idle timer, because on this Mac the screensaver locks the
screen as well. A watcher thread reads the call state every 5 s and releases
everything once the state is `none` or `locked`. `hang_up` and a reported
drop release it too. If the state can't be read, the watcher keeps the Mac
awake and logs a warning; staying awake is the safe side.

A `locked` state means no call, because locking ends a call (section 2). On
the routing path it is treated like `none`: an expected call that reads
`locked` raises "call ended", otherwise audio goes to the phone or the Mac.

**Decision: every voice call depends on reading the call state.** If
`ftcall` cannot run (no Accessibility permission for the app that launched
Claude Code, Notification Center not running, the binary missing), `speak`,
`listen` and `converse` raise, even when no call was ever placed. They do
not fall back to the Mac. Reason: the owner's standing rule against fallback
logic. Guessing "no call" when the state is unreadable is exactly how audio
would end up in an empty room. `status` reports `audio: "error"` in that
case.

**Decision: a "call expected" flag.** It is set when `call` returns
`answered` or `already connected`. It is cleared by `hang_up`, by
`not answered`, and once a "call ended" error has been raised. While it is
set, a call that has disappeared raises instead of quietly sending the
next `converse` to the Mac's speakers in an empty room. The drop is
reported once; after that the Mac is the device again. Reason: the owner's
rule that a dropped call is never a silent switch to the Mac, applied
between tool calls as well as during them. The flag lives in the process
that placed the call.

On the `facetime` route every audio path uses the call devices instead of
`SPEAK_OUTPUT_DEVICE` / `SPEAK_INPUT_DEVICE`: TTS playback
(`speak_server.py:444`), tone cues and the "Processing" ack
(`speak_server.py:418`), and the record worker's input plus its in-worker
ear-open cue (`speak_server.py:715-719`). The Mac's own speakers and mic
are untouched during a call.

**Decision:** no voice-clone archiving on the `facetime` route, even when
`SPEAK_SAVE_DIR` is set. The capture is call-codec audio, not
reference-quality material.

### A call that ends mid-operation

After every `speak`, `listen` or `converse` on the `facetime` route, the
server checks `ftcall state` again. If it is no longer `connected`, the tool
raises `RuntimeError("FaceTime call ended during <tool>")`, even if audio
was captured. A partial transcript is included in the message. It never
switches to the Mac's devices. If the state itself cannot be read, the
error names both the original failure (or the captured transcript) and the
read failure. Neither is ever dropped.

A drop during `listen` is noticed when the record worker finishes. Silence
on BlackHole ends the listen at `start_timeout_seconds` (default 45 s), or
after `silence_seconds` if the owner was mid-sentence. **Decision:** accept
that delay rather than add a watcher thread that kills workers mid-flight.
The post-check is enough to keep the result honest.

**Owner decision (rule of engagement):** when a tool raises because the call
ended mid-conversation, the session calls back. This rule is written into
the `call`, `speak`, `listen` and `converse` tool descriptions.

## 7. When to call

- **Owner decision:** no calls from 22:00 to 07:00 local time, every day.
  `SPEAK_QUIET_HOURS` holds the window (default `22:00-07:00`). The `call`
  tool enforces it. The model cannot bypass it except through
  `override_quiet_hours`.
- **Owner decision:** `override_quiet_hours=True` is used only when the
  owner has said in the current conversation that they are up ("if I'm up
  early, I'll let you know").
- **Owner decision:** after a `not answered`, the session may retry after
  one hour. The session tracks that, not the server.
- **Owner decision:** the Mac does not read the phone's Focus state. Quiet
  hours cover the night. During the day, the phone's own Focus settings
  decide whether a call rings.
- **Owner decision:** a Telegram channel will let the owner ask a session
  whether anything is waiting, and the session then calls. That channel is
  outside this server and this document.

## 8. The `hang_up` tool

```
hang_up() -> str
```

**Owner decision:** the session hangs up only when the owner asks it to. The
owner can also just hang up themselves.

Under the audio lock: if the state is `connected` or `ringing`, press `End`.
If it is `click_to_call`, press `Cancel`. Then wait up to 5 s for `none` and
return `hung up`. If the state is `none`, return `no active call`. Anything
else raises. If the press fails because the banner has already gone (the
owner hung up at the same moment), the result is still `hung up`.

## 9. `status`

`status` gains `"call": "<ftcall state>"`, or `"error: <reason>"` when the
state can't be read. `audio` can now be `"facetime"` as well as `"phone"` /
`"mac"`. It reads `"error"` when the next voice call would raise: the call
state is unreadable, a banner is half set up, or an expected call has
dropped. Every tool result's trailing
`(audio: ...)` note (`speak_server.py:94-95`) shows `facetime` on that
route.

## 10. Configuration

| Variable | Default | Meaning |
|---|---|---|
| `SPEAK_CALL_NUMBER` | unset | E.164 number to call (`+` and 8-15 digits). Validated at startup if set. `call` raises if it is unset. |
| `SPEAK_CALL_OUTPUT_DEVICE` | `BlackHole 2ch` | Device the server plays into during a call. FaceTime's Microphone must be this device. |
| `SPEAK_CALL_INPUT_DEVICE` | `BlackHole 16ch` | Device the server records from during a call. FaceTime's Output must be this device. |
| `SPEAK_QUIET_HOURS` | `22:00-07:00` | No-call window, local time, `HH:MM-HH:MM`. It may wrap midnight. Validated at startup. |

The `ftcall/ftcall` binary must exist. The server fails at startup with
the build command if it doesn't.

## 11. One-time setup on this Mac (done 2026-10-08)

1. `brew install --cask blackhole-2ch blackhole-16ch` (needs the admin
   password), then `sudo killall coreaudiod` so the devices load without a
   reboot.
2. Sign FaceTime on the Mac into the developer Apple ID. Leave iCloud and
   Messages on the owner's own account.
3. In FaceTime's Video menu, set Microphone to BlackHole 2ch and Output to
   BlackHole 16ch.
4. Grant Accessibility to Terminal (System Settings, Privacy & Security,
   Accessibility).
5. `ftcall/build.sh`.
6. Add `SPEAK_CALL_NUMBER` to the `speak` server's `env` block in
   `~/.claude.json`, then reconnect with `/mcp`.

## 12. Tests

`tests/test_speak_facetime.py` drives the Python side against a fake
`ftcall` script that plays back a scripted sequence of states. The tests
use a fictional number, and `dial` is replaced in every test. Coverage:
- answered, not answered (banner gone), ringing past the timeout
- wrong FaceTime devices, `already connected`, hang-up
- quiet hours: a window that wraps midnight, the override, and a check made
  after waiting for the lock
- routing to the call devices while connected
- a call that drops during an operation or between operations (raises, with
  no switch to the Mac)
- transient `unknown` and gap readings
- cancelling the banner after a failure
- the press race at hang-up
- state-read failures inside the drop guard
- `call` and `hang_up` holding the audio lock

The end-to-end test is a real call to the owner using the MCP tools,
including declining a call, letting one go unanswered, and checking the
banner with the Mac's display asleep.
