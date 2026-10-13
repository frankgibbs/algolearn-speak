# Design: the owner calls in to the algolearn.ai account and reaches a Claude Code session

Written 2026-10-12. Status: proposed; nothing built; awaiting the owner's
decisions in section 12. Companion to `docs/DESIGN_TELEGRAM_CALL.md` (the
outbound call), which this document extends and does not restate.

Labels:
- **Verified (read)**: read in source, with `file:line`. Repo lines are at
  commit c85c3eb (HEAD; the working tree has uncommitted audio-stats edits that
  shift a few lines of `speak_telegram.py` by a handful). Library lines are in
  the installed `.venv/lib/python3.12/site-packages` at the pins
  telethon 1.45.0, py-tgcalls 3.0.0, ntgcalls 3.0.2 (paths below are relative to
  that directory and abbreviated `pytgcalls/...`).
- **Unverified (spike)**: could only be established by a real call. Each is
  a build-phase check in section 11. Nothing marked this way is relied on
  without that check.
- **Decision**: something I decided, with the reason.
- **Owner decision needed**: section 12 only.

## 1. What this is

Today the owner's phone can only be rung by the Mac (`call`). This adds the
other direction: the owner rings **algolearn.ai** from **@frankgibbs35**, and
the call is answered by a Claude Code session, which then talks through the
existing `speak` / `listen` / `converse` over the call.

Out of scope: any change to outbound `call`; calls from anyone other than
`TELEGRAM_CALL_TARGET`; a Telegram text channel (DESIGN_TELEGRAM_CALL.md
section 7 keeps that a separate piece); starting a Claude Code session from
the daemon (section 6, option F).

## 2. What the libraries do with an incoming private call (Verified, read)

The pieces already exist in py-tgcalls; only the trigger is missing from this
repo (`speak_telegram.py` has no incoming path; its `on_update` handler only
looks at `StreamFrames` and the end statuses, lines 292-317).

1. Telethon delivers `UpdatePhoneCall` through a `Raw()` handler. A
   `PhoneCallRequested` becomes `RawCallUpdate(Type.REQUESTED)` with
   `chat_id = update.phone_call.admin_id`, i.e. the **caller's user id**
   (`pytgcalls/mtproto/telethon_client.py:106,166-176`;
   `pytgcalls/mtproto/bridged_client.py:303-309`). Before propagating, Telethon
   caches the call's `InputPhoneCall` under that id (`telethon_client.py:126-136`),
   which `accept_call` and `discard_call` need.
2. `_handle_mtproto_updates` turns a REQUESTED update into
   `self._p2p_configs[chat_id] = CallData(dhc, loop, g_a_hash)` (so
   `outgoing == False`, `CallData` `call_data.py:20-21`) and **replaces the
   update with `ChatUpdate(chat_id, INCOMING_CALL)`**, which is propagated to
   `on_update` handlers (`pytgcalls/methods/internal/handle_mtproto_updates.py:58-68,219-223`).
   `INCOMING_CALL` exists in `pytgcalls/types/chats/chat_update.py:15` and is not
   part of `LEFT_CALL` (`:19-25`), so the existing end-of-call branch
   (`speak_telegram.py:313`) does not fire on it.
3. **Accepting is the same `play()` the outbound path uses.** For a chat id
   with a non-outgoing `CallData`, `_connect_call` calls
   `accept_call` instead of `request_call` (`connect_call.py:137-150`), then waits
   for the caller's confirm (`:152-175`) for `CallConfig.timeout` seconds,
   raising `TimedOutAnswer` and discarding on timeout (`:176-182`), and then waits
   for the media connection (`:185`). `play()` returns when connected
   (`pytgcalls/methods/stream/play.py:98-104`), exactly as the outbound path
   relies on (`speak_telegram.py:442`, `_call`).
4. **Declining** an unanswered incoming call is `leave_call`: when
   `_p2p_configs[chat_id].wait_data` is not done it skips `binding.stop` and
   calls `discard_call(chat_id, False)` then pops the config
   (`pytgcalls/methods/calls/leave_call.py:22-24,47-51`).
   `discard_call(chat_id, is_missed)` offers only the Missed and Hangup
   reasons (`telethon_client.py:837-862`); Busy is not reachable through it.
5. **The caller hanging up while it rings** arrives as `PhoneCallDiscarded` ->
   `ChatUpdate(DISCARDED_CALL)` (`telethon_client.py:148-165`), which
   `_handle_mtproto_updates` propagates and, as `LEFT_CALL`, passes to
   `_clear_call` -> `_clear_cache`, which pops the `_p2p_configs` entry
   (`handle_mtproto_updates.py:78-80`, `clear_call.py:17-21`, `clear_cache.py:5-6`).
   The existing handler already maps it to `_end_call(END_REMOTE)` when
   `state != NONE` (`speak_telegram.py:313-317`).
6. The library's handler is registered by reference in `PyTgCalls.start()`
   (`pytgcalls/methods/utilities/start.py:16`), and `TelegramLink._start`
   already wraps it before `start()` (`speak_telegram.py:323-339`) to observe
   `RawCallUpdate`. That wrapper sees REQUESTED **before** the library does.
   This is the gate point (section 4).

**A hazard found while reading (Verified, read).** `_connect_call` reuses an
existing config for an outbound dial: `self._p2p_configs.setdefault(chat_id,
CallData(...))` (`connect_call.py:120-123`). If an unwanted REQUESTED from the
owner (for example while we are mid-call, or after we declined) were forwarded
to the library and not cleaned up, the entry would stay, and the **next
outbound `call` would take the `outgoing == False` branch and try to accept
instead of dial**. Therefore a REQUESTED the daemon will not take must never be
forwarded to the library (section 4), and every taken one must end with the
entry popped (rules 4 and 5 above cover decline and caller-hangs-up; the
accept path pops it in `connect_call.py:183-184`).

**Unverified (spike), all in section 11:** that a REQUESTED update actually
reaches this daemon for a call from the owner (the same `UpdatePhoneCall`
mechanism delivers ACCEPTED/CONFIRMED for outbound calls, observed live as the
`accepted` event, `speak_telegram.py:319-332`, but REQUESTED was never
observed); whether the 6 s opener quirk applies on the answering side; the real
ring window (`call_ring_timeout_ms` is in Telethon's `Config` type,
`telethon/tl/types/__init__.py:8146,8174`, so the daemon can read it at start;
the value for this account is unread); what the owner's phone shows for a
declined/missed call; what Telegram does with a second call from the same user
while one is connected; and whether other devices logged into algolearn.ai also
ring.

## 3. Architecture: extend the existing nouns, add none

Existing nouns reused: `TelegramLink` (state machine NONE/RINGING/CONNECTED,
generation, end reason, event ring), the `speak-telegram` daemon and its Unix
socket ops, `TelegramClient`, the speak server's route / `_call_expected` /
audio lock, `OPENER_MIN_SECONDS`, `call_debug`.

New (each justified):
- state `incoming` (a call from the owner is ringing, not yet connected);
- end reasons `missed` (caller hung up while it rang) and `rejected` (we
  declined: no listener, nobody claimed it, or hang_up while ringing);
- daemon op `await_call` (long poll) and op `answer`;
- one speak tool `answer(greeting, ...)`, the inbound twin of `call`;
- one console script `speak-telegram-wait` (section 6, option B).

```
owner @frankgibbs35 --rings--> algolearn.ai  (Telegram servers)
        speak-telegram daemon: TelegramLink gate -> state incoming
            -> notifies the line holder (section 6) -> its session calls answer(greeting)
            -> daemon accepts with the greeting as the opener file -> connected -> same call as outbound
```

## 4. Accepting only the owner (gate in the daemon)

**Decision: the gate is the existing `observe_mtproto` wrapper
(`speak_telegram.py:325-332`).** For a `RawCallUpdate` with `Type.REQUESTED` it
evaluates, in order, on the loop thread, synchronously:

1. `update.chat_id == self._user_id` (the caller's id as reported by Telegram's
   servers in `admin_id`, not caller-supplied; `_user_id` is the resolved
   `TELEGRAM_CALL_TARGET`, `speak_telegram.py:340`). Fails -> stranger.
2. `self._state == NONE` (no outbound ringing, no connected call, no pending
   incoming). Fails -> busy.
3. A line holder is registered (section 7). Fails -> no listener.

All three pass: set `state = INCOMING`, `generation += 1`, new `call_id`,
`_ended.clear()`, record event `incoming`, start the claim deadline
(section 8), then `await mtproto_handler(update)` so the library creates the
`CallData` and fires `INCOMING_CALL` (log-only event in `on_update`).

Any fails: **do not call `mtproto_handler`** (hazard above), record event
`incoming_rejected` with `reason` in `stranger | busy | no_listener`, and
discard via `self._calls._app.discard_call(update.chat_id, False)`
(`pytgcalls/pytgcalls.py:30`; the same `_app.discard_call` that `leave_call` uses,
`leave_call.py:48`; the `InputPhoneCall` is already cached, point 1). That is
private-API use of the same kind and fragility class as the existing wrapper;
the pins already forbid casual upgrades (DESIGN_TELEGRAM_CALL.md section 9). A
stranger's call is discarded without any other trace on their side beyond the
decline; no state, session, or audio is created for it.

Rejected alternative: filtering by `filters.` in `on_update`. The library has
already created `_p2p_configs` by then (`handle_mtproto_updates.py:58-64`).

**Glare (we are dialing out while the owner dials in):** gate rule 2 rejects the
incoming one; the outbound call stays authoritative. Real behavior of
Telegram under glare is unverified (spike, section 11).

## 5. The answer flow and the 6 s opener on the inbound path

The accept uses the same shape as the outbound path: `play(user_id,
MediaStream(<opener.wav>, params), CallConfig(timeout=ACCEPT_TIMEOUT_SECONDS))`
-> `record(...)` -> sleep `opener + 1 s` -> `play(ExternalMedia.AUDIO)` ->
pump (`speak_telegram.py:442,473-489`).

**Decision: refactor, do not copy.** `_call` splits at the ring: the part from
`self._state = CONNECTED` to the pump (lines 468-496) becomes one method used by
both `_call` and a new `_answer`. `_answer` differs only in: the starting state
is `INCOMING` (not `NONE`), no `_generation` bump (done at the gate), the
`play(...)` that accepts, its failure mapping (`TimedOutAnswer` ->
`rejected`; `CallDiscarded`/remote discard -> `missed`), and no ring-timeout
constant (the accept window is `ACCEPT_TIMEOUT_SECONDS`, defined once). It
runs as `self._ring_task`, so `_hang_up`'s cancel path
(`speak_telegram.py:563-570`) and `_end_call` can cancel a stuck accept: without
that, a caller who hangs up during the confirm wait leaves `play()` waiting out
its timeout, because the library only fails the `wait_data` future for
outgoing configs (`handle_mtproto_updates.py:35-45`).

**The opener is the answering session's greeting, synthesized in the session.**
Reasons: the daemon has no Kokoro (DESIGN_TELEGRAM_CALL.md section 2, quirk 4;
`speak_telegram.py` header); the session knows who it is and what it was
doing; and the owner decision of 2026-10-12 forbids padding a short opener, so
the same rule applies: `answer` measures the synthesized speech against
`tg.OPENER_MIN_SECONDS` (`speak_server.py` `_call_sync`, the check before
`telegram.call`) and raises `ValueError` **before accepting** if it is shorter,
and `write_opener` raises again as a backstop (`speak_telegram.py:180-194`).
The call keeps ringing while the model lengthens the greeting, within the claim
deadline.

Whether the file-first quirk is needed when we are the callee is **Unverified
(spike)**. The design keeps the file-first sequence regardless, because it is
the only sequence ever observed to work and costs nothing the owner can hear
as dead air (the greeting is real speech). If the spike shows raw frames work
from the start on the answering side, simplifying is a later, separate change.

## 6. How a session learns a call is waiting

The constraint: an MCP server only responds to requests. The speak server's
tools are `async def` handlers that run when Claude calls them
(`speak_server.py:1086-1275`); nothing in it can start a turn. The SDK in the
venv (mcp 2.1.1) can send server notifications (`mcp/server/session.py:124`,
`send_log_message :142`, `send_resource_updated :173`), but nothing in the
protocol makes the host start a model turn from them. Options, with evidence:

**A. A blocking tool the session parks in (`await_call`).** Same shape as
`listen`, which already holds a tool call for up to
`start_timeout + max_seconds` (165 s by default). Pros: no new Claude Code
feature; the parked session is an obvious "line holder". Cons: Claude Code
enforces a hard wall-clock limit per MCP call that progress notifications do
not extend (binary 2.1.296: "Hard wall-clock limit per call; progress
notifications do not extend it", `MCP_TOOL_TIMEOUT`; the default value is
**unverified**), so the session must re-call it in a loop, one model turn per
period, and is useless for anything else while parked. Suits a dedicated
phone session only.

**B. A background process whose exit or output wakes the session.** The Bash
tool's `run_in_background` "keeps running across turns and re-invokes you when
it exits" (Claude Code tool description); the `Monitor` tool turns each stdout
line of a long-running script into a notification, expiring after at most
30 minutes and re-armed by the model (Monitor description in this session).
A small console script `speak-telegram-wait` connects to the daemon, sends
`await_call`, prints one line when a call is ringing, and exits. Pros: works in
every session today with no launch flags; the session stays free for other
work; the waiter dies with the session, which is exactly the right lifetime for
"this session is the line holder" (the daemon detects the dropped socket,
section 7). Cons: the model must arm it (an instruction, section 9) and re-arm
after each call or Monitor expiry; delivery to a session that is idle at its
prompt, and to one mid-turn, is the tool's documented behavior but
**Unverified (spike)** here.

**C. MCP channel push.** Claude Code 2.1.296 contains a channels feature: an
MCP server declaring `experimental: {'claude/channel': {}}` pushes inbound
messages with `notifications/claude/channel`; they arrive as
`<channel source="...">`; it needs the session started with `--channels
server:<name>` (or `--dangerously-load-development-channels`), is off by
default for claude.ai Teams/Enterprise and gated by `channelsEnabled` /
`allowedChannelPlugins` (all from strings in the installed binary; behavior
and payload shape **unverified, must read before build**). Pros: the only
option that pushes into an idle session with no polling and no parked tool;
the speak server already runs per session and could register itself as the
line holder. Cons: a preview feature, a launch flag on every session, policy
gating, and an SDK path for a custom notification method I have not found
(`send_notification` takes typed notifications, `session.py:124`; unverified
that a custom method is accepted).

**D. Polling a tool (`status`, a new `incoming`) from a `/loop`.** The model is
not running between turns, so each poll is a full model turn; the loop skill
runs at a fixed interval. To catch a call inside a ring window of unknown but
tens of seconds (section 2, unverified) the interval would have to be well
under that, i.e. a paid turn every few seconds forever. Rejected on cost and
latency.

**E. Hooks.** Hooks run when the session already acts (prompt submit, tool
use, stop); they cannot originate a turn in an idle session. Rejected.

**F. The daemon launches a Claude Code session itself** (`claude -p ... --mcp-config`;
flags confirmed by `claude --help` on 2.1.296). It always has a taker, but the
new session has no context (resuming a session id that is also open
interactively is unverified), startup plus loading Kokoro and Whisper
(about 9 s, CLAUDE.md) eats the ring window, it needs a permission mode with
nobody present to approve, and it gives a Telegram caller identity check as the
only barrier in front of an agent with tool access. Not recommended; it is
also a second behavior for "no session listening", which the owner has ruled
out as fallback logic (section 8).

**Decision: B is the notification mechanism**, with the waiter as a console
script on the daemon's socket. C is the upgrade path if the owner is willing to
run sessions with `--channels`: it would replace only the delivery of the
notification, because the line-holder registration, claim, and `answer` are
identical (the speak server would hold the registration instead of the
waiter). B is chosen because every element it needs exists today and is
specified by the tools' own descriptions, and A and C are each blocked by a
fact that cannot be read from here (default tool timeout; channel payload and
gating).

The waiter's line is plain text, the whole instruction the model needs:
`INCOMING TELEGRAM CALL generation=7 ring_age=2s: call the speak answer tool now
with a greeting of at least 6 seconds`.

## 7. Which session takes the call: one line holder

**Decision: the daemon has exactly one inbound line holder, deterministically.**

- Registration = an open `await_call` connection from `speak-telegram-wait`,
  carrying `{pid, cwd, since}`. The daemon watches the connection; EOF
  (session closed, waiter killed) removes the registration. No heartbeat or
  lease: a dead session cannot hold the line. The daemon is already
  thread-per-connection (`_IPCServer`, `speak_telegram_daemon.py:208-209`), so
  a parked connection costs one thread.
- A second `await_call` while a holder exists is **refused** with the holder's
  `{pid, cwd, since}` in the error (no silent takeover, no first-come race
  between sessions). The model reports it to the owner. The owner moves the line
  by closing the holder's waiter (or `speak-telegram-wait --take`, an explicit
  replace, whose use the owner asks for in words).
- Only the holder's `answer` is accepted: `answer` carries the registration's
  `{pid}` and the daemon rejects any other session's `answer` during that ring.
- Rationale: three sessions racing for one phone call yields an arbitrary
  winner and two confused models. A named holder is predictable and the owner
  controls it by which session they armed.

Alternatives (owner decision, section 12): most recently armed wins; or all
armed sessions are notified and the first `answer` wins.

## 8. No session listening, and every other way a ring ends

There is one behavior per case, no substitutes:

| Situation | Behavior | End reason / event |
|---|---|---|
| No line holder when it rings | Discard immediately; the owner sees it end within about a second | gate `incoming_rejected: no_listener` |
| Holder notified, `answer` not started within `INCOMING_CLAIM_SECONDS` | Discard | `rejected`, event `claim_timeout` |
| `answer` raised before accepting (short greeting, quiet-hours N/A, daemon error) | The call keeps ringing until the claim deadline; the model may call `answer` again | no change |
| Owner hangs up while it rings | State -> none | `missed` |
| `hang_up` while `incoming` | `leave_call` discards (point 4 of section 2) | `rejected` |
| Accept fails with `TimedOutAnswer` | Already discarded by the library (`connect_call.py:176-182`) | `rejected`, event `accept_timeout` |
| Media connect fails after accept | Same as `_audio_failed`: leave, `failed`, daemon restart (`speak_telegram.py:546-555`, `speak_telegram_daemon.py:171-172`) | `failed` |
| Daemon down | Nothing rings: the owner's call simply goes to an account with no online client | none |

`INCOMING_CLAIM_SECONDS` is defined once in `speak_telegram.py`. Its value is
bounded above by the real ring window, read at daemon start from Telegram's
`Config.call_ring_timeout_ms` (unverified value); it must be less than the
window by at least the accept time. No text message is sent in any case
(out of scope, section 1).

## 9. Making the model arm the line

Two new things must reach the model without it asking: how to arm, and what to
do when the waiter fires.
- `CLAUDE.md` of this repo gets an "Inbound calls" paragraph; the tool
  descriptions of `answer` carry the rule of engagement (write a greeting of at
  least 6 s: say who you are and what session you are, then ask what they need).
- The owner starts the line by telling a session "take the phone" (or a memory
  entry makes the voice sessions do it); the session runs
  `speak-telegram-wait` with `run_in_background` and re-runs it when it
  exits. That is a usage convention, not code. Whether to bake the arming into a
  skill is a build-phase choice with no bearing on the design.

## 10. Interaction with the existing mechanisms

**Audio lock.** `answer` is the inbound twin of `call`: it takes the cross-process
audio lock for its whole hold (accept, greeting, then the same `_listen_impl`
that `call` runs, `speak_server.py` `_call_sync`), so no other session talks
in between. Cost: if another session holds the lock in a long Mac-route
`converse` (up to 165 s of listening by default), the ring can expire before
`answer` gets the lock. Decision: accept this; it is a lost call reported as
`missed`/`rejected`, not a hidden retry. `answer` acquires the lock **before**
checking the ring and, after acquiring, re-reads the daemon state; if the call
is gone it returns `not answered: the call ended before it could be picked up`.

**Routing.** `_begin_call` raises when the daemon state is neither none nor
not-running and not connected (`speak_server.py:118-119`), and `status` maps
any other state to `audio: "error"` (`:1299-1302`). Decision: during
`incoming` the call has no audio yet, so treat it like `none`: Mac-route tools
keep working and the Mac keeps its speakers until the accept. Two one-line
edits: `_begin_call` and `status` include `tg.INCOMING` with `tg.NONE`. The
route flips to telegram only when the state is connected (`:108-112`).

**Call expectation and drops.** After `answer` returns, `_call_expected` and
`_call_generation` are set exactly as `_call_sync` does for an answered call
(`speak_server.py` `_call_sync`), so a later drop raises "Telegram call ended
during ..." in each session that talks on the call, and the session calls back
with the outbound `call` (rule of engagement unchanged). `hang_up` and the
generation/end-reason comparison (`:109-110`) work unchanged on inbound calls.

**Outbound state machine.** `_call` already refuses when `state` is anything but
none/connected: `raise RuntimeError(f"a Telegram call is already {self._state}")`
(`speak_telegram.py:419-420`), so a `call` during `incoming` raises "already
incoming" with no new code. `_end_call` is reused as is; it gains the end-reason
choice (`missed` instead of `remote` when `state == INCOMING`, in the handler at
`:313-317`). `info()` (`:371-373`) gains `incoming` fields (`ring_age_s`,
`holder_pid`) only while incoming. Generation numbers keep counting both
directions in one sequence.

**Quiet hours.** Quiet hours (`SPEAK_QUIET_HOURS`, `speak_server.py` constants
and `_call_sync`) are a rule about the Mac ringing the owner's phone. An inbound
call is the owner choosing to talk. Decision: not applied to inbound; the
daemon does not know them (they live in the speak server). Owner may overrule,
section 12.

**Caffeinate, pump, VAD, `play`/`record` ops.** Unchanged: after `answer` the
call is in state connected and indistinguishable from an outbound one
(`speak_telegram.py:471,488`).

**Debugging.** The `incoming`, `incoming_rejected`, `claim_timeout`,
`accept_timeout` and `accepted` events go to the same `CallEventLog`, so
`call_debug` shows an inbound call without new tooling
(`speak_telegram.py:92-112`).

## 11. Build plan and checks

1. **Spike, no repo code (read-only scratch script, not the daemon):** with the
   daemon stopped by the owner (one Telegram login, one connection), connect the
   algolearn.ai session, log every `UpdatePhoneCall`, and have the owner ring it.
   Record: does REQUESTED arrive; `admin_id`; the ring duration from
   `Config.call_ring_timeout_ms` versus real; decline appearance on the owner's
   phone for `discard_call(…, False)` and for a caller hang-up; second call while
   connected; glare; whether other algolearn.ai devices ring.
2. Spike the accept: accept with a 6 s file, then raw frames, then try raw frames
   at once. Decides whether the file-first quirk applies (section 5).
3. Spike the waiter wake-up: `run_in_background` and `Monitor` against an idle
   session and a mid-turn session. Decides whether B stands (section 6).
4. Build: gate and `_answer` in `TelegramLink` (refactor of `_call`), ops
   `await_call` / `answer`, `speak-telegram-wait`, the `answer` tool, the two
   routing edits.
5. Tests against the existing fakes: stranger, busy, no-listener rejections never
   reach the library handler; owner path to connected; caller hang-up while
   ringing -> `missed` with `_p2p_configs` clean; claim timeout; second holder
   refused; wrong-pid `answer` refused; a following outbound `call` dials (the
   hazard in section 2). No test contacts Telegram.
6. Independent code review, tests green, the owner's go, then the launchd
   daemon is restarted by the owner.
7. End to end on real calls, including two sessions and no session.

## 12. Decisions needed from owner

Each is concrete. My recommendation is first.

1. **Delivery mechanism.** (a) Background waiter that wakes the session (B).
   Recommended: it works in every session today and leaves the session free.
   (b) Sessions started with `--channels` (C): true push, but a preview feature
   plus a flag on every session. (c) A dedicated parked phone session (A).
2. **Which session takes the call.** (a) One line holder, a second arming is
   refused (recommended). (b) The most recently armed session wins. (c) Every
   armed session is notified, first `answer` wins.
3. **No session listening.** (a) Decline at once, no text (recommended; one
   behavior, consistent with section 7 of the outbound design). (b) Decline and
   send a Telegram text to the owner, which is a new text channel.
4. **Quiet hours for inbound.** (a) Not applied (recommended: the owner is
   initiating). (b) Applied: the daemon declines 22:00-07:00, which needs the
   quiet-hours setting available to the daemon.
5. **Who speaks first on an inbound call.** (a) The answering session's own
   greeting of at least 6 s (recommended: it can name the session and the work
   context; consistent with the 2026-10-12 no-padding rule). (b) A fixed
   recorded greeting in the repo, identical for every session; this stops the
   model from needing to write 6 s of speech but loses the context.
