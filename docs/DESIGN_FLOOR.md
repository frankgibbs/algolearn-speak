# Design: the floor (one conversation at a time on the shared call)

Status: design, written 2026-10-20; deployed in commit `b5cd149` on 2026-10-20.
The client text (section 6) was rewritten by owner decision 2026-10-22, and
again by owner decisions 2026-10-26. The sign-off hang-up (section 4.10,
owner decisions 2026-10-28) is built and tested, not yet deployed.
Every build, restart and stop of `speak-telegram` remains an owner action,
approved per change.

How this document labels things:
- **Owner decision (2026-10-20)**: something the owner decided on 2026-10-20.
  It is a choice, not a claim about how the system behaves.
- **Verified (file:line)**: read in the code at commit `2fa4a20` (the tree
  this document was written against), with the evidence named.
- **Unverified, must read before build**: a claim the build must check
  first. Until then it is a question, not part of the design.
- **Design**: a choice this document makes inside the owner's decisions.
- **Design (build review, 2026-10-20)**: a choice made when the built code
  was reviewed, inside the owner's decisions; marked where it amends the
  original text.

## 1. The problem

The daemon serializes every tool op from every session through one FIFO
audio lock, taken and released per op:
- Verified: the TCP handler takes the lock right before running a tool op
  and releases it in a `finally` right after (`speak_telegram_daemon.py:752`,
  `771-772`).
- Verified: `AudioLock.acquire` grants the lock strictly in arrival order:
  a waiter goes when the lock is free and its ticket is at the head of the
  queue (`speak_telegram_daemon.py:240-247`); `release` wakes every waiter
  (`speak_telegram_daemon.py:249-255`).

A conversation is many ops. Session A runs `converse`, the lock is released,
and A's model then thinks for a few seconds before its next `converse`. A
session B that was already queued gets the lock in that gap and speaks into
the middle of A's conversation with the owner. The switch-of-session cue
(section 3.2) tells the owner what happened, but the conversation is still
broken up.

The goal: queued sessions wait until the current conversation reaches a
stopping point.

## 2. Owner decisions (2026-10-20)

1. **The floor.** The session that last played audio on the call holds the
   floor. While it holds the floor, when the lock frees, that session's
   requests go ahead of every other session's; other sessions keep waiting
   (still queued, never rejected).
2. **Automatic release on a bare `speak`.** A `speak` (say something, no
   listening after) releases the floor when it finishes. `converse` and
   `listen` keep it. There is no new flag; the agent's normal wrap-up does
   it. Rationale: the server instructions already reserve a bare `speak` for
   lines that need no reply (a last sentence, "one moment" before long silent
   work), which are exactly the moments the agent steps away.
3. **Idle timeout: 45 seconds.** If the floor holder sends no new op within
   45 s after its last op finished, the floor lapses and the next queued
   session gets the lock. This is the safety net for an agent that goes quiet
   without a closing `speak`.
4. **No "waiting: N" hint to agents.** No field tells the holder that others
   are waiting. The owner does not want the agent pressured to wrap up, as
   long as the etiquette is clear.
5. **The etiquette is explicit.** The server instructions and the relevant
   tool docstrings tell agents: other sessions may be waiting for the floor;
   at a stopping point (the topic is wrapped up, or the agent is going off to
   do background work) end with a `speak` ("Will do, talk soon"), not a
   `converse`, because a `converse` keeps the floor and holds everyone else
   off. The common wrap-up "do you want me to call you back when that's
   done?" becomes "I'll come back with the result": if the call is still
   connected when the work is done, the agent just runs `converse` (it
   queues, gets the floor when it is free, and the switch cue announces the
   project); it uses `call` only if the call has ended. **Amended by owner
   decision 2026-10-22, then by owner decisions 2026-10-26:** the etiquette
   as it now reads is in section 6.
6. **No new cue.** The existing switch-of-session cue and project-name
   announcement is what tells the owner the floor changed hands.
7. **An answered `call` gives the caller the floor** right away.
8. **HTTP whisper `transcribe` requests skip the floor:** they stay plain
   FIFO and wait only for the op holding the lock, never for a conversation.
9. **`status` does not show the floor.** It appears only in `call_debug`
   (the `floor_take` and `floor_release` events). No protocol bump; the
   protocol stays at 3.
10. **An unplanned drop keeps the floor** (the owner hangs up, or a network
    failure ends the call): the floor stays with the interrupted session for
    the 45 s idle window, so that session can call back first.
11. **A client that leaves during a hold releases the floor at once**
    (reason `client_left`) while the call stays up.
12. **A deliberate end releases the floor at once:** the agent's `hang_up`,
    or a hold expiry where the daemon hangs up.
13. **Repeated no-speech `listen` loops keep the floor.** No change now;
    watch for it on live calls.

## 3. The system today (Verified)

### 3.1 The lock

- `AudioLock` is a `threading.Condition`, a deque of opaque tickets, a
  `_held` flag and `_current` (`"<tool> (<peer>)"`)
  (`speak_telegram_daemon.py:230-259`). It knows nothing about sessions:
  `acquire(tool, peer)` takes no identity (`speak_telegram_daemon.py:240`).
- `snapshot()` returns `(held, current, len(queue))`, which `status` reports
  as `busy`, `current_tool`, `waiting` (`speak_telegram_daemon.py:257-259`,
  `1049-1051`).
- The handler validates the protocol version and the client's identity
  (`_speaker`) BEFORE it takes the lock (`speak_telegram_daemon.py:732-740`),
  so the session id is known at the moment the op queues. It parses the
  identity again inside `run_locked` (`speak_telegram_daemon.py:1074`,
  `1077`).
- A client that left while its op was queued is skipped once the op reaches
  the lock: the handler returns, and the `finally` releases
  (`speak_telegram_daemon.py:754-756`, `771-772`).
- The HTTP whisper API's `transcribe_pcm` takes the same lock as
  `TRANSCRIBE_TOOL` with no session (`speak_telegram_daemon.py:1031`,
  `1039-1040`).
- `SpeakDaemon.__init__` builds the `SpeakService` and then the `AudioLock`;
  the service holds no reference to the lock
  (`speak_telegram_daemon.py:992-993`, `346-358`).

### 3.2 The turn (switch of session)

- `SpeakService._turn` is `(call key, Speaker)` of the last session that
  played audio on the call, or None (`speak_telegram_daemon.py:354-358`). The
  call key is `"<boot id>:<link generation>"` (`speak_telegram_daemon.py:360-361`).
- `_take_turn` runs at the start of every `speak`, `listen` and `converse`,
  after the call check and the empty-text check, before any audio
  (`speak_telegram_daemon.py:557-559`, `567-568`, `574-576`). When the
  previous turn is on the same call and from another session, it plays the
  switch cue and the spoken project name, records `session_switch`, and sets
  `sess.switched_from`; in every case it then sets `_turn` to this session
  (`speak_telegram_daemon.py:417-440`). A drop during the announcement raises
  through `guard_call` before `_turn` is set (`speak_telegram_daemon.py:433`).
- An answered `call` sets `_turn` to the caller (the greeting is the call's
  first turn), then listens under the same lock hold
  (`speak_telegram_daemon.py:623-630`). `already connected` and `hang_up` do
  not touch `_turn` (`speak_telegram_daemon.py:599-601`, `632-636`).

### 3.3 How ops end

- `speak` streams the speech, then returns; a hold expiry hangs up through
  `_hold_expired` (`speak_telegram_daemon.py:556-564`, `514-528`).
- `converse` is speak then listen in one `guard_call`; after a hold expiry it
  does not listen and hangs up (`speak_telegram_daemon.py:572-590`).
- A client that leaves during a hold makes `_speak` raise
  (`_client_left_during_hold`, `speak_telegram_daemon.py:488-489`, `506-512`);
  the call stays up.
- `listen` with no speech raises `NoSpeech` (a `TimeoutError`) after the
  timeout beep; the call stays up (`speak_telegram_daemon.py:543-545`).
- `hang_up` ends the call through the link (`speak_telegram_daemon.py:632-636`).
- The link ends a call on its own loop thread in `_end_call`
  (`speak_telegram.py:570-588`); nothing in the link calls back into the
  daemon when that happens. The daemon reads the state with `link.info()` /
  `link.state()` (`speak_telegram.py:563-568`).
- Call events go to the link's ring buffer through `link.record_event`
  (`speak_telegram.py:452-454`), read by `call_debug`.

### 3.4 The client's instructions today

- `_INSTRUCTIONS` (`speak_server.py:356-402`) reserves a bare `speak()` for
  "words that need no reply (a last sentence before hang_up, a quick "one
  moment" before long silent work)" (`speak_server.py:365-367`) and says
  tools queue for the one shared call (`speak_server.py:393-394`).
- `_QUEUE_NOTE` (`speak_server.py:431-445`) is appended to the `speak`,
  `listen` and `converse` docstrings (`speak_server.py:492`, `510`, `525`).
  It describes a plain FIFO.
- No instruction or docstring mentions calling back after background work
  (`grep -i "call you back" speak_server.py`: no match).

## 4. The mechanism

### 4.1 Where the floor lives

**Design:** the floor lives in `AudioLock`, under its `Condition`, because
the rule that decides who goes next must be read and changed under the same
mutex the waiters wait on. No new class: `AudioLock` gains the floor state
and four operations; `SpeakService` gains a reference to the lock (passed to
its constructor; `SpeakDaemon.__init__` builds the lock first).

Floor state (all under `_cond`):

| Field | Meaning |
|---|---|
| `_floor` | The holder's `Speaker` (session id and project), or None |
| `_floor_since` | `time.monotonic()` when the holder took it (for the event) |
| `_idle_deadline` | `time.monotonic()` at which the floor lapses, or None while the holder's op is running |
| `_current_session` | The session id of the op holding the lock, or None |

`FLOOR_IDLE_SECONDS = 45.0` (owner decision 3) is defined once in
`speak_protocol.py`, beside `HOLD_MAX_SECONDS` (`speak_protocol.py:107`), so
the client's instruction text and the daemon use the same number. It is not
a wire field. `AudioLock` takes the idle seconds as a constructor argument
(default `FLOOR_IDLE_SECONDS`) so tests can use a fraction of a second, and
a `record_event` callable (the daemon passes `link.record_event`).

### 4.2 Queue order

Each ticket carries the session id of its op. The TCP handler passes the
identity it already validated before the lock
(`speak_telegram_daemon.py:736`) to `acquire(tool, peer, speaker)`.

**Design (owner decision 1):** a ticket may take the lock when the lock is
not held and the ticket is the first **eligible** ticket in the queue:
- While the floor is live (a holder is set and `_idle_deadline` is None or in
  the future), the eligible tickets are the holder's own. Among them, arrival
  order. Every other session's ticket stays queued. If the holder has nothing
  queued, the lock stays free and the others keep waiting.
- With no live floor, every ticket is eligible: plain FIFO, as today.
- A `transcribe` ticket (the HTTP whisper API, `speak_telegram_daemon.py:1031`)
  carries no session and is always eligible (owner decision 8): with a live
  floor, the eligible tickets are the holder's plus every `transcribe`
  ticket, in arrival order among them. A `transcribe` neither takes nor
  releases the floor, and its `release()` does not touch the idle deadline
  (it is not the holder's op).

A granted ticket is removed with `deque.remove` (no longer always the head).

**Design (build review, 2026-10-20):** the holder is active again from the
moment its op **arrives**, not when it is granted: `acquire` first observes
a lapse that is already due, then queues the ticket, and when the ticket is
the holder's it clears `_idle_deadline` at once. Without this, an op holding
the lock past the deadline (a long `transcribe`) let the floor lapse while
the holder's next op was already queued, and the session queued before it
went first.

**Design (build review, 2026-10-20):** a wait inside `acquire` that raises
(any `BaseException`) removes its ticket and wakes the other waiters before
re-raising, so a ticket that will never take the lock cannot block the queue.
When that ticket was the holder's and the holder has no other op queued or
holding the lock, it also restarts the idle count from that moment (its
arrival had stopped it), so the floor cannot stick.

**Design (build review, 2026-10-20):** once the daemon is restarting or
stopping, `AudioLock.close()` makes every ticket eligible (plain FIFO, the
floor no longer orders the queue); section 4.9.

### 4.3 The 45 s timer: who wakes the queue

**Design (owner decision 3):** `release()` sets `_idle_deadline = now +
FLOOR_IDLE_SECONDS` when the op that held the lock was the floor holder's
(`_current_session == _floor.session_id`), then `notify_all` as today. So the
45 s count from the end of the holder's last op that held the lock (a queued
op skipped because its client left also counts, since it held the lock), and
stop again the moment the holder's next op arrives (section 4.2).

Nobody needs a timer thread. A waiter that is not eligible waits with a
timeout: `cond.wait(timeout = _idle_deadline - now)` when a deadline is set,
`cond.wait()` (no timeout) otherwise, and re-checks on every wake. When the
deadline passes, the waiters wake by themselves. The first thread that finds
the floor past its deadline clears it, records `floor_release` with reason
`idle` (section 4.6) and calls `notify_all`, so every waiter re-checks and the
head of the queue goes. When nobody is waiting, nobody observes the lapse at
the deadline; the next `acquire` (or `take_floor`) observes it and records it
then. The holder coming back after a lapse is just another queued session:
if it is first in the queue it goes, and its audio takes the floor again.

### 4.4 Taking the floor

**Design (owner decision 1):** "the session that last played audio on the
call" is exactly what `_take_turn` already tracks (`speak_telegram_daemon.py:440`).
`_take_turn` calls `lock.take_floor(sess.speaker)` where it sets `_turn`, so
the floor moves with the turn:
- Same rules as the turn: an op refused before it plays (no call, empty
  text) does not take the floor; a call that drops during the switch
  announcement raises before the floor moves.
- `take_floor` by the current holder changes nothing (it already holds it)
  and records no event. `take_floor` by another session replaces the holder; that can only
  happen when the floor was not live, because otherwise the other session
  could not have held the lock. It records `floor_take` (section 4.6). In
  the build a floor that is not live is always cleared before another
  session's op is granted (a yield clears it; a lapse is recorded by the
  thread that observes it, at the latest by the next `acquire`), so
  `from_project` is null in practice.
- **Design (build review, 2026-10-20):** `take_floor` raises `RuntimeError`
  unless the lock is held by an op of that same session; only the op
  holding the lock may take the floor.
- The switch cue and project name `_take_turn` already plays are the
  owner's signal that the floor changed hands (owner decision 6). No new cue.
- **An answered `call` (owner decision 7):** `call` calls
  `lock.take_floor(sess.speaker)` where it sets `_turn` to the caller
  (`speak_telegram_daemon.py:624`), before the listen that follows under the
  same lock hold (`speak_telegram_daemon.py:628-630`). Any other `call`
  outcome (not answered, busy, already connected, quiet hours, a short
  greeting) does not touch the floor.

### 4.5 Releasing the floor

`lock.yield_floor(reason)` clears the floor (when the caller is the holder),
records `floor_release`, and lets the lock's `release()` that follows wake
the queue in plain FIFO order.

| When | Reason | Basis |
|---|---|---|
| A `speak` that took the turn ends, except when the call ended under it | `speak` | Owner decision 2. Called in `SpeakService.speak` after `_take_turn` returned (in its `except` when the speech raised), skipped when the exception is the call-ended error (next row). When the speech played through and no other tool op with its client still there is queued at that moment, the daemon then ends the call (section 4.10, owner decisions 2026-10-28) |
| `converse` or `listen` ends, including with a no-speech `TimeoutError` | (none: the floor is kept) | Owner decisions 2 and 13 |
| The call drops unplanned during or between the holder's ops (the owner hangs up, a network failure; `_end_call` on the link's loop thread, `speak_telegram.py:570-588`), including during a `speak` | (none: kept until the 45 s lapse) | Owner decision 10. The op that saw the drop raises `Telegram call ended during <tool>` and releases the lock; that release starts the 45 s (section 4.3). The interrupted session's `call` is eligible and goes ahead of every other session; its answered call takes the floor again (section 4.4). No hook from the link is needed: the floor is not touched at the moment of the drop |
| The holder sends nothing for 45 s after its last op ended | `idle` | Owner decision 3, section 4.3 |
| `hang_up` by the holder ends the call (`speak_telegram_daemon.py:632-636`) | `hang_up` | Owner decision 12 |
| A hold expires and the daemon hangs up (`_hold_expired`, `speak_telegram_daemon.py:514-528`) | `hold_expired` | Owner decision 12. Yielded in `_hold_expired`, whether or not the hang-up itself failed |
| The holder's client leaves during a hold and the call stays up (`_client_left_during_hold`, `speak_telegram_daemon.py:506-512`) | `client_left` | Owner decision 11. Yielded in `_client_left_during_hold` before it raises, for `speak` and `converse` alike; the `speak` `finally` then finds no floor to yield |

Owner decision 13, stated so the build does not "fix" it: an
agent that answers each no-speech `TimeoutError` by reopening the ear with
`listen()` (as the instructions tell it to, `speak_server.py:369-370`) keeps
the floor for as long as it keeps listening, because each `listen` starts
within 45 s of the previous one ending.

### 4.6 Events (`call_debug`)

Recorded through `link.record_event` (`speak_telegram.py:452-454`), so they
appear with the call's events:

| Event | Fields |
|---|---|
| `floor_take` | `project`, `from_project` (the previous holder's, or null) |
| `floor_release` | `project`, `reason` (`speak`, `idle`, `hang_up`, `hold_expired`, `client_left`), `held_s` (seconds since `floor_take`), `waiting` (tickets queued at that moment) |
| `signoff_hangup` (2026-10-28, section 4.10) | `project`, `waiting` (0), `outcome` (the link's hang-up outcome) |
| `signoff_hangup_failed` (2026-10-28, section 4.10) | `project`, `error` |

`waiting` is in the event log for diagnosis only; it is never in a tool reply
(owner decision 4). The floor is visible only through these events, not in
`status` (owner decision 9): `status` keeps its fields and their meaning
(`speak_telegram_daemon.py:1049-1051`), so `busy` can be false while
`waiting` is non-zero (the lock is free and the waiters are not eligible).
The client passes the whole `call_debug` reply through
(`speak_server.py:318-324`), so the new events need no client change.

### 4.7 Hold on / continue

While Angie is held, her op holds the lock (`docs/DESIGN_INTERRUPT.md`
section 5.2: a hold keeps the audio lock, up to `HOLD_MAX_SECONDS`, 300 s,
`speak_protocol.py:107`). The floor changes nothing during a hold: no one
else can run while the lock is held, and the idle timer does not run while
the holder's op runs. A `speak` that was held and resumed releases the floor
when it ends, like any `speak`. Hold expiry: section 4.5.

### 4.8 Clients that leave

- **A non-holder whose client leaves while queued:** its ticket is skipped
  when it reaches the lock, as today (`speak_telegram_daemon.py:754-756`).
  With the floor, it reaches the lock only once it is eligible; the floor is
  not touched.
- **The holder's queued op whose client left:** it is granted (it is
  eligible), skipped, and released; the 45 s restart from that release
  (section 4.3).
- **The holder's client leaves during a hold:** the floor is released at
  once with reason `client_left` (owner decision 11, section 4.5), and the
  lock goes to the head of the queue.
- **A `transcribe` client that leaves while queued:** skipped as today
  (`speak_telegram_daemon.py:1022-1043`); the floor is not involved.

### 4.9 Restart or stop

**Design (build review, 2026-10-20):** the daemon sets `closing` on a stop
(`SpeakDaemon.stop`) and before the restart that follows an audio failure on
an answered call (the TCP handler). Each op that reaches the lock after that
is refused with `speak daemon is restarting; try again in a minute`. With a
live floor, a non-holder's op would first wait out the idle time to reach
that refusal. So right after setting `closing`, both places call
`AudioLock.close()`: from then on every ticket is eligible, in arrival
order, and each queued op reaches its refusal at once. `close()` records no
event: nothing yielded the floor.

### 4.10 The sign-off hang-up (owner decisions 2026-10-28)

**Owner decisions 2026-10-28**, superseding the agent-side rule of
2026-10-27 (section 6):
1. The daemon hangs up, not the agent. When a `speak` that gave up the floor
   ends (release reason `speak`) and no other op is queued for the lock at
   that moment, the daemon ends the call at once, with no grace period. If
   any op is queued, the call stays up and the floor passes as before.
2. It is a deliberate end, like `hang_up`: the session's call expectation is
   cleared and nobody calls back. It records `signoff_hangup` (`project`,
   `waiting` 0) besides `floor_release` as before.
3. The `speak` reply carries `call_ended: "signoff"` and the client adds
   "[call ended: nobody else was waiting]".
4. The agent-side `status()`/`hang_up()` step goes from the instructions and
   the docstrings; `hang_up()` is "only when he asks" again.

**Owner decisions 2026-10-28, after the code review of commit 7efff53**
(they replace the build's own choices on these points):
5. Only a `speak` that played through (fully, or held and then resumed)
   hangs up when nobody is queued. A sign-off `speak` that raised while the
   call is still up does not: it releases the floor as before (reason
   `speak`; the normal release and the 45 s idle lapse apply) and leaves
   the call up, so the agent can send its sign-off again. Its error reply
   does not carry `call_ended`, and the `speak` docstring tells the agent
   that after a failed sign-off the call is still up and to send the
   `speak()` again.
6. When counting the queue at the sign-off, a queued op whose client has
   already left does not count. A queue of only departed clients counts as
   empty, so the daemon hangs up.
7. No exception is swallowed or replaced by the hang-up.

**Design (build, 2026-10-28), within those decisions:**
- **Atomicity.** `AudioLock.yield_floor` returns, after it clears the floor,
  the number of queued tickets that carry a session and whose client is
  still there, counted under the same hold of the lock's condition
  (`speak_telegram_daemon.py:375-391`). Each ticket carries the
  `client_left` probe its acquire was given; the handler passes
  `_peer_left` on its own socket (`:1010`). `_peer_left` (`:1058`) is a
  `select` with a zero timeout, then a `MSG_PEEK`, so calling it under the
  condition never blocks. `SpeakService.speak` (`:748`) calls
  `yield_floor` after the speech returned (`:774`) and, when the count is
  0, `_sign_off_hang_up` (`:776`), while the speak still holds the audio
  lock: the handler releases the lock only after `run_locked` returns
  (`:1025`, released at `:1031`). So no queued op can run between the count
  and the hang-up.
- **A late arrival.** An op that arrives after the count, while the
  hang-up is in progress, queues behind the speak, runs after the hang-up
  and finds no call: `require_call` (`:544`) raises "needs a connected
  Telegram call" (the link's `END_HUNG_UP` clears an expectation of that
  call, as after `hang_up`, so it is not "ended during"). The instructions
  tell that agent to `call()` back, and `call` waits until
  `REDIAL_SETTLE_SECONDS` (10 s) have passed since the hang-up before it
  dials, so the owner's phone rings again about 10 s after the call ended.
- **A failed speak.** When the speech raises other than with the
  call-ended error, `speak` yields the floor (reason `speak`) in its
  `except` (`:768`) and re-raises the same exception; it never reaches the
  hang-up. The call-ended error yields nothing, as before.
- **No exception masked.** The hang-up runs only on the path where the
  speech returned, outside any `except` or `finally`, so no speak
  exception is in flight while it runs; a hang-up that raises records
  `signoff_hangup_failed` (`project`, `error`) and re-raises its own error,
  which becomes the `speak`'s result (the call may still be up); the
  expectation is cleared either way, and no `call_ended` is sent.
- **HTTP transcriptions do not count.** Their ticket has no session
  (`transcribe_pcm` acquires with `None`, `:1292`), so a transcription
  queued at that moment does not keep the call up. It is not lost: it stays
  queued and runs once the speak releases the lock, after the hang-up.
- **Which cases never hang up:** a `speak` that raised (decision 5), a
  `speak` under which the call ended (no yield, the floor stays for the
  call-back), an expired hold (the daemon has already hung up and yielded
  with `hold_expired`), a client that left during a hold (yielded with
  `client_left`, the call stays up), a `speak` refused before its turn
  (empty text, no call), `converse` and `listen` (they keep the floor). Any
  queued tool op whose client is still there counts, this session's own
  included.
- **The reply.** `run_locked` adds `call_ended` (`REPLY_CALL_ENDED`,
  `CALL_ENDED_SIGNOFF` in `speak_protocol.py`) only once a sign-off
  hang-up succeeded, with `expect` null. The client puts "[call ended:
  nobody else was waiting]" on its own line after "spoke for Ns", before
  the audio note (`_ended_line`, `speak_server.py:314`). It raises on an
  unknown value (naming the speak's duration), and on an error reply that
  carries the field, with the daemon's error in its message
  (`speak_server.py:253-255`).
- **No protocol bump.** A protocol 4 client that predates the field ignores
  it and still clears its expectation from `expect`; `PROTOCOL_VERSION`
  stays 4.

## 5. Wire and protocol

No protocol bump (owner decision 9). Verified reasons it is not needed:
- The daemon already receives every tool op's session id before the lock
  (`speak_telegram_daemon.py:732-740`; `FIELD_SESSION_ID`,
  `speak_protocol.py:62`): queue order needs no new request field.
- No reply changes: no "waiting" hint (owner decision 4), and `status` does
  not show the floor (owner decision 9); the switch line the
  client prints is unchanged (`speak_server.py:227-234`).
- The etiquette is instruction and docstring text in the client
  (`speak_server.py:356-402`, `431-445`); a client without it still speaks
  protocol 3 correctly. `FLOOR_IDLE_SECONDS` in `speak_protocol.py` is a
  shared constant, not a field.

`PROTOCOL_VERSION` stays 3 (`speak_protocol.py:54`).

### 5.1 Deploy steps

1. Deploy the daemon, on the owner's word:
   `docker --context patrick compose up --build -d`.
2. On every computer: `git pull` this repo, then `/mcp` reconnect in every
   session (or start new sessions). Old clients keep working (same
   protocol); they only lack the new etiquette text until they reconnect.
   Server instructions are read when the MCP session starts, so a running
   session sees the new text only after the reconnect.

## 6. Client text (`speak_server.py`)

**Owner decisions 2026-10-26** (agreed on a call), replacing the text of
2026-10-22 and its coming-back additions of 2026-10-23:
1. Stay in `converse` while working the problem out with the owner
   (understanding it, deciding what to do).
2. When the agent is about to go do the work, whatever its length (even a
   2-minute job), it asks the owner whether he is ready to let it go, then
   signs off with ONE `speak` that promises to come back on the call. This
   replaces the 2026-10-22 rules "stay in `converse` while an agent you
   started researches for a few minutes" and "sign off with `speak` only when
   the topic has closed, the owner has nothing else, and the remaining work
   is around 15 minutes or more".
3. Coming back is owed, by voice. When the work is done: `status()` first. On
   "connected", ONE `converse` with the report, left waiting in the queue
   however long it takes, even behind another session's conversation; while
   it waits, the agent does not cancel it, resend it, text instead, or
   `call`. On "ringing", `status()` again. On "none", `call`; the call
   etiquette applies (`not answered`: wait an hour by background timer, and
   so on).
4. The instructions are written in the second person to the agent on the
   line ("you ..."), not as a description of the daemon.

**Owner decision 2026-10-27, superseded by the owner decisions of
2026-10-28** (section 4.10: the daemon hangs up after the sign-off, and the
agent does nothing more on the call). As it stood (on a call), amending the
rule "`hang_up` only when he asks": right after the one `speak` sign-off of decision 2, the agent
calls `status()`. If `busy` is false and `waiting` is 0 (no other session
holds the lock or is queued), it calls `hang_up()`; otherwise it leaves the
call open for the next session. The call etiquette's hang-up rule now reads
"`hang_up()` only when he asks, or after your sign-off when `status()` shows
nobody else". The owner accepted the race in which another session queues
between the `status()` and the `hang_up()` (his decision, 2026-10-27).
Client text only: the daemon, the floor mechanism and the protocol are
unchanged.

The two verified facts below describe the code under the superseded rule;
the two tests they name were removed with the 2026-10-28 change and replaced
by the tests of section 8.

**Verified from the code (2026-10-27), why the rule reads `busy` too:**
`waiting` is `len(_queue)` (`AudioLock.snapshot`,
`speak_telegram_daemon.py:384`), and an op leaves the queue the moment it
takes the lock (`AudioLock.acquire`, `speak_telegram_daemon.py:328`). A
session queued behind the sign-off takes the lock as soon as the `speak`
ends (the floor was given up, `speak_telegram_daemon.py:747`), so it shows
as `busy` true with `waiting` 0; a rule on `waiting` alone would hang up on
it. `tests/test_floor.py`
(`test_after_a_sign_off_a_session_queued_behind_it_shows_as_busy_not_waiting`)
checks it over the wire.

**Verified from the code (2026-10-27), what the accepted race does:**
`hang_up` is a locked op (`LOCKED_OPS`, `speak_protocol.py:100`; acquired at
`speak_telegram_daemon.py:959`), so it queues like the others. When B's op
is queued before A's `hang_up`, B gets the lock first (no floor is live
after A's sign-off) and takes the floor with its turn (`_take_turn`,
`speak_telegram_daemon.py:606`). While B's floor is live, A's queued
`hang_up` is not eligible (`_first_eligible`, `speak_telegram_daemon.py:388`):
it waits until B's floor lapses, `FLOOR_IDLE_SECONDS` (45 s,
`speak_protocol.py:143`) after B's last op released the lock (the deadline
is set at release, `speak_telegram_daemon.py:342`), or until B gives the
floor up (its own sign-off `speak`, or a `hang_up`). Then it ends the call
(`SpeakService.hang_up`, `speak_telegram_daemon.py:826`), which is in the
middle of B's conversation when B went quiet for 45 s. B then finds the call
ended ("Telegram call ended during ..." or "needs a connected Telegram
call") and calls back under the call etiquette. `tests/test_floor.py`
(`test_a_hang_up_sent_after_another_session_queued_waits_behind_its_floor`)
checks the order and the wait.

**Client text (2026-10-28):** the floor section's sign-off bullet now reads
"After your sign-off, do nothing more on the call: with nobody else waiting
the daemon hangs up ("[call ended: nobody else was waiting]"), otherwise the
next session goes."; the Calls section says "hang_up() only when he asks.";
the `speak`, `hang_up` and `call` docstrings say the same, and `status` no
longer mentions the sign-off (sections 6.1 and 6.3).

**Verified (2026-10-26):** Claude Code truncates MCP server instructions at
4096 characters. The 2026-10-23 `_INSTRUCTIONS` was 6,891 characters, so
every floor and coming-back rule sat after the cut and agents never saw
them.

**Verified 2026-10-26** in a live call from this repo: a queued converse
that waited behind another session's conversation was moved to the
background by Claude Code after 120 s, kept running (the request was not
cancelled), played when the floor was released, and its result arrived as a
task notification with the owner's words.

**Design (code review, 2026-10-26):** the no-cancel, no-resend rule of
decision 3 holds only while the `converse` waits. When it returns "Telegram
call ended during ..." or "needs a connected Telegram call", the agent calls
back with `call` as the call etiquette says. Kept from the earlier text, and
restored in the review: `override_quiet_hours=True` without asking when the
owner asks for a call; `set_call_debug(True)` and a test call without asking
when a call seems off. "Hand real work to agents" now reads "once he lets you
go, hand the work to agents", so it agrees with decision 2.

**Design (2026-10-26):** `_INSTRUCTIONS` states the floor and coming back
first, then calls, the persona and speaking, and keeps every rule agents
still need in 3,971 characters (3,826 after the 2026-10-27 hang-up rule,
3,796 since the 2026-10-28 sign-off text,
made room for by shorter wording for points the docstrings carry: the 120 s
backgrounding, the took-over recap, the persona review, `TimeoutError` and
the hold phrases, which `_HOLD_NOTE` explains; "never speak 'hold on' or
'continue' as a sentence by itself" is kept, no rule dropped); wording that the tool docstrings
already carry is cut from it. `speak_server.py` raises `RuntimeError` at
import (`check_instructions_length`) when `len(_INSTRUCTIONS)` reaches
`INSTRUCTIONS_MAX_CHARS` (4096), with a comment saying why;
`tests/test_hold.py` checks the length, the boundary (4095 passes, 4096 is
refused), the import-time refusal, the order and the 2026-10-26 wording. Kept from
2026-10-22: the 45 s idle timeout is not mentioned anywhere agent-facing
(`FLOOR_IDLE_SECONDS` still drives the daemon), and there is no "one moment"
before work. Client only; the protocol is unchanged.

### 6.1 `_INSTRUCTIONS`

The floor section, first in the text:

```
The floor (one conversation at a time):
- The session that last played audio holds the floor. converse() and listen() keep it;
  speak() gives it up. Other sessions queue until then.
- While you and the owner work the problem out (understanding it, deciding what to do),
  stay in converse().
- When you are about to go do the work, whatever its length (even two minutes), ask him
  whether he is ready to let you go. Once he says yes, sign off with ONE speak() that
  promises to come back on the call, never "by message".
- After your sign-off, do nothing more on the call: with nobody else waiting the daemon
  hangs up ("[call ended: nobody else was waiting]"), otherwise the next session goes.
- Coming back is owed, by voice. When the work is done, report it on the call before
  anything else, even after your context was summarized; a text is only in addition.
  Never ask whether he wants a call back. First status(). "connected": send ONE
  converse() with your report and leave it waiting in the queue, however long, even
  behind another session's conversation. Being backgrounded after ~120 s is normal.
  While it waits, do not cancel it, resend it, text instead or call(). If it returns
  "Telegram call ended during ..." or "needs a connected Telegram call", call() back as
  Calls says.
  "ringing": another session is dialing; status() again. "none": call().
- '[took over the call from ...]': open with one sentence recapping where you left off
  and your last question, unless his reply answers it.
```

The hang-up rule in the Calls section (2026-10-28):

```
  call back. hang_up() only when he asks.
```

### 6.2 `_QUEUE_NOTE`

The first paragraph:

```
    Status: speak/listen/converse/call/hang_up are serialized through one
    lock in the speak daemon, shared by every Claude Code session on every
    computer -- there is only one Telegram call. If another tool call is in
    progress anywhere, this one queues and waits; it is never rejected. Use
    `status()` to see what is running and how many calls are waiting; `busy`
    false with `waiting` above 0 means another session holds the floor. Your
    request still queues behind it: leave it waiting.
```

The switch paragraph is unchanged.

### 6.3 Docstrings

`speak`:

```
    speak() gives up the floor. Use it only for your sign-off, once the owner has said you may
    go do the work: one speak() that promises to come back on the call. Then do nothing more
    on the call: when no other request is queued as it ends, the daemon hangs up and the
    result's last line before the audio note is "[call ended: nobody else was waiting]";
    otherwise the call stays up and the next session takes the floor. Do not call hang_up()
    or call back for it. If the sign-off raises an error other than "Telegram call ended
    during ...", the call is still up: send the speak() again. When the work is done, come
    back on the call: status() first.
```

`hang_up`, its first lines (2026-10-28):

```
    End the Telegram call placed by `call()`. Use it only when the owner asks you to.
    After your speak() sign-off the daemon itself hangs up when no other request is
    queued, so never call it for that.
```

`listen`: "listen() keeps the floor."

`converse`:

```
    converse() keeps the floor. Coming back with a report, it may wait minutes in the queue
    behind another session; after about 120 s Claude Code moves it to the background and
    notifies you when it completes. While it waits, leave it: do not cancel it, resend it,
    text instead or call(). If it returns "Telegram call ended during ..." or "needs a
    connected Telegram call", call() back.
```

`status`, its last paragraph:

```
    Check it first when you come back with a report. "connected" means converse(), even
    when `busy` is true or `waiting` is above 0: your converse queues and you leave it
    waiting. This `busy` is not call()'s "busy" outcome. "ringing" means another session is
    dialing: check again until "connected" (converse) or "none" (call).
```

(The 2026-10-27 paragraph on checking `status` after the sign-off was removed
on 2026-10-28.)

`call`, its last line (2026-10-23, unchanged): "To come back with a result,
status() first; while the call is connected use converse(), not call()."

`call`, its rule-of-engagement paragraph (hang-up sentence changed 2026-10-28):

```
    Rule of engagement: if speak/listen/converse raises "Telegram call ended
    during ...", the call dropped mid-conversation -- call back with this tool.
    The daemon never dials sooner than 10 s after the previous call ended,
    whatever the reason (a redial 1.4 s after a hang-up left the owner's
    Telegram app dropping every incoming call until it was force-quit); a
    call placed sooner waits out the rest of those 10 s, then dials.
    Hang up with `hang_up()` only when the owner asks; the owner can also hang
    up themselves, and the daemon hangs up after a speak() sign-off when no
    other request is queued ("[call ended: nobody else was waiting]"). If the
    speak daemon cannot be reached this raises "speak daemon at <addr> is not
    reachable: ...".
```

## 7. Code changes

| File | Change |
|---|---|
| `speak_protocol.py` | `FLOOR_IDLE_SECONDS = 45.0` |
| `speak_telegram_daemon.py` | `AudioLock`: per-ticket session id, eligibility rule (4.2), timed waits and lapse (4.3), `take_floor` (refused unless the lock is held by that session), `yield_floor`, `close` (4.9), `_current_session`, constructor `idle_seconds` and `record_event`; `acquire` clears the deadline when the holder's ticket arrives and removes its ticket if its wait raises (4.2). `_Handler.handle`: passes the `Speaker` to `acquire`. `transcribe_pcm`: acquires with no session (always eligible). `SpeakService`: takes the lock; `_take_turn` and the answered branch of `call` call `take_floor`; `speak` yields with `speak` in a `finally` once the turn was taken, except on the call-ended error; `hang_up` yields with `hang_up`; `_hold_expired` yields with `hold_expired`; `_client_left_during_hold` yields with `client_left`. `SpeakDaemon.__init__`: builds the lock before the service; `SpeakDaemon.stop` and the handler's audio-failure restart call `lock.close()` right after setting `closing` (4.9) |
| `speak_telegram_daemon.py` (2026-10-28) | `acquire` takes the ticket's `client_left` probe; `yield_floor` returns the count of queued tool ops whose client is still there (None when nothing was yielded); `SpeakService.speak` calls `_sign_off_hang_up` when the speech returned and the count is 0, and only yields when it raised; `Session.call_ended`; `run_locked` adds `call_ended`; events `signoff_hangup`, `signoff_hangup_failed` (section 4.10) |
| `speak_protocol.py` (2026-10-28) | `REPLY_CALL_ENDED`, `CALL_ENDED_SIGNOFF` |
| `speak_server.py` | Section 6 text; `_ended_line` (2026-10-28) |
| `tests/` | Section 8 |

## 8. Tests to add (local venv, no CUDA, no Telegram)

`tests/test_floor.py` (new), at the `AudioLock` level with real threads and a
short `idle_seconds`:
- The holder's op queued after another session's goes first.
- With no live floor, plain FIFO, as before.
- With a live floor and nothing queued by the holder, a waiting session does
  not get the free lock until the deadline, then does; one `floor_release`
  with reason `idle` is recorded.
- The deadline is cleared while the holder's op runs and restarts at its
  release.
- `yield_floor` hands the lock to the head of the queue at the next release.
- A lapse observed by a later `acquire` (nobody waiting) is recorded once.
- A `transcribe` ticket (no session) queued behind a live floor goes as soon
  as the lock is free, ahead of other sessions' tickets, and its release
  leaves the idle deadline unchanged.
- (Build review) The holder's op that arrives before the deadline, while a
  `transcribe` holds the lock past it, goes before a session queued earlier;
  no lapse is recorded.
- (Build review) `take_floor` is refused with the lock free, held by
  another session, or held by a `transcribe`.
- (Build review) An `acquire` whose wait raises leaves no ticket behind.
- (Build review) `close()` hands the free lock to the head of the queue
  despite a live floor, in arrival order, with no `floor_release`.

`tests/test_session_switch.py` style (service level, `HoldServiceCase`
helpers):
- `converse` and `listen` take the floor; a bare `speak` takes it and ends
  with it released (`floor_release` reason `speak`).
- A `speak` during which the call drops (`ScriptedLink` ends the call
  mid-speech) raises the call-ended error and keeps the floor; no
  `floor_release` is recorded.
- A `speak` that raises for another reason after its turn was taken releases
  with `speak`.
- An empty-text `speak` or an op with no call does not take or release the
  floor.
- `listen` ending in `NoSpeech` keeps the floor.
- `hang_up` releases with `hang_up`; a hold expiry releases with
  `hold_expired` (also when the hang-up fails); a client that leaves during a
  hold of `speak` and of `converse` releases with `client_left`, once.
- An answered `call` takes the floor (`floor_take` with the caller's
  project); `not answered`, `busy` and `already connected` leave it as it
  was.
- Repeated `listen` calls ending in `NoSpeech` keep the floor throughout.
- `floor_take` carries `from_project`; the switch announcement still plays
  once per change of hands.

`tests/test_daemon_wire.py` style (a real `SpeakDaemon` on a loopback port,
`TurnLink`, the client's own sync bodies):
- Session A converses; while A's op runs, session B queues a converse; A's
  next converse, sent after B's, runs before B's; then A speaks and B's
  converse runs, with the switch line in B's result.
- A converses and goes quiet; B's queued op runs after the (shortened) idle
  time, and `call_debug` shows `floor_release` with reason `idle`.
- The call drops while A holds the floor and B is queued; A's `call`, sent
  after B's op, runs first and is answered; B still waits.
- An HTTP `POST /` to the whisper port while A holds the floor with the lock
  free is answered without waiting for A to finish its conversation.
- `status` while B waits behind a floor with the lock free: `busy` false,
  `waiting` 1 (unchanged fields, unchanged meaning).
- (Build review) B queues behind A's floor and its client leaves: once the
  floor lapses, B's op is skipped and plays nothing.
- (Build review) The daemon stops while B waits behind A's floor: B gets
  `speak daemon is restarting` at once, not after the idle time.

Added 2026-10-28 for section 4.10 (`tests/test_floor.py`):
- `AudioLock.yield_floor` counts the queued tool ops and not transcriptions;
  it counts 0 with nothing queued, and returns None without the floor; it
  skips a queued op whose `client_left` probe is true, so a queue of only
  departed clients counts 0.
- Service level (`TestSignOffHangUp`): a sign-off with nothing queued hangs
  up, clears the expectation and records `floor_release` then
  `signoff_hangup`; a held and resumed one hangs up too; `converse` and
  `listen` (with and without speech) never hang up; a drop during the speak,
  an expired hold, a client that left during a hold and an empty-text speak
  are no sign-off; a sign-off speak that raised releases the floor, leaves
  the call up and the expectation standing, and its retry signs off; a
  failed sign-off hang-up records `signoff_hangup_failed`, raises its own
  error and still clears; `run_locked` carries `call_ended` and `expect`
  null after a sign-off, and no `call_ended` on a failed speak or a failed
  hang-up.
- Over the wire (`TestSignOffHangUpOverTheWire`): a sign-off with nobody
  queued returns the "[call ended: ...]" line, clears the client's
  expectation and leaves no call; with a session queued the call stays up
  and that session takes the floor; a queued transcription does not keep the
  call up and is answered 200 after the hang-up; a failed sign-off speak's
  error is the bare error, the call stays up and the retry ends it; a queue
  holding only a session whose client closed its socket counts as empty,
  so the call ends and that op is skipped; an op queued while the sign-off's
  hang-up is in progress gets "needs a connected Telegram call"
  (`test_an_op_queued_during_the_sign_off_hang_up_finds_no_call`).
- Client (`tests/test_hold.py`): an unknown `call_ended` raises naming the
  speak's duration; an error reply carrying `call_ended` raises with the
  daemon's error.
- Removed: the two 2026-10-27 wire tests (`busy` not `waiting` after a
  sign-off, and a `hang_up` racing a queued session). The service-level
  switch tests that run several speaks on one call use a lock that always
  counts another waiting session (`tests/test_session_switch.py`).

The existing tests that build `SpeakService(link, engines, SETTINGS)`
(`tests/test_session_switch.py:57`, `tests/test_hold.py:455`) pass a lock.
Since `take_floor` requires the lock, the service-level tests
(`test_hold.py`, `test_session_switch.py`, `test_floor.py`,
`test_streamed_speech.py`) build the service with `locked_service`
(`tests/test_hold.py`): each tool op runs under the lock as its request's
session, as the handler runs it, on a lock whose queue is plain FIFO (the
floor's ordering is tested at the `AudioLock` level).

## 9. CLAUDE.md on build

CLAUDE.md is edited in the build as follows.
- **Tools table:** the `speak` row says it gives up the floor when it
  finishes; the `converse` and `listen` rows say they keep it.
- **New section "The floor (owner decisions 2026-10-20)"**, after "Switch of
  session": the decisions of section 2, the queue rule (4.2), the 45 s timer
  and who wakes the queue (4.3), the release table (4.5), the events (4.6).
- **"Switch of session":** one line that the switch cue now also marks the
  floor changing hands.
- **"Protocol":** "Tool ops ... queue in the daemon's FIFO lock and are never
  rejected" becomes FIFO with the floor holder's ops first; protocol stays 3.
- **"HTTP whisper API":** one line that transcriptions skip the floor and
  wait only for the op holding the lock.
- **"Status" (Tools table row):** unchanged; one line in the new section
  that the floor is visible only in `call_debug`.
- **"Files":** `speak_telegram_daemon.py` description names the floor in
  `AudioLock`; `speak_protocol.py` names `FLOOR_IDLE_SECONDS`.
- **"Development":** `tests/test_floor.py` in the test list.
