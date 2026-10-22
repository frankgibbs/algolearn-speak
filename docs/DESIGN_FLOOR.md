# Design: the floor (one conversation at a time on the shared call)

Status: design, written 2026-10-20; deployed in commit `b5cd149` on 2026-10-20.
The client text (section 6) was rewritten by owner decision 2026-10-22.
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
   decision 2026-10-22:** the etiquette as it now reads is in section 6.
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
| A `speak` that took the turn ends, except when the call ended under it | `speak` | Owner decision 2. Called in a `finally` in `SpeakService.speak` after `_take_turn` returned, skipped when the exception is the call-ended error (next row) |
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

**Owner decision 2026-10-22** (made on a call), replacing the text built on
2026-10-20:
- The floor etiquette is stated once, in `_INSTRUCTIONS`. Each of `speak`,
  `listen` and `converse` gets one plain line about its own floor effect.
  `_QUEUE_NOTE` keeps only its queue and `status` content.
- The 45 s idle timeout is not mentioned anywhere agent-facing: agents never
  reason about silence. (`FLOOR_IDLE_SECONDS` still drives the daemon; the
  client no longer composes it into text.)
- No "one moment" before work: `speak` is only the final sign-off.
- Stay in `converse` while the topic is alive, including while an agent you
  started researches for a few minutes; ask whether the owner has anything
  else before leaving; sign off with one `speak` only when the topic has
  closed, the owner has nothing else, and the remaining work is long (around
  15 minutes or more).

### 6.1 `_INSTRUCTIONS`

The turn bullet ends:

```
  converse() opens the ear right after the last word. listen() alone only to reopen the
  ear after a TimeoutError or when there is nothing to say first. speak() alone only for
  the final sign-off (see "the floor" below).
```

After the bullet "There is one call, shared by every session on every
computer; tools queue for it.":

```
- The floor: other sessions may be waiting for the call. The session that last played
  audio holds the floor, so the owner hears one conversation at a time. converse() and
  listen() keep the floor; speak() gives it up.
- Stay in converse() as long as the topic is alive, including while an agent you started
  does a few minutes of research: tell the owner what you started and keep the
  conversation going. He will likely have more questions, or a related thread on the
  same topic.
- Before leaving, ask whether he has anything else.
- End with ONE speak() sign-off (for example "Will do, talk soon.") only when the topic
  has closed, the owner has said he has nothing else, and the remaining work is long
  (around 15 minutes or more). Never use speak() for a quick check or a short task.
- Do not offer to call back; say you will come back with the result. When the work is
  done, check status(): if its call field says the call is still connected, converse()
  with the result (it waits its turn, and the owner hears the switch cue and your
  project's name first). Use call() only if the call has ended.
```

### 6.2 `_QUEUE_NOTE`

The first paragraph:

```
    Status: speak/listen/converse/call/hang_up are serialized through one
    lock in the speak daemon, shared by every Claude Code session on every
    computer -- there is only one Telegram call. If another tool call is in
    progress anywhere, this one queues and waits; it is never rejected. Use
    `status()` to see what is running and how many calls are waiting; `busy`
    false with `waiting` above 0 means another session holds the floor.
```

The switch paragraph is unchanged.

### 6.3 Docstrings

- `speak`: "speak() gives up the floor; use it only for the final sign-off."
  followed by the existing sentence that a conversational turn is
  `converse(text)`, never speak() then listen().
- `listen`: "listen() keeps the floor."
- `converse`: "converse() keeps the floor."

## 7. Code changes

| File | Change |
|---|---|
| `speak_protocol.py` | `FLOOR_IDLE_SECONDS = 45.0` |
| `speak_telegram_daemon.py` | `AudioLock`: per-ticket session id, eligibility rule (4.2), timed waits and lapse (4.3), `take_floor` (refused unless the lock is held by that session), `yield_floor`, `close` (4.9), `_current_session`, constructor `idle_seconds` and `record_event`; `acquire` clears the deadline when the holder's ticket arrives and removes its ticket if its wait raises (4.2). `_Handler.handle`: passes the `Speaker` to `acquire`. `transcribe_pcm`: acquires with no session (always eligible). `SpeakService`: takes the lock; `_take_turn` and the answered branch of `call` call `take_floor`; `speak` yields with `speak` in a `finally` once the turn was taken, except on the call-ended error; `hang_up` yields with `hang_up`; `_hold_expired` yields with `hold_expired`; `_client_left_during_hold` yields with `client_left`. `SpeakDaemon.__init__`: builds the lock before the service; `SpeakDaemon.stop` and the handler's audio-failure restart call `lock.close()` right after setting `closing` (4.9) |
| `speak_server.py` | Section 6 text |
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
