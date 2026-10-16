"""The Telegram call itself: one private voice call from the algolearn.ai account
to the owner, driven by raw PCM (docs/DESIGN_TELEGRAM_CALL.md sections 2-4, 8).

`TelegramLink` owns the Telethon client, py-tgcalls and at most one call, on its
own asyncio loop in a background thread. Its public methods are synchronous and
thread-safe (the speak daemon calls them from its request threads).
Instantiated only by speak_telegram_daemon.py; the thin MCP client never imports
Telethon or ntgcalls.

Built around what the 2026-10-09 spike verified:
- The call opens on a real audio FILE of at least OPENER_MIN_SECONDS (shorter
  openers gave all-zero audio both ways), waits for it to finish plus 1 s,
  then switches to raw frames.
- A 10 ms pump sends a frame every 10 ms for the whole call (silence when idle)
  and never bursts after a stall (bursts played the next audio sped up).

Speech streams (owner decision 2026-10-16, docs/DESIGN_STREAMED_SPEECH.md):
`play_stream` queues each segment as its producer (Kokoro, on the calling
thread) yields it, so playback starts after the first segment. If the producer
falls behind, the pump sends silence until the next segment arrives and
records the gap as a `speak_underrun` event; it never bursts to catch up.
- py-tgcalls raises CallDeclined / CallBusy / TimedOutAnswer / CallDiscarded
  from play() when the call is not taken (read in pytgcalls
  methods/internal/handle_mtproto_updates.py and connect_call.py).

Every call has a generation number and, once over, an end reason: "remote"
(the owner hung up or the call dropped), "hung_up" (our hang_up), "failed" (an
error tore it down), "not_answered" or "busy". The daemon compares these with a session's
expected generation to decide whether a call it was using dropped (call back) or
was hung up on request.
"""

from __future__ import annotations

import asyncio
import collections
import dataclasses
import datetime
import logging
import os
import queue
import tempfile
import threading
import time
import uuid
import wave
from collections.abc import Iterable
from typing import Callable

import ntgcalls
import numpy as np
from scipy.signal import resample_poly

from speak_protocol import (  # noqa: F401  (re-exported: the daemon and tests use tg.<name>)
    ALREADY_CONNECTED, ANSWERED, AUDIO_FAILED_PREFIX, BUSY, CONNECT_GRACE_SECONDS, CONNECTED, END_BUSY,
    END_FAILED, END_HUNG_UP, END_NOT_ANSWERED, END_REMOTE, HUNG_UP, HUNG_UP_DURING_GREETING, NO_ACTIVE_CALL,
    NONE, NOT_ANSWERED, OPENER_MIN_SECONDS, REDIAL_SETTLE_SECONDS, RING_TIMEOUT_SECONDS, RINGING, is_audio_failure,
)

log = logging.getLogger("speak.telegram")

CALL_RATE, CALL_CHANNELS = 48_000, 2
FRAME_BYTES = CALL_RATE // 100 * 2 * CALL_CHANNELS   # 10 ms of s16le stereo
VAD_RATE, VAD_BLOCK = 16_000, 512                    # Silero contract (32 ms)
NATIVE_BLOCK = VAD_BLOCK * CALL_RATE // VAD_RATE     # 1536 samples at 48 kHz
OPENER_TAIL_SECONDS = 1.0  # wait after the opener file before switching to raw frames
PUMP_MAX_LAG = 0.05        # behind by more than this -> reset the clock instead of bursting
PRE_ROLL_SECONDS = 0.5     # audio kept from just before speech starts
# Watching the owner's audio while Angie speaks or is held (docs/DESIGN_INTERRUPT.md
# section 4.4, Release 2). Only a short utterance after a pause is a command candidate.
WATCH_SILENCE_SECONDS = 0.4           # end of a command utterance (listen uses 2.0 s)
WATCH_LEADING_SILENCE_SECONDS = 0.8   # non-speech required before a candidate
WATCH_MIN_UTTERANCE_SECONDS = 0.3     # shorter is a click or a cough
WATCH_MAX_UTTERANCE_SECONDS = 2.0     # longer is conversation: dropped untranscribed
VAD_BLOCK_SECONDS = VAD_BLOCK / VAD_RATE   # 0.032
WATCH_POLL_SECONDS = 0.05             # how often the watch looks at the queue while nothing else runs
WATCH_DRAIN_GRACE_SECONDS = 30.0      # play_watched: the drain may run this long past the audio queued
HANG_UP_TIMEOUT = 8.0      # leave_call; with disconnect, stays inside compose's stop_grace_period
DISCONNECT_TIMEOUT = 5.0
EVENT_BUFFER_SIZE = 500
STATS_FAST_INTERVAL = 1.0    # amplitude summaries: every second ...
STATS_FAST_SECONDS = 10.0    # ... for the first 10 s after the call is answered ...
STATS_SLOW_INTERVAL = 10.0   # ... then every 10 s
MAX_LAG_EVENTS = 20   # pump_stall events per call, so a stalled call cannot flood the buffer
STREAM_STEP_TIMEOUT = 5.0   # play_stream: one loop round-trip to open, queue into, or stop a stream

TIMEOUT = object()   # record(): nobody spoke within start_timeout_seconds; watch_for(): no command in time
CLIENT_LEFT = object()   # watch_for(): the request's client went away during the hold


class CallEnded(RuntimeError):
    pass


class _Stream:
    """One streamed utterance (play_stream), touched on the loop thread only."""

    def __init__(self) -> None:
        self.t0 = time.monotonic()
        self.segments = 0
        self.audio_bytes = 0
        self.first_audio_ms: int | None = None   # stream start -> first segment queued
        self.audio_started = False   # the pump has sent the first segment's first frame
        self.gap_frames = 0          # silence frames sent since the queue ran dry mid-stream
        self.resume_segment = 0      # the segment queued into an empty buffer after audio started
        self.underruns = 0
        self.underrun_frames = 0


@dataclasses.dataclass(frozen=True)
class Classified:
    """What the daemon's classify made of one watch candidate."""
    phrase: str | None   # the command whose phrase the whole transcript matched exactly, else None
    text: str            # Whisper's transcript (in events only while debug is on)


@dataclasses.dataclass(frozen=True)
class PlayOutcome:
    """play_watched's result. `command` is None when everything played."""
    command: str | None
    played_bytes: int        # call PCM sent before the cut (all of it when completed)
    cut_tag: object | None   # tag of the segment that was playing at the cut; None when completed, or
                             # when the cut fell between segments (nothing queued was left unsent)
    seconds: float           # seconds of audio sent


@dataclasses.dataclass(frozen=True)
class _Hit:
    command: str
    utterance_end: float   # monotonic arrival time of the frame that ended the utterance


class _VadFeed:
    """48 kHz stereo s16 call frames -> 32 ms 16 kHz blocks, each with the VAD's event.
    Shared by the listen capture and the watch (docs/DESIGN_INTERRUPT.md section 4.5)."""

    def __init__(self, vad: Callable) -> None:
        self.vad = vad
        self.native = np.zeros(0, dtype=np.float32)

    def blocks(self, raw: bytes):
        s16 = np.frombuffer(raw, dtype=np.int16).reshape(-1, CALL_CHANNELS).mean(axis=1)
        self.native = np.concatenate([self.native, (s16 / 32768.0).astype(np.float32)])
        while len(self.native) >= NATIVE_BLOCK:
            block = resample_poly(self.native[:NATIVE_BLOCK], VAD_RATE, CALL_RATE).astype(np.float32)
            self.native = self.native[NATIVE_BLOCK:]
            yield block, self.vad(block)


class _Watcher:
    """The owner's audio while Angie speaks or is held, cut into utterances by the VAD
    (end after WATCH_SILENCE_SECONDS of quiet). An utterance is a candidate only when at
    least WATCH_LEADING_SILENCE_SECONDS of non-speech came before it and it lasted from
    WATCH_MIN_UTTERANCE_SECONDS to WATCH_MAX_UTTERANCE_SECONDS; anything else is dropped
    untranscribed. Each candidate goes to `classify` (Whisper and the phrase match, in the
    daemon). Every utterance is recorded as a watch_candidate event; the transcript is in
    it only while debug is on (link.audio_stats). Runs on the request thread."""

    def __init__(self, link: "TelegramLink", classify: Callable[[np.ndarray], Classified], accept: str,
                 mode: str) -> None:
        self.link = link
        self.classify = classify
        self.accept = accept   # the one command this state acts on (hold while playing, resume while held)
        self.mode = mode       # "playing" or "held", for the events
        self.feed = _VadFeed(link._vad_factory(WATCH_SILENCE_SECONDS))
        self.pre_roll: collections.deque[np.ndarray] = collections.deque(
            maxlen=int(PRE_ROLL_SECONDS * VAD_RATE / VAD_BLOCK))
        self.quiet_blocks = 0     # non-speech blocks since the last utterance ended (or the watch began)
        self.after_utterance = False   # an utterance has ended: its end already took WATCH_SILENCE_SECONDS
        self.speaking = False
        self.utterance: list[np.ndarray] = []
        self.utterance_blocks = 0
        self.too_long = False
        self.leading_s = 0.0

    def poll(self, timeout: float) -> _Hit | None:
        """Process the frames queued now, waiting up to `timeout` for the first one. Returns
        the first candidate that classify turned into a command, or None."""
        q = self.link._incoming
        try:
            item = q.get(timeout=timeout) if timeout > 0 else q.get_nowait()
        except queue.Empty:
            return None
        more = q.qsize()   # only what is queued now, so a steady stream cannot keep poll from returning
        while True:
            t, raw = item
            for block, event in self.feed.blocks(raw):
                hit = self._block(block, event, t)
                if hit is not None:
                    return hit
            if more <= 0:
                return None
            more -= 1
            try:
                item = q.get_nowait()
            except queue.Empty:
                return None

    def _block(self, block: np.ndarray, event: dict | None, t: float) -> _Hit | None:
        if not self.speaking:
            if event and "start" in event:
                self.speaking = True
                self.leading_s = self.quiet_blocks * VAD_BLOCK_SECONDS + (
                    WATCH_SILENCE_SECONDS if self.after_utterance else 0.0)
                self.utterance = [*self.pre_roll, block]
                self.utterance_blocks = 1
                self.too_long = False
                self.pre_roll.clear()
            else:
                self.quiet_blocks += 1
                self.pre_roll.append(block)
            return None
        self.utterance_blocks += 1
        if not self.too_long:
            self.utterance.append(block)
            if self.utterance_blocks * VAD_BLOCK_SECONDS > WATCH_MAX_UTTERANCE_SECONDS + WATCH_SILENCE_SECONDS:
                self.too_long = True   # conversation: stop keeping it, it will not be transcribed
                self.utterance = []
        if event and "end" in event:
            self.speaking = False
            self.quiet_blocks = 0
            self.after_utterance = True
            return self._candidate(t)
        return None

    def _candidate(self, t_end: float) -> _Hit | None:
        length = self.utterance_blocks * VAD_BLOCK_SECONDS - WATCH_SILENCE_SECONDS
        fields = {"mode": self.mode, "length_ms": round(length * 1000),
                  "leading_silence_ms": round(self.leading_s * 1000)}
        if self.leading_s < WATCH_LEADING_SILENCE_SECONDS:
            reason = "no_leading_pause"
        elif length < WATCH_MIN_UTTERANCE_SECONDS:
            reason = "too_short"
        elif self.too_long or length > WATCH_MAX_UTTERANCE_SECONDS:
            reason = "too_long"
        else:
            reason = None
        if reason is not None:
            self.utterance = []
            self.link._event("watch_candidate", result="dropped", reason=reason, **fields)
            return None
        audio = np.concatenate(self.utterance)
        self.utterance = []
        t0 = time.monotonic()
        c = self.classify(audio)
        acted = c.phrase is not None and c.phrase == self.accept
        fields.update(result="transcribed", transcribe_ms=round((time.monotonic() - t0) * 1000),
                      matched=c.phrase, acted=acted)
        if self.link.audio_stats:   # privacy: the transcript only while debug is on (design 4.7)
            fields["text"] = c.text
        self.link._event("watch_candidate", **fields)
        return _Hit(c.phrase, t_end) if acted else None


def describe(error: BaseException) -> str:
    """`Type: message`, or just `Type` when the message is empty (ntgcalls.TelegramServerError has none)."""
    return f"{type(error).__name__}: {error}" if str(error) else type(error).__name__


class CallEventLog:
    """Thread-safe in-memory ring buffer of structured call events (newest last).

    Every event also goes to the daemon log at INFO, so the container log carries the
    same record. Fields: ts (local ISO), call_id, event, plus event-specific extras."""

    def __init__(self, size: int = EVENT_BUFFER_SIZE) -> None:
        self._events: collections.deque[dict] = collections.deque(maxlen=size)
        self._lock = threading.Lock()

    def record(self, call_id: str | None, event: str, **fields) -> dict:
        entry = {"ts": datetime.datetime.now().isoformat(timespec="milliseconds"), "call_id": call_id,
                 "event": event, **fields}
        with self._lock:
            self._events.append(entry)
        log.info("call-event %s", entry)
        return entry

    def recent(self, n: int) -> list[dict]:
        with self._lock:
            return list(self._events)[-n:]


def dbfs(level: float) -> float | None:
    """int16 amplitude -> dB relative to full scale; None for exact silence."""
    return round(20 * np.log10(level / 32768.0), 1) if level > 0 else None


class AudioStats:
    """Peak / RMS / frame counts over a window of s16le frames. add() runs every 10 ms, so it only
    views the existing bytes (np.frombuffer) and makes one small float temporary."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.frames = self.zero_frames = 0
        self.peak = 0
        self._sumsq = 0.0
        self._samples = 0

    def add(self, data: bytes) -> None:
        a = np.frombuffer(data, dtype=np.int16, count=len(data) // 2)
        self.frames += 1
        if a.size == 0:
            self.zero_frames += 1
            return
        hi, lo = int(a.max()), int(a.min())
        peak = max(hi, -lo)
        if peak == 0:
            self.zero_frames += 1
            return
        self.peak = max(self.peak, peak)
        f = a.astype(np.float32)
        self._sumsq += float(np.dot(f, f))
        self._samples += a.size

    @property
    def rms(self) -> float:
        # Zero frames contribute samples of 0 to the mean, so count them in the denominator.
        n = self._samples + (self.zero_frames * (FRAME_BYTES // 2))
        return float(np.sqrt(self._sumsq / n)) if n else 0.0

    def fields(self, prefix: str = "") -> dict:
        rms = self.rms
        return {f"{prefix}frames": self.frames, f"{prefix}zero_frames": self.zero_frames,
                f"{prefix}peak": self.peak, f"{prefix}peak_dbfs": dbfs(self.peak),
                f"{prefix}rms": round(rms, 1), f"{prefix}rms_dbfs": dbfs(rms)}


def wav_stats(path: str) -> dict:
    """Peak / RMS of a 16-bit WAV file (the opener), over the whole file."""
    with wave.open(path, "rb") as w:
        data = w.readframes(w.getnframes())
    a = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    peak = int(np.abs(a).max()) if a.size else 0
    rms = float(np.sqrt(np.mean(a * a))) if a.size else 0.0
    return {"peak": peak, "peak_dbfs": dbfs(peak), "rms": round(rms, 1), "rms_dbfs": dbfs(rms)}


def to_call_pcm(audio: np.ndarray, rate: int) -> bytes:
    """Mono float PCM at `rate` -> 48 kHz stereo s16le bytes."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    up = resample_poly(audio, CALL_RATE, rate) if rate != CALL_RATE else audio
    s16 = (np.clip(up, -1.0, 1.0) * 32767).astype(np.int16)
    return np.repeat(s16, CALL_CHANNELS).tobytes()


def write_opener(path: str, audio: np.ndarray, rate: int) -> float:
    """The greeting as a 48 kHz mono WAV. Returns its length. Raises ValueError if it is shorter than
    OPENER_MIN_SECONDS: a short opener gives a dead call, and padding it would leave dead air after the
    greeting's question. The daemon's call op rejects short greetings before dialing, so this is a backstop."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    up = resample_poly(audio, CALL_RATE, rate) if rate != CALL_RATE else audio
    s16 = (np.clip(up, -1.0, 1.0) * 32767).astype(np.int16)
    if len(s16) < int(OPENER_MIN_SECONDS * CALL_RATE):
        raise ValueError(f"greeting is {len(s16) / CALL_RATE:.1f}s; the call opener must be at least {OPENER_MIN_SECONDS:g}s")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(CALL_RATE)
        w.writeframes(s16.tobytes())
    return len(s16) / CALL_RATE


_silero_model = None


def preload_silero_vad() -> None:
    """Load the Silero model (and torch) now. Run at link start so no load blocks the loop mid-call."""
    global _silero_model
    from silero_vad import load_silero_vad

    if _silero_model is None:
        _silero_model = load_silero_vad()


def silero_vad_factory(silence_seconds: float):
    """A Silero VADIterator (threshold 0.5, 16 kHz; model from preload_silero_vad)."""
    import torch
    from silero_vad import VADIterator

    if _silero_model is None:
        raise RuntimeError("Silero model not loaded: preload_silero_vad() runs in TelegramLink.start()")
    vad = VADIterator(_silero_model, threshold=0.5, sampling_rate=VAD_RATE,
                      min_silence_duration_ms=int(silence_seconds * 1000), speech_pad_ms=300)
    return lambda block: vad(torch.from_numpy(block.copy()), return_seconds=False)


class TelegramLink:
    """See the module docstring. `client_factory()` returns a Telethon client and
    `calls_factory(client)` a PyTgCalls; both are injected so tests never touch Telegram."""

    def __init__(
        self,
        target: str,
        client_factory: Callable,
        calls_factory: Callable,
        vad_factory: Callable = silero_vad_factory,
        vad_preload: Callable = preload_silero_vad,
    ) -> None:
        self.target = target
        self._client_factory = client_factory
        self._calls_factory = calls_factory
        self._vad_factory = vad_factory
        self._vad_preload = vad_preload
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, name="telegram-loop", daemon=True)
        self._client = None
        self._calls = None
        self._user_id: int | None = None
        self._state = NONE
        self._generation = 0
        self._end_reason: str | None = None
        self._ended = threading.Event()     # set when the current call ends (read from VAD threads too)
        self._ended_at: float | None = None # monotonic time the last call ended (REDIAL_SETTLE_SECONDS)
        self._outgoing = bytearray()
        self._drained: asyncio.Event | None = None
        self._pump_task: asyncio.Task | None = None
        self._ring_task: asyncio.Task | None = None
        self._incoming: queue.Queue[tuple[float, bytes]] = queue.Queue()   # (monotonic arrival, frame)
        self._listening = False
        self._watching = False   # play_watched / watch_for: the owner's frames are queued during playback too
        self.events = CallEventLog()
        self._incoming_seen = False
        self._call_id: str | None = None
        self._call_t0 = 0.0
        self._accepted = False   # Telegram confirmed the callee accepted this call (see _start)
        # Amplitude diagnostics (all touched on the loop thread only). Frames are always counted;
        # the events are recorded only while debug is on (set_debug sets audio_stats), so they can't
        # crowd a long call's own events out of the ring (owner decision 2026-10-12: reactive debugging).
        self.audio_stats = False
        self._phase = "opener"                # "opener" until the switch to raw frames, then "raw"
        self._in_stats = AudioStats()         # every incoming frame, listening or not
        self._out_stats = AudioStats()        # every frame the pump sends
        self._out_audio_stats = AudioStats()  # only the pump frames carrying queued audio
        self._out_silence_frames = 0
        self._opener_out_stats = AudioStats() # outgoing frames py-tgcalls reports during the opener file
        self._stats_task: asyncio.Task | None = None
        self._stats_t = 0.0
        # Pump timing per playback (loop thread only): the largest lag behind the 10 ms schedule
        # and the number of stalls over PUMP_MAX_LAG since the last play/play_stream began.
        self._lag_max = 0.0
        self._lag_stalls = 0
        self._stream: _Stream | None = None

    def _event(self, event: str, **fields) -> None:
        """Record a call event for the current call (elapsed seconds since it was requested)."""
        if self._call_id is not None:
            fields["elapsed_s"] = round(time.monotonic() - self._call_t0, 3)
        self.events.record(self._call_id, event, **fields)

    def record_event(self, event: str, **fields) -> None:
        """A call event recorded by the daemon's tool logic (e.g. hold_expire_hangup_failed)."""
        self._event(event, **fields)

    # ------------------------------------------------------------ lifecycle

    def start(self) -> None:
        try:
            self._vad_preload()   # a load failure fails start; never load a model during a call
            self._thread.start()
            self._run(self._start(), 60.0, "connecting to Telegram")
        except BaseException:
            self._close_loop()
            raise
        log.info("telegram link ready (target %s, user id %s)", self.target, self._user_id)

    async def _start(self) -> None:
        from pytgcalls.types import ChatUpdate, Device, Direction, StreamFrames

        self._drained = asyncio.Event()
        self._client = self._client_factory()
        self._calls = self._calls_factory(self._client)

        @self._calls.on_update()
        async def on_update(_, update):
            if isinstance(update, StreamFrames):
                if update.chat_id == self._user_id and update.direction & Direction.INCOMING \
                        and update.device & Device.MICROPHONE:
                    if self._state == CONNECTED:
                        for f in update.frames:
                            self._in_stats.add(f.frame)
                    if self._listening or self._watching:
                        if not self._incoming_seen:
                            self._incoming_seen = True
                            self._event("first_incoming_frame", during="listen" if self._listening else "watch")
                        now = time.monotonic()
                        for f in update.frames:
                            self._incoming.put((now, f.frame))
                elif update.chat_id == self._user_id and update.direction & Direction.OUTGOING \
                        and update.device & Device.MICROPHONE and self._state == CONNECTED and self._phase == "opener":
                    # Probe: does ntgcalls report the frames of the opener FILE? (audio_opener_out says.)
                    for f in update.frames:
                        self._opener_out_stats.add(f.frame)
            elif isinstance(update, ChatUpdate) and update.chat_id == self._user_id:
                self._event("chat_update", status=str(update.status))
                if update.status & (ChatUpdate.Status.DISCARDED_CALL | ChatUpdate.Status.LEFT_CALL | ChatUpdate.Status.BUSY_CALL):
                    if self._state == NONE:
                        return   # a late update for a call already torn down; never end the next one
                    log.info("call ended by the other side (%s)", update.status)
                    self._end_call(END_REMOTE)

        # RawCallUpdate (the callee accepting) is consumed inside py-tgcalls and never reaches
        # on_update, so observe it by wrapping the handler py-tgcalls registers in start().
        # Private API: fail at startup if it is not there.
        from pytgcalls.types import RawCallUpdate
        mtproto_handler = self._calls._handle_mtproto_updates

        async def observe_mtproto(update):
            if isinstance(update, RawCallUpdate) and update.chat_id == self._user_id \
                    and update.status & RawCallUpdate.Type.UPDATED_CALL and self._state != NONE:
                self._accepted = True
                self._event("accepted", status=str(update.status))
            await mtproto_handler(update)

        self._calls._handle_mtproto_updates = observe_mtproto

        # connect() + is_user_authorized(), not start(): start() would prompt on stdin for a
        # phone number if the session were revoked, which under a restart policy loops forever.
        await self._client.connect()
        if not await self._client.is_user_authorized():
            raise RuntimeError("TELEGRAM_SESSION is not authorized (revoked or expired); create a new session for the algolearn.ai account")
        await self._calls.start()
        self._user_id = (await self._client.get_entity(self.target)).id

    def stop(self) -> None:
        """Hang up (or abandon a ringing call) and disconnect. The caller must then os._exit:
        ntgcalls aborts in its C++ destructors at interpreter exit."""
        try:
            if self._state != NONE:
                self.hang_up()
        finally:
            try:
                self._run(self._disconnect(), DISCONNECT_TIMEOUT, "disconnecting from Telegram")
            finally:
                self._close_loop()

    def _close_loop(self) -> None:
        """Stop the loop thread and close the loop. Raises if the thread does not stop."""
        if self._thread.is_alive():
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(DISCONNECT_TIMEOUT)
            if self._thread.is_alive():
                raise RuntimeError(f"Telegram link: the loop thread did not stop within {DISCONNECT_TIMEOUT:.0f}s")
        self._loop.close()

    async def _disconnect(self) -> None:
        # Telethon's disconnect() is a coroutine when called on its running loop (here),
        # but runs synchronously and returns None when called from another thread.
        await self._client.disconnect()

    def _run(self, coro, timeout: float, what: str):
        """Run `coro` on the loop; on timeout cancel it (so it cannot finish later) and say what timed out."""
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return fut.result(timeout)
        except TimeoutError:
            fut.cancel()
            raise TimeoutError(f"Telegram link: {what} did not finish within {timeout:.0f}s") from None

    # ------------------------------------------------------------ state

    def state(self) -> str:
        return self._state

    def info(self) -> dict:
        """State plus the current/last call's generation and end reason."""
        return {"state": self._state, "generation": self._generation, "end_reason": self._end_reason}

    def _end_call(self, reason: str) -> None:
        """On the loop thread: the call is over, whoever ended it."""
        if self._state == NONE and self._ended.is_set():
            return
        self._state = NONE
        self._end_reason = reason
        self._ended_at = time.monotonic()
        self._event("ended", reason=reason, duration_s=round(time.monotonic() - self._call_t0, 3))
        self._ended.set()
        self._listening = False
        self._watching = False
        self._stop_stats()
        self._outgoing.clear()
        self._drained.set()
        if self._pump_task is not None:
            self._pump_task.cancel()
            self._pump_task = None

    async def _leave(self) -> None:
        """Best-effort leave_call while tearing down: a failure is logged, never raised over the real error."""
        from pytgcalls.exceptions import NotInCallError

        try:
            await asyncio.wait_for(self._calls.leave_call(self._user_id), HANG_UP_TIMEOUT)
        except NotInCallError:
            pass   # already gone on the library side
        except BaseException as e:
            log.error("leave_call failed while tearing the call down: %s: %s", type(e).__name__, e)

    # ------------------------------------------------------------ call / hang up

    def call(self, greeting: np.ndarray, rate: int) -> str:
        budget = REDIAL_SETTLE_SECONDS + RING_TIMEOUT_SECONDS + CONNECT_GRACE_SECONDS + len(greeting) / rate + 15.0
        return self._run(self._call(greeting, rate), budget, "placing the call")

    async def _call(self, greeting: np.ndarray, rate: int) -> str:
        from pytgcalls.exceptions import CallBusy, CallDeclined, CallDiscarded, TimedOutAnswer
        from pytgcalls.types import CallConfig, ExternalMedia, MediaStream, RecordStream
        from pytgcalls.types.raw import AudioParameters

        if self._state == CONNECTED:
            return ALREADY_CONNECTED
        if self._state != NONE:
            raise RuntimeError(f"a Telegram call is already {self._state}")
        params = AudioParameters(CALL_RATE, CALL_CHANNELS)
        fd, opener = tempfile.mkstemp(prefix="speak-telegram-opener-", suffix=".wav")
        os.close(fd)
        try:
            opener_seconds = write_opener(opener, greeting, rate)
            await self._redial_settle()
            self._generation += 1
            self._end_reason = None
            self._ended.clear()
            self._incoming_seen = False
            self._accepted = False
            self._phase = "opener"
            for st in (self._in_stats, self._out_stats, self._out_audio_stats, self._opener_out_stats):
                st.reset()
            self._out_silence_frames = 0
            self._call_id, self._call_t0 = uuid.uuid4().hex[:8], time.monotonic()
            self._event("requested", generation=self._generation, target=self.target)
            if self.audio_stats:
                self._event("opener_written", duration_s=round(opener_seconds, 3), **wav_stats(opener))
            self._state = RINGING
            self._event("ringing", ring_timeout_s=RING_TIMEOUT_SECONDS)
            self._ring_task = asyncio.current_task()
            try:
                await self._calls.play(self._user_id, MediaStream(opener, params), CallConfig(timeout=RING_TIMEOUT_SECONDS))
            except CallBusy as e:
                # Telegram discarded the call as busy: the owner is on another call, or their app
                # is still ending the previous one. Its own outcome, never "not answered".
                log.info("call not placed: CallBusy")
                self._event("busy", error=describe(e))
                self._end_call(END_BUSY)
                return BUSY
            except (CallDeclined, TimedOutAnswer, CallDiscarded) as e:
                log.info("call not answered: %s", type(e).__name__)
                self._event({CallDeclined: "declined", TimedOutAnswer: "timeout",
                             CallDiscarded: "discarded"}[type(e)], error=describe(e))
                self._end_call(END_NOT_ANSWERED)
                return NOT_ANSWERED
            except ntgcalls.TelegramServerError as e:
                if self._accepted:
                    return await self._audio_failed(e)
                # No accepted/confirmed update for this call: py-tgcalls' clear_call raised this
                # because the call was cleared before the callee took it (declined/discarded).
                log.info("call not answered: TelegramServerError before the callee accepted")
                self._event("discarded_before_accept", error=describe(e))
                self._end_call(END_NOT_ANSWERED)
                return NOT_ANSWERED
            except BaseException as e:   # includes cancellation (timeout, hang_up, daemon stop): never leave it ringing
                self._event("error", error=describe(e))
                await self._leave()
                self._end_call(END_FAILED)
                raise
            finally:
                self._ring_task = None
            self._event("answered_media_connected", opener_s=round(opener_seconds, 3))  # the greeting file starts playing now
            # Answered. Anything that fails from here tears the call down: never a live,
            # half-built call with no pump.
            try:
                self._state = CONNECTED
                self._start_stats()
                await self._calls.record(self._user_id, RecordStream(audio=True, audio_parameters=params))
                self._event("recording", opener_s=round(opener_seconds, 3))
                await asyncio.sleep(opener_seconds + OPENER_TAIL_SECONDS)
                if self._ended.is_set():
                    return self._ended_during_greeting()
                self._emit_audio_stats()   # closes the opener phase's window
                if self.audio_stats:
                    self._event("audio_opener_out", exposed=self._opener_out_stats.frames > 0,
                                **self._opener_out_stats.fields())
                self._phase = "raw"
                await self._calls.play(self._user_id, MediaStream(ExternalMedia.AUDIO, params))
                self._event("raw_frames")
                if self._ended.is_set():
                    # Ended while switching; py-tgcalls may have re-dialed, so make sure nothing is left up.
                    await self._leave()
                    return self._ended_during_greeting()
                self._pump_task = asyncio.create_task(self._pump())
                self._pump_task.add_done_callback(self._pump_done)
            except ntgcalls.TelegramServerError as e:   # the callee accepted (play() returned): a media-connect failure
                return await self._audio_failed(e)
            except BaseException as e:
                self._event("error", error=describe(e))
                await self._leave()
                self._end_call(END_FAILED)
                raise
        finally:
            os.unlink(opener)
        self._event("connected")
        log.info("call %d connected; raw-frame pump running", self._generation)
        return ANSWERED

    async def _redial_settle(self) -> None:
        """Wait out the rest of REDIAL_SETTLE_SECONDS since the last call ended, before dialing.
        Recorded as its own event (with the previous call's id) so a delayed ring is explained."""
        if self._ended_at is None:
            return
        wait = REDIAL_SETTLE_SECONDS - (time.monotonic() - self._ended_at)
        if wait > 0:
            self._event("redial_settle_wait", wait_s=round(wait, 3), gap_s=REDIAL_SETTLE_SECONDS)
            log.info("waiting %.1fs before dialing: the last call ended under %gs ago", wait, REDIAL_SETTLE_SECONDS)
            await asyncio.sleep(wait)

    # ------------------------------------------------------------ amplitude diagnostics

    def _start_stats(self) -> None:
        self._stats_t = time.monotonic()
        self._stats_task = asyncio.create_task(self._stats_loop())

    def _stop_stats(self) -> None:
        """Final summary, then stop the reporter (a no-op when it never started)."""
        if self._stats_task is not None:
            self._emit_audio_stats()
            self._stats_task.cancel()
            self._stats_task = None

    async def _stats_loop(self) -> None:
        t0 = time.monotonic()
        while True:
            elapsed = time.monotonic() - t0
            await asyncio.sleep(STATS_FAST_INTERVAL if elapsed < STATS_FAST_SECONDS else STATS_SLOW_INTERVAL)
            self._emit_audio_stats()

    def _emit_audio_stats(self) -> None:
        """audio_in / audio_out events (while audio_stats is on) for the window since the last one,
        then reset the window."""
        now = time.monotonic()
        window = round(now - self._stats_t, 3)
        self._stats_t = now
        if self.audio_stats:
            self._event("audio_in", phase=self._phase, window_s=window, **self._in_stats.fields())
            self._event("audio_out", phase=self._phase, window_s=window, audio_frames=self._out_audio_stats.frames,
                        silence_frames=self._out_silence_frames, **self._out_stats.fields(),
                        **{"audio_" + k: v for k, v in self._out_audio_stats.fields().items() if k != "frames"})
        for st in (self._in_stats, self._out_stats, self._out_audio_stats):
            st.reset()
        self._out_silence_frames = 0

    def _ended_during_greeting(self):
        """Outcome of a call that ended before the greeting finished. A hang_up from this
        side (another session) is HUNG_UP, so nobody re-dials; only the owner ending it
        is its own outcome, not a decline (owner decision 2026-10-12)."""
        if self._end_reason == END_HUNG_UP:
            self._event("hung_up_by_us_during_greeting")
            return HUNG_UP
        self._event("hung_up_during_greeting")
        return HUNG_UP_DURING_GREETING

    async def _audio_failed(self, error: BaseException) -> str:
        """The owner answered but the media connection failed (ntgcalls.TelegramServerError etc.):
        discard the call and report it distinctly from "not answered". The daemon then exits so
        compose's restart policy restarts it with fresh ntgcalls state (see SpeakDaemon)."""
        message = describe(error)
        log.error("call %d: answered but audio failed to connect: %s", self._generation, message)
        self._event("audio_failed", error=message)
        await self._leave()
        self._end_call(END_FAILED)
        return AUDIO_FAILED_PREFIX + message

    def hang_up(self) -> str:
        return self._run(self._hang_up(), HANG_UP_TIMEOUT + 2.0, "hanging up")

    async def _hang_up(self) -> str:
        if self._state == NONE:
            return NO_ACTIVE_CALL
        ring = self._ring_task
        if self._state == RINGING and ring is not None:
            # The ringing play() holds py-tgcalls' per-chat lock, so leave_call would wait on it:
            # cancel the ring; _call's handler leaves the call and ends it.
            ring.cancel()
            await asyncio.wait({ring}, timeout=HANG_UP_TIMEOUT)
            self._end_reason = END_HUNG_UP   # ended on request, not by a failure
            return HUNG_UP
        # End first, then leave: leaving stops the library's side, and the pump's next
        # send_frame would otherwise record the end as a drop before we record the hang-up
        # (seen live 2026-10-09).
        self._end_call(END_HUNG_UP)
        await self._leave()
        return HUNG_UP

    # ------------------------------------------------------------ audio out

    async def _pump(self) -> None:
        from pytgcalls.types import Device

        silence = bytes(FRAME_BYTES)
        start, n = time.monotonic(), 0
        first_sent, lag_events, was_sending = False, 0, False
        while True:
            sending = bool(self._outgoing)
            stream = self._stream
            if sending:
                if not was_sending:
                    self._event("audio_out_start", queued_ms=len(self._outgoing) // FRAME_BYTES * 10)
                    if stream is not None:
                        if stream.gap_frames:   # the producer fell behind: the pump sent this much silence
                            stream.underruns += 1
                            stream.underrun_frames += stream.gap_frames
                            self._event("speak_underrun", gap_ms=stream.gap_frames * 10,
                                        before_segment=stream.resume_segment)
                            stream.gap_frames = 0
                        stream.audio_started = True
                was_sending = True
                frame = bytes(self._outgoing[:FRAME_BYTES]).ljust(FRAME_BYTES, b"\0")
                del self._outgoing[:FRAME_BYTES]
                if not self._outgoing:
                    self._drained.set()
            else:
                if was_sending:
                    self._event("audio_out_end")
                was_sending = False
                frame = silence
                if stream is not None and stream.audio_started:
                    stream.gap_frames += 1
            await self._calls.send_frame(self._user_id, Device.MICROPHONE, frame)
            self._out_stats.add(frame)
            if sending:
                self._out_audio_stats.add(frame)
            else:
                self._out_silence_frames += 1
            if not first_sent:
                first_sent = True
                self._event("first_frame_sent")
            n += 1
            lag = time.monotonic() - (start + n * 0.01)
            if lag > self._lag_max:
                self._lag_max = lag
            if lag > PUMP_MAX_LAG:
                self._lag_stalls += 1
                if lag_events < MAX_LAG_EVENTS:   # an underrun/gap: the outgoing stream stalled this long
                    lag_events += 1
                    self._event("pump_stall", lag_ms=round(lag * 1000), sending_audio=was_sending)
                start, n = time.monotonic(), 0
            await asyncio.sleep(max(0.0, start + n * 0.01 - time.monotonic()))

    def _pump_done(self, task: asyncio.Task) -> None:
        """A pump that stops by itself (not cancelled by _end_call) means the call is over.
        NotInCallError is the library saying the call is gone (the owner hung up, usually
        noticed here before the ChatUpdate arrives, seen live 2026-10-09): a remote end.
        Anything else is a failure."""
        from pytgcalls.exceptions import NotInCallError

        if task.cancelled() or self._state == NONE:
            return
        error = task.exception()
        if isinstance(error, NotInCallError):
            log.info("call ended by the other side (the library dropped the call)")
            self._end_call(END_REMOTE)
            return
        log.error("raw-frame pump stopped (%s); ending the call", describe(error))
        self._event("pump_failed", error=describe(error))
        self._end_call(END_FAILED)
        asyncio.ensure_future(self._leave())

    def play(self, audio: np.ndarray, rate: int) -> None:
        """Queue `audio` on the call and return once it has all been sent. Raises CallEnded."""
        pcm = to_call_pcm(audio, rate)
        self._run(self._play(pcm), len(pcm) / (FRAME_BYTES * 100) + 30.0, "playing audio into the call")

    async def _play(self, pcm: bytes) -> None:
        self._require_pump()
        self._lag_max, self._lag_stalls = 0.0, 0
        self._drained.clear()
        self._outgoing.extend(pcm)
        self._event("playback_queued", audio_ms=len(pcm) // FRAME_BYTES * 10)
        await self._wait_drained()
        if self._ended.is_set():
            raise CallEnded("the Telegram call ended during playback")
        self._event("playback_done", max_pump_lag_ms=round(self._lag_max * 1000), pump_stalls=self._lag_stalls)

    def _require_pump(self) -> None:
        if self._state != CONNECTED or self._pump_task is None or self._pump_task.done():
            raise CallEnded("no connected Telegram call")

    async def _wait_drained(self) -> None:
        """Until the pump has sent everything queued (or the call ended, which also sets it)."""
        try:
            while not self._drained.is_set():
                try:
                    await asyncio.wait_for(self._drained.wait(), timeout=0.5)
                except asyncio.TimeoutError:
                    pass
        except asyncio.CancelledError:
            self._outgoing.clear()   # timed out: the rest must not play after the error
            raise

    # ------------------------------------------------------------ streamed audio out

    def play_stream(self, segments: Iterable[np.ndarray], rate: int) -> float:
        """Play audio that is produced segment by segment (owner decision 2026-10-16,
        docs/DESIGN_STREAMED_SPEECH.md). Each segment is queued on the call the moment
        `segments` yields it, while the earlier ones play; the iterator runs on the calling
        thread (Kokoro synthesis happens in its next()). Returns, once everything has been
        sent, the seconds of audio played (silence from underruns not counted).

        With no connected call this raises CallEnded before `segments` is touched. Any
        failure after that (the producer raised, the call ended, a timeout) clears what is
        still queued, so nothing plays after the error, records an aborted
        speech_stream_done, and propagates. Raises RuntimeError if `segments` is empty."""
        self._run(self._stream_begin(), STREAM_STEP_TIMEOUT, "starting streamed playback")
        samples = 0
        queued_seconds = 0.0
        try:
            it = iter(segments)
            while True:
                t0 = time.monotonic()
                audio = next(it, None)
                if audio is None:
                    break
                produce_s = time.monotonic() - t0
                pcm = to_call_pcm(audio, rate)
                samples += len(audio)
                queued_seconds += len(pcm) / (FRAME_BYTES * 100)
                self._run(self._stream_add(pcm, produce_s), STREAM_STEP_TIMEOUT, "queueing streamed audio")
            if samples == 0:
                raise RuntimeError("streamed playback got no audio")
            self._run(self._stream_finish(), queued_seconds + 30.0, "playing streamed audio into the call")
        except BaseException as e:
            self._run(self._stream_abort(describe(e)), STREAM_STEP_TIMEOUT, "stopping streamed playback")
            raise
        return samples / rate

    async def _stream_begin(self) -> None:
        self._require_pump()
        self._lag_max, self._lag_stalls = 0.0, 0
        self._stream = _Stream()

    async def _stream_add(self, pcm: bytes, produce_s: float) -> None:
        st = self._stream
        if st is None:
            raise RuntimeError("streamed audio queued with no stream open")
        if self._state != CONNECTED or self._pump_task is None or self._pump_task.done():
            raise CallEnded("the Telegram call ended during playback")
        st.segments += 1
        st.audio_bytes += len(pcm)
        if st.first_audio_ms is None:
            st.first_audio_ms = round((time.monotonic() - st.t0) * 1000)
        if not self._outgoing and st.audio_started:
            st.resume_segment = st.segments   # into an empty buffer: speech resumes with this one
        self._drained.clear()
        self._outgoing.extend(pcm)
        self._event("playback_queued", audio_ms=len(pcm) // FRAME_BYTES * 10, segment=st.segments,
                    produce_ms=round(produce_s * 1000))

    async def _stream_finish(self) -> None:
        await self._wait_drained()
        if self._ended.is_set():
            raise CallEnded("the Telegram call ended during playback")
        self._stream_done(aborted=False)

    async def _stream_abort(self, error: str) -> None:
        self._outgoing.clear()   # nothing plays after the error
        self._drained.set()
        self._stream_done(aborted=True, error=error)

    def _stream_done(self, aborted: bool, error: str | None = None, cut: str | None = None) -> None:
        """One speech_stream_done summary per streamed utterance, debug on or off."""
        st = self._stream
        if st is None:   # already summarized: the failure came after a completed finish
            return
        self._stream = None
        fields = {"segments": st.segments, "audio_ms": round(st.audio_bytes / (FRAME_BYTES * 100) * 1000),
                  "first_audio_ms": st.first_audio_ms, "underruns": st.underruns,
                  "underrun_ms": st.underrun_frames * 10, "max_pump_lag_ms": round(self._lag_max * 1000),
                  "pump_stalls": self._lag_stalls, "aborted": aborted}
        if error is not None:
            fields["error"] = error
        if cut is not None:
            fields["cut"] = cut
        self._event("speech_stream_done", **fields)

    # ------------------------------------------------------------ watched audio out (hold on / continue)

    def play_watched(self, segments: Iterable[tuple[object, np.ndarray]], rate: int,
                     classify: Callable[[np.ndarray], Classified], accept: str) -> PlayOutcome:
        """play_stream, while watching the owner's audio for the `accept` command
        (docs/DESIGN_INTERRUPT.md, Release 2). `segments` yields (tag, audio); the tag of
        the segment playing at a cut comes back in the outcome. The watch runs between
        segments (while Kokoro makes the next one it pauses; the frames wait in the queue)
        and every WATCH_POLL_SECONDS while the last ones drain.

        On the command, what is still queued is dropped at once (the cut), a playback_cut
        event records how much had played and the command latency, and the outcome names
        the command. Otherwise everything plays and the outcome's command is None.
        Raises CallEnded on a drop; any failure clears what is queued, records an aborted
        speech_stream_done and propagates, as play_stream."""
        self._run(self._stream_begin(), STREAM_STEP_TIMEOUT, "starting watched playback")
        marks: list[tuple[object, int]] = []   # (tag, cumulative call-PCM bytes at its end)
        total = 0
        queued_seconds = 0.0
        while not self._incoming.empty():   # stale frames from before this speech
            self._incoming.get_nowait()
        self._watching = True
        try:
            watcher = _Watcher(self, classify, accept, "playing")
            it = iter(segments)
            while True:
                t0 = time.monotonic()
                item = next(it, None)
                if item is None:
                    break
                tag, audio = item
                produce_s = time.monotonic() - t0
                pcm = to_call_pcm(audio, rate)
                total += len(pcm)
                marks.append((tag, total))
                queued_seconds += len(pcm) / (FRAME_BYTES * 100)
                self._run(self._stream_add(pcm, produce_s), STREAM_STEP_TIMEOUT, "queueing streamed audio")
                hit = watcher.poll(0)
                if hit is not None:
                    return self._cut(hit, marks, total)
            if total == 0:
                raise RuntimeError("streamed playback got no audio")
            deadline = time.monotonic() + queued_seconds + WATCH_DRAIN_GRACE_SECONDS
            while not self._drained.is_set():
                if self._ended.is_set():
                    raise CallEnded("the Telegram call ended during playback")
                if time.monotonic() > deadline:
                    raise TimeoutError(f"Telegram link: playing watched audio did not finish within "
                                       f"{queued_seconds + WATCH_DRAIN_GRACE_SECONDS:.0f}s")
                hit = watcher.poll(WATCH_POLL_SECONDS)
                if hit is not None:
                    return self._cut(hit, marks, total)
            self._run(self._stream_finish(), STREAM_STEP_TIMEOUT, "finishing watched playback")
            return PlayOutcome(None, total, None, total / (FRAME_BYTES * 100))
        except BaseException as e:
            self._run(self._stream_abort(describe(e)), STREAM_STEP_TIMEOUT, "stopping streamed playback")
            raise
        finally:
            self._watching = False

    def _cut(self, hit: _Hit, marks: list[tuple[object, int]], total: int) -> PlayOutcome:
        remaining = self._run(self._stream_cut(hit.command, hit.utterance_end), STREAM_STEP_TIMEOUT,
                              "cutting playback")
        played = total - remaining
        cut_tag = next((tag for tag, end in marks if end > played), None)
        return PlayOutcome(hit.command, played, cut_tag, played / (FRAME_BYTES * 100))

    async def _stream_cut(self, command: str, utterance_end: float) -> int:
        """On the loop: drop what is queued (the pump sends silence from its next frame)."""
        if self._ended.is_set():
            raise CallEnded("the Telegram call ended during playback")
        remaining = len(self._outgoing)
        self._outgoing.clear()
        self._drained.set()
        st = self._stream
        played_ms = round((st.audio_bytes - remaining) / (FRAME_BYTES * 100) * 1000) if st is not None else None
        self._stream_done(aborted=False, cut=command)
        self._event("playback_cut", command=command, played_ms=played_ms,
                    dropped_ms=round(remaining / (FRAME_BYTES * 100) * 1000),
                    command_latency_ms=round((time.monotonic() - utterance_end) * 1000))
        return remaining

    def watch_for(self, timeout_s: float, classify: Callable[[np.ndarray], Classified], accept: str,
                  client_left: Callable[[], bool]):
        """The hold: silence goes out (the pump idles) while the owner's audio is watched
        for the `accept` command. Returns that command, TIMEOUT after `timeout_s`, or
        CLIENT_LEFT as soon as `client_left()` is true (the request's client went away, so
        nobody waits for the reply). Raises CallEnded if the call ends meanwhile; a classify
        failure propagates. Records hold_start and hold_end (how: the command, expired,
        call_ended, client_left or error)."""
        if self._state != CONNECTED:
            raise CallEnded("no connected Telegram call")
        while not self._incoming.empty():
            self._incoming.get_nowait()
        t0 = time.monotonic()
        self._event("hold_start", max_s=round(timeout_s, 1))
        self._watching = True
        try:
            watcher = _Watcher(self, classify, accept, "held")
            while True:
                if self._ended.is_set():
                    self._event("hold_end", how="call_ended", held_s=round(time.monotonic() - t0, 1))
                    raise CallEnded("the Telegram call ended during the hold")
                left = timeout_s - (time.monotonic() - t0)
                if left <= 0:
                    self._event("hold_end", how="expired", held_s=round(time.monotonic() - t0, 1))
                    return TIMEOUT
                if client_left():
                    self._event("hold_end", how="client_left", held_s=round(time.monotonic() - t0, 1))
                    return CLIENT_LEFT
                try:
                    hit = watcher.poll(min(WATCH_POLL_SECONDS, left))
                except Exception as e:
                    self._event("hold_end", how="error", error=describe(e), held_s=round(time.monotonic() - t0, 1))
                    raise
                if hit is not None:
                    self._event("hold_end", how=hit.command, held_s=round(time.monotonic() - t0, 1),
                                command_latency_ms=round((time.monotonic() - hit.utterance_end) * 1000))
                    return hit.command
        finally:
            self._watching = False

    # ------------------------------------------------------------ audio in

    def record(self, max_seconds: float, silence_seconds: float, start_timeout_seconds: float,
               cue: np.ndarray | None, cue_rate: int):
        """Play the ear-open `cue`, then capture the owner until `silence_seconds` of quiet
        after speech (Silero VAD). Returns 16 kHz float32,
        or TIMEOUT if nobody spoke within `start_timeout_seconds`. Raises CallEnded."""
        if cue is not None and len(cue):
            self.play(cue, cue_rate)
        if self._state != CONNECTED:
            raise CallEnded("no connected Telegram call")
        while not self._incoming.empty():
            self._incoming.get_nowait()
        self._listening = True
        try:
            return self._capture(max_seconds, silence_seconds, start_timeout_seconds)
        finally:
            self._listening = False

    def _capture(self, max_seconds: float, silence_seconds: float, start_timeout_seconds: float):
        feed = _VadFeed(self._vad_factory(silence_seconds))
        blocks: list[np.ndarray] = []
        speaking = False
        t_open = time.monotonic()
        while True:
            if self._ended.is_set():
                raise CallEnded("the Telegram call ended during listen")
            try:
                _t, raw = self._incoming.get(timeout=0.5)
            except queue.Empty:
                raw = None
            for block, event in (feed.blocks(raw) if raw is not None else ()):
                blocks.append(block)
                if event and "start" in event and not speaking:
                    speaking = True
                    blocks = blocks[-int(PRE_ROLL_SECONDS * VAD_RATE / VAD_BLOCK):]
                if event and "end" in event and speaking:
                    return np.concatenate(blocks)
            elapsed = time.monotonic() - t_open
            if not speaking and elapsed > start_timeout_seconds:
                return TIMEOUT
            if elapsed > start_timeout_seconds + max_seconds:
                return np.concatenate(blocks) if blocks else np.zeros(0, dtype=np.float32)
