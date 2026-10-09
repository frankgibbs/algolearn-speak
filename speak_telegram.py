"""The Telegram call itself: one private voice call from the algolearn.ai account
to the owner, driven by raw PCM (docs/DESIGN_TELEGRAM_CALL.md sections 2-4, 8).

`TelegramLink` owns the Telethon client, py-tgcalls and at most one call, on its
own asyncio loop in a background thread. Its public methods are synchronous and
thread-safe (the speak-telegram daemon calls them from its socket threads).
Instantiated only by speak_telegram_daemon.py; the speak servers never import
Telethon or ntgcalls.

Built around what the 2026-10-09 spike verified:
- The call opens on a real audio FILE of at least OPENER_MIN_SECONDS (shorter
  openers gave all-zero audio both ways), waits for it to finish plus 1 s,
  then switches to raw frames.
- A 10 ms pump sends a frame every 10 ms for the whole call (silence when idle)
  and never bursts after a stall (bursts played the next audio sped up).
- py-tgcalls raises CallDeclined / CallBusy / TimedOutAnswer / CallDiscarded
  from play() when the call is not taken (read in pytgcalls
  methods/internal/handle_mtproto_updates.py and connect_call.py).

Every call has a generation number and, once over, an end reason: "remote"
(the owner hung up or the call dropped), "hung_up" (our hang_up), "failed" (an
error tore it down) or "not_answered". Speak servers compare these to decide
whether a call they were using dropped (call back) or was hung up on request.
"""

from __future__ import annotations

import asyncio
import logging
import os
import queue
import subprocess
import tempfile
import threading
import time
import wave
from typing import Callable

import numpy as np
from scipy.signal import resample_poly

log = logging.getLogger("speak.telegram")

CALL_RATE, CALL_CHANNELS = 48_000, 2
FRAME_BYTES = CALL_RATE // 100 * 2 * CALL_CHANNELS   # 10 ms of s16le stereo
VAD_RATE, VAD_BLOCK = 16_000, 512                    # Silero contract (32 ms)
NATIVE_BLOCK = VAD_BLOCK * CALL_RATE // VAD_RATE     # 1536 samples at 48 kHz
OPENER_MIN_SECONDS = 6.0   # 2 s failed, 6 s worked (2026-10-09); shorter greetings are padded with silence
OPENER_TAIL_SECONDS = 1.0  # wait after the opener file before switching to raw frames
RING_TIMEOUT_SECONDS = 60
CONNECT_GRACE_SECONDS = 30  # py-tgcalls waits for the media connection with no timeout of its own
PUMP_MAX_LAG = 0.05        # behind by more than this -> reset the clock instead of bursting
PRE_ROLL_SECONDS = 0.5     # same pre-roll as speak_audio_worker.cmd_record
HANG_UP_TIMEOUT = 8.0      # leave_call; with disconnect, stays inside launchd's stop window
DISCONNECT_TIMEOUT = 5.0

ANSWERED = "answered"
ALREADY_CONNECTED = "already connected"
NOT_ANSWERED = "not answered (declined or no answer)"
HUNG_UP = "hung up"
NO_ACTIVE_CALL = "no active call"
TIMEOUT = object()   # record(): nobody spoke within start_timeout_seconds

NONE, RINGING, CONNECTED = "none", "ringing", "connected"
END_REMOTE, END_HUNG_UP, END_FAILED, END_NOT_ANSWERED = "remote", "hung_up", "failed", "not_answered"


class CallEnded(RuntimeError):
    pass


def to_call_pcm(audio: np.ndarray, rate: int) -> bytes:
    """Mono float PCM at `rate` -> 48 kHz stereo s16le bytes."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    up = resample_poly(audio, CALL_RATE, rate) if rate != CALL_RATE else audio
    s16 = (np.clip(up, -1.0, 1.0) * 32767).astype(np.int16)
    return np.repeat(s16, CALL_CHANNELS).tobytes()


def write_opener(path: str, audio: np.ndarray, rate: int) -> float:
    """The greeting as a 48 kHz mono WAV, padded with silence to OPENER_MIN_SECONDS. Returns its length."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    up = resample_poly(audio, CALL_RATE, rate) if rate != CALL_RATE else audio
    s16 = (np.clip(up, -1.0, 1.0) * 32767).astype(np.int16)
    s16 = np.concatenate([s16, np.zeros(max(0, int(OPENER_MIN_SECONDS * CALL_RATE) - len(s16)), dtype=np.int16)])
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(CALL_RATE)
        w.writeframes(s16.tobytes())
    return len(s16) / CALL_RATE


_silero_model = None


def silero_vad_factory(silence_seconds: float):
    """A Silero VADIterator with speak_audio_worker.cmd_record's settings (model loaded once)."""
    global _silero_model
    import torch
    from silero_vad import VADIterator, load_silero_vad

    if _silero_model is None:
        _silero_model = load_silero_vad()
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
        caffeinate: str = "caffeinate",
    ) -> None:
        self.target = target
        self._client_factory = client_factory
        self._calls_factory = calls_factory
        self._vad_factory = vad_factory
        self._caffeinate_cmd = caffeinate
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, name="telegram-loop", daemon=True)
        self._client = None
        self._calls = None
        self._user_id: int | None = None
        self._state = NONE
        self._generation = 0
        self._end_reason: str | None = None
        self._ended = threading.Event()     # set when the current call ends (read from VAD threads too)
        self._outgoing = bytearray()
        self._drained: asyncio.Event | None = None
        self._pump_task: asyncio.Task | None = None
        self._ring_task: asyncio.Task | None = None
        self._caffeinate: subprocess.Popen | None = None
        self._incoming: queue.Queue[bytes] = queue.Queue()
        self._listening = False

    # ------------------------------------------------------------ lifecycle

    def start(self) -> None:
        self._thread.start()
        self._run(self._start(), 60.0, "connecting to Telegram")
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
                        and update.device & Device.MICROPHONE and self._listening:
                    for f in update.frames:
                        self._incoming.put(f.frame)
            elif isinstance(update, ChatUpdate) and update.chat_id == self._user_id and update.status & (
                    ChatUpdate.Status.DISCARDED_CALL | ChatUpdate.Status.LEFT_CALL | ChatUpdate.Status.BUSY_CALL):
                if self._state == NONE:
                    return   # a late update for a call already torn down; never end the next one
                log.info("call ended by the other side (%s)", update.status)
                self._end_call(END_REMOTE)

        # connect() + is_user_authorized(), not start(): start() would prompt on stdin for a
        # phone number if the session were revoked, which under launchd loops forever.
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
            self._run(self._disconnect(), DISCONNECT_TIMEOUT, "disconnecting from Telegram")
            self._loop.call_soon_threadsafe(self._loop.stop)

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
        self._ended.set()
        self._listening = False
        self._outgoing.clear()
        self._drained.set()
        if self._pump_task is not None:
            self._pump_task.cancel()
            self._pump_task = None
        if self._caffeinate is not None:
            self._caffeinate.terminate()
            self._caffeinate.wait(timeout=5.0)
            self._caffeinate = None

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
        budget = RING_TIMEOUT_SECONDS + CONNECT_GRACE_SECONDS + max(OPENER_MIN_SECONDS, len(greeting) / rate) + 15.0
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
            self._generation += 1
            self._end_reason = None
            self._ended.clear()
            self._state = RINGING
            self._ring_task = asyncio.current_task()
            try:
                await self._calls.play(self._user_id, MediaStream(opener, params), CallConfig(timeout=RING_TIMEOUT_SECONDS))
            except (CallDeclined, CallBusy, TimedOutAnswer, CallDiscarded) as e:
                log.info("call not answered: %s", type(e).__name__)
                self._end_call(END_NOT_ANSWERED)
                return NOT_ANSWERED
            except BaseException:   # includes cancellation (timeout, hang_up, daemon stop): never leave it ringing
                await self._leave()
                self._end_call(END_FAILED)
                raise
            finally:
                self._ring_task = None
            # Answered. Anything that fails from here tears the call down: never a live,
            # half-built call with no pump.
            try:
                self._state = CONNECTED
                self._caffeinate = subprocess.Popen([self._caffeinate_cmd, "-i", "-w", str(os.getpid())],
                                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                await self._calls.record(self._user_id, RecordStream(audio=True, audio_parameters=params))
                await asyncio.sleep(opener_seconds + OPENER_TAIL_SECONDS)
                if self._ended.is_set():
                    # Hung up during the greeting: treat it as a decline (the session then waits an hour).
                    return NOT_ANSWERED
                await self._calls.play(self._user_id, MediaStream(ExternalMedia.AUDIO, params))
                if self._ended.is_set():
                    # Ended while switching; py-tgcalls may have re-dialed, so make sure nothing is left up.
                    await self._leave()
                    return NOT_ANSWERED
                self._pump_task = asyncio.create_task(self._pump())
                self._pump_task.add_done_callback(self._pump_done)
            except BaseException:
                await self._leave()
                self._end_call(END_FAILED)
                raise
        finally:
            os.unlink(opener)
        log.info("call %d connected; raw-frame pump running", self._generation)
        return ANSWERED

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
        while True:
            if self._outgoing:
                frame = bytes(self._outgoing[:FRAME_BYTES]).ljust(FRAME_BYTES, b"\0")
                del self._outgoing[:FRAME_BYTES]
                if not self._outgoing:
                    self._drained.set()
            else:
                frame = silence
            await self._calls.send_frame(self._user_id, Device.MICROPHONE, frame)
            n += 1
            if time.monotonic() - (start + n * 0.01) > PUMP_MAX_LAG:
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
        log.error("raw-frame pump stopped (%s: %s); ending the call", type(error).__name__, error)
        self._end_call(END_FAILED)
        asyncio.ensure_future(self._leave())

    def play(self, audio: np.ndarray, rate: int) -> None:
        """Queue `audio` on the call and return once it has all been sent. Raises CallEnded."""
        pcm = to_call_pcm(audio, rate)
        self._run(self._play(pcm), len(pcm) / (FRAME_BYTES * 100) + 30.0, "playing audio into the call")

    async def _play(self, pcm: bytes) -> None:
        if self._state != CONNECTED or self._pump_task is None or self._pump_task.done():
            raise CallEnded("no connected Telegram call")
        self._drained.clear()
        self._outgoing.extend(pcm)
        try:
            while not self._drained.is_set():
                try:
                    await asyncio.wait_for(self._drained.wait(), timeout=0.5)
                except asyncio.TimeoutError:
                    pass
        except asyncio.CancelledError:
            self._outgoing.clear()   # timed out: the rest must not play after the error
            raise
        if self._ended.is_set():
            raise CallEnded("the Telegram call ended during playback")

    # ------------------------------------------------------------ audio in

    def record(self, max_seconds: float, silence_seconds: float, start_timeout_seconds: float,
               cue: np.ndarray | None, cue_rate: int):
        """Play the ear-open `cue`, then capture the owner until `silence_seconds` of quiet
        after speech (Silero, speak_audio_worker.cmd_record's rules). Returns 16 kHz float32,
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
        vad = self._vad_factory(silence_seconds)
        native = np.zeros(0, dtype=np.float32)
        blocks: list[np.ndarray] = []
        speaking = False
        t_open = time.monotonic()
        while True:
            if self._ended.is_set():
                raise CallEnded("the Telegram call ended during listen")
            try:
                raw = self._incoming.get(timeout=0.5)
            except queue.Empty:
                raw = None
            if raw is not None:
                s16 = np.frombuffer(raw, dtype=np.int16).reshape(-1, CALL_CHANNELS).mean(axis=1)
                native = np.concatenate([native, (s16 / 32768.0).astype(np.float32)])
            while len(native) >= NATIVE_BLOCK:
                block = resample_poly(native[:NATIVE_BLOCK], VAD_RATE, CALL_RATE).astype(np.float32)
                native = native[NATIVE_BLOCK:]
                blocks.append(block)
                event = vad(block)
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
