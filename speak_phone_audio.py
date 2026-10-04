"""The phone as the speak server's microphone and speaker (docs/DESIGN_PHONE_AUDIO.md).

`PhoneAudioServer` is a WebSocket server on a background thread with its own
asyncio loop. At most one phone is connected. The server's worker threads
(the ones that run speak/listen/converse) call its synchronous methods:
`connected()`, `play()`, `play_segments()`, `record()`, `set_state()`.

Protocol (design section 3): text frames are JSON control messages; binary
frames are raw PCM16 little-endian mono -- phone->server microphone audio at
16 kHz, server->phone speaker audio at 24 kHz.

No fallbacks: a hello with other rates is refused, a disconnect mid-call
raises RuntimeError, a `played` that does not arrive within the segment's
duration + PLAYED_GRACE_SECONDS raises, a phone whose mic never delivers a
block raises. This module never imports sounddevice (no PortAudio).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import queue
import threading
import time
from typing import Callable

import numpy as np
from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed

from speak_audio_worker import VAD_SPEECH_PAD_MS, VAD_THRESHOLD

log = logging.getLogger("speak.phone")

__version__ = "0.1.0"  # reported in `ready`; keep in step with pyproject.toml

MIC_RATE = 16_000          # phone -> server (Whisper/Silero contract)
SPK_RATE = 24_000          # server -> phone (Kokoro rate)
VAD_FRAME = 512            # Silero frame at 16 kHz (32 ms)
PLAYED_GRACE_SECONDS = 10.0
HELLO_TIMEOUT_SECONDS = 10.0
CONTROL_TIMEOUT_SECONDS = 5.0   # bound on any control-message send
ALIVE_PROBE_SECONDS = 2.0       # how long a stale slot holder gets to answer a ping
CLOSE_TIMEOUT_SECONDS = 2.0
PING_INTERVAL_SECONDS = 5.0
MAX_MISSED_PONGS = 2
MIC_FIRST_BLOCK_TIMEOUT_SECONDS = 5.0
SEND_CHUNK_BYTES = 32_768
STATES = ("idle", "speaking", "listening", "processing")

# Close codes. 1013 = "try again later", 1008 = policy violation, 1002 = protocol error.
CLOSE_BUSY = 1013
CLOSE_POLICY = 1008
CLOSE_PROTOCOL = 1002

TIMEOUT = object()  # record() result when nobody spoke within start_timeout_seconds


class _Disconnected(Exception):
    """Internal: the phone's connection ended. Re-raised as RuntimeError naming the operation."""


class _ProtocolError(Exception):
    pass


def _default_vad_factory() -> Callable[[float], Callable[[np.ndarray], dict | None]]:
    """Load Silero once; the returned factory builds a fresh VADIterator per
    record() with exactly the worker's parameters (cmd_record)."""
    import torch
    from silero_vad import VADIterator, load_silero_vad

    model = load_silero_vad()

    def factory(silence_seconds: float) -> Callable[[np.ndarray], dict | None]:
        it = VADIterator(
            model,
            threshold=VAD_THRESHOLD,
            sampling_rate=MIC_RATE,
            min_silence_duration_ms=int(silence_seconds * 1000),
            speech_pad_ms=VAD_SPEECH_PAD_MS,
        )
        return lambda frame: it(torch.from_numpy(frame.copy()), return_seconds=False)

    return factory


def pcm16_from_float(audio: np.ndarray) -> bytes:
    return (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def float_from_pcm16(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


class PhoneAudioServer:
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8772,
        vad_factory: Callable[[float], Callable[[np.ndarray], dict | None]] | None = None,
        played_grace_seconds: float = PLAYED_GRACE_SECONDS,
        mic_first_block_timeout_seconds: float = MIC_FIRST_BLOCK_TIMEOUT_SECONDS,
        ping_interval_seconds: float = PING_INTERVAL_SECONDS,
        control_timeout_seconds: float = CONTROL_TIMEOUT_SECONDS,
    ) -> None:
        self.host = host
        self.port = port
        self._vad_factory = vad_factory
        self._played_grace = played_grace_seconds
        self._mic_first_block_timeout = mic_first_block_timeout_seconds
        self._ping_interval = ping_interval_seconds
        self._control_timeout = control_timeout_seconds

        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._stop: asyncio.Event | None = None
        self._started = threading.Event()
        self._start_error: BaseException | None = None

        self._state_lock = threading.Lock()
        self._ws: ServerConnection | None = None   # claimed at accept
        self._ready = False                         # True after a valid hello
        self._state = "idle"
        self._next_id = 0
        self._pending: dict[int, concurrent.futures.Future] = {}
        self._mic_q: queue.Queue = queue.Queue()
        self._mic_active = False
        self._missed = 0

    # ------------------------------------------------------------ lifecycle

    def start(self) -> None:
        """Start the server thread; returns once the socket is bound (raises if it could not be)."""
        if self._vad_factory is None:
            self._vad_factory = _default_vad_factory()
        self._thread = threading.Thread(target=self._thread_main, name="phone-audio", daemon=True)
        self._thread.start()
        self._started.wait()
        if self._start_error is not None:
            raise RuntimeError(
                f"phone audio server failed to bind {self.host}:{self.port}: {self._start_error!r}; "
                f"set SPEAK_PHONE_PORT to a free port (and enter that port in the phone app)"
            ) from self._start_error

    def stop(self) -> None:
        if self._loop is None or self._stop is None:
            return
        self._loop.call_soon_threadsafe(self._stop.set)
        assert self._thread is not None
        self._thread.join(timeout=10.0)

    def _thread_main(self) -> None:
        asyncio.run(self._serve())

    async def _serve(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._stop = asyncio.Event()
        try:
            server = await serve(self._handler, self.host, self.port, ping_interval=None, max_size=2**20, close_timeout=CLOSE_TIMEOUT_SECONDS)
        except BaseException as e:  # reported to start()
            self._start_error = e
            self._started.set()
            return
        self.port = server.sockets[0].getsockname()[1]
        log.info("phone audio server listening on %s:%d", self.host, self.port)
        self._started.set()
        async with server:
            await self._stop.wait()

    # ------------------------------------------------------------ connection handler

    async def _handler(self, ws: ServerConnection) -> None:
        if not await self._claim(ws):
            await ws.close(CLOSE_BUSY, "busy")
            return
        ping_task: asyncio.Task | None = None
        try:
            if not await self._hello(ws):
                return
            with self._state_lock:
                self._ready = True
                state = self._state
            await ws.send(json.dumps({"type": "state", "value": state}))
            log.info("phone connected")
            self._missed = 0
            ping_task = asyncio.create_task(self._pinger(ws))
            async for message in ws:
                await self._on_message(ws, message)
        except _ProtocolError as e:
            log.warning("phone protocol error: %s", e)
            await ws.close(CLOSE_PROTOCOL, str(e)[:100])
        except ConnectionClosed:
            pass
        finally:
            if ping_task is not None:
                ping_task.cancel()
            self._on_gone(ws)

    async def _claim(self, ws: ServerConnection) -> bool:
        """Take the single slot. A held slot may belong to a phone that dropped
        silently (WiFi lost, no FIN) and would otherwise refuse that same phone's
        reconnect for ~25 s of ping timeouts, so the holder is pinged first: a
        live one gets this connection refused `busy`; an unresponsive one is
        aborted and replaced."""
        with self._state_lock:
            old = self._ws
            if old is None:
                self._ws = ws
                return True
        if await self._alive(old):
            return False
        log.warning("replacing unresponsive phone connection")
        self._abort(old)
        self._on_gone(old)
        with self._state_lock:
            if self._ws is None:
                self._ws = ws
                return True
        return False

    async def _alive(self, ws: ServerConnection) -> bool:
        try:
            pong = await asyncio.wait_for(ws.ping(), ALIVE_PROBE_SECONDS)
            await asyncio.wait_for(pong, ALIVE_PROBE_SECONDS)
        except (asyncio.TimeoutError, ConnectionClosed):
            return False
        return True

    @staticmethod
    def _abort(ws: ServerConnection) -> None:
        """Drop the TCP connection without a close handshake (the link is dead)."""
        ws.transport.abort()

    async def _hello(self, ws: ServerConnection) -> bool:
        """True if accepted (ready sent); False if refused (connection closed)."""
        try:
            raw = await asyncio.wait_for(ws.recv(), HELLO_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            raise _ProtocolError("no hello") from None
        if isinstance(raw, bytes):
            raise _ProtocolError("first message must be a hello")
        hello = self._parse(raw)
        if hello.get("type") != "hello":
            raise _ProtocolError("first message must be a hello")
        if hello.get("mic_rate") != MIC_RATE or hello.get("spk_rate") != SPK_RATE:
            await ws.close(CLOSE_POLICY, "bad-rates")
            return False
        await ws.send(json.dumps({"type": "ready", "server": "algolearn-speak", "version": __version__}))
        return True

    @staticmethod
    def _parse(raw: str) -> dict:
        try:
            msg = json.loads(raw)
        except ValueError as e:
            raise _ProtocolError(f"bad JSON: {e}") from None
        if not isinstance(msg, dict) or "type" not in msg:
            raise _ProtocolError("control message must be an object with a type")
        return msg

    async def _on_message(self, ws: ServerConnection, message: str | bytes) -> None:
        if isinstance(message, bytes):
            if self._mic_active:
                self._mic_q.put(message)
            # else: a block already in flight when mic_stop was sent -- expected, dropped.
            return
        msg = self._parse(message)
        kind = msg["type"]
        if kind == "pong":
            self._missed = 0
        elif kind == "ping":
            await ws.send(json.dumps({"type": "pong"}))
        elif kind == "played":
            with self._state_lock:
                fut = self._pending.pop(msg.get("id"), None)
            if fut is None:
                raise _ProtocolError(f"played for unknown segment {msg.get('id')!r}")
            fut.set_result(None)
        else:
            raise _ProtocolError(f"unknown message type {kind!r}")

    async def _pinger(self, ws: ServerConnection) -> None:
        while True:
            await asyncio.sleep(self._ping_interval)
            if self._missed >= MAX_MISSED_PONGS:
                log.warning("phone missed %d pongs; disconnecting", self._missed)
                try:
                    await asyncio.wait_for(ws.close(CLOSE_POLICY, "ping timeout"), self._control_timeout)
                except asyncio.TimeoutError:
                    self._abort(ws)
                return
            self._missed += 1
            try:
                await asyncio.wait_for(ws.send(json.dumps({"type": "ping"})), self._control_timeout)
            except asyncio.TimeoutError:
                log.warning("ping send blocked for %.0fs; aborting phone connection", self._control_timeout)
                self._abort(ws)
                return

    def _on_gone(self, ws: ServerConnection) -> None:
        with self._state_lock:
            if self._ws is not ws:
                return  # a refused (busy) connection never owned the slot
            self._ws = None
            self._ready = False
            pending, self._pending = self._pending, {}
        log.info("phone disconnected")
        for fut in pending.values():
            fut.set_exception(_Disconnected())
        self._mic_q.put(_Disconnected())

    # ------------------------------------------------------------ sync API (worker threads)

    def connected(self) -> bool:
        with self._state_lock:
            return self._ready

    def _require(self, op: str) -> ServerConnection:
        with self._state_lock:
            ws = self._ws if self._ready else None
        if ws is None:
            raise RuntimeError(f"phone is not connected ({op})")
        return ws

    def _run(self, coro, op: str, ws: ServerConnection, timeout: float):
        """Run a send coroutine on the loop and wait at most `timeout`. A send
        that does not finish means the link is dead with a full buffer: abort
        the connection (which also unblocks the pinger) and raise."""
        assert self._loop is not None
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return fut.result(timeout=timeout)
        except ConnectionClosed:
            raise RuntimeError(f"phone disconnected during {op}") from None
        except concurrent.futures.TimeoutError:
            fut.cancel()
            self._loop.call_soon_threadsafe(self._abort, ws)
            raise RuntimeError(f"phone unresponsive during {op}") from None

    def set_state(self, value: str) -> None:
        """Record the state and tell the phone. This is a status display, not
        audio, so it is deliberately non-fatal: with no phone, or a phone that
        is already gone or too slow, the value is kept (sent on the next hello)
        and the problem is logged -- the call that matters (play/record) raises
        for that disconnect itself, and a failing status send must never mask
        that or break call cleanup."""
        if value not in STATES:
            raise ValueError(f"unknown state {value!r}")
        with self._state_lock:
            self._state = value
            ws = self._ws if self._ready else None
        if ws is not None:
            try:
                self._run(ws.send(json.dumps({"type": "state", "value": value})), "state update", ws, self._control_timeout)
            except RuntimeError as e:
                log.warning("could not send state %r to phone: %s", value, e)

    def play(self, pcm_f32: np.ndarray, rate: int) -> None:
        self.play_segments([pcm_f32], rate)

    def play_segments(self, segments: list[np.ndarray], rate: int, state: str = "speaking") -> None:
        """Send each float32 mono segment (one cue or one TTS sentence) as
        play_start / PCM16 chunks / play_end, then block until the phone has
        reported `played` for every one. Segment i must be played within the
        cumulative duration up to i plus the grace, measured from the send."""
        if rate != SPK_RATE:
            raise ValueError(f"phone speaker rate is {SPK_RATE}, got {rate}")
        ws = self._require("speak")
        self.set_state(state)
        t0 = time.monotonic()
        waits: list[tuple[int, concurrent.futures.Future, float]] = []
        cumulative = 0.0
        for seg in segments:
            seg = np.asarray(seg, dtype=np.float32).reshape(-1)
            with self._state_lock:
                seg_id = self._next_id
                self._next_id += 1
                fut: concurrent.futures.Future = concurrent.futures.Future()
                self._pending[seg_id] = fut
            seg_seconds = len(seg) / rate
            self._run(
                self._send_segment(ws, seg_id, pcm16_from_float(seg), rate), "speak", ws,
                self._control_timeout + seg_seconds + self._played_grace,
            )
            cumulative += seg_seconds
            waits.append((seg_id, fut, t0 + cumulative + self._played_grace))
        for seg_id, fut, deadline in waits:
            try:
                fut.result(timeout=max(0.0, deadline - time.monotonic()))
            except _Disconnected:
                raise RuntimeError("phone disconnected during speak") from None
            except concurrent.futures.TimeoutError:
                with self._state_lock:
                    self._pending.pop(seg_id, None)
                raise RuntimeError(
                    f"phone did not report played for segment {seg_id} within its duration + {self._played_grace:.0f}s"
                ) from None

    @staticmethod
    async def _send_segment(ws: ServerConnection, seg_id: int, data: bytes, rate: int) -> None:
        await ws.send(json.dumps({"type": "play_start", "rate": rate}))
        for i in range(0, len(data), SEND_CHUNK_BYTES):
            await ws.send(data[i:i + SEND_CHUNK_BYTES])
        await ws.send(json.dumps({"type": "play_end", "id": seg_id}))

    def record(
        self, max_seconds: float, silence_seconds: float, start_timeout_seconds: float, cue: np.ndarray | None = None,
    ) -> np.ndarray | object:
        """The phone counterpart of speak_audio_worker.cmd_record. Plays `cue`
        (float32 at SPK_RATE) and waits for `played`, sends mic_start, runs VAD
        on the incoming 16 kHz blocks with the worker's parameters and 0.5 s
        pre-roll, sends mic_stop, and returns the 16 kHz float32 capture --
        or TIMEOUT if nobody spoke within start_timeout_seconds of mic_start.
        Hard cap: start_timeout_seconds + max_seconds."""
        ws = self._require("listen")
        assert self._vad_factory is not None
        vad = self._vad_factory(silence_seconds)
        cap_seconds = start_timeout_seconds + max_seconds
        if cue is not None:
            # badge says "listening" from the cue on (no speaking->listening flicker)
            self.play_segments([cue], SPK_RATE, state="listening")
        self.set_state("listening")
        while not self._mic_q.empty():
            self._mic_q.get_nowait()
        self._mic_active = True
        self._run(ws.send(json.dumps({"type": "mic_start"})), "listen", ws, self._control_timeout)
        t_open = time.monotonic()
        try:
            return self._vad_loop(vad, t_open, cap_seconds, start_timeout_seconds)
        finally:
            self._mic_active = False
            if self.connected():
                try:
                    self._run(ws.send(json.dumps({"type": "mic_stop"})), "listen", ws, self._control_timeout)
                except RuntimeError as e:
                    # Not raised: this runs in a finally and must not mask the
                    # loop's own result or error. The phone is stopping its mic
                    # on its own disconnect/abort; logged so it is never silent.
                    log.warning("could not send mic_stop: %s", e)

    def _vad_loop(self, vad, t_open: float, cap_seconds: float, start_timeout_seconds: float):
        frames: list[np.ndarray] = []
        buf = b""
        speaking = False
        got_block = False
        frame_bytes = VAD_FRAME * 2
        while True:
            elapsed = time.monotonic() - t_open
            if not got_block and elapsed > self._mic_first_block_timeout:
                raise RuntimeError(f"phone sent no microphone audio within {self._mic_first_block_timeout:.0f}s of mic_start")
            if not speaking and elapsed > start_timeout_seconds:
                return TIMEOUT
            if elapsed > cap_seconds:
                break
            try:
                item = self._mic_q.get(timeout=0.05)
            except queue.Empty:
                continue
            if isinstance(item, _Disconnected):
                raise RuntimeError("phone disconnected during listen")
            got_block = True
            buf += item
            while len(buf) >= frame_bytes:
                frame = float_from_pcm16(buf[:frame_bytes])
                buf = buf[frame_bytes:]
                frames.append(frame)
                event = vad(frame)
                if event and "start" in event and not speaking:
                    speaking = True
                    frames = frames[-int(0.5 * MIC_RATE / VAD_FRAME):]  # ~0.5 s pre-roll
                if event and "end" in event and speaking:
                    return np.concatenate(frames)
        return np.concatenate(frames)
