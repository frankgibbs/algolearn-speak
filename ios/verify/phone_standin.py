"""A protocol-faithful stand-in for the iOS app (docs/DESIGN_PHONE_AUDIO.md section 3).

Used only because the simulator app could not start its audio engine on this
Mac (CoreAudio input stalled machine-wide, see VERIFY.md). It speaks the exact
wire protocol: hello -> ready, state, play_start / PCM16 / play_end -> played
(sent after the segment's real duration, like a player draining its queue),
mic_start -> 512-sample PCM16 blocks at 16 kHz paced in real time -> mic_stop,
ping/pong.

  python phone_standin.py --mic-wav speech16k.wav [--die-after-mic-start 1.0] [--host 127.0.0.1]

The mic feed is 0.5 s of silence, then the WAV, then silence until mic_stop.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
import wave

import numpy as np
from websockets.asyncio.client import connect

T0 = time.time()


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')}.{int((time.time() % 1) * 1000):03d} standin {msg}", flush=True)


def load_wav16(path: str) -> np.ndarray:
    with wave.open(path, "rb") as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2, "need 16 kHz mono PCM16"
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8772)
    ap.add_argument("--mic-wav", required=True)
    ap.add_argument("--die-after-mic-start", type=float, default=None, help="close the socket N s after mic_start (mid-call disconnect test)")
    ap.add_argument("--die-after-play-start", type=float, default=None, help="close the socket N s after the first play_start")
    args = ap.parse_args()
    speech = load_wav16(args.mic_wav)

    async with connect(f"ws://{args.host}:{args.port}", max_size=2**20) as ws:
        await ws.send(json.dumps({"type": "hello", "app": "algolearn-speak-ios", "version": "1", "mic_rate": 16000, "spk_rate": 24000}))
        log(f"ready: {await ws.recv()}")
        seg_bytes = 0
        seg_t0 = 0.0
        mic_task: asyncio.Task | None = None
        play_count = 0

        async def mic_feed() -> None:
            # 0.5 s silence, the speech, then silence until cancelled -- paced at 32 ms per block.
            silence = np.zeros(512, dtype="<i2")
            blocks = [silence] * 16 + [speech[i:i + 512] for i in range(0, len(speech) - 511, 512)]
            n = 0
            t_start = time.monotonic()
            while True:
                blk = blocks[n] if n < len(blocks) else silence
                await ws.send(blk.tobytes())
                n += 1
                if n == 17:
                    log("mic: speech starts")
                if n == len(blocks):
                    log("mic: speech done")
                await asyncio.sleep(max(0.0, t_start + n * 0.032 - time.monotonic()))

        async def played_later(seg_id: int, seconds: float) -> None:
            await asyncio.sleep(seconds)
            await ws.send(json.dumps({"type": "played", "id": seg_id}))
            log(f"played {seg_id} sent ({seconds:.2f}s segment)")

        async for message in ws:
            if isinstance(message, bytes):
                seg_bytes += len(message)
                continue
            msg = json.loads(message)
            kind = msg["type"]
            if kind == "ping":
                await ws.send(json.dumps({"type": "pong"}))
            elif kind == "state":
                log(f"state: {msg['value']}")
            elif kind == "play_start":
                seg_bytes = 0
                seg_t0 = time.time()
                play_count += 1
                log(f"play_start #{play_count} rate={msg['rate']}")
                if args.die_after_play_start is not None and play_count == 1:
                    async def die():
                        await asyncio.sleep(args.die_after_play_start)
                        log("closing socket mid-speak (test)")
                        await ws.close()
                    asyncio.create_task(die())
            elif kind == "play_end":
                seconds = seg_bytes / 2 / 24000
                log(f"play_end id={msg['id']} {seg_bytes} bytes = {seconds:.2f}s audio, received in {time.time() - seg_t0:.3f}s")
                asyncio.create_task(played_later(msg["id"], seconds))
            elif kind == "mic_start":
                log("mic_start")
                mic_task = asyncio.create_task(mic_feed())
                if args.die_after_mic_start is not None:
                    async def die2():
                        await asyncio.sleep(args.die_after_mic_start)
                        log("closing socket mid-listen (test)")
                        await ws.close()
                    asyncio.create_task(die2())
            elif kind == "mic_stop":
                log("mic_stop")
                if mic_task:
                    mic_task.cancel()
                    mic_task = None
            else:
                log(f"other: {msg}")
    log("connection closed")


if __name__ == "__main__":
    asyncio.run(main())
