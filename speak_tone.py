"""Tone synthesis for algolearn-speak's cues (ear open, ear closed, the
"processing" chime), played into the Telegram call by speak_server.py.

No hardware access here: this module only does numpy math.
"""

from __future__ import annotations

import numpy as np

# Kokoro's output rate. Cues are synthesized at this rate too so a cue and a
# TTS chunk share one playback path into the call.
TONE_RATE = 24_000


def tone(freq_hz: float, seconds: float, volume: float = 0.2) -> np.ndarray:
    """A sine tone with a 10 ms fade in/out, as mono float32 PCM at TONE_RATE."""
    t = np.arange(int(TONE_RATE * seconds)) / TONE_RATE
    env = np.minimum(1.0, np.minimum(t, seconds - t) / 0.01)  # 10 ms fade in/out
    return (volume * env * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def cue_pcm(freq_hz: float, seconds: float = 0.12, volume: float = 0.2, lead_silence: float = 0.0) -> np.ndarray:
    """A tone optionally preceded by silence, as one PCM buffer ready to play."""
    audio = tone(freq_hz, seconds, volume)
    if lead_silence:
        audio = np.concatenate([np.zeros(int(TONE_RATE * lead_silence), dtype=np.float32), audio])
    return audio
