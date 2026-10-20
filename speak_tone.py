"""Tone synthesis for algolearn-speak's cues (the ear-open beep, the rising
"captured" chime, the no-speech timeout beep and the session-switch double
beep), played into the Telegram call by the speak daemon
(speak_telegram_daemon.py). Owner decisions 2026-10-16: the ear-closed beep
and the spoken ack are gone; the rising chime stays, played after Whisper has
the transcript. Owner decision 2026-10-20: a switch of session on the call is
announced with a cue and the new session's project name.

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


def chime_pcm() -> np.ndarray:
    """The rising two-note chime (660 Hz then 990 Hz): the owner's words were captured.
    Distinct from the single-tone ear-open and timeout beeps."""
    return np.concatenate([tone(660.0, 0.10), tone(990.0, 0.14)])


def switch_pcm() -> np.ndarray:
    """The session-switch cue: two short 523 Hz beeps (80 ms each, 60 ms apart), then 150 ms
    of silence before the project name is spoken. Distinct from the single-tone ear-open
    (880 Hz) and timeout (330 Hz) beeps and from the rising chime."""
    beep = tone(523.0, 0.08)
    gap = np.zeros(int(TONE_RATE * 0.06), dtype=np.float32)
    tail = np.zeros(int(TONE_RATE * 0.15), dtype=np.float32)
    return np.concatenate([beep, gap, beep, tail])
