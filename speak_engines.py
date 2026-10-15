"""Kokoro (TTS) and Whisper (STT) on CUDA, both at full precision, for the speak
daemon (docs/DESIGN_REMOTE_DAEMON.md section 6).

- TTS: kokoro's reference PyTorch `KPipeline` (hexgrad/Kokoro-82M), fp32 on the GPU.
- STT: faster-whisper (CTranslate2) large-v3-turbo, compute_type float32 on the GPU,
  greedy decoding, no built-in VAD (Silero has already gated the audio).

Owner decision 2026-10-15: both models run at full precision. No CPU fallback:
`device="cuda"` raises when no GPU is visible, and a model that loads in any other
dtype fails the load.
"""

from __future__ import annotations

import logging
import time

import numpy as np

log = logging.getLogger("speak.engines")

TTS_RATE = 24_000              # Kokoro output rate
STT_RATE = 16_000              # Whisper input rate
SENTENCE_SPLIT = r"(?<=[.!?])\s+"
WHISPER_COMPUTE_TYPE = "float32"


class Engines:
    """Loaded once, synchronously, at daemon start (before Telegram connects and before
    the port is bound), so a reachable daemon always has its models."""

    def __init__(self, kokoro_model: str, voice: str, speed: float, whisper_model: str, language: str) -> None:
        self.kokoro_model = kokoro_model
        self.voice = voice
        self.speed = speed
        self.whisper_model = whisper_model
        self.language = language
        self._kokoro = None
        self._whisper = None

    def load(self) -> None:
        import torch
        from faster_whisper import WhisperModel
        from kokoro import KPipeline

        t0 = time.time()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available to torch; the speak daemon needs the GPU")
        self._kokoro = KPipeline(lang_code="a", repo_id=self.kokoro_model, device="cuda")
        dtypes = {p.dtype for p in self._kokoro.model.parameters()}
        if dtypes != {torch.float32}:
            raise RuntimeError(f"Kokoro loaded with parameter dtypes {dtypes}; fp32 is required")
        self._whisper = WhisperModel(self.whisper_model, device="cuda", device_index=0,
                                     compute_type=WHISPER_COMPUTE_TYPE)
        # Warm both so the first speak/listen does not pay CUDA kernel selection or the voice load.
        self.synthesize("Ready.")
        self.transcribe(np.zeros(STT_RATE, dtype=np.float32))
        log.info("engines ready in %.1fs (kokoro=%s voice=%s, whisper=%s %s on %s)", time.time() - t0,
                 self.kokoro_model, self.voice, self.whisper_model, WHISPER_COMPUTE_TYPE,
                 torch.cuda.get_device_name(0))

    def synthesize(self, text: str) -> np.ndarray:
        """All of `text` as mono float32 at TTS_RATE. Synthesis finishes before any of it
        is played, so inference never overlaps outgoing speech (design section 4.6)."""
        chunks = [result.audio.detach().cpu().numpy().astype(np.float32).reshape(-1)
                  for result in self._kokoro(text, voice=self.voice, speed=self.speed, split_pattern=SENTENCE_SPLIT)]
        if not chunks:
            raise RuntimeError(f"Kokoro produced no audio for {text!r}")
        return np.concatenate(chunks)

    def transcribe(self, audio: np.ndarray) -> str:
        """16 kHz mono float32 -> text."""
        segments, _info = self._whisper.transcribe(audio, language=self.language, beam_size=1, vad_filter=False)
        return "".join(segment.text for segment in segments).strip()
