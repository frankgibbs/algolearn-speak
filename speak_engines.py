"""Kokoro (TTS) and Whisper (STT) on CUDA, both at full precision, for the speak
daemon (docs/DESIGN_REMOTE_DAEMON.md section 6).

- TTS: kokoro's reference PyTorch `KPipeline` (hexgrad/Kokoro-82M), fp32 on the GPU.
- STT: faster-whisper (CTranslate2) large-v3-turbo, compute_type float32 on the GPU,
  greedy decoding, no built-in VAD (Silero has already gated the audio).

Owner decision 2026-10-15: both models run at full precision. No CPU fallback:
`device="cuda"` raises when no GPU is visible, and a model that loads in any other
dtype fails the load.

Voices (docs/DESIGN_PERSONAS.md, owner decisions 2026-10-22): the English voice packs of
the Kokoro repo, exactly ENGLISH_VOICES, are loaded on the CPU at start; each synthesis takes
a Voice (a pack, one voice or a weighted blend, and a speed) resolved from the speaking
project's persona. The British voices (bf_, bm_) speak with American pronunciation: the one
KPipeline is lang_code "a" (owner decision 2026-10-22; a British path may come later).
"""

from __future__ import annotations

import collections
import logging
import os
import re
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

log = logging.getLogger("speak.engines")

TTS_RATE = 24_000              # Kokoro output rate
STT_RATE = 16_000              # Whisper input rate
SENTENCE_SPLIT = r"(?<=[.!?])\s+"
WHISPER_COMPUTE_TYPE = "float32"
WHISPER_DEVICE = "cuda"        # also reported by the daemon's GET /health (port 9000)
BLEND_CACHE_SIZE = 64          # voice blends kept (docs/DESIGN_PERSONAS.md 4.5)
# The English voices of hexgrad/Kokoro-82M (ids starting af_, am_, bf_, bm_; owner decision 14),
# the full set the daemon must load (owner decision 2026-10-22): load_voices fetches exactly
# these and refuses to start, naming the missing ones, when the snapshot (from the Hub, or the
# HF cache when the Hub is unreachable) lacks any. Listed from the repo 2026-10-22: 28 of its
# 54 voices.
ENGLISH_VOICES = (
    "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica", "af_kore", "af_nicole", "af_nova",
    "af_river", "af_sarah", "af_sky",
    "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael", "am_onyx", "am_puck",
    "am_santa",
    "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
    "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
)


def split_sentences(text: str) -> list[str]:
    """`text` split into the sentences Kokoro synthesizes one by one: the same
    re.split(SENTENCE_SPLIT, text.strip()) KPipeline does (kokoro 0.9.4 pipeline.py:366),
    so Result.text_index indexes this list. Raises on text with nothing to say."""
    pieces = re.split(SENTENCE_SPLIT, text.strip())
    if not all(p.strip() for p in pieces):
        raise ValueError(f"text has an empty sentence: {text!r}")
    return pieces


@dataclass(frozen=True)
class Voice:
    """What Kokoro speaks with: a voice pack (a CPU float32 torch.FloatTensor, one voice or a
    blend from Engines.voice_pack) and a speed. Built per persona by the daemon
    (docs/DESIGN_PERSONAS.md 4.5)."""

    pack: Any
    speed: float


def blend_key(blend: Iterable[tuple[str, float]]) -> tuple[tuple[str, float], ...]:
    """The canonical cache key of a blend: its (id, weight) pairs sorted, so input order does not matter."""
    return tuple(sorted(blend))


class Engines:
    """Loaded once, synchronously, at daemon start (before Telegram connects and before
    the port is bound), so a reachable daemon always has its models.

    The voice and speed are not settings of the engines: every synthesis takes a Voice,
    resolved from the speaking project's persona (docs/DESIGN_PERSONAS.md 4.5)."""

    def __init__(self, kokoro_model: str, whisper_model: str, language: str) -> None:
        self.kokoro_model = kokoro_model
        self.whisper_model = whisper_model
        self.language = language
        self.packs: dict[str, Any] = {}     # English voice id -> CPU float32 pack, filled by load
        self._blends: collections.OrderedDict[tuple, Any] = collections.OrderedDict()
        self._blends_lock = threading.Lock()
        self._kokoro = None
        self._whisper = None

    @property
    def voices(self) -> list[str]:
        """The English voice ids this daemon can speak with, sorted."""
        return sorted(self.packs)

    def load(self, default_voice: str, default_speed: float) -> None:
        """Load and warm both models and every English voice pack. `default_voice` (SPEAK_VOICE)
        must be one of the English voices; the warm-up speaks with it."""
        import torch
        from faster_whisper import WhisperModel
        from kokoro import KPipeline

        t0 = time.time()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available to torch; the speak daemon needs the GPU")
        self.load_voices(default_voice)
        # lang_code "a" (American English) for every voice, the British ones included: owner
        # decision 2026-10-22 (docs/DESIGN_PERSONAS.md 4.5). Kokoro only warns "Language
        # mismatch" for that in load_single_voice, which this daemon does not call.
        self._kokoro = KPipeline(lang_code="a", repo_id=self.kokoro_model, device="cuda")
        # KPipeline swallows an espeak-ng failure (a warning, fallback=None) and then silently
        # drops every out-of-vocabulary word from speech. Fail the load instead.
        if self._kokoro.g2p.fallback is None:
            raise RuntimeError("Kokoro's espeak-ng fallback did not load (is espeak-ng installed?); "
                               "out-of-vocabulary words would be skipped silently")
        dtypes = {p.dtype for p in self._kokoro.model.parameters()}
        if dtypes != {torch.float32}:
            raise RuntimeError(f"Kokoro loaded with parameter dtypes {dtypes}; fp32 is required")
        self._whisper = WhisperModel(self.whisper_model, device=WHISPER_DEVICE, device_index=0,
                                     compute_type=WHISPER_COMPUTE_TYPE)
        # Warm both so the first speak/listen does not pay CUDA kernel selection.
        self.synthesize("Ready.", Voice(self.voice_pack(((default_voice, 1.0),)), default_speed))
        self.transcribe(np.zeros(STT_RATE, dtype=np.float32))
        log.info("engines ready in %.1fs (kokoro=%s, %d English voices, default %s, whisper=%s %s on %s)",
                 time.time() - t0, self.kokoro_model, len(self.packs), default_voice, self.whisper_model,
                 WHISPER_COMPUTE_TYPE, torch.cuda.get_device_name(0))

    def load_voices(self, default_voice: str) -> None:
        """Fetch the ENGLISH_VOICES packs of the Kokoro repo (snapshot_download into the HF
        cache; when the Hub is unreachable it serves the packs already in the cache) and load
        each once on the CPU as float32 into self.packs. Raises, loading nothing, when
        `default_voice` is not one of ENGLISH_VOICES, when the fetch fails, or when any of
        ENGLISH_VOICES is missing from the snapshot (named in the error): no partial list."""
        import torch
        from huggingface_hub import snapshot_download

        if default_voice not in ENGLISH_VOICES:
            raise RuntimeError(f"SPEAK_VOICE {default_voice!r} is not an English voice of {self.kokoro_model}; "
                               f"English voices: {', '.join(ENGLISH_VOICES)}")
        root = snapshot_download(self.kokoro_model, allow_patterns=[f"voices/{vid}.pt" for vid in ENGLISH_VOICES])
        directory = os.path.join(root, "voices")
        files = {vid: os.path.join(directory, f"{vid}.pt") for vid in ENGLISH_VOICES}
        missing = [vid for vid, path in files.items() if not os.path.isfile(path)]
        if missing:
            raise RuntimeError(f"{len(missing)} of the {len(ENGLISH_VOICES)} English voice packs of "
                               f"{self.kokoro_model} are missing from {directory} (the Hub unreachable and the "
                               f"HF cache incomplete?): {', '.join(missing)}")
        packs = {}
        for vid in ENGLISH_VOICES:
            # As KPipeline.load_single_voice does for a .pt path (kokoro 0.9.4 pipeline.py:136-148),
            # without its per-voice hub call: the snapshot already holds the file.
            pack = torch.load(files[vid], weights_only=True, map_location="cpu").float()
            if not isinstance(pack, torch.FloatTensor):
                raise RuntimeError(f"voice pack {vid} loaded as {pack.type()}, not a CPU torch.FloatTensor")
            packs[vid] = pack
        self.packs = packs

    def voice_pack(self, blend: Iterable[tuple[str, float]]):
        """The pack for a blend of (voice id, weight) pairs (weights summing to 1, from the
        persona store): the loaded pack itself for a single voice, else the weighted sum on the
        CPU in float32, checked to be a torch.FloatTensor (KPipeline takes it as voice= unchanged,
        kokoro 0.9.4 pipeline.py:157-167). Blends are kept in an LRU of BLEND_CACHE_SIZE."""
        import torch

        key = blend_key(blend)
        if not key:
            raise ValueError("a voice blend needs at least one voice")
        if len(key) == 1:
            return self.packs[key[0][0]]
        with self._blends_lock:
            pack = self._blends.get(key)
            if pack is not None:
                self._blends.move_to_end(key)
                return pack
        pack = sum(float(weight) * self.packs[vid] for vid, weight in key).float()
        if not isinstance(pack, torch.FloatTensor):
            raise RuntimeError(f"voice blend {key} is {pack.type()}, not a CPU torch.FloatTensor")
        with self._blends_lock:
            self._blends[key] = pack
            self._blends.move_to_end(key)
            while len(self._blends) > BLEND_CACHE_SIZE:
                self._blends.popitem(last=False)
        return pack

    def synthesize_segments(self, text: str, voice_of: Callable[[], Voice]) -> Iterator[tuple[int, np.ndarray]]:
        """`text` as mono float32 at TTS_RATE, one array per Kokoro segment, each with the index
        of its sentence in split_sentences(text), yielded as soon as Kokoro has made it. Kokoro
        is called once per sentence and `voice_of()` is asked before each one, so a persona
        change lands on the next sentence synthesized (docs/DESIGN_PERSONAS.md 4.5); Kokoro
        infers each sentence independently anyway (kokoro 0.9.4 pipeline.py:369-383). A sentence
        over 510 phonemes gives several segments with the same index; a sentence with no
        phonemes gives none. speak plays each segment while the next is synthesized
        (docs/DESIGN_STREAMED_SPEECH.md) and uses the index to cut and resume at a sentence
        (docs/DESIGN_INTERRUPT.md). Raises if Kokoro yields nothing for the whole text."""
        produced = False
        for index, sentence in enumerate(split_sentences(text)):
            voice = voice_of()
            for result in self._kokoro(sentence, voice=voice.pack, speed=voice.speed, split_pattern=SENTENCE_SPLIT):
                if result.text_index != 0:
                    raise RuntimeError(f"Kokoro split sentence {index} of {text!r} again "
                                       f"(text_index {result.text_index})")
                produced = True
                yield index, result.audio.detach().cpu().numpy().astype(np.float32).reshape(-1)
        if not produced:
            raise RuntimeError(f"Kokoro produced no audio for {text!r}")

    def synthesize(self, text: str, voice: Voice) -> np.ndarray:
        """All of `text` as one mono float32 array at TTS_RATE, in one voice: the call greeting
        (one opener file of at least OPENER_MIN_SECONDS, measured before dialing), the switch
        announcement and the warm-up. Speech on a connected call streams instead
        (synthesize_segments). Raises if Kokoro yields nothing."""
        pieces = [result.audio.detach().cpu().numpy().astype(np.float32).reshape(-1)
                  for result in self._kokoro(text, voice=voice.pack, speed=voice.speed, split_pattern=SENTENCE_SPLIT)]
        if not pieces:
            raise RuntimeError(f"Kokoro produced no audio for {text!r}")
        return np.concatenate(pieces)

    def transcribe(self, audio: np.ndarray) -> str:
        """16 kHz mono float32 -> text. Used by listen and by the daemon's POST / (port 9000)."""
        segments, _info = self._whisper.transcribe(audio, language=self.language, beam_size=1, vad_filter=False)
        return "".join(segment.text for segment in segments).strip()
