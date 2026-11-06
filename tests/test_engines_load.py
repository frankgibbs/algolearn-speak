"""Engines.load: Whisper's compute type (docs/DESIGN_REMOTE_DAEMON.md section 12).

The models are faked: the test checks what load asks CTranslate2 for and that it refuses
a Whisper that runs any other compute type. No GPU, no model download.
"""

from __future__ import annotations

import os
import sys
import types
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

import speak_engines as eng  # noqa: E402


class FakeKPipeline:
    def __init__(self, lang_code, repo_id, device):
        self.g2p = types.SimpleNamespace(fallback=object())
        self.model = torch.nn.Linear(2, 2)


class FakeWhisperModel:
    reported = None   # the compute type the fake CTranslate2 model says it runs
    calls: list = []

    def __init__(self, model, device, device_index, compute_type):
        FakeWhisperModel.calls.append({"model": model, "device": device, "device_index": device_index,
                                       "compute_type": compute_type})
        self.model = types.SimpleNamespace(
            compute_type=compute_type if FakeWhisperModel.reported is None else FakeWhisperModel.reported)


class TestWhisperComputeType(unittest.TestCase):
    def setUp(self):
        FakeWhisperModel.calls = []
        FakeWhisperModel.reported = None
        self.engines = eng.Engines("hexgrad/Kokoro-82M", "large-v3-turbo", "en")
        patches = [
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.get_device_name", return_value="fake GPU"),
            mock.patch("kokoro.KPipeline", FakeKPipeline),
            mock.patch("faster_whisper.WhisperModel", FakeWhisperModel),
            mock.patch.object(self.engines, "load_voices"),
            mock.patch.object(self.engines, "voice_pack"),
            mock.patch.object(self.engines, "synthesize"),
            mock.patch.object(self.engines, "transcribe"),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def test_the_compute_type_is_int8_float16(self):
        self.assertEqual(eng.WHISPER_COMPUTE_TYPE, "int8_float16")

    def test_load_asks_for_it_on_gpu0(self):
        self.engines.load("af_heart", 1.0)
        self.assertEqual(FakeWhisperModel.calls, [{"model": "large-v3-turbo", "device": "cuda", "device_index": 0,
                                                   "compute_type": "int8_float16"}])

    def test_load_refuses_a_whisper_running_another_compute_type(self):
        FakeWhisperModel.reported = "int8_float32"
        with self.assertRaisesRegex(RuntimeError, "compute_type 'int8_float32'.*'int8_float16' is required"):
            self.engines.load("af_heart", 1.0)
        self.engines.transcribe.assert_not_called()


if __name__ == "__main__":
    unittest.main()
