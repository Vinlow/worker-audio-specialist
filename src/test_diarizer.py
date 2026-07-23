import copy
import os
import sys
import threading
import types
import unittest
from unittest.mock import patch

from diarizer import (
    SpeakerDiarizer,
    _canonicalize_turns,
    build_diarization_sidecar,
)


class DiarizationSidecarTest(unittest.TestCase):
    def test_canonicalizes_labels_by_first_appearance(self):
        turns, mapping = _canonicalize_turns(
            [
                (2.0, 3.0, "B"),
                (0.0, 1.0, "A"),
                (3.0, 4.0, "A"),
            ]
        )

        self.assertEqual(mapping, {"A": "SPEAKER_00", "B": "SPEAKER_01"})
        self.assertEqual(
            [turn["speaker_id"] for turn in turns],
            ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"],
        )

    def test_builds_attributions_without_mutating_np_sbv2_words(self):
        words = [
            {
                "word": "Hello",
                "start": 0.1,
                "end": 0.4,
                "onset_start": 0.05,
                "offset_end": 0.45,
            },
            {
                "word": "there",
                "start": 0.5,
                "end": 0.8,
                "onset_start": 0.45,
                "offset_end": 0.9,
            },
        ]
        original = copy.deepcopy(words)
        sidecar = build_diarization_sidecar(
            words,
            [
                {
                    "start_sec": 0.0,
                    "end_sec": 0.48,
                    "speaker_id": "SPEAKER_00",
                },
                {
                    "start_sec": 0.48,
                    "end_sec": 1.0,
                    "speaker_id": "SPEAKER_01",
                },
            ],
        )

        self.assertEqual(words, original)
        self.assertFalse(sidecar["transcript_geometry_mutated"])
        self.assertFalse(sidecar["boundary_authority"])
        self.assertEqual(
            [item["speaker_id"] for item in sidecar["word_attributions"]],
            ["SPEAKER_00", "SPEAKER_01"],
        )

    def test_marks_overlapping_speech_without_forcing_two_speakers(self):
        sidecar = build_diarization_sidecar(
            [{"word": "wow", "start": 1.0, "end": 1.5}],
            [
                {
                    "start_sec": 0.9,
                    "end_sec": 1.5,
                    "speaker_id": "SPEAKER_00",
                },
                {
                    "start_sec": 1.2,
                    "end_sec": 1.6,
                    "speaker_id": "SPEAKER_01",
                },
            ],
            [
                {
                    "start_sec": 0.9,
                    "end_sec": 1.5,
                    "speaker_id": "SPEAKER_00",
                }
            ],
        )

        attribution = sidecar["word_attributions"][0]
        self.assertEqual(attribution["speaker_id"], "SPEAKER_00")
        self.assertTrue(attribution["overlap"])

    def test_unknown_when_no_turn_overlaps_word(self):
        sidecar = build_diarization_sidecar(
            [{"word": "late", "start": 10.0, "end": 10.4}],
            [
                {
                    "start_sec": 0.0,
                    "end_sec": 1.0,
                    "speaker_id": "SPEAKER_00",
                }
            ],
        )

        self.assertEqual(sidecar["word_attributions"][0]["status"], "UNKNOWN")
        self.assertIsNone(sidecar["word_attributions"][0]["speaker_id"])

    def test_missing_token_fails_sidecar_without_touching_words(self):
        words = [{"word": "safe", "start": 0.0, "end": 0.2}]
        original = copy.deepcopy(words)

        with patch.dict(
            os.environ,
            {"HUGGINGFACE_TOKEN": "", "HF_TOKEN": ""},
            clear=False,
        ):
            sidecar = SpeakerDiarizer().diarize("unused.wav", words)

        self.assertEqual(sidecar["status"], "FAILED")
        self.assertIn("MISSING_HUGGINGFACE_TOKEN", sidecar["error"])
        self.assertFalse(sidecar["boundary_authority"])
        self.assertFalse(sidecar["transcript_geometry_mutated"])
        self.assertEqual(words, original)

    def test_rejects_inverted_speaker_hints_before_model_load(self):
        sidecar = SpeakerDiarizer().diarize(
            "unused.wav",
            [],
            min_speakers=3,
            max_speakers=2,
        )

        self.assertEqual(sidecar["status"], "FAILED")
        self.assertIn("DIARIZATION_MIN_SPEAKERS_EXCEEDS_MAX", sidecar["error"])

    def test_concurrent_setup_publishes_one_complete_pipeline(self):
        entered = threading.Event()
        release = threading.Event()
        load_calls = []
        errors = []

        class FakePipeline:
            @classmethod
            def from_pretrained(cls, *_args, **_kwargs):
                load_calls.append(1)
                entered.set()
                if not release.wait(timeout=2):
                    raise RuntimeError("test coordination timeout")
                return cls()

            def to(self, _device):
                return self

        fake_pyannote = types.ModuleType("pyannote")
        fake_audio = types.ModuleType("pyannote.audio")
        fake_core = types.ModuleType("pyannote.audio.core")
        fake_task = types.ModuleType("pyannote.audio.core.task")
        fake_audio.Pipeline = FakePipeline
        fake_task.Problem = type("Problem", (), {})
        fake_task.Resolution = type("Resolution", (), {})
        fake_task.Specifications = type("Specifications", (), {})

        diarizer = SpeakerDiarizer()

        def setup():
            try:
                diarizer.setup("cpu")
            except Exception as exc:
                errors.append(exc)

        with patch.dict(
            os.environ,
            {"HUGGINGFACE_TOKEN": "test-token"},
            clear=False,
        ), patch.dict(
            sys.modules,
            {
                "pyannote": fake_pyannote,
                "pyannote.audio": fake_audio,
                "pyannote.audio.core": fake_core,
                "pyannote.audio.core.task": fake_task,
            },
        ), patch(
            "torch.serialization.add_safe_globals",
        ):
            first = threading.Thread(target=setup)
            second = threading.Thread(target=setup)
            first.start()
            self.assertTrue(entered.wait(timeout=2))
            second.start()
            release.set()
            first.join(timeout=2)
            second.join(timeout=2)

        self.assertEqual(errors, [])
        self.assertEqual(load_calls, [1])
        self.assertIsNotNone(diarizer.pipeline)
        self.assertEqual(diarizer.device, "cpu")


if __name__ == "__main__":
    unittest.main()
