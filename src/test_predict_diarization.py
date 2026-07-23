import copy
import time
import unittest
from types import SimpleNamespace

from predict import Predictor


class _FakeWhisperModel:
    def __init__(self):
        word = SimpleNamespace(
            word=" hello",
            start=0.1,
            end=0.4,
            probability=0.95,
        )
        self.segment = SimpleNamespace(
            id=0,
            seek=0,
            start=0.1,
            end=0.4,
            text=" hello",
            tokens=[1],
            temperature=0.0,
            avg_logprob=-0.1,
            compression_ratio=1.0,
            no_speech_prob=0.0,
            words=[word],
        )

    def transcribe(self, *_args, **_kwargs):
        return iter([self.segment]), SimpleNamespace(language="en")


class _RecordingDiarizer:
    def __init__(self, order=None):
        self.words_seen = None
        self.order = order

    def diarize(self, _audio, words, **_kwargs):
        if self.order is not None:
            self.order.append("diarize")
        self.words_seen = copy.deepcopy(words)
        return {
            "schema_version": "w2l-speaker-diarization-v1",
            "status": "COMPLETED",
            "boundary_authority": False,
            "transcript_geometry_mutated": False,
        }


class _SlowClapScorer:
    def __init__(self, order):
        self.order = order

    def score(self, _audio, _queries):
        self.order.append("clap-start")
        time.sleep(0.05)
        self.order.append("clap-end")
        return {"duration": 1.0, "device": "cpu", "scores": {}}


class PredictorDiarizationContractTest(unittest.TestCase):
    def _predictor(self):
        predictor = Predictor()
        predictor.models["base"] = _FakeWhisperModel()
        predictor.diarizer = _RecordingDiarizer()
        return predictor

    def test_default_path_does_not_emit_or_run_diarization(self):
        predictor = self._predictor()

        result = predictor.predict(
            "unused.wav",
            model_name="base",
            word_timestamps=True,
        )

        self.assertNotIn("speaker_diarization", result)
        self.assertIsNone(predictor.diarizer.words_seen)

    def test_opt_in_sidecar_sees_but_does_not_mutate_whisper_words(self):
        predictor = self._predictor()

        result = predictor.predict(
            "unused.wav",
            model_name="base",
            word_timestamps=True,
            diarize=True,
        )

        expected_words = [
            {
                "word": " hello",
                "start": 0.1,
                "end": 0.4,
                "probability": 0.95,
            }
        ]
        self.assertEqual(result["word_timestamps"], expected_words)
        self.assertEqual(predictor.diarizer.words_seen, expected_words)
        self.assertEqual(result["speaker_diarization"]["status"], "COMPLETED")

    def test_turn_only_diarization_can_run_without_word_timestamps(self):
        predictor = self._predictor()

        result = predictor.predict(
            "unused.wav",
            model_name="base",
            word_timestamps=False,
            diarize=True,
        )

        self.assertNotIn("word_timestamps", result)
        self.assertEqual(predictor.diarizer.words_seen, [])
        self.assertEqual(result["speaker_diarization"]["status"], "COMPLETED")

    def test_diarization_waits_for_clap_instead_of_overlapping_gpu_phase(self):
        order = []
        predictor = self._predictor()
        predictor.clap_scorer = _SlowClapScorer(order)
        predictor.diarizer = _RecordingDiarizer(order)

        result = predictor.predict(
            "unused.wav",
            model_name="base",
            word_timestamps=True,
            clap_queries={"laughter": "people laughing"},
            diarize=True,
        )

        self.assertEqual(order, ["clap-start", "clap-end", "diarize"])
        self.assertIsNotNone(result["clap_scores"])


if __name__ == "__main__":
    unittest.main()
