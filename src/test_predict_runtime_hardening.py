import os
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from aligner import Wav2Vec2Aligner
from model_manifest import WHISPER_MODEL_REVISIONS
from predict import Predictor


class _OneWordWhisperModel:
    def transcribe(self, *_args, **_kwargs):
        word = SimpleNamespace(
            word=" hello",
            start=0.1,
            end=0.4,
            probability=0.95,
        )
        segment = SimpleNamespace(
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
        return iter([segment]), SimpleNamespace(language="en")


class _ReturningClapScorer:
    def __init__(self, result):
        self.result = result

    def score(self, *_args, **_kwargs):
        return self.result


class _RaisingClapScorer:
    def score(self, *_args, **_kwargs):
        raise RuntimeError("private scorer detail")


class PredictorRuntimeHardeningTest(unittest.TestCase):
    def test_resource_exhaustion_classifier_is_narrow(self):
        self.assertTrue(Predictor._is_resource_exhaustion(RuntimeError("CUDA out of memory")))
        self.assertTrue(Predictor._is_resource_exhaustion(RuntimeError("failed to allocate 2 GiB")))
        self.assertFalse(Predictor._is_resource_exhaustion(RuntimeError("401 gated model")))
        self.assertFalse(Predictor._is_resource_exhaustion(RuntimeError("network timed out")))

    def test_non_resource_model_failure_does_not_evict_residents(self):
        predictor = Predictor()
        resident = object()
        predictor.models = {"small": resident}

        with patch("predict.WhisperModel", side_effect=RuntimeError("401 gated model")):
            with self.assertRaisesRegex(ValueError, "Failed to load model"):
                predictor._load_model_locked("large-v3")

        self.assertEqual(predictor.models, {"small": resident})

    def test_resource_failure_evicts_one_lru_and_retries(self):
        predictor = Predictor()
        predictor.models = {"small": object()}
        loaded = object()

        with patch(
            "predict.WhisperModel",
            side_effect=[RuntimeError("CUDA out of memory"), loaded],
        ) as constructor:
            result = predictor._load_model_locked("large-v3")

        self.assertIs(result, loaded)
        self.assertEqual(list(predictor.models), ["large-v3"])
        self.assertEqual(constructor.call_count, 2)

    def test_whisper_load_is_revision_pinned_and_runtime_offline(self):
        predictor = Predictor()
        loaded = object()

        with patch("predict.rp_cuda.is_available", return_value=False), patch(
            "predict.WhisperModel",
            return_value=loaded,
        ) as constructor:
            self.assertIs(predictor._load_model_locked("large-v3"), loaded)

        constructor.assert_called_once_with(
            "large-v3",
            device="cpu",
            compute_type="int8",
            revision=WHISPER_MODEL_REVISIONS["large-v3"],
            local_files_only=True,
        )

    def test_configured_warmup_runs_in_background(self):
        predictor = Predictor()
        warmed = threading.Event()
        predictor.clap_scorer.warmup = warmed.set

        with patch.dict(os.environ, {"AUDIO_WORKER_PRELOAD": "clap"}):
            predictor.setup()

        self.assertIsNotNone(predictor._warmup_thread)
        predictor._warmup_thread.join(timeout=2)
        self.assertTrue(warmed.is_set())

    @staticmethod
    def _predictor_with_clap(scorer):
        predictor = Predictor()
        predictor.models["base"] = _OneWordWhisperModel()
        predictor.clap_scorer = scorer
        return predictor

    def test_structured_clap_failure_preserves_transcript_and_diagnostics(self):
        predictor = self._predictor_with_clap(
            _ReturningClapScorer(
                {
                    "schema_version": "w2l-clap-scores-v2",
                    "status": "FAILED",
                    "error_code": "CUDA_OUT_OF_MEMORY",
                    "error": "bounded failure",
                    "retryable": True,
                    "scores": {},
                    "duration": 1.0,
                }
            )
        )

        result = predictor.predict(
            "unused.wav",
            model_name="base",
            word_timestamps=True,
            clap_queries={"speech": "a person speaking"},
        )

        self.assertEqual(result["transcription"], "hello")
        self.assertIsNone(result["clap_scores"])
        self.assertEqual(
            result["clap_diagnostics"]["error_code"],
            "CUDA_OUT_OF_MEMORY",
        )
        self.assertTrue(result["clap_diagnostics"]["retryable"])
        self.assertNotIn("scores", result["clap_diagnostics"])

    def test_unexpected_clap_thread_exception_has_stable_retryable_code(self):
        predictor = self._predictor_with_clap(_RaisingClapScorer())

        result = predictor.predict(
            "unused.wav",
            model_name="base",
            clap_queries={"speech": "a person speaking"},
        )

        self.assertIsNone(result["clap_scores"])
        self.assertEqual(
            result["clap_diagnostics"]["error_code"],
            "CLAP_THREAD_FAILED",
        )
        self.assertTrue(result["clap_diagnostics"]["retryable"])
        self.assertNotIn("private scorer detail", repr(result))


class AlignerInferenceSerializationTest(unittest.TestCase):
    def test_two_align_calls_cannot_enter_model_path_together(self):
        aligner = Wav2Vec2Aligner()
        first_entered = threading.Event()
        release_first = threading.Event()
        second_entered = threading.Event()
        call_count = 0
        call_count_lock = threading.Lock()

        def fake_align(*_args, **_kwargs):
            nonlocal call_count
            with call_count_lock:
                call_count += 1
                current = call_count
            if current == 1:
                first_entered.set()
                self.assertTrue(release_first.wait(timeout=2))
            else:
                second_entered.set()
            return []

        aligner._align = fake_align
        first = threading.Thread(target=lambda: aligner.align("a.wav", []))
        second = threading.Thread(target=lambda: aligner.align("b.wav", []))
        first.start()
        self.assertTrue(first_entered.wait(timeout=2))
        second.start()
        self.assertFalse(second_entered.wait(timeout=0.1))
        release_first.set()
        first.join(timeout=2)
        second.join(timeout=2)

        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertTrue(second_entered.is_set())


if __name__ == "__main__":
    unittest.main()
