import copy
import unittest
from types import SimpleNamespace

from aligner import Wav2Vec2Aligner
from predict import Predictor


def _word(text, start, end):
    return SimpleNamespace(
        word=text,
        start=start,
        end=end,
        probability=0.95,
    )


class _FakeWhisperModel:
    def __init__(self, language="en", words=None):
        self.language = language
        self.words = words or [_word(" hello", 0.1, 0.4)]

    def transcribe(self, *_args, **_kwargs):
        segment = SimpleNamespace(
            id=0,
            seek=0,
            start=self.words[0].start,
            end=self.words[-1].end,
            text="".join(word.word for word in self.words),
            tokens=[1],
            temperature=0.0,
            avg_logprob=-0.1,
            compression_ratio=1.0,
            no_speech_prob=0.0,
            words=self.words,
        )
        return iter([segment]), SimpleNamespace(language=self.language)


class _FakeAligner:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.setup_calls = []
        self.align_calls = []

    @staticmethod
    def normalize_language_code(language_code):
        return Wav2Vec2Aligner.normalize_language_code(language_code)

    @staticmethod
    def supports_language(language_code):
        return Wav2Vec2Aligner.supports_language(language_code)

    def setup(self, device="cuda"):
        self.setup_calls.append(device)

    def align(self, audio_path, words, language_code="en"):
        self.align_calls.append(
            {
                "audio_path": audio_path,
                "words": copy.deepcopy(words),
                "language_code": language_code,
            }
        )
        if self.error is not None:
            raise self.error
        return copy.deepcopy(self.result)


class PredictorAlignmentAuthorityTest(unittest.TestCase):
    def _predictor(self, language="en", words=None, aligner=None):
        predictor = Predictor()
        predictor.models["base"] = _FakeWhisperModel(
            language=language,
            words=words,
        )
        predictor.aligner = aligner or _FakeAligner()
        return predictor

    def test_unsupported_language_preserves_whisper_geometry(self):
        words = [_word(" über", 0.1, 0.4)]
        aligner = _FakeAligner(result=[])
        predictor = self._predictor(
            language="de",
            words=words,
            aligner=aligner,
        )

        result = predictor.predict(
            "unused.wav",
            model_name="base",
            word_timestamps=True,
            force_align=True,
        )

        self.assertEqual(
            result["word_timestamps"],
            [
                {
                    "word": " über",
                    "start": 0.1,
                    "end": 0.4,
                    "probability": 0.95,
                }
            ],
        )
        self.assertFalse(result["word_timestamps_aligned"])
        self.assertEqual(
            result["alignment"]["status"],
            "UNSUPPORTED_LANGUAGE",
        )
        self.assertEqual(result["alignment"]["detected_language"], "de")
        self.assertEqual(result["alignment"]["aligned_words"], 0)
        self.assertEqual(result["alignment"]["fallback_words"], 1)
        self.assertFalse(result["alignment"]["per_word_authority"])
        self.assertEqual(aligner.setup_calls, [])
        self.assertEqual(aligner.align_calls, [])

    def test_supported_language_reports_partial_per_word_authority(self):
        words = [
            _word(" hello", 0.1, 0.4),
            _word(" 2026", 0.5, 0.8),
        ]
        aligned_result = [
            {
                "word": " hello",
                "start": 0.12,
                "end": 0.38,
                "onset_start": 0.1,
                "offset_end": 0.4,
                "probability": 0.95,
                "alignment_status": "ALIGNED_SUPPORTED",
                "alignment_authority": True,
                "alignment_model_id": "test-aligner",
                "alignment_language": "en",
            },
            {
                "word": " 2026",
                "start": 0.5,
                "end": 0.8,
                "probability": 0.95,
                "alignment_status": "FALLBACK_UNALIGNED",
                "alignment_authority": False,
                "alignment_model_id": "test-aligner",
                "alignment_reason": "NO_MODEL_VOCABULARY",
            },
        ]
        aligner = _FakeAligner(result=aligned_result)
        predictor = self._predictor(
            language="en-US",
            words=words,
            aligner=aligner,
        )

        result = predictor.predict(
            "unused.wav",
            model_name="base",
            word_timestamps=True,
            force_align=True,
        )

        self.assertTrue(result["word_timestamps_aligned"])
        self.assertEqual(result["alignment"]["status"], "PARTIAL")
        self.assertEqual(result["alignment"]["detected_language"], "en")
        self.assertEqual(result["alignment"]["aligned_words"], 1)
        self.assertEqual(result["alignment"]["fallback_words"], 1)
        self.assertEqual(result["alignment"]["aligned_word_fraction"], 0.5)
        self.assertTrue(result["alignment"]["per_word_authority"])
        self.assertEqual(len(aligner.setup_calls), 1)
        self.assertEqual(aligner.align_calls[0]["language_code"], "en")
        self.assertEqual(
            result["word_timestamps"][0]["alignment_status"],
            "ALIGNED_SUPPORTED",
        )
        self.assertEqual(
            result["word_timestamps"][1]["alignment_status"],
            "FALLBACK_UNALIGNED",
        )

    def test_alignment_failure_preserves_paid_whisper_result(self):
        words = [_word(" hello", 0.1, 0.4)]
        aligner = _FakeAligner(error=RuntimeError("synthetic failure"))
        predictor = self._predictor(
            language="en",
            words=words,
            aligner=aligner,
        )

        result = predictor.predict(
            "unused.wav",
            model_name="base",
            word_timestamps=True,
            force_align=True,
        )

        self.assertEqual(
            result["word_timestamps"],
            [
                {
                    "word": " hello",
                    "start": 0.1,
                    "end": 0.4,
                    "probability": 0.95,
                }
            ],
        )
        self.assertFalse(result["word_timestamps_aligned"])
        self.assertEqual(result["alignment"]["status"], "FAILED")
        self.assertEqual(
            result["alignment"]["failure_type"],
            "RuntimeError",
        )
        self.assertEqual(len(aligner.align_calls), 1)

    def test_non_alignment_path_keeps_legacy_shape(self):
        predictor = self._predictor(language="de")

        result = predictor.predict(
            "unused.wav",
            model_name="base",
            word_timestamps=True,
            force_align=False,
        )

        self.assertNotIn("alignment", result)
        self.assertNotIn("word_timestamps_aligned", result)

    def test_aligner_rejects_unsupported_language_before_model_load(self):
        aligner = Wav2Vec2Aligner()

        with self.assertRaisesRegex(
            ValueError,
            "does not support alignment language",
        ):
            aligner.align(
                "unused.wav",
                [{"word": "über", "start": 0.1, "end": 0.4}],
                language_code="de",
            )

        self.assertIsNone(aligner.model)


if __name__ == "__main__":
    unittest.main()
