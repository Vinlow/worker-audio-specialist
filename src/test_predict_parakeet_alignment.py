import copy
import unittest
from unittest.mock import Mock

from predict import Predictor
import test_parakeet_acoustic_alignment as fixtures


class PredictorParakeetAlignmentTest(unittest.TestCase):
    def predictor(self):
        evidence = fixtures.ParakeetAcousticAlignmentTest()
        source = evidence.source()
        predictor = Predictor()
        predictor.parakeet_transcriber.transcribe = Mock(return_value=source)
        predictor.aligner = fixtures.AlignerProbe(evidence.aligned(source))
        return predictor, source

    def test_explicit_request_runs_recognition_then_acoustic_alignment_once(self):
        predictor, source = self.predictor()
        before = copy.deepcopy(source)
        result = predictor.predict("source.wav", asr_backend="parakeet", language="en",
                                   word_timestamps=True, force_align=True)
        predictor.parakeet_transcriber.transcribe.assert_called_once_with(
            "source.wav", language_hint="en", include_word_timestamps=True)
        self.assertEqual(result["alignment"]["status"], "ALIGNED_SUPPORTED")
        self.assertEqual(source, before)
        self.assertEqual(predictor.models, {})

    def test_native_route_does_not_load_aligner(self):
        predictor, source = self.predictor()
        result = predictor.predict("source.wav", asr_backend="parakeet", word_timestamps=True)
        self.assertIs(result, source)
        self.assertEqual(predictor.aligner.calls, [])

    def test_invalid_alignment_request_is_rejected_before_recognition(self):
        for language, timestamps in ((None, True), ("de", True), ("en", False)):
            with self.subTest(language=language, timestamps=timestamps):
                predictor, _ = self.predictor()
                with self.assertRaisesRegex(ValueError, "explicit language=en"):
                    predictor.predict("source.wav", asr_backend="parakeet", language=language,
                                      word_timestamps=timestamps, force_align=True)
                predictor.parakeet_transcriber.transcribe.assert_not_called()


if __name__ == "__main__":
    unittest.main()
