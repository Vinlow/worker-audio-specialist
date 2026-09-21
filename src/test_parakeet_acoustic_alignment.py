import copy
import unittest

from parakeet_acoustic_alignment import ParakeetAcousticAlignment


class AlignerProbe:
    def __init__(self, words=None, error=None):
        self.words = words
        self.error = error
        self.calls = []

    def setup(self, device):
        self.calls.append(("setup", device))

    def align(self, audio, words, language_code):
        self.calls.append(("align", audio, copy.deepcopy(words), language_code))
        if self.error:
            raise self.error
        return copy.deepcopy(self.words)


class ParakeetAcousticAlignmentTest(unittest.TestCase):
    def source(self):
        return {
            "asr_backend": "parakeet", "transcription": "Restart cut",
            "word_timestamps_aligned": False,
            "asr_backend_evidence": {"language_hint": "en", "model_detected_language": None,
                "language_status": "HINT_ONLY", "language_authority": "CALLER_HINT_MODEL_DETECTION_MISSING",
                "audio_duration_seconds": 4, "timestamp_authority": "DIRECTIONAL_NOT_NP_SBV2"},
            "word_timestamps": [
                {"word": "Restart", "start": 1, "end": 1.5, "probability": None,
                 "timestamp_authority": "DIRECTIONAL_NOT_NP_SBV2", "timestamp_source": "PARAKEET_NATIVE_TOKEN_DURATION"},
                {"word": " cut", "start": 1.5, "end": 2, "probability": None,
                 "timestamp_authority": "DIRECTIONAL_NOT_NP_SBV2", "timestamp_source": "PARAKEET_NATIVE_TOKEN_DURATION"},
            ],
        }

    def aligned(self, source):
        return [{**word, "start": word["start"] + .1, "end": word["end"] - .1,
                 "onset_start": word["start"], "offset_end": word["end"],
                 "alignment_authority": True, "alignment_status": "ALIGNED_SUPPORTED"}
                for word in source["word_timestamps"]]

    def test_keeps_native_measurements_and_language_uncertainty_separate(self):
        source = self.source()
        before = copy.deepcopy(source)
        probe = AlignerProbe(self.aligned(source))
        result = ParakeetAcousticAlignment.apply("source.wav", source, probe, "cpu")
        self.assertEqual(source, before)
        self.assertEqual(result["asr_backend_evidence"], before["asr_backend_evidence"])
        self.assertEqual(result["alignment"]["status"], "ALIGNED_SUPPORTED")
        self.assertIsNone(result["alignment"]["detected_language"])
        self.assertFalse(result["alignment"]["natural_landing_authority"])
        for word, native in zip(result["word_timestamps"], before["word_timestamps"]):
            self.assertEqual(word["word"], native["word"])
            self.assertIsNone(word["probability"])
            self.assertEqual(word["native_timing"]["start"], native["start"])
            self.assertEqual(word["timestamp_authority"], "NP_SBV2_ACOUSTIC")

    def test_partial_alignment_retains_unsupported_native_word(self):
        source = self.source()
        words = self.aligned(source)
        words[1] = {**words[1], "alignment_authority": False,
                    "alignment_status": "FALLBACK_UNALIGNED", "start": 3}
        result = ParakeetAcousticAlignment.apply("source.wav", source, AlignerProbe(words), "cpu")
        self.assertEqual(result["alignment"]["status"], "PARTIAL")
        self.assertEqual(result["alignment"]["aligned_words"], 1)
        self.assertEqual(result["word_timestamps"][1]["start"], 1.5)
        self.assertFalse(result["word_timestamps"][1]["alignment_authority"])
        self.assertEqual(result["word_timestamps"][1]["timestamp_authority"], "DIRECTIONAL_NOT_NP_SBV2")

    def test_refuses_identity_loss_and_invalid_acoustic_geometry(self):
        source = self.source()
        for mutation in ({"word": "Stop"}, {"end": 10}, {"start": float("nan")}, {"onset_start": 2}):
            with self.subTest(mutation=mutation):
                words = self.aligned(source)
                words[0].update(mutation)
                result = ParakeetAcousticAlignment.apply("source.wav", source, AlignerProbe(words), "cpu")
                self.assertEqual(result["alignment"]["status"], "FAILED")
                self.assertFalse(result["word_timestamps_aligned"])
                self.assertEqual(result["word_timestamps"], source["word_timestamps"])

    def test_failure_preserves_paid_recognition_without_retry(self):
        source = self.source()
        probe = AlignerProbe(error=RuntimeError("alignment failed"))
        result = ParakeetAcousticAlignment.apply("source.wav", source, probe, "cpu")
        self.assertEqual(result["transcription"], source["transcription"])
        self.assertEqual(result["word_timestamps"], source["word_timestamps"])
        self.assertEqual(len(probe.calls), 2)
        self.assertEqual(result["alignment"]["status"], "FAILED")

    def test_rejects_individually_valid_but_overlapping_acoustic_words(self):
        source = self.source()
        words = self.aligned(source)
        words[0].update({"end": 1.8, "offset_end": 1.9})
        result = ParakeetAcousticAlignment.apply("source.wav", source, AlignerProbe(words), "cpu")
        self.assertEqual(result["alignment"]["status"], "FAILED")
        self.assertEqual(result["word_timestamps"], source["word_timestamps"])

    def test_language_mismatch_and_missing_hint_cannot_gain_authority(self):
        for evidence in ({"model_detected_language": "de"}, {"language_hint": None}, {"language_status": "MODEL_HINT_MISMATCH"}):
            with self.subTest(evidence=evidence):
                source = self.source()
                source["asr_backend_evidence"].update(evidence)
                probe = AlignerProbe([])
                result = ParakeetAcousticAlignment.apply("source.wav", source, probe, "cpu")
                self.assertEqual(result["alignment"]["status"], "UNSUPPORTED_LANGUAGE")
                self.assertEqual(probe.calls, [])


if __name__ == "__main__":
    unittest.main()
