import copy
import sys
import types
import unittest

if "aligner" not in sys.modules:
    aligner_stub = types.ModuleType("aligner")
    aligner_stub.ALIGNMENT_MODEL_ID = (
        "torchaudio/WAV2VEC2_ASR_LARGE_LV60K_960H"
    )
    aligner_stub.ALIGNMENT_SCHEMA_VERSION = "w2l-forced-alignment-v1"
    sys.modules["aligner"] = aligner_stub

from supplied_text_aligner import SuppliedTextAligner


class _RecordingAligner:
    def __init__(self):
        self.words = None

    def align(self, _audio_path, words, language_code="en"):
        self.words = copy.deepcopy(words)
        return [
            {
                **word,
                "start": word["start"] + 0.01,
                "end": word["end"] + 0.01,
                "onset_start": word["start"],
                "offset_end": word["end"] + 0.02,
                "alignment_status": "ALIGNED_SUPPORTED",
                "alignment_authority": True,
                "alignment_score_mean": -0.1,
                "alignment_score_min": -0.2,
            }
            for word in words
        ]


class SuppliedTextAlignerTest(unittest.TestCase):
    def test_routes_exact_supplied_words_through_acoustic_alignment(self):
        acoustic = _RecordingAligner()
        result = SuppliedTextAligner(acoustic).align(
            "unused.wav",
            [
                {"start": 0, "end": 1.5, "text": "Restart, cut that."},
                {"start": 2, "end": 3, "text": "Clean take."},
            ],
        )

        self.assertEqual(
            [word["word"] for word in acoustic.words],
            ["Restart,", "cut", "that.", "Clean", "take."],
        )
        self.assertEqual(result["alignment"]["status"], "ALIGNED_SUPPORTED")
        self.assertEqual(result["alignment"]["aligned_words"], 5)
        self.assertEqual(
            result["alignment"]["supplied_text_admission"]["status"],
            "ACCEPTED",
        )
        self.assertRegex(result["alignment"]["supplied_text_sha256"], r"^[a-f0-9]{64}$")
        self.assertEqual(
            [word["supplied_word_ordinal"] for word in result["word_timestamps"]],
            list(range(5)),
        )

    def test_rejects_overlap_instead_of_inventing_speaker_order(self):
        with self.assertRaisesRegex(ValueError, "invalid ordered geometry"):
            SuppliedTextAligner.validate_segments(
                [
                    {"start": 0, "end": 2, "text": "one"},
                    {"start": 1.5, "end": 3, "text": "two"},
                ]
            )

    def test_marks_partial_alignment_without_promoting_fallback_geometry(self):
        acoustic = _RecordingAligner()
        original_align = acoustic.align

        def partial(_audio_path, words, language_code="en"):
            result = original_align(_audio_path, words, language_code)
            result[0]["alignment_status"] = "FALLBACK_UNALIGNED"
            result[0]["alignment_authority"] = False
            return result

        acoustic.align = partial
        result = SuppliedTextAligner(acoustic).align(
            "unused.wav",
            [{"start": 0, "end": 1, "text": "one two"}],
        )
        self.assertEqual(result["alignment"]["status"], "PARTIAL")
        self.assertEqual(result["alignment"]["fallback_words"], 1)
        self.assertEqual(
            result["alignment"]["supplied_text_admission"]["status"],
            "REJECTED",
        )

    def test_rejects_acoustically_unsupported_command_text(self):
        acoustic = _RecordingAligner()
        original_align = acoustic.align

        def weak_command(_audio_path, words, language_code="en"):
            result = original_align(_audio_path, words, language_code)
            result[0]["alignment_score_mean"] = -3.0
            return result

        acoustic.align = weak_command
        result = SuppliedTextAligner(acoustic).align(
            "unused.wav",
            [{"start": 0, "end": 1, "text": "restart now"}],
        )
        admission = result["alignment"]["supplied_text_admission"]
        self.assertEqual(admission["status"], "REJECTED")
        self.assertIn(
            "LOW_COMMAND_ACOUSTIC_SCORE",
            admission["violations"][0]["reason_codes"],
        )


if __name__ == "__main__":
    unittest.main()
