import copy
import json
from pathlib import Path
import unittest

from alignment_window_stitcher import (
    AlignmentWindowCandidate, AlignmentWindowCoverage, AlignmentWindowStitcher,
)


def candidate(text, start, end, onset, offset, window=0, bounds=(0, 60)):
    return AlignmentWindowCandidate(
        {"word": text, "start": start, "end": end, "onset_start": onset,
         "offset_end": offset, "alignment_authority": True,
         "alignment_status": "ALIGNED_SUPPORTED", "probability": 0.91},
        window, *bounds,
    )


class AlignmentWindowStitcherTest(unittest.TestCase):
    def test_measured_german_quiet_seam_requires_joint_context(self):
        fixture = json.loads((Path(__file__).parent / 'test_fixtures' /
                              'alignment-german-quiet-seam.json').read_text(encoding='utf8'))
        baseline = [[AlignmentWindowCandidate(**item) for item in row]
                    for row in fixture['baseline']]
        with self.assertRaisesRegex(ValueError, 'No coherent acoustic window join at word 2'):
            AlignmentWindowStitcher.stitch(baseline, [None] * len(baseline))
        choices = [[AlignmentWindowCandidate(**item) for item in row]
                   for row in fixture['jointContext']]
        before = copy.deepcopy(choices)
        result = AlignmentWindowStitcher.stitch(choices, [None] * len(choices))
        self.assertEqual(result, [row[index].word for row, index in
                                  zip(choices, fixture['expectedChoice'])])
        self.assertEqual(choices, before)
        self.assertEqual([word['word'] for word in result],
                         [row[0].word['word'] for row in baseline])

    def test_quiet_minecraft_seam_gets_joint_context_without_global_widening(self):
        words = [(2735, 1369.14, 1370.04), (2736, 1386.58, 1387.1)]
        original = AlignmentWindowCoverage._regular_windows(1603.833333, 60, 7)
        planned = AlignmentWindowCoverage.plan_windows(words, 1603.833333, 60, 7, 0.5)
        added = [window for window in planned if window not in original]
        self.assertEqual(len(added), 1)
        self.assertAlmostEqual(added[0][0], 1348.12)
        self.assertAlmostEqual(added[0][1], 1408.12)
        self.assertTrue(all(window in planned for window in original))
        self.assertTrue(AlignmentWindowCoverage._contains_pair(added[0], 1369.14, 1387.1, 0.5))
        self.assertEqual(planned, sorted(planned))

    def test_joint_context_does_not_add_work_for_covered_speech_or_long_silence(self):
        for words in ([(0, 10, 11), (1, 12, 13)],
                      [(0, 10, 11), (1, 100, 101)]):
            self.assertEqual(
                AlignmentWindowCoverage.plan_windows(words, 120, 60, 5, 0.5),
                AlignmentWindowCoverage._regular_windows(120, 60, 5),
            )

    def test_joint_context_can_reuse_a_bridge_for_multiple_neighboring_words(self):
        words = [(0, 44, 45), (1, 61, 62), (2, 64, 65)]
        planned = AlignmentWindowCoverage.plan_windows(words, 113, 60, 7, 0.5)
        original = AlignmentWindowCoverage._regular_windows(113, 60, 7)
        self.assertEqual(len(planned), len(original) + 1)
        for left, right in zip(words, words[1:]):
            self.assertTrue(any(AlignmentWindowCoverage._contains_pair(
                window, left[1], right[2], 0.5) for window in planned))

    def test_overlap_covers_long_native_words_before_gpu_work(self):
        # Source-backed native timing from the 97-minute Minecraft regression.
        words = [(4490, 1980.32, 1984.56), (6698, 3132.20, 3143.12)]
        self.assertEqual(
            AlignmentWindowCoverage.first_uncovered_word(
                words, 5843.32, 60, 5, 0.5,
            ),
            4490,
        )
        self.assertEqual(
            AlignmentWindowCoverage.select_overlap(
                words, 5843.32, 60, 5, 0.5,
            ),
            12.0,
        )

    def test_uncoverable_word_fails_before_measured_timing_is_claimed(self):
        with self.assertRaisesRegex(ValueError, "No complete-context acoustic window for word 9"):
            AlignmentWindowCoverage.select_overlap(
                [(9, 20, 80)], 120, 60, 5, 0.5,
            )

    def test_real_model_overlap_candidates_repair_both_retained_seams(self):
        fixture = json.loads((Path(__file__).parent / "test_fixtures" /
                              "alignment-window-seams.json").read_text(encoding="utf-8-sig"))
        for case in fixture["cases"]:
            with self.subTest(case=case["name"]):
                choices = [[AlignmentWindowCandidate(**item) for item in row]
                           for row in case["candidates"]]
                # First-window ownership reproduces the actual rejected join.
                with self.assertRaisesRegex(ValueError, "No coherent acoustic window join"):
                    AlignmentWindowStitcher.stitch([[row[0]] for row in choices], [None] * len(choices))
                result = AlignmentWindowStitcher.stitch(choices, [None] * len(choices))
                self.assertEqual(result, [row[index].word for row, index in
                                          zip(choices, case["expectedChoice"])])

    def test_chooses_complete_path_instead_of_independent_best_words(self):
        # Locally best first-word candidate cannot join the second word.
        choices = [
            [candidate("one", 25, 26, 24, 27),
             candidate("one", 21, 22, 20, 23, 1, (10, 70))],
            [candidate("two", 23, 24, 22, 25, 1, (10, 70))],
        ]
        before = copy.deepcopy(choices)
        result = AlignmentWindowStitcher.stitch(choices, [None, None])
        self.assertEqual(result, [choices[0][1].word, choices[1][0].word])
        self.assertEqual(choices, before)
        self.assertIsNot(result[0], choices[0][1].word)

    def test_preserves_raw_provider_overlap_as_failure_without_an_alternative(self):
        # Exact retained raw lexical/acoustic boundary from the rejected NLE run.
        choices = [
            [candidate("I've", 444.4398132710904, 444.9399799933311,
                       443.89963321107035, 445, 7, (385, 445))],
            [candidate("actually", 444.5615205068356, 444.8416138712904,
                       444.52150716905635, 444.9616538846282, 8, (440, 500))],
        ]
        before = copy.deepcopy(choices)
        with self.assertRaisesRegex(ValueError, "No coherent acoustic window join at word 1"):
            AlignmentWindowStitcher.stitch(choices, [None, None])
        self.assertEqual(choices, before)

    def test_rejects_retained_backward_onset_even_with_ordered_lexical_times(self):
        choices = [
            [candidate("motivated,", 389.55985328442813, 389.75991997332443,
                       389.55985328442813, 390, 6, (330, 390))],
            [candidate("having", 390.8019339779927, 391.0420140046682,
                       389.54151383794596, 391.12204068022675, 7, (385, 445))],
        ]
        with self.assertRaisesRegex(ValueError, "No coherent acoustic window join"):
            AlignmentWindowStitcher.stitch(choices, [None, None])

    def test_selects_existing_alternative_without_averaging_or_clamping(self):
        choices = [
            [candidate("one", 59, 59.4, 58.8, 60),
             candidate("one", 59.1, 59.3, 58.9, 59.5, 1, (55, 115))],
            [candidate("two", 59.6, 59.8, 59.3, 59.9, 1, (55, 115))],
        ]
        result = AlignmentWindowStitcher.stitch(choices, [None, None])
        self.assertEqual(result, [choices[0][1].word, choices[1][0].word])

    def test_uses_more_context_when_both_paths_are_coherent(self):
        choices = [[candidate("one", 58, 58.4, 57.8, 58.5),
                    candidate("one", 58.1, 58.3, 57.9, 58.6, 1, (55, 115))]]
        self.assertEqual(AlignmentWindowStitcher.stitch(choices, [None]), [choices[0][1].word])

    def test_does_not_switch_back_to_an_earlier_window(self):
        choices = [[candidate("one", 58, 58.4, 57.8, 58.5, 1, (55, 115))],
                   [candidate("two", 59, 59.4, 58.8, 59.5)]]
        with self.assertRaisesRegex(ValueError, "No coherent acoustic window join"):
            AlignmentWindowStitcher.stitch(choices, [None, None])

    def test_keeps_explicit_non_vocabulary_tokens_without_breaking_order_check(self):
        fallback = {"word": "30", "start": 2, "end": 2.4,
                    "alignment_authority": False, "alignment_status": "FALLBACK_UNALIGNED"}
        choices = [[candidate("count", 1, 1.5, 0.8, 1.8)], [],
                   [candidate("times", 3, 3.5, 2.8, 3.8)]]
        result = AlignmentWindowStitcher.stitch(choices, [None, fallback, None])
        self.assertEqual([word["word"] for word in result], ["count", "30", "times"])
        self.assertEqual(result[1], fallback)
        choices[2] = [candidate("times", 1.2, 1.4, 1, 1.6)]
        with self.assertRaisesRegex(ValueError, "No coherent acoustic window join"):
            AlignmentWindowStitcher.stitch(choices, [None, fallback, None])

    def test_missing_candidate_is_not_silently_demoted_to_fallback(self):
        with self.assertRaisesRegex(ValueError, "No complete acoustic candidate"):
            AlignmentWindowStitcher.stitch([[]], [None])
        with self.assertRaisesRegex(ValueError, "Fallback lacks explicit non-authority"):
            AlignmentWindowStitcher.stitch([[]], [{"word": "word", "alignment_authority": True}])

    def test_invalid_geometry_or_changed_text_cannot_enter_a_path(self):
        bad = [candidate("word", 1, 1, 0.8, 1.2),
               candidate("word", 1, 2, -0.1, 2.2),
               candidate("word", 1, 2, 0.8, 61)]
        for item in bad:
            with self.subTest(word=item.word), self.assertRaisesRegex(ValueError, "Invalid acoustic candidate"):
                AlignmentWindowStitcher.stitch([[item]], [None])
        with self.assertRaisesRegex(ValueError, "Invalid acoustic candidate"):
            AlignmentWindowStitcher.stitch([[candidate("one", 1, 2, 0.8, 2.2),
                                             candidate("two", 1, 2, 0.8, 2.2)]], [None])


if __name__ == "__main__":
    unittest.main()
