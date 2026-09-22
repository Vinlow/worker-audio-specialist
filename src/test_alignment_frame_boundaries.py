"""Pin real torchaudio half-open spans at EOF and adjacent speech boundaries."""
import unittest

import torch
from torchaudio.functional import merge_tokens

from aligner import Wav2Vec2Aligner
from alignment_window_stitcher import AlignmentWindowCandidate, AlignmentWindowStitcher


class AlignmentFrameBoundariesTest(unittest.TestCase):
    def envelope(self, frames, ordinal):
        tokens = torch.tensor(frames)
        spans = merge_tokens(tokens, torch.ones(len(frames)))
        span = spans[ordinal]
        return span, Wav2Vec2Aligner._acoustic_frame_envelope(tokens, span.start, span.end)

    def test_word_on_last_frame_ends_at_eof_without_an_extra_frame(self):
        span, envelope = self.envelope([0, 0, 2, 2], 0)
        self.assertEqual((span.start, span.end), (2, 4))
        self.assertEqual(envelope, (0, 4))

    def test_adjacent_next_word_is_not_included_or_skipped(self):
        _, envelope = self.envelope([0, 2, 2, 3, 0, 0], 0)
        self.assertEqual(envelope, (0, 3))

    def test_all_trailing_blanks_belong_to_the_measured_envelope(self):
        _, envelope = self.envelope([2, 0, 0, 3], 0)
        self.assertEqual(envelope, (0, 3))
        _, eof_envelope = self.envelope([2, 0, 0], 0)
        self.assertEqual(eof_envelope, (0, 3))

    def test_invalid_span_is_rejected_without_clamping(self):
        for start, end in [(-1, 1), (1, 1), (1, 5)]:
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                Wav2Vec2Aligner._acoustic_frame_envelope([0, 2, 0], start, end)

    def test_real_28_second_clock_keeps_eof_inside_strict_stitcher(self):
        # Both settled Whisper reads failed on this measured 1399-frame clock.
        self.assertGreater(28 / 1399 * 1399, 28)
        clock = lambda frame: Wav2Vec2Aligner._frame_time(frame, 1399, 0, 28)
        candidate = AlignmentWindowCandidate({
            "word": "week.", "start": clock(1334), "end": clock(1343),
            "onset_start": clock(1323), "offset_end": clock(1399),
            "alignment_authority": True,
        }, 0, 0, 28)
        actual = AlignmentWindowStitcher.stitch([[candidate]], [None])
        self.assertEqual(actual[0]["offset_end"], 28)

    def test_frame_clock_preserves_absolute_endpoints_and_order(self):
        for start, end, count in [(0, 28, 1399), (55, 115.115, 3004), (110, 115.115, 255)]:
            with self.subTest(start=start, end=end, count=count):
                values = [Wav2Vec2Aligner._frame_time(frame, count, start, end)
                          for frame in range(count + 1)]
                self.assertEqual(values[0], start)
                self.assertEqual(values[-1], end)
                self.assertTrue(all(left < right for left, right in zip(values, values[1:])))
                self.assertTrue(all(start <= value <= end for value in values))

    def test_invalid_frame_clock_cannot_be_clamped_into_the_source(self):
        for frame, count, start, end in [
            (-1, 1399, 0, 28), (1400, 1399, 0, 28), (1, 0, 0, 28),
            (1.5, 1399, 0, 28), (True, 1399, 0, 28), (1, True, 0, 28),
            (1, 1399, 28, 0), (1, 1399, 0, float("inf")),
            (1, 1399, float("nan"), 28), (1, 1399, -1, 28),
        ]:
            with self.subTest(frame=frame, count=count, start=start, end=end), self.assertRaises(ValueError):
                Wav2Vec2Aligner._frame_time(frame, count, start, end)


if __name__ == "__main__":
    unittest.main()
