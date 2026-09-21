"""Pin real torchaudio half-open spans at EOF and adjacent speech boundaries."""
import unittest

import torch
from torchaudio.functional import merge_tokens

from aligner import Wav2Vec2Aligner


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


if __name__ == "__main__":
    unittest.main()
