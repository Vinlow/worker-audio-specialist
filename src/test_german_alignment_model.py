"""Lexical authority checks run in the hosted source gate without model deps."""
import unittest
from german_alignment_model import GermanAlignmentModel


class GermanAlignmentLexicalTest(unittest.TestCase):
    def test_umlauts_case_and_sharp_s_keep_all_spoken_letters(self):
        for original, expected in [(" über,", "über"), ("GRÖẞE!", "grösse"),
                                   ("Straße.", "strasse"), ("fu\u0308r", "für"),
                                   ("geht’s", "geht's"), ("Minecraft", "minecraft")]:
            with self.subTest(original=original):
                self.assertEqual(GermanAlignmentModel.normalize_word(original), expected)

    def test_unknown_letter_or_number_cannot_grant_partial_word_authority(self):
        for word in ("2026", "13-jähriger", "résumé", "şaka", "gute²", "---", "''"):
            with self.subTest(word=word):
                self.assertEqual(GermanAlignmentModel.normalize_word(word), "")

    def test_words_have_only_nonblank_known_ctc_tokens(self):
        for word in ("Tür", "Öl", "Straße", "Bau-Projekt"):
            projection = GermanAlignmentModel.normalize_word(word)
            self.assertTrue(projection)
            self.assertTrue(all(GermanAlignmentModel.labels.index(char) > 4
                                for char in projection))


if __name__ == "__main__":
    unittest.main()
