import unittest

from parakeet_transcriber import (
    extract_model_language,
    normalize_language_code,
    strip_control_tokens,
    supports_language,
    token_timestamps_to_words,
    words_to_directional_segments,
)


class _FakeTokenizer:
    def __init__(self, tokens):
        self.tokens = tokens

    def convert_ids_to_tokens(self, token_ids):
        return [self.tokens[token_id] for token_id in token_ids]


class ParakeetTranscriberContractTest(unittest.TestCase):
    def test_language_support_is_explicit_and_primary_subtag_aware(self):
        self.assertEqual(normalize_language_code("de-DE"), "de")
        self.assertTrue(supports_language("de-DE"))
        self.assertTrue(supports_language("uk"))
        self.assertFalse(supports_language("hi"))
        self.assertFalse(supports_language(None))

    def test_extracts_first_supported_model_language_control_token(self):
        tokenizer = _FakeTokenizer(
            {
                1: "<|startoftranscript|>",
                2: "<|de|>",
                3: " Hallo",
                4: "<|en|>",
            }
        )

        language, observed = extract_model_language(
            [1, 2, 3, 4],
            tokenizer,
        )

        self.assertEqual(language, "de")
        self.assertEqual(observed, ["de", "en"])

    def test_control_tokens_never_leak_into_transcript(self):
        self.assertEqual(
            strip_control_tokens(
                "<|startoftranscript|> <|de|> Hallo, Welt!"
            ),
            "Hallo, Welt!",
        )

    def test_coalesces_subwords_and_punctuation_without_fake_probability(self):
        words = token_timestamps_to_words(
            [
                {"token": "<|de|>", "start": 0.0, "end": 0.0},
                {"token": "Hal", "start": 0.1, "end": 0.2},
                {"token": "lo", "start": 0.2, "end": 0.3},
                {"token": ",", "start": 0.3, "end": 0.3},
                {"token": " Welt", "start": 0.5, "end": 0.8},
                {"token": "!", "start": 0.8, "end": 0.8},
            ]
        )

        self.assertEqual(
            words,
            [
                {
                    "word": "Hallo,",
                    "start": 0.1,
                    "end": 0.3,
                    "probability": None,
                    "timestamp_source": "PARAKEET_NATIVE_TOKEN_DURATION",
                    "timestamp_authority": "DIRECTIONAL_NOT_NP_SBV2",
                },
                {
                    "word": " Welt!",
                    "start": 0.5,
                    "end": 0.8,
                    "probability": None,
                    "timestamp_source": "PARAKEET_NATIVE_TOKEN_DURATION",
                    "timestamp_authority": "DIRECTIONAL_NOT_NP_SBV2",
                },
            ],
        )

    def test_builds_sentence_segments_from_directional_words(self):
        words = token_timestamps_to_words(
            [
                {"token": "Hi", "start": 0.1, "end": 0.2},
                {"token": ".", "start": 0.2, "end": 0.2},
                {"token": " Next", "start": 0.5, "end": 0.8},
                {"token": "!", "start": 0.8, "end": 0.8},
            ]
        )

        segments = words_to_directional_segments(words, "", 1.0)

        self.assertEqual([segment["text"] for segment in segments], ["Hi.", "Next!"])
        self.assertEqual(
            [segment["timestamp_authority"] for segment in segments],
            [
                "PARAKEET_NATIVE_TOKEN_DURATION_DIRECTIONAL",
                "PARAKEET_NATIVE_TOKEN_DURATION_DIRECTIONAL",
            ],
        )

    def test_text_only_fallback_does_not_claim_word_authority(self):
        segments = words_to_directional_segments([], "Hello.", 2.5)

        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]["start"], 0.0)
        self.assertEqual(segments[0]["end"], 2.5)
        self.assertEqual(
            segments[0]["timestamp_authority"],
            "AUDIO_EXTENT_FALLBACK_NOT_WORD_AUTHORITY",
        )


if __name__ == "__main__":
    unittest.main()
