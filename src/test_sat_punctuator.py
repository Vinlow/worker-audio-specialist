import unittest

from sat_punctuator import (
    SAT_BOUNDARY_THRESHOLD,
    SAT_MODEL_ID,
    SAT_MODEL_REVISION,
    SAT_REQUEST_SCHEMA_VERSION,
    SAT_TOKENIZER_ID,
    SAT_TOKENIZER_REVISION,
    normalize_language,
    token_sha256,
    validate_probe_request,
)


def request_for(token_ids=None, *, terminal_tail=True):
    token_ids = token_ids or [17, 42, 99]
    return {
        "schemaVersion": SAT_REQUEST_SCHEMA_VERSION,
        "sourceFingerprint": "a" * 64,
        "language": "es-ES",
        "candidate": {
            "modelId": SAT_MODEL_ID,
            "modelRevision": SAT_MODEL_REVISION,
            "tokenizerId": SAT_TOKENIZER_ID,
            "tokenizerRevision": SAT_TOKENIZER_REVISION,
            "boundaryThreshold": SAT_BOUNDARY_THRESHOLD,
        },
        "window": {
            "startToken": 64,
            "endTokenExclusive": 64 + len(token_ids),
            "terminalTail": terminal_tail,
        },
        "inputTokenIds": token_ids,
        "inputTokenSha256": token_sha256(token_ids),
        "terminalAnchors": [
            {"ordinal": 8, "terminalTokenIndex": 65},
        ],
    }


class SaTPunctuatorContractTest(unittest.TestCase):
    def test_language_is_primary_subtag_and_launch_gated(self):
        self.assertEqual("es", normalize_language("es-ES"))
        with self.assertRaisesRegex(ValueError, "does not support"):
            normalize_language("ja")

    def test_window_request_binds_candidate_tokens_and_absolute_anchors(self):
        normalized = validate_probe_request(request_for())
        self.assertEqual("es", normalized["language"])
        self.assertEqual(64, normalized["window"]["startToken"])
        self.assertEqual(
            65,
            normalized["terminalAnchors"][0]["terminalTokenIndex"],
        )

    def test_candidate_identity_drift_fails_closed(self):
        request = request_for()
        request["candidate"]["boundaryThreshold"] = 0.25
        with self.assertRaisesRegex(ValueError, "identity drifted"):
            validate_probe_request(request)

    def test_token_hash_and_window_cardinality_fail_closed(self):
        request = request_for()
        request["inputTokenSha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "token identity drifted"):
            validate_probe_request(request)

        request = request_for()
        request["window"]["endTokenExclusive"] += 1
        with self.assertRaisesRegex(ValueError, "cardinality drifted"):
            validate_probe_request(request)

    def test_boolean_is_not_accepted_as_a_token_id(self):
        request = request_for()
        request["inputTokenIds"][1] = True
        with self.assertRaisesRegex(ValueError, "invalid tokenizer id"):
            validate_probe_request(request)

    def test_complete_window_requires_exact_source_grid_geometry(self):
        with self.assertRaisesRegex(ValueError, "510 tokens"):
            validate_probe_request(
                request_for(terminal_tail=False)
            )

    def test_terminal_anchors_must_follow_source_order(self):
        request = request_for()
        request["terminalAnchors"] = [
            {"ordinal": 8, "terminalTokenIndex": 66},
            {"ordinal": 9, "terminalTokenIndex": 65},
        ]
        with self.assertRaisesRegex(ValueError, "anchor is invalid"):
            validate_probe_request(request)


if __name__ == "__main__":
    unittest.main()
