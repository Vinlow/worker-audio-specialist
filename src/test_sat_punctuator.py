import unittest

from sat_punctuator import (
    SAT_BATCH_REQUEST_SCHEMA_VERSION,
    SAT_BOUNDARY_THRESHOLD,
    SAT_MODEL_ID,
    SAT_MODEL_REVISION,
    SAT_REQUEST_SCHEMA_VERSION,
    SAT_TOKENIZER_ID,
    SAT_TOKENIZER_REVISION,
    normalize_language,
    token_sha256,
    validate_batch_request,
    validate_probe_request,
)


def request_for(
    token_ids=None,
    *,
    terminal_tail=True,
    start_token=64,
    ordinal=8,
):
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
            "startToken": start_token,
            "endTokenExclusive": start_token + len(token_ids),
            "terminalTail": terminal_tail,
        },
        "inputTokenIds": token_ids,
        "inputTokenSha256": token_sha256(token_ids),
        "terminalAnchors": [
            {
                "ordinal": ordinal,
                "terminalTokenIndex": start_token + 1,
            },
        ],
    }


def batch_request_for():
    complete = request_for(
        [17] * 510,
        terminal_tail=False,
        start_token=64,
        ordinal=8,
    )
    tail = request_for(
        [42, 99, 101],
        terminal_tail=True,
        start_token=574,
        ordinal=9,
    )
    return {
        "schemaVersion": SAT_BATCH_REQUEST_SCHEMA_VERSION,
        "sourceFingerprint": complete["sourceFingerprint"],
        "language": complete["language"],
        "candidate": complete["candidate"],
        "windows": [
            {
                key: complete[key]
                for key in (
                    "window",
                    "inputTokenIds",
                    "inputTokenSha256",
                    "terminalAnchors",
                )
            },
            {
                key: tail[key]
                for key in (
                    "window",
                    "inputTokenIds",
                    "inputTokenSha256",
                    "terminalAnchors",
                )
            },
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

    def test_batch_accepts_complete_windows_plus_one_final_tail(self):
        normalized = validate_batch_request(batch_request_for())
        self.assertEqual(2, len(normalized["windows"]))
        self.assertFalse(normalized["windows"][0]["window"]["terminalTail"])
        self.assertTrue(normalized["windows"][1]["window"]["terminalTail"])

    def test_batch_rejects_more_than_padded_capacity(self):
        request = batch_request_for()
        request["windows"] = request["windows"][:1] * 9
        with self.assertRaisesRegex(ValueError, "between one and 8"):
            validate_batch_request(request)

    def test_batch_rejects_nonfinal_tail(self):
        request = batch_request_for()
        request["windows"].reverse()
        with self.assertRaisesRegex(ValueError, "tail must be the final"):
            validate_batch_request(request)

    def test_batch_rejects_duplicate_terminal_ordinals(self):
        request = batch_request_for()
        request["windows"][1]["terminalAnchors"][0]["ordinal"] = 8
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_batch_request(request)

    def test_batch_rejects_terminal_ordinal_regression(self):
        request = batch_request_for()
        request["windows"][1]["terminalAnchors"][0]["ordinal"] = 7
        with self.assertRaisesRegex(ValueError, "global source order"):
            validate_batch_request(request)

    def test_batch_window_cannot_override_top_level_source_identity(self):
        request = batch_request_for()
        request["windows"][0]["sourceFingerprint"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "unexpected keys"):
            validate_batch_request(request)

    def test_batch_rejects_nonconsecutive_complete_windows(self):
        request = batch_request_for()
        second = request_for(
            [42] * 510,
            terminal_tail=False,
            start_token=192,
            ordinal=9,
        )
        request["windows"][1] = {
            key: second[key]
            for key in (
                "window",
                "inputTokenIds",
                "inputTokenSha256",
                "terminalAnchors",
            )
        }
        with self.assertRaisesRegex(ValueError, "must be consecutive"):
            validate_batch_request(request)


if __name__ == "__main__":
    unittest.main()
