from __future__ import annotations

import math
from pathlib import Path
import sys
import tempfile
import unittest


MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))

from active_speaker_contracts import (  # noqa: E402
    AUTHORITY,
    CROP_AUTHORITY,
    ContractViolation,
    DeterministicShotTracker,
    FaceBox,
    FaceDetection,
    LRASD_EXECUTED_SOURCE_FILES,
    RawScoreLedger,
    SourceInterval,
    assert_no_decision_authority,
    box_iou,
    content_identity,
    lrasd_source_identity,
    success_envelope,
    validate_content_identity,
    validate_git_revision,
    validate_sha256,
)


class HashContractTests(unittest.TestCase):
    def test_hashes_and_revisions_require_exact_lowercase_width(self) -> None:
        self.assertEqual(validate_sha256("a" * 64, "fixture"), "a" * 64)
        self.assertEqual(validate_git_revision("b" * 40), "b" * 40)
        for invalid in ("a" * 63, "A" * 64, "g" * 64):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ContractViolation):
                    validate_sha256(invalid, "fixture")
        with self.assertRaises(ContractViolation):
            validate_git_revision("b" * 39)
        self.assertEqual(
            validate_content_identity(f"sha256:{'c' * 64}", "fixture"),
            f"sha256:{'c' * 64}",
        )
        for invalid_identity in (
            f"sha256:{'c' * 63}",
            f"sha256:{'C' * 64}",
            "sha512:" + "c" * 64,
            "sha256:anything",
        ):
            with self.subTest(invalid_identity=invalid_identity):
                with self.assertRaises(ContractViolation):
                    validate_content_identity(invalid_identity, "fixture")

    def test_source_interval_is_exact_and_positive(self) -> None:
        interval = SourceInterval("d" * 64, 500_000, 1_500_000)
        self.assertEqual(interval.duration_microseconds, 1_000_000)
        self.assertEqual(
            interval.as_json(),
            {
                "endMicrosecondsExclusive": 1_500_000,
                "originalSourceVideoSha256": "d" * 64,
                "startMicrosecondsInclusive": 500_000,
            },
        )
        for start, end in ((-1, 1), (1, 1), (2, 1)):
            with self.subTest(start=start, end=end):
                with self.assertRaises(ContractViolation):
                    SourceInterval("d" * 64, start, end)

    def test_source_identity_is_ordered_and_content_sensitive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index, relative_path in enumerate(LRASD_EXECUTED_SOURCE_FILES):
                path = root / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"source-{index}\n".encode())
            first_identity, first_manifest = lrasd_source_identity(root)
            second_identity, second_manifest = lrasd_source_identity(root)
            self.assertEqual(first_identity, second_identity)
            self.assertEqual(first_manifest, second_manifest)
            self.assertEqual(
                [item["path"] for item in first_manifest],
                list(LRASD_EXECUTED_SOURCE_FILES),
            )
            (root / LRASD_EXECUTED_SOURCE_FILES[-1]).write_bytes(b"changed\n")
            changed_identity, _ = lrasd_source_identity(root)
            self.assertNotEqual(first_identity, changed_identity)

    def test_source_identity_rejects_symlinked_executed_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target.py"
            target.write_text("pass\n", encoding="utf-8")
            for relative_path in LRASD_EXECUTED_SOURCE_FILES:
                path = root / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("pass\n", encoding="utf-8")
            symlink = root / LRASD_EXECUTED_SOURCE_FILES[0]
            symlink.unlink()
            symlink.symlink_to(target)
            with self.assertRaises(ContractViolation):
                lrasd_source_identity(root)

    def test_content_identity_is_canonical(self) -> None:
        self.assertEqual(
            content_identity({"a": 1, "b": 2}),
            content_identity({"b": 2, "a": 1}),
        )


class TrackingTests(unittest.TestCase):
    @staticmethod
    def detection(
        frame: int,
        shot: int,
        x: float,
        *,
        score: float = 0.99,
    ) -> FaceDetection:
        return FaceDetection(
            frame_index=frame,
            shot_index=shot,
            box=FaceBox(x, 0, x + 10, 10),
            detection_score=score,
        )

    def test_box_iou(self) -> None:
        self.assertEqual(box_iou(FaceBox(0, 0, 10, 10), FaceBox(0, 0, 10, 10)), 1)
        self.assertEqual(box_iou(FaceBox(0, 0, 10, 10), FaceBox(20, 0, 30, 10)), 0)
        self.assertAlmostEqual(
            box_iou(FaceBox(0, 0, 10, 10), FaceBox(5, 0, 15, 10)),
            1 / 3,
        )

    def test_tracking_is_deterministic_across_detection_input_order(self) -> None:
        detections = [
            self.detection(0, 0, 0),
            self.detection(0, 0, 50),
            self.detection(1, 0, 1),
            self.detection(1, 0, 51),
        ]
        tracker = DeterministicShotTracker(minimum_detection_frames=1)
        forward = tracker.track(detections)
        reverse = tracker.track(reversed(detections))
        self.assertEqual(forward, reverse)
        self.assertEqual([track.track_id for track in forward], [
            "shot-0000-track-0000",
            "shot-0000-track-0001",
        ])

    def test_track_identity_resets_at_every_shot(self) -> None:
        tracker = DeterministicShotTracker(minimum_detection_frames=1)
        tracks = tracker.track(
            [
                self.detection(0, 0, 0),
                self.detection(1, 0, 0),
                self.detection(2, 1, 0),
                self.detection(3, 1, 0),
            ]
        )
        self.assertEqual(
            [track.track_id for track in tracks],
            ["shot-0000-track-0000", "shot-0001-track-0000"],
        )
        self.assertEqual([track.shot_index for track in tracks], [0, 1])

    def test_track_does_not_bridge_a_gap_beyond_the_wall(self) -> None:
        tracker = DeterministicShotTracker(
            maximum_gap_frames=1,
            minimum_detection_frames=1,
        )
        tracks = tracker.track(
            [self.detection(0, 0, 0), self.detection(2, 0, 0)]
        )
        self.assertEqual(len(tracks), 2)

    def test_short_tracks_are_not_admitted(self) -> None:
        tracker = DeterministicShotTracker(minimum_detection_frames=2)
        self.assertEqual(tracker.track([self.detection(0, 0, 0)]), ())

    def test_invalid_box_and_nonfinite_detection_fail_closed(self) -> None:
        with self.assertRaises(ContractViolation):
            FaceBox(0, 0, 0, 10)
        with self.assertRaises(ContractViolation):
            self.detection(0, 0, 0, score=math.nan)


class RawModelOutputContractTests(unittest.TestCase):
    def test_ledger_names_only_raw_speaking_score_and_exact_pts(self) -> None:
        ledger = RawScoreLedger.build({"track-a": [3, 4]}, {"track-a": [-0.2, 1.4]})
        self.assertEqual(
            ledger,
            [
                {
                    "samples": [
                        {
                            "frameIndex": 3,
                            "pts": {"denominator": 25, "numerator": 3},
                            "rawSpeakingScore": -0.2,
                        },
                        {
                            "frameIndex": 4,
                            "pts": {"denominator": 25, "numerator": 4},
                            "rawSpeakingScore": 1.4,
                        },
                    ],
                    "trackId": "track-a",
                }
            ],
        )
        assert_no_decision_authority(ledger)

    def test_missing_short_and_nonfinite_scores_fail_closed(self) -> None:
        fixtures = [
            ({"a": [0]}, {}, ContractViolation),
            ({"a": [0, 1]}, {"a": [0.1]}, ContractViolation),
            ({"a": [0]}, {"a": [math.nan]}, ContractViolation),
            ({"a": [1, 0]}, {"a": [0.1, 0.2]}, ContractViolation),
        ]
        for frames, scores, expected in fixtures:
            with self.subTest(frames=frames, scores=scores):
                with self.assertRaises(expected):
                    RawScoreLedger.build(frames, scores)

    def test_decision_and_probability_fields_are_forbidden(self) -> None:
        for key in (
            "probability",
            "speaker_probability",
            "speakingConfidence",
            "confidence_score",
            "isSpeaking",
            "speakingScore",
            "speakerDecision",
            "selected_track",
            "cropBox",
            "activeSpeaker",
        ):
            with self.subTest(key=key):
                with self.assertRaises(ContractViolation):
                    assert_no_decision_authority({key: None})

    def test_success_envelope_remains_diagnostic_only(self) -> None:
        identities = {
            "clockIdentity": f"sha256:{'a' * 64}",
            "modelIdentity": f"sha256:{'b' * 64}",
            "observationIdentity": f"sha256:{'c' * 64}",
            "runIdentity": f"sha256:{'e' * 64}",
            "runtimeIdentity": f"sha256:{'d' * 64}",
        }
        result = success_envelope(
            identities=identities,
            clocks={"video": {"frameCount": 1}},
            tracks=[],
            score_ledger=[],
            outputs={},
            measurements={},
        )
        self.assertEqual(result["authority"], AUTHORITY)
        self.assertEqual(result["cropAuthority"], CROP_AUTHORITY)
        self.assertIn("not a probability", result["rawScoreSemantics"])
        assert_no_decision_authority(result)

    def test_success_envelope_requires_closed_identity_set(self) -> None:
        with self.assertRaises(ContractViolation):
            success_envelope(
                identities={"modelIdentity": "sha256:a"},
                clocks={},
                tracks=[],
                score_ledger=[],
                outputs={},
                measurements={},
            )

    def test_success_envelope_rejects_truncated_content_identity(self) -> None:
        with self.assertRaises(ContractViolation):
            success_envelope(
                identities={
                    "clockIdentity": "sha256:a",
                    "modelIdentity": f"sha256:{'b' * 64}",
                    "observationIdentity": f"sha256:{'c' * 64}",
                    "runIdentity": f"sha256:{'e' * 64}",
                    "runtimeIdentity": f"sha256:{'d' * 64}",
                },
                clocks={},
                tracks=[],
                score_ledger=[],
                outputs={},
                measurements={},
            )


if __name__ == "__main__":
    unittest.main()
