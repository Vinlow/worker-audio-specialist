from __future__ import annotations

import inspect
import json
from pathlib import Path
import stat
import sys
import tempfile
import unittest


MODULE_ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(MODULE_ROOT))
sys.path.insert(0, str(TEST_ROOT))

from active_speaker_contracts import (  # noqa: E402
    ContractViolation,
    FaceBox,
    content_identity,
    sha256_bytes,
)
from active_speaker_media import TrackGeometry  # noqa: E402
from active_speaker_media_v2 import (  # noqa: E402
    SuppliedTrackMediaProcessor,
    edge_padded_median_13,
)
from active_speaker_runtime_v2 import (  # noqa: E402
    MFCC_LOOKAHEAD_SAMPLES,
    RUNTIME_V2_CLOSURE_FILES,
    _parser,
    _preprocessing_policy_v2,
    _reauthenticate_media_inputs,
    _run,
    _runtime_identity_v2,
    _track_audio_samples_v2,
    _validate_authenticated_geometry_lineage,
    _view_policy,
    _write_failure_receipt,
)
from active_speaker_supplied_tracks import (  # noqa: E402
    SuppliedTrackLimits,
    V2_RUNTIME_VERSION,
    load_v1_observation_receipt,
    load_supplied_track_manifest,
)
from v2_test_fixtures import (  # noqa: E402
    bind_manifest_to_v1_receipt,
    build_base_manifest,
    build_mirrored_manifest,
    build_v1_observation_receipt,
    write_manifest,
)

try:  # The host contract suite remains dependency-light; the image runs this gate.
    import cv2
    import numpy
except ImportError:
    cv2 = None
    numpy = None


class FakeSamples:
    ndim = 1

    def __init__(self, values: list[int]) -> None:
        self.values = values
        self.shape = (len(values),)

    def __getitem__(self, key: slice) -> "FakeSamples":
        return FakeSamples(self.values[key])


class FakeNumpy:
    @staticmethod
    def pad(samples: FakeSamples, padding, *, mode: str) -> FakeSamples:
        if mode != "constant" or padding[0] != 0:
            raise AssertionError("unexpected fake pad contract")
        return FakeSamples(samples.values + [0] * padding[1])


def track_geometry(frame_indexes: tuple[int, ...]) -> TrackGeometry:
    count = len(frame_indexes)
    return TrackGeometry(
        track_id="track-a",
        shot_index=0,
        frame_indexes=frame_indexes,
        face_boxes=tuple(FaceBox(1, 1, 2, 2) for _ in range(count)),
        crop_center_x=tuple(1.5 for _ in range(count)),
        crop_center_y=tuple(1.5 for _ in range(count)),
        crop_half_size=tuple(0.5 for _ in range(count)),
        observed_detection_frames=frame_indexes,
    )


class V2RuntimeContractTests(unittest.TestCase):
    def test_cli_is_explicit_supplied_v2_and_has_no_detector_arguments(self) -> None:
        parser = _parser()
        command_action = next(
            action for action in parser._actions if getattr(action, "choices", None)
        )
        self.assertEqual(set(command_action.choices), {"run-supplied-v2", "source-identity-v2"})
        run_parser = command_action.choices["run-supplied-v2"]
        destinations = {action.dest for action in run_parser._actions}
        self.assertIn("supplied_tracks", destinations)
        self.assertIn("supplied_tracks_sha256", destinations)
        self.assertIn("base_observation_result", destinations)
        self.assertIn("base_observation_result_sha256", destinations)
        self.assertIn("lineage_source_tracks", destinations)
        self.assertNotIn("yunet", destinations)
        self.assertNotIn("yunet_sha256", destinations)
        self.assertNotIn("face_score_threshold", destinations)
        self.assertNotIn("minimum_track_iou", destinations)
        required = {
            action.dest
            for action in run_parser._actions
            if getattr(action, "required", False)
        }
        self.assertIn("base_observation_result", required)
        self.assertIn("base_observation_result_sha256", required)

    def test_base_and_derived_runs_close_geometry_through_same_v1_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            initial_base = build_base_manifest()
            receipt_json = build_v1_observation_receipt(initial_base)
            receipt_path = root / "v1-result.json"
            receipt_sha = write_manifest(receipt_path, receipt_json)
            base_json = bind_manifest_to_v1_receipt(
                initial_base,
                receipt_json,
                receipt_sha256=receipt_sha,
                receipt_bytes=receipt_path.stat().st_size,
            )
            base_path = root / "base.json"
            base_sha = write_manifest(base_path, base_json)
            base = load_supplied_track_manifest(
                base_path,
                expected_sha256=base_sha,
                limits=SuppliedTrackLimits(),
            )
            observation = load_v1_observation_receipt(
                receipt_path,
                expected_sha256=receipt_sha,
                maximum_bytes=32 * 1024 * 1024,
            )
            self.assertIs(
                _validate_authenticated_geometry_lineage(base, None, observation),
                base,
            )

            mirror_json = build_mirrored_manifest(
                base_json,
                source_file_sha256=base_sha,
            )
            mirror_path = root / "mirror.json"
            mirror_sha = write_manifest(mirror_path, mirror_json)
            mirror = load_supplied_track_manifest(
                mirror_path,
                expected_sha256=mirror_sha,
                limits=SuppliedTrackLimits(),
            )
            self.assertIs(
                _validate_authenticated_geometry_lineage(
                    mirror,
                    base,
                    observation,
                ),
                base,
            )

    def test_runtime_reauthenticates_base_receipt_and_lineage_after_inference(self) -> None:
        source = inspect.getsource(_run)
        inference = source.index('"lrasdTwoViewInference"')
        final_observation = source.index('"finalBaseObservationAuthentication"')
        final_lineage = source.rindex("_validate_authenticated_geometry_lineage(")
        receipt_record = source.index('supplied_track_record["baseObservation"]')
        self.assertLess(inference, final_observation)
        self.assertLess(final_observation, final_lineage)
        self.assertLess(final_lineage, receipt_record)

    def test_runtime_source_has_no_detector_or_tracker_execution(self) -> None:
        source = inspect.getsource(_run)
        self.assertNotIn("detect_faces_and_shots", source)
        self.assertNotIn("DeterministicShotTracker", source)
        processor = object.__new__(SuppliedTrackMediaProcessor)
        with self.assertRaisesRegex(ContractViolation, "forbidden in v2"):
            processor.detect_faces_and_shots()

    def test_view_and_preprocessing_policies_bind_exact_two_view_semantics(self) -> None:
        view_policy = _view_policy()
        self.assertEqual(view_policy["views"], ["CANONICAL", "HORIZONTAL_MIRROR"])
        self.assertEqual(view_policy["mirrorAxis"], "width")
        self.assertIn("math-fsum", view_policy["viewAggregation"])
        preprocessing = _preprocessing_policy_v2()
        self.assertEqual(
            preprocessing["faceInput"]["geometrySource"],
            "authenticated-dense-supplied-track-manifest-v2",
        )
        self.assertNotIn("detector", preprocessing["faceInput"])
        self.assertEqual(
            preprocessing["faceInput"]["horizontalBoundsRounding"],
            "floor-left-ceil-right-mirror-equivariant-v1",
        )
        self.assertEqual(
            preprocessing["audio"]["trackMfccLookaheadSamples"],
            MFCC_LOOKAHEAD_SAMPLES,
        )

    def test_edge_smoothing_is_exactly_horizontal_mirror_equivariant_at_boundaries(self) -> None:
        width = 100.0
        source = (7.0, 40.0, 11.0, 19.0, 25.0, 31.0, 37.0, 43.0, 49.0, 55.0, 61.0, 67.0, 73.0)
        mirrored = tuple(width - value for value in source)
        smoothed_source = edge_padded_median_13(source)
        smoothed_mirror = edge_padded_median_13(mirrored)
        self.assertEqual(
            smoothed_mirror,
            tuple(width - value for value in smoothed_source),
        )
        self.assertEqual(smoothed_source[0], 7.0)
        self.assertEqual(smoothed_source[-1], 73.0)

    @unittest.skipUnless(numpy is not None and cv2 is not None, "requires pinned image media dependencies")
    def test_transformed_boxes_produce_exact_mirrored_crops_at_track_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base_json = build_base_manifest(frame_count=13)
            base_json.pop("contentIdentity")
            for frame in base_json["tracks"][0]["frames"]:
                frame["faceBox"]["x1"] += 0.2
                frame["faceBox"]["x2"] += 0.3
            base_json["contentIdentity"] = content_identity(base_json)
            base_path = root / "base.json"
            base_hash = write_manifest(base_path, base_json)
            base = load_supplied_track_manifest(
                base_path,
                expected_sha256=base_hash,
                limits=SuppliedTrackLimits(),
            )
            mirror_json = build_mirrored_manifest(base_json, source_file_sha256=base_hash)
            mirror_path = root / "mirror.json"
            mirror_hash = write_manifest(mirror_path, mirror_json)
            mirror = load_supplied_track_manifest(
                mirror_path,
                expected_sha256=mirror_hash,
                limits=SuppliedTrackLimits(),
            )
            processor = object.__new__(SuppliedTrackMediaProcessor)
            processor.numpy = numpy
            processor.cv2 = cv2
            source_geometry = processor.geometry_from_manifest(base)[0]
            mirror_geometry = processor.geometry_from_manifest(mirror)[0]
            pixels = (
                numpy.arange(60 * 100 * 3, dtype=numpy.uint32) % 251
            ).astype(numpy.uint8).reshape((60, 100, 3))
            mirrored_pixels = numpy.ascontiguousarray(pixels[:, ::-1, :])
            for local_index in (0, 6, 12):
                with self.subTest(local_index=local_index):
                    source_crop = processor._official_face_crop(
                        pixels, source_geometry, local_index
                    )
                    mirror_crop = processor._official_face_crop(
                        mirrored_pixels, mirror_geometry, local_index
                    )
                    numpy.testing.assert_array_equal(
                        mirror_crop,
                        numpy.ascontiguousarray(source_crop[:, ::-1]),
                    )

    def test_track_audio_has_exact_lookahead_and_only_zero_pads_at_eof(self) -> None:
        geometry = track_geometry((1, 2))
        samples = FakeSamples(list(range(4 * 640)))
        available = _track_audio_samples_v2(samples, geometry, FakeNumpy())
        self.assertEqual(len(available.values), 2 * 640 + MFCC_LOOKAHEAD_SAMPLES)
        self.assertEqual(available.values[0], 640)
        self.assertEqual(available.values[-1], 3 * 640 + MFCC_LOOKAHEAD_SAMPLES - 1)

        eof_samples = FakeSamples(list(range(3 * 640)))
        padded = _track_audio_samples_v2(eof_samples, geometry, FakeNumpy())
        self.assertEqual(len(padded.values), 2 * 640 + MFCC_LOOKAHEAD_SAMPLES)
        self.assertEqual(padded.values[-MFCC_LOOKAHEAD_SAMPLES:], [0] * MFCC_LOOKAHEAD_SAMPLES)

    def test_media_inputs_are_reauthenticated_after_model_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared = root / "prepared.mp4"
            source = root / "source.mp4"
            prepared.write_bytes(b"prepared")
            source.write_bytes(b"source")
            _reauthenticate_media_inputs(
                input_video=prepared,
                input_sha256=sha256_bytes(b"prepared"),
                input_bytes=8,
                source_video=source,
                source_sha256=sha256_bytes(b"source"),
                source_bytes=6,
            )
            prepared.write_bytes(b"tampered")
            with self.assertRaisesRegex(ContractViolation, "SHA-256 mismatch"):
                _reauthenticate_media_inputs(
                    input_video=prepared,
                    input_sha256=sha256_bytes(b"prepared"),
                    input_bytes=8,
                    source_video=source,
                    source_sha256=sha256_bytes(b"source"),
                    source_bytes=6,
                )

    def test_v2_runtime_identity_binds_only_declared_additive_closure(self) -> None:
        first, manifest = _runtime_identity_v2(f"sha256:{'a' * 64}")
        second, second_manifest = _runtime_identity_v2(f"sha256:{'b' * 64}")
        self.assertNotEqual(first, second)
        self.assertEqual(manifest, second_manifest)
        self.assertEqual({item["path"] for item in manifest}, set(RUNTIME_V2_CLOSURE_FILES))
        for required in (
            "Dockerfile.v2",
            "active_speaker_runtime_v2.py",
            "active_speaker_supplied_tracks.py",
            "active_speaker_model_v2.py",
        ):
            self.assertIn(required, {item["path"] for item in manifest})
        self.assertNotIn("active_speaker_runtime.py", RUNTIME_V2_CLOSURE_FILES)

    def test_v2_failure_receipt_is_system_failure_atomic_and_no_clobber(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_failure_receipt(root, ValueError("broken"))
            path = root / "failure.json"
            receipt = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(receipt["reasonClass"], "SYSTEM_FAILURE")
            self.assertEqual(receipt["runtimeVersion"], V2_RUNTIME_VERSION)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o400)
            with self.assertRaisesRegex(ContractViolation, "no-clobber"):
                _write_failure_receipt(root, RuntimeError("later"))

    def test_v2_dockerfile_is_root_owned_read_only_offline_and_nonroot(self) -> None:
        dockerfile = (MODULE_ROOT / "Dockerfile.v2").read_text(encoding="utf-8")
        self.assertIn("--chown=0:0 --chmod=0444", dockerfile)
        self.assertIn("PYTHONDONTWRITEBYTECODE=1", dockerfile)
        self.assertIn("HF_HUB_OFFLINE=1", dockerfile)
        self.assertIn("USER 65532:65532", dockerfile)
        self.assertIn("active_speaker_runtime_v2.py", dockerfile)
        self.assertIn(
            'org.web2labs.starforge.active-speaker.contract="v2-authenticated-v1-lineage"',
            dockerfile,
        )
        self.assertNotIn("yunet", dockerfile.lower())


if __name__ == "__main__":
    unittest.main()
