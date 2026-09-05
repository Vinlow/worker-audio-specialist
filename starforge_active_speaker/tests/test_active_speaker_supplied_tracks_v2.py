from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest


MODULE_ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(MODULE_ROOT))
sys.path.insert(0, str(TEST_ROOT))

from active_speaker_contracts import (  # noqa: E402
    ContractViolation,
    canonical_json_bytes,
    content_identity,
    sha256_bytes,
)
from active_speaker_supplied_tracks import (  # noqa: E402
    LRASD_V2_EXECUTED_SOURCE_FILES,
    SUPPLIED_TRACK_SCHEMA_VERSION,
    V2_OBSERVATION_SCHEMA_VERSION,
    SuppliedTrackLimits,
    V2RawScoreLedger,
    load_v1_observation_receipt,
    load_supplied_track_manifest,
    lrasd_v2_source_identity,
    success_envelope_v2,
    validate_base_observation_lineage,
    validate_geometry_lineage,
)
from v2_test_fixtures import (  # noqa: E402
    bind_manifest_to_v1_receipt,
    build_base_manifest,
    build_mirrored_manifest,
    build_v1_observation_receipt,
    write_manifest,
)


class SuppliedTrackManifestTests(unittest.TestCase):
    def load(self, root: Path, manifest: dict[str, object]):
        path = root / "tracks.json"
        file_hash = write_manifest(path, manifest)
        return load_supplied_track_manifest(
            path,
            expected_sha256=file_hash,
            limits=SuppliedTrackLimits(),
        )

    def mutate_and_reseal(self, manifest: dict[str, object], action) -> dict[str, object]:
        result = deepcopy(manifest)
        result.pop("contentIdentity")
        action(result)
        result["contentIdentity"] = content_identity(result)
        return result

    def test_complete_base_manifest_binds_clock_producer_tracks_and_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest = build_base_manifest()
            loaded = self.load(Path(temporary), manifest)
            self.assertEqual(loaded.file_bytes, len(canonical_json_bytes(manifest)) + 1)
            self.assertEqual(loaded.content_identity, manifest["contentIdentity"])
            self.assertEqual(loaded.clock_identity, manifest["clockIdentity"])
            self.assertEqual((loaded.width, loaded.height, loaded.frame_count), (100, 60, 4))
            self.assertEqual(len(loaded.tracks), 1)
            self.assertEqual(loaded.track_frame_count, 4)
            validate_geometry_lineage(loaded, None)

    def test_only_complete_authenticated_empty_manifest_is_valid_no_face(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            empty = build_base_manifest(include_track=False)
            loaded = self.load(root, empty)
            self.assertEqual(loaded.tracks, ())
            incomplete = self.mutate_and_reseal(empty, lambda value: value.update(status="PARTIAL"))
            with self.assertRaisesRegex(ContractViolation, "must be COMPLETE"):
                self.load(root, incomplete)

    def test_file_hash_clock_identity_and_content_identity_each_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = build_base_manifest()
            path = root / "tracks.json"
            file_hash = write_manifest(path, manifest)
            self.assertEqual(file_hash, sha256_bytes(path.read_bytes()))
            with self.assertRaisesRegex(ContractViolation, "SHA-256 mismatch"):
                load_supplied_track_manifest(
                    path,
                    expected_sha256="f" * 64,
                    limits=SuppliedTrackLimits(),
                )
            bad_clock = deepcopy(manifest)
            bad_clock["clockIdentity"] = f"sha256:{'a' * 64}"
            bad_clock["contentIdentity"] = content_identity(
                {key: value for key, value in bad_clock.items() if key != "contentIdentity"}
            )
            with self.assertRaisesRegex(ContractViolation, "clock identity"):
                self.load(root, bad_clock)
            bad_content = deepcopy(manifest)
            bad_content["contentIdentity"] = f"sha256:{'b' * 64}"
            with self.assertRaisesRegex(ContractViolation, "content identity"):
                self.load(root, bad_content)

    def test_duplicate_keys_nonfinite_numbers_and_symlink_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            duplicate = root / "duplicate.json"
            duplicate.write_bytes(b'{"status":"COMPLETE","status":"COMPLETE"}')
            with self.assertRaisesRegex(ContractViolation, "duplicate key"):
                load_supplied_track_manifest(
                    duplicate,
                    expected_sha256=sha256_bytes(duplicate.read_bytes()),
                    limits=SuppliedTrackLimits(),
                )
            nonfinite = root / "nonfinite.json"
            nonfinite.write_bytes(b'{"value":NaN}')
            with self.assertRaisesRegex(ContractViolation, "non-finite"):
                load_supplied_track_manifest(
                    nonfinite,
                    expected_sha256=sha256_bytes(nonfinite.read_bytes()),
                    limits=SuppliedTrackLimits(),
                )
            target = root / "target.json"
            write_manifest(target, build_base_manifest())
            link = root / "link.json"
            link.symlink_to(target)
            with self.assertRaisesRegex(ContractViolation, "non-symlink"):
                load_supplied_track_manifest(
                    link,
                    expected_sha256=sha256_bytes(target.read_bytes()),
                    limits=SuppliedTrackLimits(),
                )

    def test_shot_processed_frame_and_track_coverage_fail_closed(self) -> None:
        mutations = {
            "shots": lambda value: value["clock"]["shots"][0].update(endFrameExclusive=3),
            "processed-frame": lambda value: value["producer"]["processedFrames"].pop(),
            "track-gap": lambda value: value["tracks"][0]["frames"][1].update(frameIndex=2),
            "cross-shot": lambda value: value["tracks"][0].update(shotIndex=1),
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for label, mutation in mutations.items():
                with self.subTest(label=label):
                    fixture = self.mutate_and_reseal(build_base_manifest(), mutation)
                    with self.assertRaises(ContractViolation):
                        self.load(root, fixture)

    def test_box_bounds_duplicate_tracks_and_resource_caps_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            outside = self.mutate_and_reseal(
                build_base_manifest(),
                lambda value: value["tracks"][0]["frames"][0]["faceBox"].update(x2=101),
            )
            with self.assertRaisesRegex(ContractViolation, "outside"):
                self.load(root, outside)
            duplicate = self.mutate_and_reseal(
                build_base_manifest(),
                lambda value: value["tracks"].append(deepcopy(value["tracks"][0])),
            )
            with self.assertRaisesRegex(ContractViolation, "unique"):
                self.load(root, duplicate)
            path = root / "tracks.json"
            file_hash = write_manifest(path, duplicate)
            with self.assertRaisesRegex(ContractViolation, "maximum track count"):
                load_supplied_track_manifest(
                    path,
                    expected_sha256=file_hash,
                    limits=SuppliedTrackLimits(maximum_tracks=1),
                )
            for invalid_limit in (1.5, True, 257):
                with self.subTest(invalid_limit=invalid_limit):
                    with self.assertRaisesRegex(ContractViolation, "maximum tracks"):
                        SuppliedTrackLimits(maximum_tracks=invalid_limit)

    def test_base_and_exact_horizontal_mirror_lineage_are_authenticated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base_json = build_base_manifest()
            base_path = root / "base.json"
            base_hash = write_manifest(base_path, base_json)
            base = load_supplied_track_manifest(
                base_path, expected_sha256=base_hash, limits=SuppliedTrackLimits()
            )
            mirror_json = build_mirrored_manifest(
                base_json,
                source_file_sha256=base_hash,
            )
            mirror_path = root / "mirror.json"
            mirror_hash = write_manifest(mirror_path, mirror_json)
            mirror = load_supplied_track_manifest(
                mirror_path, expected_sha256=mirror_hash, limits=SuppliedTrackLimits()
            )
            validate_geometry_lineage(base, None)
            validate_geometry_lineage(mirror, base)
            self.assertEqual(mirror.tracks[0].frames[0].face_box.x1, 70)
            self.assertEqual(mirror.tracks[0].frames[0].face_box.x2, 90)

    def test_mirror_lineage_rejects_missing_wrong_or_modified_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base_json = build_base_manifest()
            base_path = root / "base.json"
            base_hash = write_manifest(base_path, base_json)
            base = load_supplied_track_manifest(
                base_path, expected_sha256=base_hash, limits=SuppliedTrackLimits()
            )
            mirror_json = build_mirrored_manifest(base_json, source_file_sha256=base_hash)
            mirror_path = root / "mirror.json"
            mirror_hash = write_manifest(mirror_path, mirror_json)
            mirror = load_supplied_track_manifest(
                mirror_path, expected_sha256=mirror_hash, limits=SuppliedTrackLimits()
            )
            with self.assertRaisesRegex(ContractViolation, "requires"):
                validate_geometry_lineage(mirror, None)
            changed_json = self.mutate_and_reseal(
                mirror_json,
                lambda value: value["tracks"][0]["frames"][0]["faceBox"].update(x1=69),
            )
            changed_path = root / "changed.json"
            changed_hash = write_manifest(changed_path, changed_json)
            changed = load_supplied_track_manifest(
                changed_path, expected_sha256=changed_hash, limits=SuppliedTrackLimits()
            )
            with self.assertRaisesRegex(ContractViolation, "exact frozen box transform"):
                validate_geometry_lineage(changed, base)

            for label, mutation, message in (
                (
                    "detector",
                    lambda value: value["producer"]["detector"].update(
                        modelSha256="9" * 64
                    ),
                    "detector provenance",
                ),
                (
                    "tracking-policy",
                    lambda value: value["producer"]["trackingPolicy"].update(
                        maximumGapFrames=14
                    ),
                    "tracking policy",
                ),
            ):
                with self.subTest(label=label):
                    modified_json = self.mutate_and_reseal(mirror_json, mutation)
                    modified_path = root / f"{label}.json"
                    modified_hash = write_manifest(modified_path, modified_json)
                    modified = load_supplied_track_manifest(
                        modified_path,
                        expected_sha256=modified_hash,
                        limits=SuppliedTrackLimits(),
                    )
                    with self.assertRaisesRegex(ContractViolation, message):
                        validate_geometry_lineage(modified, base)


class BaseObservationProvenanceTests(unittest.TestCase):
    def authenticated_base(self, root: Path):
        initial_manifest = build_base_manifest()
        receipt_json = build_v1_observation_receipt(initial_manifest)
        receipt_path = root / "v1-result.json"
        receipt_sha256 = write_manifest(receipt_path, receipt_json)
        manifest_json = bind_manifest_to_v1_receipt(
            initial_manifest,
            receipt_json,
            receipt_sha256=receipt_sha256,
            receipt_bytes=receipt_path.stat().st_size,
        )
        manifest_path = root / "base.json"
        manifest_sha256 = write_manifest(manifest_path, manifest_json)
        observation = load_v1_observation_receipt(
            receipt_path,
            expected_sha256=receipt_sha256,
            maximum_bytes=32 * 1024 * 1024,
        )
        manifest = load_supplied_track_manifest(
            manifest_path,
            expected_sha256=manifest_sha256,
            limits=SuppliedTrackLimits(),
        )
        return manifest_json, manifest, observation, receipt_path, receipt_sha256

    def test_base_manifest_is_exactly_bound_to_authenticated_v1_observation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, manifest, observation, _, _ = self.authenticated_base(root)
            validate_base_observation_lineage(manifest, observation)
            self.assertEqual(
                manifest.producer["geometryLineage"]["sourceObservation"],
                observation.lineage_record(),
            )

    def test_forged_resealed_boxes_fail_against_v1_observation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_json, _, observation, _, _ = self.authenticated_base(root)
            manifest_json.pop("contentIdentity")
            manifest_json["tracks"][0]["frames"][0]["faceBox"]["x1"] += 1
            manifest_json["contentIdentity"] = content_identity(manifest_json)
            forged_path = root / "forged.json"
            forged_hash = write_manifest(forged_path, manifest_json)
            forged = load_supplied_track_manifest(
                forged_path,
                expected_sha256=forged_hash,
                limits=SuppliedTrackLimits(),
            )
            with self.assertRaisesRegex(ContractViolation, "geometry differs"):
                validate_base_observation_lineage(forged, observation)

    def test_receipt_hash_and_recomputed_identity_mismatch_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, _, receipt_path, receipt_sha256 = self.authenticated_base(root)
            with self.assertRaisesRegex(ContractViolation, "SHA-256 mismatch"):
                load_v1_observation_receipt(
                    receipt_path,
                    expected_sha256="0" * 64,
                    maximum_bytes=32 * 1024 * 1024,
                )
            receipt_json = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt_json["identities"]["clockIdentity"] = f"sha256:{'f' * 64}"
            tampered_path = root / "tampered-result.json"
            tampered_hash = write_manifest(tampered_path, receipt_json)
            self.assertNotEqual(tampered_hash, receipt_sha256)
            with self.assertRaisesRegex(ContractViolation, "clock identity"):
                load_v1_observation_receipt(
                    tampered_path,
                    expected_sha256=tampered_hash,
                    maximum_bytes=32 * 1024 * 1024,
                )

    def test_coordinated_resealed_receipt_and_manifest_cannot_cross_trusted_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_json, _, _, receipt_path, trusted_receipt_sha = (
                self.authenticated_base(root)
            )
            forged_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            forged_receipt["tracks"][0]["frames"][0]["faceBox"]["x1"] += 1
            forged_receipt["identities"]["observationIdentity"] = content_identity(
                {
                    "clockIdentity": forged_receipt["identities"]["clockIdentity"],
                    "modelIdentity": forged_receipt["identities"]["modelIdentity"],
                    "scoreLedger": forged_receipt["scoreLedger"],
                    "trackingPolicy": forged_receipt["measurements"]["trackingPolicy"],
                    "tracks": forged_receipt["tracks"],
                }
            )
            forged_receipt["identities"]["runIdentity"] = content_identity(
                {
                    "clockIdentity": forged_receipt["identities"]["clockIdentity"],
                    "modelIdentity": forged_receipt["identities"]["modelIdentity"],
                    "observationIdentity": forged_receipt["identities"][
                        "observationIdentity"
                    ],
                    "outputs": forged_receipt["outputs"],
                    "runtimeIdentity": forged_receipt["identities"]["runtimeIdentity"],
                }
            )
            forged_receipt_path = root / "forged-resealed-result.json"
            forged_receipt_sha = write_manifest(
                forged_receipt_path,
                forged_receipt,
            )
            forged_manifest = deepcopy(manifest_json)
            forged_manifest["tracks"] = deepcopy(forged_receipt["tracks"])
            forged_manifest = bind_manifest_to_v1_receipt(
                forged_manifest,
                forged_receipt,
                receipt_sha256=forged_receipt_sha,
                receipt_bytes=forged_receipt_path.stat().st_size,
            )
            forged_manifest_path = root / "forged-resealed-manifest.json"
            forged_manifest_sha = write_manifest(forged_manifest_path, forged_manifest)
            load_supplied_track_manifest(
                forged_manifest_path,
                expected_sha256=forged_manifest_sha,
                limits=SuppliedTrackLimits(),
            )
            with self.assertRaisesRegex(ContractViolation, "SHA-256 mismatch"):
                load_v1_observation_receipt(
                    forged_receipt_path,
                    expected_sha256=trusted_receipt_sha,
                    maximum_bytes=32 * 1024 * 1024,
                )

    def test_v1_internal_source_runtime_and_score_closures_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, _, receipt_path, _ = self.authenticated_base(root)
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))

            def reseal(value):
                identities = value["identities"]
                identities["modelIdentity"] = content_identity(
                    value["measurements"]["model"]
                )
                runtime = value["measurements"]["runtime"]
                identities["runtimeIdentity"] = content_identity(
                    {
                        "audioWorkerBaseImageId": runtime["audioWorkerBaseImageId"],
                        "runtimeClosure": runtime["runtimeClosure"],
                        "runtimeVersion": runtime["runtimeVersion"],
                    }
                )
                identities["observationIdentity"] = content_identity(
                    {
                        "clockIdentity": identities["clockIdentity"],
                        "modelIdentity": identities["modelIdentity"],
                        "scoreLedger": value["scoreLedger"],
                        "trackingPolicy": value["measurements"]["trackingPolicy"],
                        "tracks": value["tracks"],
                    }
                )
                identities["runIdentity"] = content_identity(
                    {
                        "clockIdentity": identities["clockIdentity"],
                        "modelIdentity": identities["modelIdentity"],
                        "observationIdentity": identities["observationIdentity"],
                        "outputs": value["outputs"],
                        "runtimeIdentity": identities["runtimeIdentity"],
                    }
                )

            attacks = (
                (
                    "source identity",
                    lambda value: value["measurements"]["model"].update(
                        lrasdSourceSha256="0" * 64
                    ),
                ),
                (
                    "runtime closure",
                    lambda value: value["measurements"]["runtime"][
                        "runtimeClosure"
                    ].reverse(),
                ),
                (
                    "score clock",
                    lambda value: value["scoreLedger"][0]["samples"][0].update(
                        frameIndex=1
                    ),
                ),
            )
            for index, (message, attack) in enumerate(attacks):
                with self.subTest(message=message):
                    tampered = deepcopy(receipt)
                    attack(tampered)
                    reseal(tampered)
                    path = root / f"internal-attack-{index}.json"
                    digest = write_manifest(path, tampered)
                    with self.assertRaises(ContractViolation):
                        load_v1_observation_receipt(
                            path,
                            expected_sha256=digest,
                            maximum_bytes=32 * 1024 * 1024,
                        )

    def test_derived_manifest_roots_to_same_v1_observation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base_json, base, observation, _, _ = self.authenticated_base(root)
            base_path = root / "base.json"
            base_hash = sha256_bytes(base_path.read_bytes())
            mirror_json = build_mirrored_manifest(
                base_json,
                source_file_sha256=base_hash,
            )
            mirror_path = root / "mirror.json"
            mirror_hash = write_manifest(mirror_path, mirror_json)
            mirror = load_supplied_track_manifest(
                mirror_path,
                expected_sha256=mirror_hash,
                limits=SuppliedTrackLimits(),
            )
            validate_geometry_lineage(mirror, base)
            validate_base_observation_lineage(base, observation)

            mirror_json.pop("contentIdentity")
            mirror_json["producer"]["geometryLineage"]["sourceObservation"][
                "identities"
            ]["runIdentity"] = f"sha256:{'0' * 64}"
            mirror_json["contentIdentity"] = content_identity(mirror_json)
            bad_path = root / "bad-mirror.json"
            bad_hash = write_manifest(bad_path, mirror_json)
            bad_mirror = load_supplied_track_manifest(
                bad_path,
                expected_sha256=bad_hash,
                limits=SuppliedTrackLimits(),
            )
            with self.assertRaisesRegex(ContractViolation, "base observation differs"):
                validate_geometry_lineage(bad_mirror, base)


class LrasdV2SourceClosureTests(unittest.TestCase):
    def test_v2_source_closure_matches_actual_namespace_package_sources(self) -> None:
        self.assertNotIn("model/__init__.py", LRASD_V2_EXECUTED_SOURCE_FILES)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for relative_path in LRASD_V2_EXECUTED_SOURCE_FILES:
                path = root / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"# {relative_path}\n", encoding="utf-8")
            identity, manifest = lrasd_v2_source_identity(root)
            self.assertTrue(identity)
            self.assertEqual(
                [item["path"] for item in manifest], list(LRASD_V2_EXECUTED_SOURCE_FILES)
            )

    def test_shadow_module_initializer_and_bytecode_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for relative_path in LRASD_V2_EXECUTED_SOURCE_FILES:
                path = root / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("pass\n", encoding="utf-8")
            (root / "model.py").write_text("pass\n", encoding="utf-8")
            with self.assertRaisesRegex(ContractViolation, "executable shadow"):
                lrasd_v2_source_identity(root)
            (root / "model.py").unlink()
            initializer = root / "model" / "__init__.py"
            initializer.write_text("pass\n", encoding="utf-8")
            with self.assertRaisesRegex(ContractViolation, "unexpected package initializer"):
                lrasd_v2_source_identity(root)
            initializer.unlink()
            cache = root / "model" / "__pycache__"
            cache.mkdir()
            bytecode = cache / "Model.cpython-311.pyc"
            bytecode.write_bytes(b"bytecode")
            with self.assertRaisesRegex(ContractViolation, "bytecode"):
                lrasd_v2_source_identity(root)
            bytecode.unlink()
            cache.rmdir()
            extension = root / "model" / "Model.abi3.so"
            extension.write_bytes(b"extension")
            with self.assertRaisesRegex(ContractViolation, "executable shadow"):
                lrasd_v2_source_identity(root)


class V2ScoreReceiptContractTests(unittest.TestCase):
    def test_component_ledger_preserves_views_and_recomputable_exact_mean(self) -> None:
        ledger = V2RawScoreLedger.build(
            {"track-a": [2, 3]},
            {"track-a": [-0.5, 1.0]},
            {"track-a": [0.5, 3.0]},
            {"track-a": [0.0, 2.0]},
        )
        samples = ledger[0]["samples"]
        self.assertEqual(samples[0]["rawViewLogits"], {"canonical": -0.5, "horizontalMirror": 0.5})
        self.assertEqual(samples[0]["rawSpeakingScore"], 0.0)
        self.assertEqual(samples[1]["rawSpeakingScore"], 2.0)

    def test_component_mismatch_nonfinite_and_wrong_mean_fail_closed(self) -> None:
        fixtures = (
            ({}, {"track-a": [1.0]}, {"track-a": [1.0]}, {"track-a": [1.0]}),
            ({"track-a": [0]}, {"track-a": [math.nan]}, {"track-a": [1.0]}, {"track-a": [1.0]}),
            ({"track-a": [0]}, {"track-a": [1.0]}, {"track-a": [3.0]}, {"track-a": [1.0]}),
        )
        for fixture in fixtures:
            with self.subTest(fixture=fixture):
                with self.assertRaises(ContractViolation):
                    V2RawScoreLedger.build(*fixture)

    def test_v2_envelope_requires_track_identity_and_stays_diagnostic(self) -> None:
        identities = {
            "clockIdentity": f"sha256:{'a' * 64}",
            "modelIdentity": f"sha256:{'b' * 64}",
            "observationIdentity": f"sha256:{'c' * 64}",
            "runIdentity": f"sha256:{'d' * 64}",
            "runtimeIdentity": f"sha256:{'e' * 64}",
            "trackIdentity": f"sha256:{'f' * 64}",
        }
        result = success_envelope_v2(
            identities=identities,
            clocks={},
            supplied_tracks={"schemaVersion": SUPPLIED_TRACK_SCHEMA_VERSION},
            tracks=[],
            score_ledger=[],
            outputs={},
            measurements={},
        )
        self.assertEqual(result["schemaVersion"], V2_OBSERVATION_SCHEMA_VERSION)
        self.assertEqual(result["cropAuthority"], "NONE")
        self.assertIn("horizontal-mirror", result["rawScoreSemantics"])
        with self.assertRaisesRegex(ContractViolation, "exactly bind"):
            success_envelope_v2(
                identities={key: value for key, value in identities.items() if key != "trackIdentity"},
                clocks={},
                supplied_tracks={},
                tracks=[],
                score_ledger=[],
                outputs={},
                measurements={},
            )


if __name__ == "__main__":
    unittest.main()
