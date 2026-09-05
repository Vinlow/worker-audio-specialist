from __future__ import annotations

import json
from pathlib import Path
import stat
import sys
import tempfile
import unittest


MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))

from active_speaker_contracts import ContractViolation  # noqa: E402
from active_speaker_runtime import (  # noqa: E402
    RUNTIME_CLOSURE_FILES,
    _parser,
    _preprocessing_policy,
    _runtime_identity,
    _write_failure_receipt,
    _write_new_json,
)


class AtomicReceiptTests(unittest.TestCase):
    def test_json_publication_is_atomic_read_only_and_no_clobber(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "result.json"
            _write_new_json(target, {"value": 1})
            self.assertEqual(json.loads(target.read_text(encoding="utf-8")), {"value": 1})
            self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o400)
            self.assertEqual(list(root.glob(".result.json.*.tmp")), [])

            with self.assertRaisesRegex(ContractViolation, "no-clobber"):
                _write_new_json(target, {"value": 2})
            self.assertEqual(json.loads(target.read_text(encoding="utf-8")), {"value": 1})
            self.assertEqual(list(root.glob(".result.json.*.tmp")), [])

    def test_failure_receipt_is_structured_and_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_failure_receipt(root, ValueError("first"))
            failure = json.loads((root / "failure.json").read_text(encoding="utf-8"))
            self.assertEqual(failure["error"], {"message": "first", "type": "ValueError"})
            with self.assertRaises(ContractViolation):
                _write_failure_receipt(root, RuntimeError("second"))
            self.assertEqual(
                json.loads((root / "failure.json").read_text(encoding="utf-8"))["error"],
                {"message": "first", "type": "ValueError"},
            )


class RuntimeClosureTests(unittest.TestCase):
    def test_dockerfile_keeps_runtime_closure_root_owned_and_read_only(self) -> None:
        dockerfile = (MODULE_ROOT / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("--chown=0:0 --chmod=0444", dockerfile)
        self.assertIn("chmod 0555 /opt/starforge-active-speaker", dockerfile)
        self.assertNotIn("chown -R 65532", dockerfile)

    def test_receipt_bound_detection_defaults_are_frozen(self) -> None:
        parser = _parser()
        run_parser = next(
            action
            for action in parser._actions
            if getattr(action, "choices", None) is not None
        ).choices["run"]
        defaults = {
            action.dest: action.default
            for action in run_parser._actions
            if action.dest in {"face_score_threshold", "maximum_track_gap_frames"}
        }
        self.assertEqual(
            defaults,
            {"face_score_threshold": 0.7, "maximum_track_gap_frames": 15},
        )

    def test_runtime_identity_binds_full_declared_closure_and_base_image(self) -> None:
        first_id, manifest = _runtime_identity(f"sha256:{'a' * 64}")
        second_id, second_manifest = _runtime_identity(f"sha256:{'b' * 64}")
        self.assertNotEqual(first_id, second_id)
        self.assertEqual(manifest, second_manifest)
        self.assertEqual(
            {item["path"] for item in manifest},
            set(RUNTIME_CLOSURE_FILES),
        )
        for required in ("Dockerfile", "requirements.lock.txt", "LR-ASD-LICENSE.txt"):
            self.assertIn(required, {item["path"] for item in manifest})

    def test_preprocessing_policy_names_exact_clock_and_feature_contracts(self) -> None:
        policy = _preprocessing_policy()
        self.assertEqual(
            policy["audio"]["decoderTail"],
            "trim-to-selected-stream-presentation-duration-before-origin-alignment-v1",
        )
        self.assertEqual(
            policy["audio"]["timelineValidation"],
            "packet-coverage-and-decoded-sample-clock-v1",
        )
        self.assertEqual(policy["audio"]["sampleRateHz"], 16_000)
        self.assertEqual(policy["mfcc"]["rowsPerVideoFrame"], 4)
        self.assertEqual(policy["video"]["frameZero"], "first-decoded-selected-video-frame")
        self.assertEqual(
            policy["video"]["timelineValidation"],
            "packet-coverage-and-decoded-cfr-clock-v1",
        )


if __name__ == "__main__":
    unittest.main()
