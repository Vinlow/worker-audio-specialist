"""Mirror-invariant v2 wrapper around the frozen one-view LR-ASD adapter."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sys
from typing import Any, Sequence

from active_speaker_contracts import ContractViolation
from active_speaker_model import LrasdModelRunner


@dataclass(frozen=True)
class TwoViewTrackScores:
    canonical: tuple[float, ...]
    horizontal_mirror: tuple[float, ...]
    mean: tuple[float, ...]


def average_two_view_scores(
    canonical_scores: Sequence[float],
    mirrored_scores: Sequence[float],
) -> tuple[float, ...]:
    canonical = tuple(float(value) for value in canonical_scores)
    mirrored = tuple(float(value) for value in mirrored_scores)
    if not canonical or len(canonical) != len(mirrored):
        raise ContractViolation("two-view LR-ASD ledgers must have the same non-zero length")
    if not all(math.isfinite(value) for value in (*canonical, *mirrored)):
        raise ContractViolation("two-view LR-ASD component ledger is non-finite")
    result = tuple(
        math.fsum((canonical_score, mirrored_score)) / 2.0
        for canonical_score, mirrored_score in zip(canonical, mirrored, strict=True)
    )
    if not all(math.isfinite(value) for value in result):
        raise ContractViolation("two-view LR-ASD mean ledger is non-finite")
    return result


class MirrorInvariantLrasdModelRunner(LrasdModelRunner):
    """Scores the same face crop in canonical and horizontally mirrored views."""

    def __init__(self, *, lrasd_root: Path, checkpoint: Path, device: str) -> None:
        super().__init__(lrasd_root=lrasd_root, checkpoint=checkpoint, device=device)
        self._validate_import_origins(lrasd_root)

    @staticmethod
    def _validate_import_origins(lrasd_root: Path) -> None:
        expected = {
            "loss": lrasd_root / "loss.py",
            "model.Classifier": lrasd_root / "model" / "Classifier.py",
            "model.Encoder": lrasd_root / "model" / "Encoder.py",
            "model.Model": lrasd_root / "model" / "Model.py",
        }
        for module_name, expected_path in expected.items():
            module = sys.modules.get(module_name)
            origin = getattr(module, "__file__", None)
            if not isinstance(origin, str):
                raise ContractViolation(f"LR-ASD module {module_name} has no exact source origin")
            try:
                actual_path = Path(origin).resolve(strict=True)
                resolved_expected = expected_path.resolve(strict=True)
            except OSError as error:
                raise ContractViolation(
                    f"LR-ASD module {module_name} source origin does not resolve"
                ) from error
            if actual_path != resolved_expected:
                raise ContractViolation(
                    f"LR-ASD module {module_name} loaded outside the authenticated closure"
                )
        model_namespace = sys.modules.get("model")
        namespace_paths = getattr(model_namespace, "__path__", None)
        namespace_spec = getattr(model_namespace, "__spec__", None)
        if (
            model_namespace is None
            or getattr(model_namespace, "__file__", None) is not None
            or getattr(namespace_spec, "origin", None) is not None
            or namespace_paths is None
        ):
            raise ContractViolation("LR-ASD model must remain the pinned implicit namespace")
        try:
            resolved_paths = tuple(
                Path(path).resolve(strict=True) for path in namespace_paths
            )
            expected_namespace = (lrasd_root.joinpath("model").resolve(strict=True),)
        except OSError as error:
            raise ContractViolation("LR-ASD model namespace path does not resolve") from error
        if resolved_paths != expected_namespace:
            raise ContractViolation("LR-ASD model namespace includes an unauthenticated path")

    def score_track_two_view(
        self,
        *,
        audio_feature: Any,
        visual_feature: Any,
    ) -> TwoViewTrackScores:
        """Return canonical, mirror, and exact arithmetic-mean raw logits."""

        if getattr(visual_feature, "ndim", None) != 3:
            raise ContractViolation("LR-ASD visual feature must have shape [frames, H, W]")
        if tuple(visual_feature.shape[1:]) != (112, 112):
            raise ContractViolation("LR-ASD visual face crops must be exactly 112x112")
        if int(visual_feature.shape[0]) < 1:
            raise ContractViolation("LR-ASD track must contain at least one admitted frame")

        canonical_scores = tuple(
            self.score_track(
                audio_feature=audio_feature,
                visual_feature=visual_feature,
            )
        )
        mirrored_feature = self.numpy.ascontiguousarray(visual_feature[:, :, ::-1])
        flags = getattr(mirrored_feature, "flags", None)
        if flags is None or not bool(flags["C_CONTIGUOUS"]):
            raise ContractViolation("horizontal-mirror LR-ASD crop tensor is not contiguous")
        mirrored_scores = tuple(
            self.score_track(
                audio_feature=audio_feature,
                visual_feature=mirrored_feature,
            )
        )
        return TwoViewTrackScores(
            canonical=canonical_scores,
            horizontal_mirror=mirrored_scores,
            mean=average_two_view_scores(canonical_scores, mirrored_scores),
        )
