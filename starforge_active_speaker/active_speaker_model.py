"""Device-safe strict LR-ASD model adapter for the isolated lab runtime."""

from __future__ import annotations

import importlib
import math
from pathlib import Path
import random
import sys
from typing import Any, Sequence

from active_speaker_contracts import ContractViolation


LRASD_CONTEXT_SECONDS = (1, 2, 3, 4, 5, 6)


def validate_checkpoint_state_dict(
    *,
    expected_state: Any,
    loaded_state: Any,
    is_tensor: Any,
) -> None:
    """Validate exact checkpoint keys, tensor values, and shapes without PyTorch I/O."""

    if not isinstance(loaded_state, dict):
        raise ContractViolation("LR-ASD checkpoint must be a direct state dictionary")
    if not loaded_state:
        raise ContractViolation("LR-ASD checkpoint state dictionary is empty")
    if not all(isinstance(key, str) for key in loaded_state):
        raise ContractViolation("LR-ASD checkpoint keys must all be strings")
    if not all(is_tensor(value) for value in loaded_state.values()):
        raise ContractViolation("LR-ASD checkpoint values must all be tensors")

    expected_keys = set(expected_state)
    received_keys = set(loaded_state)
    if expected_keys != received_keys:
        missing = sorted(expected_keys - received_keys)
        unexpected = sorted(received_keys - expected_keys)
        raise ContractViolation(
            "LR-ASD checkpoint key closure mismatch; "
            f"missing={missing}, unexpected={unexpected}"
        )
    mismatched_shapes = []
    for key in sorted(expected_keys):
        expected_shape = tuple(getattr(expected_state[key], "shape", ()))
        received_shape = tuple(getattr(loaded_state[key], "shape", ()))
        if expected_shape != received_shape:
            mismatched_shapes.append(
                {
                    "expected": expected_shape,
                    "key": key,
                    "received": received_shape,
                }
            )
    if mismatched_shapes:
        raise ContractViolation(
            f"LR-ASD checkpoint tensor shape mismatch: {mismatched_shapes}"
        )


class LrasdModelRunner:
    """Loads an external LR-ASD checkout and checkpoint without weakening either."""

    def __init__(self, *, lrasd_root: Path, checkpoint: Path, device: str) -> None:
        try:
            import numpy
            import torch
        except ImportError as error:
            raise ContractViolation(
                f"LR-ASD runtime dependency is unavailable: {error.name}"
            ) from error

        self.numpy = numpy
        self.torch = torch
        self.device = self._resolve_device(device)
        self._configure_determinism()
        self.closure = self._load_strict_closure(lrasd_root, checkpoint)

    def _resolve_device(self, requested: str) -> Any:
        torch = self.torch
        if requested == "auto":
            resolved = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            resolved = requested
        if resolved == "cuda" and not torch.cuda.is_available():
            raise ContractViolation("CUDA was requested but torch.cuda.is_available() is false")
        if resolved not in {"cpu", "cuda"}:
            raise ContractViolation(f"unsupported LR-ASD device: {resolved}")
        return torch.device(resolved)

    def _configure_determinism(self) -> None:
        torch = self.torch
        random.seed(0)
        self.numpy.random.seed(0)
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
        torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    def _load_strict_closure(self, lrasd_root: Path, checkpoint: Path) -> Any:
        torch = self.torch
        root_text = str(lrasd_root)
        if root_text in sys.path:
            sys.path.remove(root_text)
        sys.path.insert(0, root_text)

        conflicting_modules = [
            name
            for name in ("loss", "model", "model.Classifier", "model.Encoder", "model.Model")
            if name in sys.modules
        ]
        if conflicting_modules:
            raise ContractViolation(
                "LR-ASD modules were imported before closure validation: "
                + ", ".join(conflicting_modules)
            )

        try:
            model_module = importlib.import_module("model.Model")
            loss_module = importlib.import_module("loss")
        except Exception as error:
            raise ContractViolation(f"failed to import strict LR-ASD source: {error}") from error

        class CheckpointClosure(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = model_module.ASD_Model()
                self.lossAV = loss_module.lossAV()
                self.lossV = loss_module.lossV()

        closure = CheckpointClosure()
        try:
            loaded = torch.load(
                checkpoint,
                map_location="cpu",
                weights_only=True,
            )
        except Exception as error:
            raise ContractViolation(
                f"checkpoint failed torch.load(weights_only=True): {error}"
            ) from error
        validate_checkpoint_state_dict(
            expected_state=closure.state_dict(),
            loaded_state=loaded,
            is_tensor=torch.is_tensor,
        )
        try:
            closure.load_state_dict(loaded, strict=True)
        except Exception as error:
            raise ContractViolation(f"strict LR-ASD state load failed: {error}") from error

        closure.eval()
        closure.to(self.device)
        return closure

    def dependency_versions(self) -> dict[str, str]:
        return {
            "numpy": str(self.numpy.__version__),
            "torch": str(self.torch.__version__),
        }

    def score_track(self, *, audio_feature: Any, visual_feature: Any) -> list[float]:
        """Return one uncalibrated class-1 logit per admitted video frame."""

        numpy = self.numpy
        torch = self.torch
        if getattr(audio_feature, "ndim", None) != 2 or audio_feature.shape[1] != 13:
            raise ContractViolation("LR-ASD audio feature must have shape [rows, 13]")
        if getattr(visual_feature, "ndim", None) != 3:
            raise ContractViolation("LR-ASD visual feature must have shape [frames, H, W]")
        if tuple(visual_feature.shape[1:]) != (112, 112):
            raise ContractViolation("LR-ASD visual face crops must be exactly 112x112")
        frame_count = int(visual_feature.shape[0])
        if frame_count < 1:
            raise ContractViolation("LR-ASD track must contain at least one admitted frame")
        if int(audio_feature.shape[0]) != frame_count * 4:
            raise ContractViolation(
                "LR-ASD audio feature must contain exactly four 10ms rows per 25fps frame"
            )

        per_context: list[list[float]] = []
        with torch.inference_mode():
            for context_seconds in LRASD_CONTEXT_SECONDS:
                context_frames = context_seconds * 25
                context_audio_rows = context_seconds * 100
                context_scores: list[float] = []
                for frame_start in range(0, frame_count, context_frames):
                    frame_end = min(frame_count, frame_start + context_frames)
                    audio_start = frame_start * 4
                    audio_end = min(
                        int(audio_feature.shape[0]),
                        audio_start + context_audio_rows,
                    )
                    input_audio = torch.as_tensor(
                        audio_feature[audio_start:audio_end],
                        dtype=torch.float32,
                        device=self.device,
                    ).unsqueeze(0)
                    input_visual = torch.as_tensor(
                        visual_feature[frame_start:frame_end],
                        dtype=torch.float32,
                        device=self.device,
                    ).unsqueeze(0)

                    audio_embedding = self.closure.model.forward_audio_frontend(
                        input_audio
                    )
                    visual_embedding = self.closure.model.forward_visual_frontend(
                        input_visual
                    )
                    if audio_embedding.shape[1] != visual_embedding.shape[1]:
                        raise ContractViolation(
                            "LR-ASD audio and visual embeddings have different frame counts"
                        )
                    fused = self.closure.model.forward_audio_visual_backend(
                        audio_embedding,
                        visual_embedding,
                    )
                    logits = self.closure.lossAV.FC(fused.squeeze(1))
                    raw_scores = logits[:, 1].detach().to("cpu", dtype=torch.float64)
                    chunk = [float(value) for value in raw_scores.tolist()]
                    expected_chunk = frame_end - frame_start
                    if len(chunk) != expected_chunk:
                        raise ContractViolation(
                            "LR-ASD emitted an unexpected score count: "
                            f"expected {expected_chunk}, received {len(chunk)}"
                        )
                    if not all(math.isfinite(value) for value in chunk):
                        raise ContractViolation("LR-ASD emitted a non-finite raw score")
                    context_scores.extend(chunk)
                if len(context_scores) != frame_count:
                    raise ContractViolation(
                        "LR-ASD context pass did not cover every admitted frame"
                    )
                per_context.append(context_scores)

        scores = [
            math.fsum(context[index] for context in per_context) / len(per_context)
            for index in range(frame_count)
        ]
        if len(scores) != frame_count or not all(math.isfinite(value) for value in scores):
            raise ContractViolation("LR-ASD averaged raw score ledger is invalid")
        return scores

    def prepare_features(
        self,
        *,
        audio_samples: Any,
        visual_feature: Any,
        sample_rate_hz: int,
    ) -> tuple[Any, Any]:
        if sample_rate_hz != 16_000:
            raise ContractViolation("LR-ASD audio must be exactly 16kHz")
        try:
            import python_speech_features
        except ImportError as error:
            raise ContractViolation("python-speech-features is unavailable") from error

        audio_feature = python_speech_features.mfcc(
            audio_samples,
            sample_rate_hz,
            numcep=13,
            winlen=0.025,
            winstep=0.010,
        )
        available_frames = int(audio_feature.shape[0]) // 4
        admitted_frames = min(int(visual_feature.shape[0]), available_frames)
        if admitted_frames < 1:
            raise ContractViolation("track has no synchronized LR-ASD feature frames")
        # A normal MFCC tail may be shorter than one complete four-row video frame,
        # but a larger loss is an A/V contract failure rather than an interpolation.
        if int(visual_feature.shape[0]) - admitted_frames > 1:
            raise ContractViolation("track loses more than one frame during MFCC alignment")
        return (
            audio_feature[: admitted_frames * 4, :],
            visual_feature[:admitted_frames, :, :],
        )
