"""Optional speaker-diarization sidecar for the Audio Specialist worker.

The transcription and NP-SBV2 word geometry are immutable inputs here.  This
module never rewrites ``word_timestamps``; it emits a separate, versioned
artifact that attributes existing word ordinals to anonymous, chunk-local
speaker IDs.

Why a sidecar instead of ``word["speaker"]``:
  * Starforge hashes transcript chunks into source/prefix identity.
  * Diarization is probabilistic and may be unavailable.
  * Speaker labels can permute across independently processed chunks.
  * Speaker evidence must never become cutting/boundary authority.
"""

from __future__ import annotations

import os
import time
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DIARIZATION_SCHEMA_VERSION = "w2l-speaker-diarization-v1"
DEFAULT_MODEL_ID = "pyannote/speaker-diarization-3.1"
MIN_OVERLAP_SEC = 0.01


def _finite_number(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _round_sec(value: float) -> float:
    return round(float(value), 3)


def _iter_annotation(annotation: Any) -> Iterable[Tuple[float, float, str]]:
    """Yield ``(start, end, raw_label)`` across pyannote 3.x/4.x shapes."""

    if annotation is None:
        return
    if hasattr(annotation, "itertracks"):
        for segment, _track, label in annotation.itertracks(yield_label=True):
            yield float(segment.start), float(segment.end), str(label)
        return
    for item in annotation:
        if not isinstance(item, (tuple, list)) or len(item) < 2:
            continue
        segment, label = item[0], item[-1]
        if hasattr(segment, "start") and hasattr(segment, "end"):
            yield float(segment.start), float(segment.end), str(label)


def _canonicalize_turns(
    raw_turns: Iterable[Tuple[float, float, str]],
    label_map: Optional[Dict[str, str]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """Sort turns and map arbitrary model labels to first-seen speaker IDs."""

    cleaned: List[Tuple[float, float, str]] = []
    for raw_start, raw_end, raw_label in raw_turns:
        start = _finite_number(raw_start)
        end = _finite_number(raw_end)
        if start is None or end is None or end <= start:
            continue
        cleaned.append((start, end, str(raw_label)))
    cleaned.sort(key=lambda item: (item[0], item[1], item[2]))

    canonical = dict(label_map or {})
    turns: List[Dict[str, Any]] = []
    for start, end, raw_label in cleaned:
        if raw_label not in canonical:
            canonical[raw_label] = f"SPEAKER_{len(canonical):02d}"
        turns.append(
            {
                "start_sec": _round_sec(start),
                "end_sec": _round_sec(end),
                "speaker_id": canonical[raw_label],
            }
        )
    return turns, canonical


def _speaker_overlaps(
    start: float,
    end: float,
    turns: Sequence[Dict[str, Any]],
) -> Dict[str, float]:
    overlaps: Dict[str, float] = {}
    for turn in turns:
        turn_start = float(turn["start_sec"])
        turn_end = float(turn["end_sec"])
        overlap = min(end, turn_end) - max(start, turn_start)
        if overlap < MIN_OVERLAP_SEC:
            continue
        speaker = str(turn["speaker_id"])
        overlaps[speaker] = overlaps.get(speaker, 0.0) + overlap
    return overlaps


def build_diarization_sidecar(
    words: Sequence[Dict[str, Any]],
    regular_turns: Sequence[Dict[str, Any]],
    exclusive_turns: Optional[Sequence[Dict[str, Any]]] = None,
    *,
    model_id: str = DEFAULT_MODEL_ID,
    package_version: str = "unknown",
    device: str = "unknown",
    processing_sec: float = 0.0,
) -> Dict[str, Any]:
    """Attach speaker evidence to word ordinals without mutating word geometry."""

    assignment_turns = list(exclusive_turns or regular_turns)
    attributions: List[Dict[str, Any]] = []
    for word_index, word in enumerate(words):
        start = _finite_number(word.get("start"))
        end = _finite_number(word.get("end"))
        if start is None or end is None or end <= start:
            attributions.append(
                {
                    "word_index": word_index,
                    "status": "INVALID_WORD_GEOMETRY",
                    "speaker_id": None,
                    "confidence": 0.0,
                    "overlap": False,
                }
            )
            continue

        assignment_overlaps = _speaker_overlaps(start, end, assignment_turns)
        regular_overlaps = _speaker_overlaps(start, end, regular_turns)
        ranked = sorted(
            assignment_overlaps.items(),
            key=lambda item: (-item[1], item[0]),
        )
        duration = max(end - start, 1e-6)
        if not ranked:
            attributions.append(
                {
                    "word_index": word_index,
                    "status": "UNKNOWN",
                    "speaker_id": None,
                    "confidence": 0.0,
                    "overlap": len(regular_overlaps) > 1,
                }
            )
            continue

        best_speaker, best_overlap = ranked[0]
        attributions.append(
            {
                "word_index": word_index,
                "status": "ATTRIBUTED",
                "speaker_id": best_speaker,
                "confidence": round(min(1.0, best_overlap / duration), 4),
                "overlap": len(regular_overlaps) > 1,
            }
        )

    speakers = sorted(
        {
            str(turn["speaker_id"])
            for turn in regular_turns
            if turn.get("speaker_id") is not None
        }
    )
    return {
        "schema_version": DIARIZATION_SCHEMA_VERSION,
        "status": "COMPLETED",
        "model": model_id,
        "pyannote_audio_version": package_version,
        "device": device,
        "identity_scope": "CHUNK_LOCAL_UNSTABLE",
        "boundary_authority": False,
        "transcript_geometry_mutated": False,
        "speaker_count": len(speakers),
        "speakers": speakers,
        "turns": list(regular_turns),
        "exclusive_turns": list(exclusive_turns or []),
        "word_attributions": attributions,
        "processing_sec": round(max(0.0, float(processing_sec)), 3),
    }


class SpeakerDiarizer:
    """Lazy pyannote pipeline wrapper with fail-soft sidecar output."""

    def __init__(self, model_id: Optional[str] = None):
        self.model_id = model_id or os.environ.get(
            "PYANNOTE_DIARIZATION_MODEL",
            DEFAULT_MODEL_ID,
        )
        self.pipeline = None
        self.device = None

    @staticmethod
    def _package_version() -> str:
        try:
            return version("pyannote.audio")
        except PackageNotFoundError:
            return "missing"

    def setup(self, device: str) -> None:
        if self.pipeline is not None and self.device == device:
            return

        token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("MISSING_HUGGINGFACE_TOKEN")

        # Keep customer audio out of optional package telemetry by default.
        os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "0")
        import torch
        # PyTorch 2.6+ safely defaults checkpoint loading to
        # ``weights_only=True``. The official pyannote segmentation-3.0
        # checkpoint contains TorchVersion metadata, so allowlist that one
        # inert framework value instead of disabling safe loading for the
        # entire checkpoint.
        from pyannote.audio import Pipeline
        from pyannote.audio.core.task import Problem, Resolution, Specifications
        from torch.torch_version import TorchVersion

        if hasattr(torch.serialization, "add_safe_globals"):
            torch.serialization.add_safe_globals(
                [TorchVersion, Specifications, Problem, Resolution]
            )

        print(
            f"[SpeakerDiarizer] loading {self.model_id} on {device}...",
            flush=True,
        )
        try:
            pipeline = Pipeline.from_pretrained(self.model_id, token=token)
        except TypeError:
            # pyannote.audio 3.x calls this argument use_auth_token. Keeping the
            # fallback makes a future Community-1 challenger a model/package
            # change rather than a sidecar-contract rewrite.
            pipeline = Pipeline.from_pretrained(
                self.model_id,
                use_auth_token=token,
            )
        if pipeline is None:
            raise RuntimeError("PYANNOTE_MODEL_UNAVAILABLE_OR_GATED")
        pipeline.to(torch.device(device))
        self.pipeline = pipeline
        self.device = device
        print("[SpeakerDiarizer] model loaded", flush=True)

    def diarize(
        self,
        audio_path: str,
        words: Sequence[Dict[str, Any]],
        *,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> Dict[str, Any]:
        started = time.perf_counter()
        device = "cuda"
        try:
            if min_speakers is not None and min_speakers < 1:
                raise ValueError("DIARIZATION_MIN_SPEAKERS_MUST_BE_POSITIVE")
            if max_speakers is not None and max_speakers < 1:
                raise ValueError("DIARIZATION_MAX_SPEAKERS_MUST_BE_POSITIVE")
            if (
                min_speakers is not None
                and max_speakers is not None
                and min_speakers > max_speakers
            ):
                raise ValueError("DIARIZATION_MIN_SPEAKERS_EXCEEDS_MAX")

            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.setup(device)
            kwargs: Dict[str, int] = {}
            if min_speakers is not None and min_speakers > 0:
                kwargs["min_speakers"] = int(min_speakers)
            if max_speakers is not None and max_speakers > 0:
                kwargs["max_speakers"] = int(max_speakers)

            output = self.pipeline(audio_path, **kwargs)
            regular_annotation = getattr(output, "speaker_diarization", output)
            regular, label_map = _canonicalize_turns(
                _iter_annotation(regular_annotation)
            )
            exclusive_annotation = getattr(
                output,
                "exclusive_speaker_diarization",
                None,
            )
            exclusive, _ = _canonicalize_turns(
                _iter_annotation(exclusive_annotation),
                label_map,
            )
            return build_diarization_sidecar(
                words,
                regular,
                exclusive,
                model_id=self.model_id,
                package_version=self._package_version(),
                device=device,
                processing_sec=time.perf_counter() - started,
            )
        except Exception as error:  # Sidecar failure must not lose transcription.
            error_text = str(error).replace("\n", " ")[:300]
            return {
                "schema_version": DIARIZATION_SCHEMA_VERSION,
                "status": "FAILED",
                "model": self.model_id,
                "pyannote_audio_version": self._package_version(),
                "device": device,
                "identity_scope": "CHUNK_LOCAL_UNSTABLE",
                "boundary_authority": False,
                "transcript_geometry_mutated": False,
                "error_code": type(error).__name__,
                "error": error_text,
                "processing_sec": round(time.perf_counter() - started, 3),
            }
