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

import json
import os
import tempfile
import threading
import time
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from hf_auth import install_legacy_use_auth_token_compat
from model_manifest import (
    PYANNOTE_EMBEDDING_REPO,
    PYANNOTE_EMBEDDING_REVISION,
    PYANNOTE_PIPELINE_REPO,
    PYANNOTE_PIPELINE_REVISION,
    PYANNOTE_SEGMENTATION_REPO,
    PYANNOTE_SEGMENTATION_REVISION,
)
from model_load_lock import serialized_model_load


DIARIZATION_SCHEMA_VERSION = "w2l-speaker-diarization-v1"
DEFAULT_PIPELINE_REPO = PYANNOTE_PIPELINE_REPO
DEFAULT_PIPELINE_REVISION = PYANNOTE_PIPELINE_REVISION
DEFAULT_MODEL_ID = f"{DEFAULT_PIPELINE_REPO}@{DEFAULT_PIPELINE_REVISION}"
SEGMENTATION_MODEL_REPO = PYANNOTE_SEGMENTATION_REPO
SEGMENTATION_MODEL_REVISION = PYANNOTE_SEGMENTATION_REVISION
EMBEDDING_MODEL_REPO = PYANNOTE_EMBEDDING_REPO
EMBEDDING_MODEL_REVISION = PYANNOTE_EMBEDDING_REVISION
PINNED_MODEL_DEPENDENCIES = {
    "segmentation": (
        f"{SEGMENTATION_MODEL_REPO}@{SEGMENTATION_MODEL_REVISION}"
    ),
    "embedding": f"{EMBEDDING_MODEL_REPO}@{EMBEDDING_MODEL_REVISION}",
}
MIN_OVERLAP_SEC = 0.01
_INFERENCE_LOCK = threading.Lock()

_KNOWN_ERROR_CODES = {
    "DIARIZATION_EMPTY_OUTPUT",
    "DIARIZATION_MAX_SPEAKERS_MUST_BE_POSITIVE",
    "DIARIZATION_MIN_SPEAKERS_EXCEEDS_MAX",
    "DIARIZATION_MIN_SPEAKERS_MUST_BE_POSITIVE",
    "MISSING_HUGGINGFACE_TOKEN",
    "PYANNOTE_LOCAL_CACHE_INCOMPLETE",
    "PYANNOTE_PIPELINE_CONFIG_INVALID",
    "PYANNOTE_PIPELINE_DEPENDENCY_MISMATCH",
    "PYANNOTE_MODEL_UNAVAILABLE_OR_GATED",
}


def _log_event(event: str, **fields: Any) -> None:
    """Emit path/token-free, machine-readable lifecycle diagnostics."""

    payload = {
        "component": "speaker_diarizer",
        "event": event,
        **fields,
    }
    try:
        print(
            "[SpeakerDiarizer] " + json.dumps(payload, sort_keys=True),
            flush=True,
        )
    except (OSError, ValueError):
        # Diagnostics must never turn an optional sidecar into a job failure.
        pass


def _stable_error_code(error: Exception, stage: str) -> str:
    """Collapse dependency details into a small, stable public taxonomy."""

    message = str(error).strip()
    if message in _KNOWN_ERROR_CODES:
        return message
    if isinstance(error, (ImportError, ModuleNotFoundError)):
        return "DIARIZATION_DEPENDENCY_UNAVAILABLE"
    return {
        "VALIDATION": "DIARIZATION_INVALID_ARGUMENT",
        "LOAD": "DIARIZATION_MODEL_LOAD_FAILED",
        "INFERENCE": "DIARIZATION_INFERENCE_FAILED",
        "POSTPROCESSING": "DIARIZATION_OUTPUT_INVALID",
    }.get(stage, "DIARIZATION_INTERNAL_FAILED")


def _finite_number(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _round_sec(value: float) -> float:
    return round(float(value), 3)


def _record_discard(stats: Optional[Dict[str, int]]) -> None:
    if stats is not None:
        stats["discarded_invalid_turn_count"] = (
            stats.get("discarded_invalid_turn_count", 0) + 1
        )


def _iter_annotation(
    annotation: Any,
    stats: Optional[Dict[str, int]] = None,
) -> Iterable[Tuple[Any, Any, Any]]:
    """Yield ``(start, end, raw_label)`` across pyannote 3.x/4.x shapes."""

    if annotation is None:
        return
    if hasattr(annotation, "itertracks"):
        for segment, _track, label in annotation.itertracks(yield_label=True):
            if not hasattr(segment, "start") or not hasattr(segment, "end"):
                _record_discard(stats)
                continue
            yield segment.start, segment.end, label
        return
    for item in annotation:
        if not isinstance(item, (tuple, list)) or len(item) < 2:
            _record_discard(stats)
            continue
        segment, label = item[0], item[-1]
        if hasattr(segment, "start") and hasattr(segment, "end"):
            yield segment.start, segment.end, label
        else:
            _record_discard(stats)


def _canonicalize_turns(
    raw_turns: Iterable[Tuple[Any, Any, Any]],
    label_map: Optional[Dict[str, str]] = None,
    *,
    stats: Optional[Dict[str, int]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """Sort turns and map arbitrary model labels to first-seen speaker IDs."""

    cleaned: List[Tuple[float, float, str]] = []
    for raw_turn in raw_turns:
        try:
            raw_start, raw_end, raw_label = raw_turn
        except (TypeError, ValueError):
            _record_discard(stats)
            continue
        start = _finite_number(raw_start)
        end = _finite_number(raw_end)
        if (
            start is None
            or end is None
            or end <= start
            or raw_label is None
            or not str(raw_label).strip()
        ):
            _record_discard(stats)
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


def _merge_intervals(
    intervals: Sequence[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    merged: List[Tuple[float, float]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
            continue
        previous_start, previous_end = merged[-1]
        merged[-1] = (previous_start, max(previous_end, end))
    return merged


def _speaker_intervals(
    start: float,
    end: float,
    turns: Sequence[Dict[str, Any]],
) -> Dict[str, List[Tuple[float, float]]]:
    intervals: Dict[str, List[Tuple[float, float]]] = {}
    for turn in turns:
        if not isinstance(turn, dict) or turn.get("speaker_id") is None:
            continue
        turn_start = _finite_number(turn.get("start_sec"))
        turn_end = _finite_number(turn.get("end_sec"))
        if turn_start is None or turn_end is None or turn_end <= turn_start:
            continue
        clipped_start = max(start, turn_start)
        clipped_end = min(end, turn_end)
        if clipped_end <= clipped_start:
            continue
        speaker = str(turn["speaker_id"])
        intervals.setdefault(speaker, []).append((clipped_start, clipped_end))
    return {
        speaker: _merge_intervals(speaker_intervals)
        for speaker, speaker_intervals in intervals.items()
    }


def _speaker_overlaps(
    start: float,
    end: float,
    turns: Sequence[Dict[str, Any]],
) -> Dict[str, float]:
    """Return union coverage per speaker, never double-counting duplicate turns."""

    coverages = {
        speaker: sum(
            interval_end - interval_start
            for interval_start, interval_end in intervals
        )
        for speaker, intervals in _speaker_intervals(start, end, turns).items()
    }
    return {
        speaker: coverage_sec
        for speaker, coverage_sec in coverages.items()
        if coverage_sec + 1e-9 >= MIN_OVERLAP_SEC
    }


def _intersect_intervals(
    left: Sequence[Tuple[float, float]],
    right: Sequence[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    intersections: List[Tuple[float, float]] = []
    left_index = 0
    right_index = 0
    while left_index < len(left) and right_index < len(right):
        overlap_start = max(left[left_index][0], right[right_index][0])
        overlap_end = min(left[left_index][1], right[right_index][1])
        if overlap_end > overlap_start:
            intersections.append((overlap_start, overlap_end))
        if left[left_index][1] <= right[right_index][1]:
            left_index += 1
        else:
            right_index += 1
    return intersections


def _simultaneous_coverage(
    speaker_intervals: Dict[str, List[Tuple[float, float]]],
) -> Tuple[Dict[str, float], float]:
    """Measure only material pairwise interval intersections."""

    speakers = sorted(speaker_intervals)
    per_speaker: Dict[str, List[Tuple[float, float]]] = {}
    simultaneous_intervals: List[Tuple[float, float]] = []
    for left_index, left_speaker in enumerate(speakers):
        for right_speaker in speakers[left_index + 1 :]:
            intersections = _intersect_intervals(
                speaker_intervals[left_speaker],
                speaker_intervals[right_speaker],
            )
            intersection_sec = sum(
                end - start for start, end in intersections
            )
            if intersection_sec + 1e-9 < MIN_OVERLAP_SEC:
                continue
            per_speaker.setdefault(left_speaker, []).extend(intersections)
            per_speaker.setdefault(right_speaker, []).extend(intersections)
            simultaneous_intervals.extend(intersections)

    simultaneous_by_speaker = {
        speaker: sum(end - start for start, end in _merge_intervals(intervals))
        for speaker, intervals in per_speaker.items()
    }
    simultaneous_sec = sum(
        end - start
        for start, end in _merge_intervals(simultaneous_intervals)
    )
    return simultaneous_by_speaker, simultaneous_sec


def _candidate_rows(
    speaker_ids: Sequence[str],
    coverages: Dict[str, float],
    simultaneous_coverages: Dict[str, float],
    duration: float,
) -> List[Dict[str, Any]]:
    rows = []
    for speaker_id in speaker_ids:
        coverage_sec = max(0.0, coverages.get(speaker_id, 0.0))
        simultaneous_sec = max(
            0.0,
            simultaneous_coverages.get(speaker_id, 0.0),
        )
        rows.append(
            {
                "speaker_id": speaker_id,
                "coverage_sec": _round_sec(coverage_sec),
                "coverage_fraction": round(
                    min(1.0, coverage_sec / duration),
                    4,
                ),
                "simultaneous_coverage_sec": _round_sec(simultaneous_sec),
                "simultaneous_coverage_fraction": round(
                    min(1.0, simultaneous_sec / duration),
                    4,
                ),
            }
        )
    return sorted(
        rows,
        key=lambda row: (-row["coverage_sec"], row["speaker_id"]),
    )


def build_diarization_sidecar(
    words: Sequence[Dict[str, Any]],
    regular_turns: Sequence[Dict[str, Any]],
    exclusive_turns: Optional[Sequence[Dict[str, Any]]] = None,
    *,
    model_id: str = DEFAULT_MODEL_ID,
    package_version: str = "unknown",
    device: str = "unknown",
    processing_sec: float = 0.0,
    load_sec: float = 0.0,
    inference_wait_sec: float = 0.0,
    inference_sec: float = 0.0,
    discarded_invalid_turn_count: int = 0,
) -> Dict[str, Any]:
    """Attach speaker evidence to word ordinals without mutating word geometry."""

    assignment_turns = list(exclusive_turns or regular_turns)
    attributions: List[Dict[str, Any]] = []
    for word_index, word in enumerate(words):
        start = _finite_number(word.get("start")) if isinstance(word, dict) else None
        end = _finite_number(word.get("end")) if isinstance(word, dict) else None
        if start is None or end is None or end <= start:
            attributions.append(
                {
                    "word_index": word_index,
                    "status": "INVALID_WORD_GEOMETRY",
                    "attribution_reason": "INVALID_WORD_GEOMETRY",
                    "speaker_id": None,
                    "coverage_fraction": 0.0,
                    # Compatibility alias: this has always been temporal
                    # coverage, not a calibrated model probability.
                    "confidence": 0.0,
                    "overlap": False,
                    "simultaneous_overlap_fraction": 0.0,
                    "sequential_handoff": False,
                    "candidate_speaker_ids": [],
                    "candidate_speakers": [],
                }
            )
            continue

        duration = max(end - start, 1e-6)
        regular_intervals = _speaker_intervals(start, end, regular_turns)
        regular_overlaps = {
            speaker: sum(
                interval_end - interval_start
                for interval_start, interval_end in intervals
            )
            for speaker, intervals in regular_intervals.items()
        }
        simultaneous_by_speaker, simultaneous_sec = _simultaneous_coverage(
            regular_intervals
        )
        ambiguous_speaker_ids = sorted(simultaneous_by_speaker)
        is_ambiguous_overlap = len(ambiguous_speaker_ids) > 1
        materially_covered_speakers = sorted(
            speaker
            for speaker, coverage_sec in regular_overlaps.items()
            if coverage_sec + 1e-9 >= MIN_OVERLAP_SEC
        )
        sequential_handoff = (
            not is_ambiguous_overlap
            and len(materially_covered_speakers) > 1
        )

        assignment_overlaps = _speaker_overlaps(start, end, assignment_turns)
        ranked = sorted(
            assignment_overlaps.items(),
            key=lambda item: (-item[1], item[0]),
        )
        if is_ambiguous_overlap:
            candidate_speakers = _candidate_rows(
                ambiguous_speaker_ids,
                regular_overlaps,
                simultaneous_by_speaker,
                duration,
            )
            coverage_fraction = max(
                (
                    candidate["coverage_fraction"]
                    for candidate in candidate_speakers
                ),
                default=0.0,
            )
            attributions.append(
                {
                    "word_index": word_index,
                    # Keep the v1 status vocabulary stable. The additive
                    # reason carries the more precise ambiguity semantics.
                    "status": "UNKNOWN",
                    "attribution_reason": "AMBIGUOUS_OVERLAP",
                    "speaker_id": None,
                    "coverage_fraction": coverage_fraction,
                    "confidence": coverage_fraction,
                    "overlap": True,
                    "simultaneous_overlap_fraction": round(
                        min(1.0, simultaneous_sec / duration),
                        4,
                    ),
                    "sequential_handoff": False,
                    "candidate_speaker_ids": [
                        candidate["speaker_id"]
                        for candidate in candidate_speakers
                    ],
                    "candidate_speakers": candidate_speakers,
                }
            )
            continue

        candidate_ids = sorted(
            set(materially_covered_speakers) | set(assignment_overlaps)
        )
        candidate_coverages = dict(regular_overlaps)
        for speaker_id, coverage_sec in assignment_overlaps.items():
            candidate_coverages.setdefault(speaker_id, coverage_sec)
        candidate_speakers = _candidate_rows(
            candidate_ids,
            candidate_coverages,
            simultaneous_by_speaker,
            duration,
        )
        if not ranked:
            attributions.append(
                {
                    "word_index": word_index,
                    "status": "UNKNOWN",
                    "attribution_reason": "NO_SPEAKER_COVERAGE",
                    "speaker_id": None,
                    "coverage_fraction": 0.0,
                    "confidence": 0.0,
                    "overlap": False,
                    "simultaneous_overlap_fraction": 0.0,
                    "sequential_handoff": sequential_handoff,
                    "candidate_speaker_ids": [
                        candidate["speaker_id"]
                        for candidate in candidate_speakers
                    ],
                    "candidate_speakers": candidate_speakers,
                }
            )
            continue

        best_speaker, best_overlap = ranked[0]
        coverage_fraction = round(
            min(1.0, best_overlap / duration),
            4,
        )
        attributions.append(
            {
                "word_index": word_index,
                "status": "ATTRIBUTED",
                "attribution_reason": None,
                "speaker_id": best_speaker,
                "coverage_fraction": coverage_fraction,
                "confidence": coverage_fraction,
                "overlap": False,
                "simultaneous_overlap_fraction": 0.0,
                "sequential_handoff": sequential_handoff,
                "candidate_speaker_ids": [
                    candidate["speaker_id"]
                    for candidate in candidate_speakers
                ],
                "candidate_speakers": candidate_speakers,
            }
        )

    speakers = sorted(
        {
            str(turn["speaker_id"])
            for turn in [*regular_turns, *(exclusive_turns or [])]
            if isinstance(turn, dict) and turn.get("speaker_id") is not None
        }
    )
    attribution_count = sum(
        item["status"] == "ATTRIBUTED" for item in attributions
    )
    ambiguity_count = sum(
        item.get("attribution_reason") == "AMBIGUOUS_OVERLAP"
        for item in attributions
    )
    unknown_count = sum(
        item["status"] == "UNKNOWN"
        and item.get("attribution_reason") != "AMBIGUOUS_OVERLAP"
        for item in attributions
    )
    invalid_word_count = sum(
        item["status"] == "INVALID_WORD_GEOMETRY"
        for item in attributions
    )
    word_count = len(words)

    has_turn_output = bool(regular_turns or exclusive_turns)
    if not has_turn_output:
        quality_status = "EMPTY_OUTPUT"
    elif ambiguity_count or unknown_count or invalid_word_count:
        quality_status = "PARTIAL"
    else:
        quality_status = "COMPLETED"
    status = "FAILED" if word_count > 0 and not has_turn_output else "COMPLETED"
    stage = "POSTPROCESSING" if status == "FAILED" else "COMPLETE"

    def fraction(count: int) -> float:
        if word_count == 0:
            return 0.0
        return round(count / word_count, 4)

    safe_processing_sec = _finite_number(processing_sec) or 0.0
    safe_load_sec = _finite_number(load_sec) or 0.0
    safe_inference_wait_sec = _finite_number(inference_wait_sec) or 0.0
    safe_inference_sec = _finite_number(inference_sec) or 0.0
    return {
        "schema_version": DIARIZATION_SCHEMA_VERSION,
        "status": status,
        "quality_status": quality_status,
        "stage": stage,
        "model": model_id,
        "model_dependencies": (
            dict(PINNED_MODEL_DEPENDENCIES)
            if model_id == DEFAULT_MODEL_ID
            else {}
        ),
        "model_load_policy": (
            "BAKED_CACHE_ONLY"
            if model_id == DEFAULT_MODEL_ID
            else "CONFIGURED_OVERRIDE"
        ),
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
        "word_count": word_count,
        "attribution_count": attribution_count,
        "ambiguity_count": ambiguity_count,
        "unknown_count": unknown_count,
        "invalid_word_count": invalid_word_count,
        "attribution_fraction": fraction(attribution_count),
        "ambiguity_fraction": fraction(ambiguity_count),
        "unknown_fraction": fraction(unknown_count),
        "invalid_word_fraction": fraction(invalid_word_count),
        "coverage_fraction": fraction(
            attribution_count + ambiguity_count
        ),
        "discarded_invalid_turn_count": max(
            0,
            int(discarded_invalid_turn_count),
        ),
        "timing": {
            "load_sec": round(max(0.0, safe_load_sec), 3),
            "inference_wait_sec": round(
                max(0.0, safe_inference_wait_sec),
                3,
            ),
            "inference_sec": round(max(0.0, safe_inference_sec), 3),
            "processing_sec": round(max(0.0, safe_processing_sec), 3),
        },
        "processing_sec": round(max(0.0, safe_processing_sec), 3),
        **(
            {
                "error_code": "DIARIZATION_EMPTY_OUTPUT",
                "error": "DIARIZATION_EMPTY_OUTPUT",
            }
            if status == "FAILED"
            else {}
        ),
    }


def _capture_torch_precision_state(
    torch_module: Any,
) -> List[Tuple[Any, str, Any]]:
    """Snapshot pyannote-visible process-wide precision controls."""

    owners_and_attributes = []
    backends = getattr(torch_module, "backends", None)
    cuda_backend = getattr(backends, "cuda", None)
    cuda_matmul = getattr(cuda_backend, "matmul", None)
    cudnn = getattr(backends, "cudnn", None)
    if cuda_matmul is not None:
        owners_and_attributes.append((cuda_matmul, "allow_tf32"))
    if cudnn is not None:
        owners_and_attributes.extend(
            [
                (cudnn, "allow_tf32"),
                (cudnn, "deterministic"),
                (cudnn, "benchmark"),
            ]
        )

    state: List[Tuple[Any, str, Any]] = []
    for owner, attribute in owners_and_attributes:
        try:
            state.append((owner, attribute, getattr(owner, attribute)))
        except (AttributeError, RuntimeError):
            continue
    return state


def _restore_torch_precision_state(
    state: Sequence[Tuple[Any, str, Any]],
) -> None:
    first_error: Optional[Exception] = None
    for owner, attribute, value in state:
        try:
            setattr(owner, attribute, value)
        except Exception as error:  # Restore every remaining process flag.
            if first_error is None:
                first_error = error
    if first_error is not None:
        raise first_error


def _pin_pipeline_dependencies(
    config: Any,
    segmentation_path: str,
    embedding_path: str,
) -> Dict[str, Any]:
    """Validate the reviewed pipeline graph and replace mutable model refs."""

    if not isinstance(config, dict):
        raise RuntimeError("PYANNOTE_PIPELINE_CONFIG_INVALID")
    pipeline_config = config.get("pipeline")
    if not isinstance(pipeline_config, dict):
        raise RuntimeError("PYANNOTE_PIPELINE_CONFIG_INVALID")
    params = pipeline_config.get("params")
    if not isinstance(params, dict):
        raise RuntimeError("PYANNOTE_PIPELINE_CONFIG_INVALID")

    expected_repositories = {
        "segmentation": SEGMENTATION_MODEL_REPO,
        "embedding": EMBEDDING_MODEL_REPO,
    }
    for dependency, expected_repository in expected_repositories.items():
        configured = params.get(dependency)
        if (
            not isinstance(configured, str)
            or configured.split("@", 1)[0] != expected_repository
        ):
            raise RuntimeError("PYANNOTE_PIPELINE_DEPENDENCY_MISMATCH")

    params["segmentation"] = segmentation_path
    params["embedding"] = embedding_path
    return config


def _load_reviewed_pipeline(
    Pipeline: Any,
    token: Optional[str],
) -> Any:
    """Load the reviewed pipeline graph with all three revisions immutable."""

    import yaml
    from huggingface_hub import hf_hub_download

    auth_kwargs = {"token": token} if token else {}

    def cached_artifact(repo_id: str, filename: str, revision: str) -> str:
        artifact_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="model",
            revision=revision,
            local_files_only=True,
            **auth_kwargs,
        )
        artifact_path = os.fspath(artifact_path)
        if not os.path.isfile(artifact_path):
            raise RuntimeError("PYANNOTE_LOCAL_CACHE_INCOMPLETE")
        return artifact_path

    pipeline_config_source = cached_artifact(
        DEFAULT_PIPELINE_REPO,
        "config.yaml",
        DEFAULT_PIPELINE_REVISION,
    )
    segmentation_source = cached_artifact(
        SEGMENTATION_MODEL_REPO,
        "pytorch_model.bin",
        SEGMENTATION_MODEL_REVISION,
    )
    embedding_source = cached_artifact(
        EMBEDDING_MODEL_REPO,
        "pytorch_model.bin",
        EMBEDDING_MODEL_REVISION,
    )
    if "pyannote" not in embedding_source.lower():
        # PretrainedSpeakerEmbedding uses this marker to select the local
        # pyannote Model backend instead of a network-backed adapter.
        raise RuntimeError("PYANNOTE_PIPELINE_CONFIG_INVALID")
    with open(pipeline_config_source, "r", encoding="utf-8") as source:
        config = yaml.safe_load(source)

    with tempfile.TemporaryDirectory(prefix="w2l-pyannote-") as temp_dir:
        pinned_config = _pin_pipeline_dependencies(
            config,
            segmentation_source,
            embedding_source,
        )
        config_path = os.path.join(temp_dir, "config.yaml")
        with open(config_path, "w", encoding="utf-8") as destination:
            yaml.safe_dump(pinned_config, destination, sort_keys=False)
        os.chmod(config_path, 0o600)
        return Pipeline.from_pretrained(
            config_path,
            use_auth_token=token,
        )


def _load_configured_pipeline(
    Pipeline: Any,
    model_id: str,
    token: Optional[str],
) -> Any:
    if model_id == DEFAULT_MODEL_ID:
        return _load_reviewed_pipeline(Pipeline, token)
    if token:
        try:
            return Pipeline.from_pretrained(model_id, token=token)
        except TypeError:
            # Preserve the environment override for pyannote.audio 3.x while
            # leaving a future token-keyword pipeline as an explicit challenger.
            return Pipeline.from_pretrained(
                model_id,
                use_auth_token=token,
            )
    return Pipeline.from_pretrained(model_id)


class SpeakerDiarizer:
    """Lazy pyannote pipeline wrapper with fail-soft sidecar output."""

    def __init__(self, model_id: Optional[str] = None):
        self.model_id = model_id or os.environ.get(
            "PYANNOTE_DIARIZATION_MODEL",
            DEFAULT_MODEL_ID,
        )
        self.pipeline = None
        self.device = None
        self.load_sec = 0.0
        self._setup_lock = threading.Lock()
        # Precision flags are process-wide, so all diarizer instances share
        # the same single-flight inference boundary.
        self._inference_lock = _INFERENCE_LOCK

    @staticmethod
    def _package_version() -> str:
        try:
            return version("pyannote.audio")
        except PackageNotFoundError:
            return "missing"

    def setup(self, device: str) -> None:
        if self.pipeline is not None and self.device == device:
            return

        with self._setup_lock:
            if self.pipeline is not None and self.device == device:
                return

            token = (
                os.environ.get("HUGGINGFACE_TOKEN")
                or os.environ.get("HF_TOKEN")
                or None
            )

            # Keep customer audio out of optional package telemetry by default.
            os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "0")
            import torch
            # PyTorch 2.6+ safely defaults checkpoint loading to
            # ``weights_only=True``. The official pyannote segmentation-3.0
            # checkpoint contains TorchVersion metadata, so allowlist that one
            # inert framework value instead of disabling safe loading for the
            # entire checkpoint.
            # pyannote.audio 3.3.2 still imports hf_hub_download with the
            # removed use_auth_token keyword. Install the process-local
            # token->token adapter before pyannote copies that callable.
            install_legacy_use_auth_token_compat()
            from pyannote.audio import Pipeline
            from pyannote.audio.core.task import (
                Problem,
                Resolution,
                Specifications,
            )
            from torch.torch_version import TorchVersion

            if hasattr(torch.serialization, "add_safe_globals"):
                torch.serialization.add_safe_globals(
                    [TorchVersion, Specifications, Problem, Resolution]
                )

            with serialized_model_load("speaker-diarizer"):
                if self.pipeline is not None and self.device == device:
                    return
                load_started = time.perf_counter()
                _log_event(
                    "stage_started",
                    stage="LOAD",
                    device=device,
                )
                pipeline = _load_configured_pipeline(
                    Pipeline,
                    self.model_id,
                    token,
                )
                if pipeline is None:
                    raise RuntimeError("PYANNOTE_MODEL_UNAVAILABLE_OR_GATED")
                pipeline.to(torch.device(device))
                self.pipeline = pipeline
                self.device = device
                self.load_sec = time.perf_counter() - load_started
                _log_event(
                    "stage_completed",
                    stage="LOAD",
                    device=device,
                    load_sec=round(self.load_sec, 3),
                )

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
        stage = "VALIDATION"
        load_sec = 0.0
        inference_wait_sec = 0.0
        inference_sec = 0.0
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

            stage = "LOAD"
            load_started = time.perf_counter()
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.setup(device)
            load_sec = time.perf_counter() - load_started
            kwargs: Dict[str, int] = {}
            if min_speakers is not None and min_speakers > 0:
                kwargs["min_speakers"] = int(min_speakers)
            if max_speakers is not None and max_speakers > 0:
                kwargs["max_speakers"] = int(max_speakers)

            stage = "INFERENCE"
            inference_wait_started = time.perf_counter()
            _log_event(
                "stage_started",
                stage=stage,
                device=device,
            )
            with self._inference_lock:
                inference_wait_sec = (
                    time.perf_counter() - inference_wait_started
                )
                precision_state = _capture_torch_precision_state(torch)
                inference_started = time.perf_counter()
                try:
                    output = self.pipeline(audio_path, **kwargs)
                finally:
                    inference_sec = time.perf_counter() - inference_started
                    _restore_torch_precision_state(precision_state)
            _log_event(
                "stage_completed",
                stage=stage,
                device=device,
                inference_wait_sec=round(inference_wait_sec, 3),
                inference_sec=round(inference_sec, 3),
            )

            stage = "POSTPROCESSING"
            regular_annotation = getattr(output, "speaker_diarization", output)
            regular_stats: Dict[str, int] = {}
            regular, label_map = _canonicalize_turns(
                _iter_annotation(regular_annotation, regular_stats),
                stats=regular_stats,
            )
            exclusive_annotation = getattr(
                output,
                "exclusive_speaker_diarization",
                None,
            )
            exclusive_stats: Dict[str, int] = {}
            exclusive, _ = _canonicalize_turns(
                _iter_annotation(exclusive_annotation, exclusive_stats),
                label_map,
                stats=exclusive_stats,
            )
            processing_sec = time.perf_counter() - started
            sidecar = build_diarization_sidecar(
                words,
                regular,
                exclusive,
                model_id=self.model_id,
                package_version=self._package_version(),
                device=device,
                processing_sec=processing_sec,
                load_sec=load_sec,
                inference_wait_sec=inference_wait_sec,
                inference_sec=inference_sec,
                discarded_invalid_turn_count=(
                    regular_stats.get("discarded_invalid_turn_count", 0)
                    + exclusive_stats.get("discarded_invalid_turn_count", 0)
                ),
            )
            processing_sec = time.perf_counter() - started
            rounded_processing_sec = round(processing_sec, 3)
            sidecar["processing_sec"] = rounded_processing_sec
            sidecar["timing"]["processing_sec"] = rounded_processing_sec
            log_fields = {
                "stage": sidecar["stage"],
                "status": sidecar["status"],
                "quality_status": sidecar["quality_status"],
                "device": device,
                "load_sec": round(load_sec, 3),
                "inference_wait_sec": round(inference_wait_sec, 3),
                "inference_sec": round(inference_sec, 3),
                "processing_sec": round(processing_sec, 3),
            }
            if sidecar["status"] == "FAILED":
                log_fields["error_code"] = sidecar["error_code"]
                _log_event("request_failed", **log_fields)
            else:
                _log_event("request_completed", **log_fields)
            return sidecar
        except Exception as error:  # Sidecar failure must not lose transcription.
            if stage == "LOAD" and load_sec == 0.0:
                load_sec = time.perf_counter() - load_started
            processing_sec = time.perf_counter() - started
            error_code = _stable_error_code(error, stage)
            _log_event(
                "request_failed",
                stage=stage,
                status="FAILED",
                device=device,
                error_code=error_code,
                load_sec=round(load_sec, 3),
                inference_wait_sec=round(inference_wait_sec, 3),
                inference_sec=round(inference_sec, 3),
                processing_sec=round(processing_sec, 3),
            )
            return {
                "schema_version": DIARIZATION_SCHEMA_VERSION,
                "status": "FAILED",
                "quality_status": "EMPTY_OUTPUT",
                "stage": stage,
                "model": self.model_id,
                "model_dependencies": (
                    dict(PINNED_MODEL_DEPENDENCIES)
                    if self.model_id == DEFAULT_MODEL_ID
                    else {}
                ),
                "model_load_policy": (
                    "BAKED_CACHE_ONLY"
                    if self.model_id == DEFAULT_MODEL_ID
                    else "CONFIGURED_OVERRIDE"
                ),
                "pyannote_audio_version": self._package_version(),
                "device": device,
                "identity_scope": "CHUNK_LOCAL_UNSTABLE",
                "boundary_authority": False,
                "transcript_geometry_mutated": False,
                "error_code": error_code,
                # Preserve the legacy field without exposing arbitrary
                # dependency messages that can contain paths or credentials.
                "error": error_code,
                "timing": {
                    "load_sec": round(max(0.0, load_sec), 3),
                    "inference_wait_sec": round(
                        max(0.0, inference_wait_sec),
                        3,
                    ),
                    "inference_sec": round(max(0.0, inference_sec), 3),
                    "processing_sec": round(
                        max(0.0, processing_sec),
                        3,
                    ),
                },
                "processing_sec": round(max(0.0, processing_sec), 3),
            }
