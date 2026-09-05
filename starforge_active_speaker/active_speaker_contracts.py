"""Pure contracts for the isolated Starforge active-speaker lab runtime.

This module intentionally depends only on the Python standard library so its
validation, tracking, and output semantics can be tested without media files,
model weights, OpenCV, NumPy, or PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "starforge-active-speaker-observation-v1"
AUTHORITY = "DIAGNOSTIC_OBSERVATION_ONLY"
CROP_AUTHORITY = "NONE"
VIDEO_FRAMES_PER_SECOND = 25
AUDIO_SAMPLE_RATE_HZ = 16_000
AUDIO_CHANNELS = 1
AUDIO_SAMPLES_PER_VIDEO_FRAME = AUDIO_SAMPLE_RATE_HZ // VIDEO_FRAMES_PER_SECOND

LRASD_EXECUTED_SOURCE_FILES = (
    "loss.py",
    "model/Classifier.py",
    "model/Encoder.py",
    "model/Model.py",
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CONTENT_ID_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_GIT_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")


class ContractViolation(RuntimeError):
    """Raised when an input cannot satisfy the fail-closed lab contract."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def content_identity(value: Any) -> str:
    return f"sha256:{sha256_bytes(canonical_json_bytes(value))}"


def validate_sha256(value: str, label: str) -> str:
    if _SHA256_RE.fullmatch(value) is None:
        raise ContractViolation(f"{label} must be exactly 64 lowercase hex characters")
    return value


def validate_content_identity(value: str, label: str) -> str:
    if _CONTENT_ID_RE.fullmatch(value) is None:
        raise ContractViolation(
            f"{label} must be exactly sha256: followed by 64 lowercase hex characters"
        )
    return value


def validate_git_revision(value: str) -> str:
    if _GIT_REVISION_RE.fullmatch(value) is None:
        raise ContractViolation(
            "LR-ASD revision must be an exact 40-character lowercase Git commit"
        )
    return value


@dataclass(frozen=True)
class SourceInterval:
    """Immutable source provenance for an extracted diagnostic input."""

    source_video_sha256: str
    start_microseconds: int
    end_microseconds: int

    def __post_init__(self) -> None:
        validate_sha256(self.source_video_sha256, "original source video SHA-256")
        if self.start_microseconds < 0:
            raise ContractViolation("source interval start must be non-negative")
        if self.end_microseconds <= self.start_microseconds:
            raise ContractViolation("source interval end must be after its start")

    @property
    def duration_microseconds(self) -> int:
        return self.end_microseconds - self.start_microseconds

    def as_json(self) -> dict[str, Any]:
        return {
            "endMicrosecondsExclusive": self.end_microseconds,
            "originalSourceVideoSha256": self.source_video_sha256,
            "startMicrosecondsInclusive": self.start_microseconds,
        }


def strict_regular_file(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise ContractViolation(f"{label} must not be a symbolic link: {path}")
    if not path.is_file():
        raise ContractViolation(f"{label} is not a regular file: {path}")
    return path


def sha256_file(path: Path, label: str) -> tuple[str, int]:
    strict_regular_file(path, label)
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def require_file_hash(path: Path, expected_sha256: str, label: str) -> int:
    expected = validate_sha256(expected_sha256, f"{label} SHA-256")
    actual, size = sha256_file(path, label)
    if actual != expected:
        raise ContractViolation(
            f"{label} SHA-256 mismatch: expected {expected}, received {actual}"
        )
    return size


def lrasd_source_manifest(root: Path) -> list[dict[str, Any]]:
    if root.is_symlink() or not root.is_dir():
        raise ContractViolation(f"LR-ASD root must be a non-symlink directory: {root}")

    entries: list[dict[str, Any]] = []
    for relative_path in LRASD_EXECUTED_SOURCE_FILES:
        path = root / relative_path
        digest, size = sha256_file(path, f"LR-ASD source {relative_path}")
        entries.append(
            {
                "bytes": size,
                "path": relative_path,
                "sha256": digest,
            }
        )
    return entries


def lrasd_source_identity(root: Path) -> tuple[str, list[dict[str, Any]]]:
    manifest = lrasd_source_manifest(root)
    return sha256_bytes(canonical_json_bytes(manifest)), manifest


@dataclass(frozen=True)
class FaceBox:
    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        values = (self.x1, self.y1, self.x2, self.y2)
        if not all(math.isfinite(value) for value in values):
            raise ContractViolation("face box coordinates must be finite")
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ContractViolation("face box must have positive width and height")

    def as_json(self) -> dict[str, float]:
        return {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
        }


@dataclass(frozen=True)
class FaceDetection:
    frame_index: int
    shot_index: int
    box: FaceBox
    detection_score: float

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            raise ContractViolation("face detection frame index must be non-negative")
        if self.shot_index < 0:
            raise ContractViolation("face detection shot index must be non-negative")
        if not math.isfinite(self.detection_score):
            raise ContractViolation("face detection score must be finite")


@dataclass(frozen=True)
class FaceTrack:
    track_id: str
    shot_index: int
    detections: tuple[FaceDetection, ...]

    def __post_init__(self) -> None:
        if not self.detections:
            raise ContractViolation("face track must contain at least one detection")
        if any(item.shot_index != self.shot_index for item in self.detections):
            raise ContractViolation("face track cannot cross a shot boundary")
        frames = [item.frame_index for item in self.detections]
        if frames != sorted(frames) or len(frames) != len(set(frames)):
            raise ContractViolation(
                "face track detection frames must be strictly increasing"
            )


def box_iou(left: FaceBox, right: FaceBox) -> float:
    intersection_width = max(0.0, min(left.x2, right.x2) - max(left.x1, right.x1))
    intersection_height = max(0.0, min(left.y2, right.y2) - max(left.y1, right.y1))
    intersection = intersection_width * intersection_height
    union = (
        (left.x2 - left.x1) * (left.y2 - left.y1)
        + (right.x2 - right.x1) * (right.y2 - right.y1)
        - intersection
    )
    if union <= 0:
        raise ContractViolation("face box union must be positive")
    return intersection / union


class DeterministicShotTracker:
    """Greedy deterministic IoU tracking that never carries identity over a cut."""

    def __init__(
        self,
        *,
        minimum_iou: float = 0.5,
        maximum_gap_frames: int = 10,
        minimum_detection_frames: int = 11,
    ) -> None:
        if not math.isfinite(minimum_iou) or not 0 < minimum_iou <= 1:
            raise ContractViolation("minimum IoU must be within (0, 1]")
        if maximum_gap_frames < 0:
            raise ContractViolation("maximum gap frames must be non-negative")
        if minimum_detection_frames < 1:
            raise ContractViolation("minimum detection frames must be positive")
        self.minimum_iou = minimum_iou
        self.maximum_gap_frames = maximum_gap_frames
        self.minimum_detection_frames = minimum_detection_frames

    @staticmethod
    def _detection_key(item: FaceDetection) -> tuple[float, ...]:
        return (
            item.box.x1,
            item.box.y1,
            item.box.x2,
            item.box.y2,
            -item.detection_score,
        )

    def track(self, detections: Iterable[FaceDetection]) -> tuple[FaceTrack, ...]:
        grouped: dict[tuple[int, int], list[FaceDetection]] = {}
        for detection in detections:
            grouped.setdefault(
                (detection.shot_index, detection.frame_index), []
            ).append(detection)

        mutable_tracks: list[dict[str, Any]] = []
        next_track_by_shot: dict[int, int] = {}

        for (shot_index, frame_index), frame_detections in sorted(grouped.items()):
            ordered_detections = sorted(frame_detections, key=self._detection_key)
            candidate_track_indexes = [
                index
                for index, track in enumerate(mutable_tracks)
                if track["shotIndex"] == shot_index
                and frame_index - track["detections"][-1].frame_index
                <= self.maximum_gap_frames
            ]

            candidates: list[tuple[float, str, int, int]] = []
            for track_index in candidate_track_indexes:
                track = mutable_tracks[track_index]
                last_box = track["detections"][-1].box
                for detection_index, detection in enumerate(ordered_detections):
                    overlap = box_iou(last_box, detection.box)
                    if overlap >= self.minimum_iou:
                        candidates.append(
                            (
                                -overlap,
                                track["trackId"],
                                detection_index,
                                track_index,
                            )
                        )

            used_tracks: set[int] = set()
            used_detections: set[int] = set()
            for _, _, detection_index, track_index in sorted(candidates):
                if track_index in used_tracks or detection_index in used_detections:
                    continue
                mutable_tracks[track_index]["detections"].append(
                    ordered_detections[detection_index]
                )
                used_tracks.add(track_index)
                used_detections.add(detection_index)

            for detection_index, detection in enumerate(ordered_detections):
                if detection_index in used_detections:
                    continue
                next_index = next_track_by_shot.get(shot_index, 0)
                next_track_by_shot[shot_index] = next_index + 1
                mutable_tracks.append(
                    {
                        "trackId": f"shot-{shot_index:04d}-track-{next_index:04d}",
                        "shotIndex": shot_index,
                        "detections": [detection],
                    }
                )

        tracks = [
            FaceTrack(
                track_id=item["trackId"],
                shot_index=item["shotIndex"],
                detections=tuple(item["detections"]),
            )
            for item in mutable_tracks
            if len(item["detections"]) >= self.minimum_detection_frames
        ]
        return tuple(sorted(tracks, key=lambda item: item.track_id))


@dataclass(frozen=True)
class RawScoreSample:
    track_id: str
    frame_index: int
    raw_speaking_score: float

    def __post_init__(self) -> None:
        if not self.track_id:
            raise ContractViolation("raw score sample must name a track")
        if self.frame_index < 0:
            raise ContractViolation("raw score frame index must be non-negative")
        if not math.isfinite(self.raw_speaking_score):
            raise ContractViolation("raw speaking score must be finite")

    def as_json(self) -> dict[str, Any]:
        return {
            "frameIndex": self.frame_index,
            "pts": {
                "denominator": VIDEO_FRAMES_PER_SECOND,
                "numerator": self.frame_index,
            },
            "rawSpeakingScore": self.raw_speaking_score,
        }


class RawScoreLedger:
    """Builds a closed one-score-per-admitted-frame observation ledger."""

    @staticmethod
    def build(
        admitted_frames_by_track: Mapping[str, Sequence[int]],
        scores_by_track: Mapping[str, Sequence[float]],
    ) -> list[dict[str, Any]]:
        if set(admitted_frames_by_track) != set(scores_by_track):
            raise ContractViolation(
                "raw score tracks must exactly match admitted face tracks"
            )

        result: list[dict[str, Any]] = []
        for track_id in sorted(admitted_frames_by_track):
            frames = list(admitted_frames_by_track[track_id])
            scores = list(scores_by_track[track_id])
            if not frames:
                raise ContractViolation("admitted face track must not be empty")
            if frames != sorted(frames) or len(frames) != len(set(frames)):
                raise ContractViolation(
                    "admitted face track frames must be strictly increasing"
                )
            if len(frames) != len(scores):
                raise ContractViolation(
                    f"raw score count mismatch for {track_id}: "
                    f"expected {len(frames)}, received {len(scores)}"
                )
            samples = [
                RawScoreSample(track_id, frame, float(score)).as_json()
                for frame, score in zip(frames, scores, strict=True)
            ]
            result.append(
                {
                    "samples": samples,
                    "trackId": track_id,
                }
            )
        return result


def assert_no_decision_authority(value: Any) -> None:
    exact_forbidden_keys = {
        "activespeaker",
        "classification",
        "crop",
        "cropbox",
        "cropcenter",
        "cropcoordinates",
        "cropinstruction",
        "croppath",
        "decision",
        "isspeaking",
        "selectedface",
        "selectedspeaker",
        "selectedtrack",
        "speakerdecision",
        "speakinglabel",
        "speakingprobability",
        "speakingstatus",
        "targetface",
        "targetspeaker",
        "targettrack",
    }

    def normalized_key(key: Any) -> str:
        return re.sub(r"[^a-z0-9]", "", str(key).lower())

    def key_is_forbidden(key: Any) -> bool:
        normalized = normalized_key(key)
        if normalized in exact_forbidden_keys:
            return True
        if "confidence" in normalized or "probability" in normalized:
            return True
        if "decision" in normalized:
            return True
        if normalized.endswith("speakingscore") and normalized != "rawspeakingscore":
            return True
        return False

    def visit(node: Any) -> None:
        if isinstance(node, Mapping):
            for key, child in node.items():
                if key_is_forbidden(key):
                    raise ContractViolation(
                        f"diagnostic observation cannot contain decision field {key}"
                    )
                visit(child)
        elif isinstance(node, Sequence) and not isinstance(
            node, (str, bytes, bytearray)
        ):
            for child in node:
                visit(child)

    visit(value)


def success_envelope(
    *,
    identities: Mapping[str, str],
    clocks: Mapping[str, Any],
    tracks: Sequence[Mapping[str, Any]],
    score_ledger: Sequence[Mapping[str, Any]],
    outputs: Mapping[str, Any],
    measurements: Mapping[str, Any],
) -> dict[str, Any]:
    required_identities = {
        "clockIdentity",
        "modelIdentity",
        "observationIdentity",
        "runIdentity",
        "runtimeIdentity",
    }
    if set(identities) != required_identities:
        raise ContractViolation(
            "success identities must exactly bind clock, model, observation, runtime, and run"
        )
    for name, identity in identities.items():
        validate_content_identity(identity, f"success identity {name}")

    result = {
        "authority": AUTHORITY,
        "clocks": dict(clocks),
        "cropAuthority": CROP_AUTHORITY,
        "identities": dict(identities),
        "measurements": dict(measurements),
        "outputs": dict(outputs),
        "rawScoreSemantics": (
            "Uncalibrated LR-ASD class-1 logit averaged across explicit context "
            "windows; it is not a probability, confidence, speaker identity, "
            "crop decision, or production authority."
        ),
        "schemaVersion": SCHEMA_VERSION,
        "scoreLedger": list(score_ledger),
        "tracks": list(tracks),
    }
    assert_no_decision_authority(result)
    return result
