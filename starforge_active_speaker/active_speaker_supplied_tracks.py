"""Strict supplied-track and v2 receipt contracts for the isolated ASD lab.

The v2 scorer consumes a complete, content-addressed track manifest.  It never
repairs producer output or turns malformed evidence into an empty face set.
This module stays standard-library-only so the authority boundary is testable
without OpenCV, NumPy, PyTorch, or model assets.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping, Sequence

from active_speaker_contracts import (
    AUDIO_CHANNELS,
    AUDIO_SAMPLE_RATE_HZ,
    AUDIO_SAMPLES_PER_VIDEO_FRAME,
    AUTHORITY,
    CROP_AUTHORITY,
    ContractViolation,
    FaceBox,
    LRASD_EXECUTED_SOURCE_FILES,
    SourceInterval,
    VIDEO_FRAMES_PER_SECOND,
    assert_no_decision_authority,
    canonical_json_bytes,
    content_identity,
    sha256_bytes,
    sha256_file,
    validate_content_identity,
    validate_git_revision,
    validate_sha256,
)


SUPPLIED_TRACK_SCHEMA_VERSION = "starforge-active-speaker-supplied-tracks-v2"
V2_OBSERVATION_SCHEMA_VERSION = "starforge-active-speaker-observation-v2"
V2_FAILURE_SCHEMA_VERSION = "starforge-active-speaker-failure-v2"
V2_RUNTIME_VERSION = "starforge-active-speaker-local-v2"
V1_OBSERVATION_SCHEMA_VERSION = "starforge-active-speaker-observation-v1"
V1_RUNTIME_VERSION = "starforge-active-speaker-local-v1"
V1_RUNTIME_CLOSURE_FILES = (
    "Dockerfile",
    "LR-ASD-LICENSE.txt",
    "active_speaker_contracts.py",
    "active_speaker_media.py",
    "active_speaker_model.py",
    "active_speaker_runtime.py",
    "requirements.lock.txt",
)
LRASD_LICENSE_SHA256 = "1ea9714e15424fb28d551675751172ba635b45c9d137a203a9988937e8215931"
CANONICAL_TRACK_PRODUCER_KIND = "starforge-canonical-face-tracks-v1"
MIRROR_TRACK_PRODUCER_KIND = "starforge-horizontal-mirror-face-tracks-v1"
_TRACK_ID_RE = re.compile(r"^shot-([0-9]{4,})-track-[0-9]{4,}$")

LRASD_V2_EXECUTED_SOURCE_FILES = LRASD_EXECUTED_SOURCE_FILES


@dataclass(frozen=True)
class SuppliedTrackLimits:
    maximum_manifest_bytes: int = 32 * 1024 * 1024
    maximum_input_bytes: int = 2 * 1024 * 1024 * 1024
    maximum_checkpoint_bytes: int = 64 * 1024 * 1024
    maximum_frames: int = 4_500
    maximum_frame_pixels: int = 3_840 * 2_160
    maximum_tracks: int = 256
    maximum_track_frames: int = 100_000

    def __post_init__(self) -> None:
        values = (
            ("maximum manifest bytes", self.maximum_manifest_bytes, 32 * 1024 * 1024),
            ("maximum input bytes", self.maximum_input_bytes, 2 * 1024 * 1024 * 1024),
            ("maximum checkpoint bytes", self.maximum_checkpoint_bytes, 64 * 1024 * 1024),
            ("maximum frames", self.maximum_frames, 4_500),
            ("maximum frame pixels", self.maximum_frame_pixels, 3_840 * 2_160),
            ("maximum tracks", self.maximum_tracks, 256),
            ("maximum track frames", self.maximum_track_frames, 100_000),
        )
        for label, value, ceiling in values:
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 1
                or value > ceiling
            ):
                raise ContractViolation(
                    f"v2 {label} must be an integer within [1, {ceiling}]"
                )

    def as_json(self) -> dict[str, int]:
        return {
            "maximumCheckpointBytes": self.maximum_checkpoint_bytes,
            "maximumFramePixels": self.maximum_frame_pixels,
            "maximumFrames": self.maximum_frames,
            "maximumInputBytes": self.maximum_input_bytes,
            "maximumManifestBytes": self.maximum_manifest_bytes,
            "maximumTrackFrames": self.maximum_track_frames,
            "maximumTracks": self.maximum_tracks,
        }


@dataclass(frozen=True)
class SuppliedShot:
    shot_index: int
    start_frame: int
    end_frame: int

    def contains(self, frame_index: int) -> bool:
        return self.start_frame <= frame_index < self.end_frame

    def as_json(self) -> dict[str, int]:
        return {
            "endFrameExclusive": self.end_frame,
            "shotIndex": self.shot_index,
            "startFrameInclusive": self.start_frame,
        }


@dataclass(frozen=True)
class SuppliedTrackFrame:
    frame_index: int
    face_box: FaceBox
    is_detector_observation: bool

    def as_json(self) -> dict[str, Any]:
        return {
            "faceBox": self.face_box.as_json(),
            "frameIndex": self.frame_index,
            "isDetectorObservation": self.is_detector_observation,
            "pts": {
                "denominator": VIDEO_FRAMES_PER_SECOND,
                "numerator": self.frame_index,
            },
        }


@dataclass(frozen=True)
class SuppliedTrack:
    track_id: str
    shot_index: int
    frames: tuple[SuppliedTrackFrame, ...]

    def as_json(self) -> dict[str, Any]:
        return {
            "frames": [frame.as_json() for frame in self.frames],
            "shotIndex": self.shot_index,
            "trackId": self.track_id,
        }


@dataclass(frozen=True)
class V1ObservationReceipt:
    file_sha256: str
    file_bytes: int
    identities: Mapping[str, str]
    clocks: Mapping[str, Any]
    tracks: tuple[Mapping[str, Any], ...]
    tracking_policy: Mapping[str, Any]
    model: Mapping[str, Any]
    runtime: Mapping[str, Any]

    def lineage_record(self) -> dict[str, Any]:
        return {
            "bytes": self.file_bytes,
            "identities": dict(self.identities),
            "schemaVersion": V1_OBSERVATION_SCHEMA_VERSION,
            "sha256": self.file_sha256,
        }


@dataclass(frozen=True)
class SuppliedTrackManifest:
    file_sha256: str
    file_bytes: int
    content_identity: str
    clock_identity: str
    clock: Mapping[str, Any]
    producer: Mapping[str, Any]
    shots: tuple[SuppliedShot, ...]
    tracks: tuple[SuppliedTrack, ...]

    @property
    def frame_count(self) -> int:
        return int(self.clock["video"]["frameCount"])

    @property
    def width(self) -> int:
        return int(self.clock["video"]["width"])

    @property
    def height(self) -> int:
        return int(self.clock["video"]["height"])

    @property
    def track_frame_count(self) -> int:
        return sum(len(track.frames) for track in self.tracks)

    def receipt_record(self) -> dict[str, Any]:
        return {
            "bytes": self.file_bytes,
            "clockIdentity": self.clock_identity,
            "contentIdentity": self.content_identity,
            "producer": dict(self.producer),
            "schemaVersion": SUPPLIED_TRACK_SCHEMA_VERSION,
            "sha256": self.file_sha256,
            "status": "COMPLETE",
            "trackCount": len(self.tracks),
            "trackFrameCount": self.track_frame_count,
        }


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractViolation(f"{label} must be an object")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], label: str
) -> None:
    received = set(value)
    if received != expected:
        raise ContractViolation(
            f"{label} keys mismatch: missing={sorted(expected - received)}, "
            f"unexpected={sorted(received - expected)}"
        )


def _require_integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ContractViolation(f"{label} must be an integer >= {minimum}")
    return value


def _require_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractViolation(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ContractViolation(f"{label} must be a finite number")
    return result


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractViolation(f"{label} must be a non-empty string")
    return value


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractViolation(f"supplied-track JSON contains duplicate key {key}")
        result[key] = value
    return result


def _reject_nonfinite_json(token: str) -> None:
    raise ContractViolation(f"supplied-track JSON contains non-finite number {token}")


def _read_bounded_regular_file(
    path: Path,
    maximum_bytes: int,
    *,
    label: str = "supplied-track manifest",
) -> bytes:
    try:
        before = path.stat(follow_symlinks=False)
    except OSError as error:
        raise ContractViolation(f"cannot stat {label}: {error}") from error
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ContractViolation(f"{label} must be a non-symlink regular file")
    if before.st_size < 1 or before.st_size > maximum_bytes:
        raise ContractViolation(
            f"{label} bytes must be within [1, {maximum_bytes}]"
        )
    try:
        with path.open("rb") as handle:
            opened = os.fstat(handle.fileno())
            if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
                raise ContractViolation(f"{label} changed while opening")
            payload = handle.read(maximum_bytes + 1)
            after = os.fstat(handle.fileno())
    except OSError as error:
        raise ContractViolation(f"cannot read {label}: {error}") from error
    if len(payload) > maximum_bytes or len(payload) != before.st_size:
        raise ContractViolation(f"{label} size changed while reading")
    if (after.st_size, after.st_mtime_ns) != (before.st_size, before.st_mtime_ns):
        raise ContractViolation(f"{label} changed while reading")
    return payload


def _decode_json(payload: bytes) -> Mapping[str, Any]:
    try:
        text = payload.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_json,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractViolation("supplied-track manifest is not strict UTF-8 JSON") from error
    return _require_mapping(value, "supplied-track manifest")


def _parse_pts(value: Any, frame_index: int, label: str) -> None:
    mapping = _require_mapping(value, label)
    _require_exact_keys(mapping, {"denominator", "numerator"}, label)
    denominator = _require_integer(mapping["denominator"], f"{label} denominator", minimum=1)
    numerator = _require_integer(mapping["numerator"], f"{label} numerator")
    if denominator != VIDEO_FRAMES_PER_SECOND or numerator != frame_index:
        raise ContractViolation(f"{label} must equal exact frameIndex/25")


def _parse_shots(value: Any, frame_count: int) -> tuple[SuppliedShot, ...]:
    if not isinstance(value, list) or not value:
        raise ContractViolation("supplied shots must be a non-empty array")
    shots: list[SuppliedShot] = []
    next_start = 0
    for expected_index, item in enumerate(value):
        mapping = _require_mapping(item, f"shot {expected_index}")
        _require_exact_keys(
            mapping,
            {"endFrameExclusive", "shotIndex", "startFrameInclusive"},
            f"shot {expected_index}",
        )
        shot_index = _require_integer(mapping["shotIndex"], "shot index")
        start = _require_integer(mapping["startFrameInclusive"], "shot start")
        end = _require_integer(mapping["endFrameExclusive"], "shot end", minimum=1)
        if shot_index != expected_index or start != next_start or end <= start:
            raise ContractViolation("supplied shots must be contiguous ordered half-open ranges")
        shots.append(SuppliedShot(shot_index, start, end))
        next_start = end
    if next_start != frame_count:
        raise ContractViolation("supplied shots must cover every prepared video frame exactly")
    return tuple(shots)


def _validate_clock(
    value: Any, limits: SuppliedTrackLimits
) -> tuple[Mapping[str, Any], tuple[SuppliedShot, ...]]:
    clock = _require_mapping(value, "supplied clock")
    _require_exact_keys(
        clock,
        {"audio", "inputStreams", "preparedInput", "shots", "sourceInterval", "video"},
        "supplied clock",
    )
    video = _require_mapping(clock["video"], "supplied video clock")
    _require_exact_keys(video, {"frameCount", "frameRate", "height", "width"}, "video clock")
    frame_count = _require_integer(video["frameCount"], "video frame count", minimum=1)
    width = _require_integer(video["width"], "video width", minimum=2)
    height = _require_integer(video["height"], "video height", minimum=2)
    if frame_count > limits.maximum_frames or width * height > limits.maximum_frame_pixels:
        raise ContractViolation("supplied video exceeds v2 preflight limits")
    frame_rate = _require_mapping(video["frameRate"], "video frame rate")
    _require_exact_keys(frame_rate, {"denominator", "numerator"}, "video frame rate")
    if frame_rate != {"denominator": 1, "numerator": VIDEO_FRAMES_PER_SECOND}:
        raise ContractViolation("supplied video frame rate must be exactly 25/1")
    _validate_clock_audio(clock["audio"], frame_count)
    _validate_clock_input(clock["preparedInput"], clock["inputStreams"], limits)
    _validate_source_interval(clock["sourceInterval"], frame_count)
    return clock, _parse_shots(clock["shots"], frame_count)


def _validate_clock_audio(value: Any, frame_count: int) -> None:
    audio = _require_mapping(value, "supplied audio clock")
    _require_exact_keys(audio, {"channels", "sampleCount", "sampleRateHz"}, "audio clock")
    channels = _require_integer(audio["channels"], "audio channels", minimum=1)
    sample_rate = _require_integer(audio["sampleRateHz"], "audio sample rate", minimum=1)
    samples = _require_integer(audio["sampleCount"], "audio sample count", minimum=1)
    if channels != AUDIO_CHANNELS or sample_rate != AUDIO_SAMPLE_RATE_HZ:
        raise ContractViolation("supplied audio clock must be exact 16kHz mono")
    if samples != frame_count * AUDIO_SAMPLES_PER_VIDEO_FRAME:
        raise ContractViolation("supplied audio samples must exactly cover the video clock")


def _validate_clock_input(
    prepared_value: Any,
    streams_value: Any,
    limits: SuppliedTrackLimits,
) -> None:
    prepared = _require_mapping(prepared_value, "prepared input")
    _require_exact_keys(prepared, {"bytes", "sha256"}, "prepared input")
    prepared_bytes = _require_integer(prepared["bytes"], "prepared input bytes", minimum=1)
    if prepared_bytes > limits.maximum_input_bytes:
        raise ContractViolation("prepared input exceeds v2 byte limit")
    validate_sha256(_require_string(prepared["sha256"], "prepared input SHA-256"), "prepared input SHA-256")
    streams = _require_mapping(streams_value, "input streams")
    _require_exact_keys(streams, {"audioStreamIndex", "origins", "videoStreamIndex"}, "input streams")
    audio_index = _require_integer(streams["audioStreamIndex"], "audio stream index")
    video_index = _require_integer(streams["videoStreamIndex"], "video stream index")
    if audio_index == video_index:
        raise ContractViolation("supplied audio and video stream indexes must differ")
    origins = _require_mapping(streams["origins"], "input stream origins")
    _require_exact_keys(
        origins,
        {"audioOffsetFromVideoFrameZero", "audioPresentationDuration", "audioStream", "timelineValidation", "videoStream"},
        "input stream origins",
    )


def _validate_source_interval(value: Any, frame_count: int) -> None:
    source = _require_mapping(value, "source interval")
    _require_exact_keys(
        source,
        {"endMicrosecondsExclusive", "originalSourceVideoBytes", "originalSourceVideoSha256", "startMicrosecondsInclusive"},
        "source interval",
    )
    interval = SourceInterval(
        _require_string(source["originalSourceVideoSha256"], "source video SHA-256"),
        _require_integer(source["startMicrosecondsInclusive"], "source interval start"),
        _require_integer(source["endMicrosecondsExclusive"], "source interval end", minimum=1),
    )
    _require_integer(source["originalSourceVideoBytes"], "source video bytes", minimum=1)
    canonical_duration = frame_count * 1_000_000 // VIDEO_FRAMES_PER_SECOND
    if abs(interval.duration_microseconds - canonical_duration) > 1_000_000 // VIDEO_FRAMES_PER_SECOND:
        raise ContractViolation("source interval differs from supplied video by more than one frame")


def _validate_tracking_policy(value: Any) -> Mapping[str, Any]:
    policy = _require_mapping(value, "producer tracking policy")
    expected = {
        "faceScoreThreshold",
        "maximumGapFrames",
        "minimumDetectionFrames",
        "minimumIou",
        "shotCutThreshold",
        "shotDetection",
        "tracker",
    }
    _require_exact_keys(policy, expected, "producer tracking policy")
    threshold = _require_number(policy["faceScoreThreshold"], "face score threshold")
    minimum_iou = _require_number(policy["minimumIou"], "minimum track IoU")
    shot_threshold = _require_number(policy["shotCutThreshold"], "shot cut threshold")
    if not 0 < threshold <= 1 or not 0 < minimum_iou <= 1 or shot_threshold <= 0:
        raise ContractViolation("producer tracking thresholds are outside valid ranges")
    _require_integer(policy["maximumGapFrames"], "maximum track gap frames")
    _require_integer(policy["minimumDetectionFrames"], "minimum detection frames", minimum=1)
    if policy["shotDetection"] != "64x36-gray-mean-absolute-difference-v1":
        raise ContractViolation("producer shot detection policy is not exact v1")
    if policy["tracker"] != "deterministic-greedy-shot-bounded-iou-v1":
        raise ContractViolation("producer tracking policy is not exact v1")
    return policy


def _validate_source_observation_record(value: Any) -> Mapping[str, Any]:
    record = _require_mapping(value, "base v1 observation record")
    _require_exact_keys(
        record,
        {"bytes", "identities", "schemaVersion", "sha256"},
        "base v1 observation record",
    )
    if record["schemaVersion"] != V1_OBSERVATION_SCHEMA_VERSION:
        raise ContractViolation("base observation schemaVersion must be exact v1")
    _require_integer(record["bytes"], "base observation bytes", minimum=1)
    validate_sha256(
        _require_string(record["sha256"], "base observation SHA-256"),
        "base observation SHA-256",
    )
    identities = _require_mapping(record["identities"], "base observation identities")
    expected_identities = {
        "clockIdentity",
        "modelIdentity",
        "observationIdentity",
        "runIdentity",
        "runtimeIdentity",
    }
    _require_exact_keys(identities, expected_identities, "base observation identities")
    for name, identity in identities.items():
        validate_content_identity(
            _require_string(identity, f"base observation {name}"),
            f"base observation {name}",
        )
    return record


def _validate_geometry_lineage(value: Any, prepared_input_sha256: str) -> None:
    lineage = _require_mapping(value, "producer geometry lineage")
    kind = lineage.get("kind")
    if kind == "BASE_OBSERVED":
        _require_exact_keys(
            lineage,
            {"inputSha256", "kind", "sourceObservation"},
            "base geometry lineage",
        )
        bound_input = validate_sha256(
            _require_string(lineage["inputSha256"], "base lineage input SHA-256"),
            "base lineage input SHA-256",
        )
        if bound_input != prepared_input_sha256:
            raise ContractViolation("base geometry lineage binds a different prepared input")
        _validate_source_observation_record(lineage["sourceObservation"])
        return
    if kind != "HORIZONTAL_MIRROR_DERIVED":
        raise ContractViolation("producer geometry lineage kind is unsupported")
    _require_exact_keys(
        lineage,
        {
            "derivedInputSha256",
            "kind",
            "sourceInputSha256",
            "sourceManifestContentIdentity",
            "sourceManifestSha256",
            "sourceObservation",
            "transform",
        },
        "horizontal-mirror geometry lineage",
    )
    derived_input = validate_sha256(
        _require_string(lineage["derivedInputSha256"], "derived input SHA-256"),
        "derived input SHA-256",
    )
    if derived_input != prepared_input_sha256:
        raise ContractViolation("mirror geometry lineage binds a different derived input")
    source_input = validate_sha256(
        _require_string(lineage["sourceInputSha256"], "source input SHA-256"),
        "source input SHA-256",
    )
    if source_input == derived_input:
        raise ContractViolation(
            "mirror geometry lineage must bind distinct source and derived inputs"
        )
    validate_sha256(
        _require_string(lineage["sourceManifestSha256"], "source manifest SHA-256"),
        "source manifest SHA-256",
    )
    validate_content_identity(
        _require_string(
            lineage["sourceManifestContentIdentity"],
            "source manifest content identity",
        ),
        "source manifest content identity",
    )
    _validate_source_observation_record(lineage["sourceObservation"])
    transform = _require_mapping(lineage["transform"], "mirror lineage transform")
    _require_exact_keys(
        transform,
        {"kind", "topology", "x1", "x2", "y"},
        "mirror lineage transform",
    )
    if transform != {
        "kind": "HORIZONTAL_MIRROR_PIXEL_BOX_V1",
        "topology": "preserve-track-frame-shot-pts-observation-v1",
        "x1": "width-minus-source-x2",
        "x2": "width-minus-source-x1",
        "y": "unchanged",
    }:
        raise ContractViolation("mirror geometry lineage transform is not the frozen v1 transform")


def _validate_producer(
    value: Any,
    frame_count: int,
    prepared_input_sha256: str,
) -> Mapping[str, Any]:
    producer = _require_mapping(value, "supplied-track producer")
    _require_exact_keys(
        producer,
        {
            "detector",
            "geometryLineage",
            "kind",
            "processedFrames",
            "runtimeIdentity",
            "sourceClosureSha256",
            "trackingPolicy",
        },
        "supplied-track producer",
    )
    _validate_geometry_lineage(producer["geometryLineage"], prepared_input_sha256)
    lineage_kind = producer["geometryLineage"]["kind"]
    expected_kind = (
        CANONICAL_TRACK_PRODUCER_KIND
        if lineage_kind == "BASE_OBSERVED"
        else MIRROR_TRACK_PRODUCER_KIND
    )
    if producer["kind"] != expected_kind:
        raise ContractViolation("supplied-track producer kind disagrees with geometry lineage")
    validate_content_identity(_require_string(producer["runtimeIdentity"], "producer runtime identity"), "producer runtime identity")
    validate_sha256(_require_string(producer["sourceClosureSha256"], "producer source closure"), "producer source closure")
    policy = _validate_tracking_policy(producer["trackingPolicy"])
    detector = _require_mapping(producer["detector"], "producer detector")
    _require_exact_keys(
        detector,
        {
            "evidenceRole",
            "fallbackPolicy",
            "kind",
            "modelSha256",
            "scoreThreshold",
        },
        "producer detector",
    )
    if detector["kind"] != "YuNet-2023mar" or detector["fallbackPolicy"] != "NONE":
        raise ContractViolation("supplied-track producer must use strict YuNet with no fallback")
    expected_evidence_role = (
        "DIRECT_OBSERVATION"
        if lineage_kind == "BASE_OBSERVED"
        else "AUTHENTICATED_SOURCE_MANIFEST"
    )
    if detector["evidenceRole"] != expected_evidence_role:
        raise ContractViolation(
            "producer detector evidence role disagrees with geometry lineage"
        )
    validate_sha256(_require_string(detector["modelSha256"], "producer detector model"), "producer detector model")
    detector_threshold = _require_number(detector["scoreThreshold"], "detector score threshold")
    if detector_threshold != float(policy["faceScoreThreshold"]):
        raise ContractViolation("producer detector and tracking thresholds disagree")
    _validate_processed_frames(producer["processedFrames"], frame_count)
    return producer


def _validate_processed_frames(value: Any, frame_count: int) -> None:
    if not isinstance(value, list) or len(value) != frame_count:
        raise ContractViolation("producer processed-frame ledger must cover every frame")
    for frame_index, item in enumerate(value):
        mapping = _require_mapping(item, f"processed frame {frame_index}")
        _require_exact_keys(mapping, {"frameIndex", "pts"}, f"processed frame {frame_index}")
        if _require_integer(mapping["frameIndex"], "processed frame index") != frame_index:
            raise ContractViolation("producer processed-frame ledger is not contiguous")
        _parse_pts(mapping["pts"], frame_index, "processed frame PTS")


def _parse_track_frame(
    value: Any,
    *,
    expected_frame: int | None,
    width: int,
    height: int,
) -> SuppliedTrackFrame:
    frame = _require_mapping(value, "supplied track frame")
    _require_exact_keys(frame, {"faceBox", "frameIndex", "isDetectorObservation", "pts"}, "supplied track frame")
    frame_index = _require_integer(frame["frameIndex"], "track frame index")
    if expected_frame is not None and frame_index != expected_frame:
        raise ContractViolation("supplied track frame indexes must be contiguous")
    _parse_pts(frame["pts"], frame_index, "track frame PTS")
    observed = frame["isDetectorObservation"]
    if not isinstance(observed, bool):
        raise ContractViolation("isDetectorObservation must be boolean")
    box = _require_mapping(frame["faceBox"], "track face box")
    _require_exact_keys(box, {"x1", "x2", "y1", "y2"}, "track face box")
    face_box = FaceBox(*(_require_number(box[key], f"face box {key}") for key in ("x1", "y1", "x2", "y2")))
    if face_box.x1 < 0 or face_box.y1 < 0 or face_box.x2 > width or face_box.y2 > height:
        raise ContractViolation("supplied track face box is outside prepared video bounds")
    return SuppliedTrackFrame(frame_index, face_box, observed)


def _parse_tracks(
    value: Any,
    *,
    shots: tuple[SuppliedShot, ...],
    width: int,
    height: int,
    limits: SuppliedTrackLimits,
) -> tuple[SuppliedTrack, ...]:
    if not isinstance(value, list):
        raise ContractViolation("supplied tracks must be an array")
    if len(value) > limits.maximum_tracks:
        raise ContractViolation("supplied tracks exceed maximum track count")
    result: list[SuppliedTrack] = []
    track_ids: set[str] = set()
    total_frames = 0
    for item in value:
        track = _require_mapping(item, "supplied track")
        _require_exact_keys(track, {"frames", "shotIndex", "trackId"}, "supplied track")
        track_id = _require_string(track["trackId"], "supplied track ID")
        track_id_match = _TRACK_ID_RE.fullmatch(track_id)
        if track_id_match is None or len(track_id) > 128 or track_id in track_ids:
            raise ContractViolation("supplied track IDs must be unique canonical shot-local IDs")
        shot_index = _require_integer(track["shotIndex"], "supplied track shot index")
        if shot_index >= len(shots):
            raise ContractViolation("supplied track references a missing shot")
        if int(track_id_match.group(1)) != shot_index:
            raise ContractViolation("supplied track ID shot component disagrees with shotIndex")
        frames_value = track["frames"]
        if not isinstance(frames_value, list) or not frames_value:
            raise ContractViolation("supplied track must contain at least one frame")
        frames: list[SuppliedTrackFrame] = []
        for frame_value in frames_value:
            expected = None if not frames else frames[-1].frame_index + 1
            parsed = _parse_track_frame(frame_value, expected_frame=expected, width=width, height=height)
            if not shots[shot_index].contains(parsed.frame_index):
                raise ContractViolation("supplied track frame crosses its shot boundary")
            frames.append(parsed)
        if not any(frame.is_detector_observation for frame in frames):
            raise ContractViolation("non-empty supplied track has no detector observation")
        total_frames += len(frames)
        if total_frames > limits.maximum_track_frames:
            raise ContractViolation("supplied tracks exceed maximum total track frames")
        track_ids.add(track_id)
        result.append(SuppliedTrack(track_id, shot_index, tuple(frames)))
    if [track.track_id for track in result] != sorted(track_ids):
        raise ContractViolation("supplied tracks must be ordered by trackId")
    return tuple(result)


def load_supplied_track_manifest(
    path: Path,
    *,
    expected_sha256: str,
    limits: SuppliedTrackLimits,
) -> SuppliedTrackManifest:
    """Authenticate and parse one exact supplied-track manifest from one read."""

    expected_hash = validate_sha256(expected_sha256, "supplied-track manifest SHA-256")
    payload = _read_bounded_regular_file(path, limits.maximum_manifest_bytes)
    actual_hash = sha256_bytes(payload)
    if actual_hash != expected_hash:
        raise ContractViolation(
            "supplied-track manifest SHA-256 mismatch: "
            f"expected {expected_hash}, received {actual_hash}"
        )
    root = _decode_json(payload)
    _require_exact_keys(
        root,
        {"clock", "clockIdentity", "contentIdentity", "producer", "schemaVersion", "status", "tracks"},
        "supplied-track manifest",
    )
    if root["schemaVersion"] != SUPPLIED_TRACK_SCHEMA_VERSION:
        raise ContractViolation("supplied-track manifest schemaVersion is not v2")
    if root["status"] != "COMPLETE":
        raise ContractViolation("supplied-track producer status must be COMPLETE")
    clock, shots = _validate_clock(root["clock"], limits)
    clock_identity = validate_content_identity(
        _require_string(root["clockIdentity"], "supplied clock identity"),
        "supplied clock identity",
    )
    if content_identity(clock) != clock_identity:
        raise ContractViolation("supplied clock identity does not match its canonical content")
    producer = _validate_producer(
        root["producer"],
        int(clock["video"]["frameCount"]),
        str(clock["preparedInput"]["sha256"]),
    )
    tracks = _parse_tracks(
        root["tracks"],
        shots=shots,
        width=int(clock["video"]["width"]),
        height=int(clock["video"]["height"]),
        limits=limits,
    )
    supplied_identity = validate_content_identity(
        _require_string(root["contentIdentity"], "supplied-track content identity"),
        "supplied-track content identity",
    )
    identity_projection = {key: value for key, value in root.items() if key != "contentIdentity"}
    if content_identity(identity_projection) != supplied_identity:
        raise ContractViolation("supplied-track content identity does not match canonical content")
    return SuppliedTrackManifest(
        file_sha256=actual_hash,
        file_bytes=len(payload),
        content_identity=supplied_identity,
        clock_identity=clock_identity,
        clock=clock,
        producer=producer,
        shots=shots,
        tracks=tracks,
    )


def _validate_hash_bytes_record(value: Any, label: str) -> Mapping[str, Any]:
    record = _require_mapping(value, label)
    _require_exact_keys(record, {"bytes", "sha256"}, label)
    _require_integer(record["bytes"], f"{label} bytes", minimum=1)
    validate_sha256(
        _require_string(record["sha256"], f"{label} SHA-256"),
        f"{label} SHA-256",
    )
    return record


def _validate_v1_runtime(value: Any) -> Mapping[str, Any]:
    runtime = _require_mapping(value, "base v1 observation runtime")
    _require_exact_keys(
        runtime,
        {
            "audioWorkerBaseImageId",
            "baseAudioWorkerBuildSha",
            "dependencies",
            "runtimeClosure",
            "runtimeVersion",
            "tools",
        },
        "base v1 observation runtime",
    )
    if runtime["runtimeVersion"] != V1_RUNTIME_VERSION:
        raise ContractViolation("base observation runtimeVersion must be exact v1")
    validate_content_identity(
        _require_string(
            runtime["audioWorkerBaseImageId"],
            "base observation Audio-Worker image ID",
        ),
        "base observation Audio-Worker image ID",
    )
    _require_string(runtime["baseAudioWorkerBuildSha"], "base observation build SHA")
    _require_mapping(runtime["dependencies"], "base observation dependencies")
    _require_mapping(runtime["tools"], "base observation tools")
    closure = runtime["runtimeClosure"]
    if not isinstance(closure, list):
        raise ContractViolation("base observation runtime closure must be an array")
    closure_paths: list[str] = []
    for index, item in enumerate(closure):
        record = _require_mapping(item, f"base observation runtime closure {index}")
        _require_exact_keys(
            record,
            {"bytes", "path", "sha256"},
            f"base observation runtime closure {index}",
        )
        _require_integer(record["bytes"], "runtime closure bytes", minimum=1)
        closure_paths.append(_require_string(record["path"], "runtime closure path"))
        validate_sha256(
            _require_string(record["sha256"], "runtime closure SHA-256"),
            "runtime closure SHA-256",
        )
    if tuple(closure_paths) != V1_RUNTIME_CLOSURE_FILES:
        raise ContractViolation("base observation runtime closure is not exact v1")
    license_record = closure[V1_RUNTIME_CLOSURE_FILES.index("LR-ASD-LICENSE.txt")]
    if license_record["sha256"] != LRASD_LICENSE_SHA256:
        raise ContractViolation("base observation LR-ASD license differs from pinned v1")
    return runtime


def _v1_preprocessing_policy() -> dict[str, Any]:
    return {
        "audio": {
            "channels": AUDIO_CHANNELS,
            "decoderTail": (
                "trim-to-selected-stream-presentation-duration-before-origin-alignment-v1"
            ),
            "originAlignment": "selected-stream-start-relative-to-video-frame-zero-v1",
            "sampleRateHz": AUDIO_SAMPLE_RATE_HZ,
            "sampleType": "signed-16-bit-little-endian",
            "timelineValidation": "packet-coverage-and-decoded-sample-clock-v1",
        },
        "faceInput": {
            "color": "grayscale",
            "detector": "YuNet-2023mar",
            "interpolation": "linear-boxes-median-center-size-13-v1",
            "modelHeight": 112,
            "modelWidth": 112,
            "upstreamCropScale": 0.4,
        },
        "mfcc": {
            "coefficients": 13,
            "rowsPerVideoFrame": 4,
            "windowMilliseconds": 25,
            "windowStepMilliseconds": 10,
        },
        "video": {
            "frameRate": {"denominator": 1, "numerator": VIDEO_FRAMES_PER_SECOND},
            "frameZero": "first-decoded-selected-video-frame",
            "normalization": "setpts-start-fps-near-setpts-frame-index-v1",
            "timelineValidation": "packet-coverage-and-decoded-cfr-clock-v1",
        },
    }


def _validate_v1_model(value: Any) -> Mapping[str, Any]:
    model = _require_mapping(value, "base v1 observation model")
    _require_exact_keys(
        model,
        {
            "checkpoint",
            "contextSeconds",
            "device",
            "lrasdRevision",
            "lrasdSource",
            "lrasdSourceSha256",
            "preprocessingPolicy",
            "stateLoad",
            "yunet",
        },
        "base v1 observation model",
    )
    _validate_hash_bytes_record(model["checkpoint"], "base observation checkpoint")
    _validate_hash_bytes_record(model["yunet"], "base observation YuNet")
    if model["contextSeconds"] != [1, 2, 3, 4, 5, 6]:
        raise ContractViolation("base observation LR-ASD contexts are not frozen v1")
    _require_string(model["device"], "base observation model device")
    validate_git_revision(
        _require_string(model["lrasdRevision"], "base observation LR-ASD revision")
    )
    expected_source_sha = validate_sha256(
        _require_string(
            model["lrasdSourceSha256"],
            "base observation LR-ASD source SHA-256",
        ),
        "base observation LR-ASD source SHA-256",
    )
    source = model["lrasdSource"]
    if not isinstance(source, list):
        raise ContractViolation("base observation LR-ASD source closure must be an array")
    source_paths: list[str] = []
    for index, item in enumerate(source):
        record = _require_mapping(item, f"base observation LR-ASD source {index}")
        _require_exact_keys(
            record,
            {"bytes", "path", "sha256"},
            f"base observation LR-ASD source {index}",
        )
        _require_integer(record["bytes"], "LR-ASD source bytes", minimum=1)
        source_paths.append(_require_string(record["path"], "LR-ASD source path"))
        validate_sha256(
            _require_string(record["sha256"], "LR-ASD source SHA-256"),
            "LR-ASD source SHA-256",
        )
    if tuple(source_paths) != LRASD_EXECUTED_SOURCE_FILES:
        raise ContractViolation("base observation LR-ASD source closure is not exact v1")
    if sha256_bytes(canonical_json_bytes(source)) != expected_source_sha:
        raise ContractViolation("base observation LR-ASD source identity is inconsistent")
    if model["preprocessingPolicy"] != _v1_preprocessing_policy():
        raise ContractViolation("base observation preprocessing policy is not exact v1")
    if model["stateLoad"] != {"strict": True, "weightsOnly": True}:
        raise ContractViolation("base observation model state load is not strict v1")
    return model


def _validate_v1_track_score_closure(
    tracks: Any,
    score_ledger: Any,
) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(tracks, list) or not isinstance(score_ledger, list):
        raise ContractViolation("base observation tracks and score ledger must be arrays")
    if len(tracks) != len(score_ledger):
        raise ContractViolation("base observation score ledger is not closed over tracks")
    parsed_tracks: list[Mapping[str, Any]] = []
    previous_track_id: str | None = None
    for track_index, (track_value, ledger_value) in enumerate(
        zip(tracks, score_ledger, strict=True)
    ):
        track = _require_mapping(track_value, f"base observation track {track_index}")
        _require_exact_keys(
            track,
            {"frames", "shotIndex", "trackId"},
            f"base observation track {track_index}",
        )
        track_id = _require_string(track["trackId"], "base observation track ID")
        if _TRACK_ID_RE.fullmatch(track_id) is None:
            raise ContractViolation("base observation track ID is not canonical")
        if previous_track_id is not None and track_id <= previous_track_id:
            raise ContractViolation("base observation tracks are not strictly ordered")
        previous_track_id = track_id
        _require_integer(track["shotIndex"], "base observation track shot index")
        frames = track["frames"]
        if not isinstance(frames, list) or not frames:
            raise ContractViolation("base observation track frames must be non-empty")

        ledger = _require_mapping(
            ledger_value,
            f"base observation score ledger {track_index}",
        )
        _require_exact_keys(
            ledger,
            {"samples", "trackId"},
            f"base observation score ledger {track_index}",
        )
        if ledger["trackId"] != track_id:
            raise ContractViolation("base observation score ledger track ID mismatch")
        samples = ledger["samples"]
        if not isinstance(samples, list) or len(samples) != len(frames):
            raise ContractViolation("base observation score count differs from track frames")
        for frame_index, (frame_value, sample_value) in enumerate(
            zip(frames, samples, strict=True)
        ):
            frame = _require_mapping(
                frame_value,
                f"base observation track frame {frame_index}",
            )
            _require_exact_keys(
                frame,
                {"faceBox", "frameIndex", "isDetectorObservation", "pts"},
                f"base observation track frame {frame_index}",
            )
            sample = _require_mapping(
                sample_value,
                f"base observation score sample {frame_index}",
            )
            _require_exact_keys(
                sample,
                {"frameIndex", "pts", "rawSpeakingScore"},
                f"base observation score sample {frame_index}",
            )
            if sample["frameIndex"] != frame["frameIndex"] or sample["pts"] != frame["pts"]:
                raise ContractViolation("base observation score clock differs from track clock")
            _require_number(sample["rawSpeakingScore"], "base observation raw speaking score")
        parsed_tracks.append(track)
    return tuple(parsed_tracks)


def _validate_v1_outputs(value: Any) -> Mapping[str, Any]:
    outputs = _require_mapping(value, "base v1 observation outputs")
    _require_exact_keys(
        outputs,
        {"annotatedReview", "canonicalAudio", "canonicalVideo"},
        "base v1 observation outputs",
    )
    for name, value_record in outputs.items():
        record = _require_mapping(value_record, f"base observation output {name}")
        expected = {"bytes", "path", "sha256"}
        if name == "annotatedReview":
            expected.add("validatedClock")
        _require_exact_keys(record, expected, f"base observation output {name}")
        _require_integer(record["bytes"], f"base observation output {name} bytes", minimum=1)
        _require_string(record["path"], f"base observation output {name} path")
        validate_sha256(
            _require_string(record["sha256"], f"base observation output {name} SHA-256"),
            f"base observation output {name} SHA-256",
        )
        if name == "annotatedReview":
            _require_mapping(record["validatedClock"], "annotated review validated clock")
    return outputs


def load_v1_observation_receipt(
    path: Path,
    *,
    expected_sha256: str,
    maximum_bytes: int,
) -> V1ObservationReceipt:
    """Authenticate and recompute every identity in one sealed v1 observation."""

    expected_hash = validate_sha256(expected_sha256, "base v1 observation SHA-256")
    payload = _read_bounded_regular_file(
        path,
        maximum_bytes,
        label="base v1 observation receipt",
    )
    actual_hash = sha256_bytes(payload)
    if actual_hash != expected_hash:
        raise ContractViolation(
            "base v1 observation SHA-256 mismatch: "
            f"expected {expected_hash}, received {actual_hash}"
        )
    root = _decode_json(payload)
    _require_exact_keys(
        root,
        {
            "authority",
            "clocks",
            "cropAuthority",
            "identities",
            "measurements",
            "outputs",
            "rawScoreSemantics",
            "schemaVersion",
            "scoreLedger",
            "tracks",
        },
        "base v1 observation receipt",
    )
    if root["schemaVersion"] != V1_OBSERVATION_SCHEMA_VERSION:
        raise ContractViolation("base observation schemaVersion must be exact v1")
    if root["authority"] != AUTHORITY or root["cropAuthority"] != CROP_AUTHORITY:
        raise ContractViolation("base observation carries unexpected authority")
    _require_string(root["rawScoreSemantics"], "base observation score semantics")
    identities = _require_mapping(root["identities"], "base observation identities")
    expected_identity_keys = {
        "clockIdentity",
        "modelIdentity",
        "observationIdentity",
        "runIdentity",
        "runtimeIdentity",
    }
    _require_exact_keys(identities, expected_identity_keys, "base observation identities")
    for name, identity in identities.items():
        validate_content_identity(
            _require_string(identity, f"base observation {name}"),
            f"base observation {name}",
        )
    clocks = _require_mapping(root["clocks"], "base observation clocks")
    if content_identity(clocks) != identities["clockIdentity"]:
        raise ContractViolation("base observation clock identity is inconsistent")
    measurements = _require_mapping(root["measurements"], "base observation measurements")
    _require_exact_keys(
        measurements,
        {
            "detectedFaceCount",
            "detectedTrackCount",
            "input",
            "model",
            "runtime",
            "scoredTrackCount",
            "stageMilliseconds",
            "totalMilliseconds",
            "trackingPolicy",
        },
        "base observation measurements",
    )
    model = _validate_v1_model(measurements["model"])
    if content_identity(model) != identities["modelIdentity"]:
        raise ContractViolation("base observation model identity is inconsistent")
    runtime = _validate_v1_runtime(measurements["runtime"])
    runtime_projection = {
        "audioWorkerBaseImageId": runtime["audioWorkerBaseImageId"],
        "runtimeClosure": runtime["runtimeClosure"],
        "runtimeVersion": runtime["runtimeVersion"],
    }
    if content_identity(runtime_projection) != identities["runtimeIdentity"]:
        raise ContractViolation("base observation runtime identity is inconsistent")
    tracking_policy = _validate_tracking_policy(measurements["trackingPolicy"])
    tracks = root["tracks"]
    score_ledger = root["scoreLedger"]
    parsed_tracks = _validate_v1_track_score_closure(tracks, score_ledger)
    observation_projection = {
        "clockIdentity": identities["clockIdentity"],
        "modelIdentity": identities["modelIdentity"],
        "scoreLedger": score_ledger,
        "trackingPolicy": tracking_policy,
        "tracks": tracks,
    }
    if content_identity(observation_projection) != identities["observationIdentity"]:
        raise ContractViolation("base observation identity is inconsistent")
    outputs = _validate_v1_outputs(root["outputs"])
    run_projection = {
        "clockIdentity": identities["clockIdentity"],
        "modelIdentity": identities["modelIdentity"],
        "observationIdentity": identities["observationIdentity"],
        "outputs": outputs,
        "runtimeIdentity": identities["runtimeIdentity"],
    }
    if content_identity(run_projection) != identities["runIdentity"]:
        raise ContractViolation("base observation run identity is inconsistent")
    input_record = _validate_hash_bytes_record(
        measurements["input"],
        "base observation input",
    )
    if input_record != clocks.get("preparedInput"):
        raise ContractViolation("base observation input differs from its clock")
    scored_track_count = _require_integer(
        measurements["scoredTrackCount"],
        "base observation scored track count",
    )
    if scored_track_count != len(tracks) or len(score_ledger) != len(tracks):
        raise ContractViolation("base observation scored track closure is incomplete")
    _require_integer(measurements["detectedFaceCount"], "detected face count")
    detected_tracks = _require_integer(
        measurements["detectedTrackCount"],
        "detected track count",
    )
    if detected_tracks < scored_track_count:
        raise ContractViolation("base observation detected track count is inconsistent")
    _require_mapping(measurements["stageMilliseconds"], "base observation stage timings")
    _require_number(measurements["totalMilliseconds"], "base observation total timing")
    assert_no_decision_authority(root)
    return V1ObservationReceipt(
        file_sha256=actual_hash,
        file_bytes=len(payload),
        identities=dict(identities),
        clocks=clocks,
        tracks=parsed_tracks,
        tracking_policy=tracking_policy,
        model=model,
        runtime=runtime,
    )


def validate_base_observation_lineage(
    base_manifest: SuppliedTrackManifest,
    observation: V1ObservationReceipt,
) -> None:
    """Prove that BASE_OBSERVED geometry is exactly the sealed v1 result."""

    lineage = base_manifest.producer["geometryLineage"]
    if lineage["kind"] != "BASE_OBSERVED":
        raise ContractViolation("base observation can bind only BASE_OBSERVED geometry")
    if lineage["sourceObservation"] != observation.lineage_record():
        raise ContractViolation("base geometry lineage binds a different v1 observation")
    if base_manifest.clock != observation.clocks:
        raise ContractViolation("base supplied-track clock differs from v1 observation")
    manifest_tracks = tuple(track.as_json() for track in base_manifest.tracks)
    if manifest_tracks != observation.tracks:
        raise ContractViolation("base supplied-track geometry differs from v1 observation")
    producer = base_manifest.producer
    if producer["trackingPolicy"] != observation.tracking_policy:
        raise ContractViolation("base tracking policy differs from v1 observation")
    detector = producer["detector"]
    expected_detector = {
        "evidenceRole": "DIRECT_OBSERVATION",
        "fallbackPolicy": "NONE",
        "kind": observation.model["preprocessingPolicy"]["faceInput"]["detector"],
        "modelSha256": observation.model["yunet"]["sha256"],
        "scoreThreshold": observation.tracking_policy["faceScoreThreshold"],
    }
    if detector != expected_detector:
        raise ContractViolation("base YuNet detector differs from v1 observation")
    if producer["runtimeIdentity"] != observation.identities["runtimeIdentity"]:
        raise ContractViolation("base producer runtime differs from v1 observation")
    expected_source_closure = sha256_bytes(
        canonical_json_bytes(observation.runtime["runtimeClosure"])
    )
    if producer["sourceClosureSha256"] != expected_source_closure:
        raise ContractViolation("base producer source closure differs from v1 observation")


def validate_geometry_lineage(
    manifest: SuppliedTrackManifest,
    source_manifest: SuppliedTrackManifest | None,
) -> None:
    """Authenticate base or mechanically mirrored geometry provenance."""

    lineage = manifest.producer["geometryLineage"]
    if lineage["kind"] == "BASE_OBSERVED":
        if source_manifest is not None:
            raise ContractViolation("base geometry must not supply a lineage source manifest")
        return
    if source_manifest is None:
        raise ContractViolation("mirror-derived geometry requires its exact source manifest")
    if source_manifest.producer["geometryLineage"]["kind"] != "BASE_OBSERVED":
        raise ContractViolation("mirror-derived geometry source must be BASE_OBSERVED")
    if lineage["sourceManifestSha256"] != source_manifest.file_sha256:
        raise ContractViolation("mirror lineage source manifest file SHA-256 mismatch")
    if lineage["sourceManifestContentIdentity"] != source_manifest.content_identity:
        raise ContractViolation("mirror lineage source manifest content identity mismatch")
    if lineage["sourceInputSha256"] != source_manifest.clock["preparedInput"]["sha256"]:
        raise ContractViolation("mirror lineage source input SHA-256 mismatch")
    if lineage["derivedInputSha256"] != manifest.clock["preparedInput"]["sha256"]:
        raise ContractViolation("mirror lineage derived input SHA-256 mismatch")
    source_observation = source_manifest.producer["geometryLineage"][
        "sourceObservation"
    ]
    if lineage["sourceObservation"] != source_observation:
        raise ContractViolation(
            "mirror lineage base observation differs from its source manifest"
        )
    _validate_mirror_topology(manifest, source_manifest)


def _validate_mirror_topology(
    manifest: SuppliedTrackManifest,
    source_manifest: SuppliedTrackManifest,
) -> None:
    if (manifest.width, manifest.height, manifest.frame_count) != (
        source_manifest.width,
        source_manifest.height,
        source_manifest.frame_count,
    ):
        raise ContractViolation("mirror lineage video geometry differs from its source")
    if tuple(shot.as_json() for shot in manifest.shots) != tuple(
        shot.as_json() for shot in source_manifest.shots
    ):
        raise ContractViolation("mirror lineage shot topology differs from its source")
    for clock_key in ("audio", "inputStreams", "shots", "sourceInterval", "video"):
        if manifest.clock[clock_key] != source_manifest.clock[clock_key]:
            raise ContractViolation(
                f"mirror lineage {clock_key} clock differs from its source"
            )
    if len(manifest.tracks) != len(source_manifest.tracks):
        raise ContractViolation("mirror lineage track count differs from its source")
    source_detector = dict(source_manifest.producer["detector"])
    target_detector = dict(manifest.producer["detector"])
    source_detector.pop("evidenceRole")
    target_detector.pop("evidenceRole")
    if target_detector != source_detector:
        raise ContractViolation("mirror lineage detector provenance differs from its source")
    if manifest.producer["trackingPolicy"] != source_manifest.producer["trackingPolicy"]:
        raise ContractViolation("mirror lineage tracking policy differs from its source")
    for target_track, source_track in zip(manifest.tracks, source_manifest.tracks, strict=True):
        if (target_track.track_id, target_track.shot_index, len(target_track.frames)) != (
            source_track.track_id,
            source_track.shot_index,
            len(source_track.frames),
        ):
            raise ContractViolation("mirror lineage changed track identity or topology")
        for target_frame, source_frame in zip(
            target_track.frames, source_track.frames, strict=True
        ):
            expected = (
                manifest.width - source_frame.face_box.x2,
                source_frame.face_box.y1,
                manifest.width - source_frame.face_box.x1,
                source_frame.face_box.y2,
            )
            actual = (
                target_frame.face_box.x1,
                target_frame.face_box.y1,
                target_frame.face_box.x2,
                target_frame.face_box.y2,
            )
            if (
                target_frame.frame_index != source_frame.frame_index
                or target_frame.is_detector_observation
                != source_frame.is_detector_observation
                or actual != expected
            ):
                raise ContractViolation("mirror lineage is not the exact frozen box transform")


def lrasd_v2_source_identity(root: Path) -> tuple[str, list[dict[str, Any]]]:
    if root.is_symlink() or not root.is_dir():
        raise ContractViolation(f"LR-ASD root must be a non-symlink directory: {root}")
    model_root = root / "model"
    if model_root.is_symlink() or not model_root.is_dir():
        raise ContractViolation("LR-ASD v2 model namespace must be a non-symlink directory")
    _reject_lrasd_shadow_code(root, "model", allowed_name=None, allow_directory=model_root)
    _reject_lrasd_shadow_code(root, "loss", allowed_name="loss.py")
    if (root / "model" / "__init__.py").exists():
        raise ContractViolation(
            "LR-ASD v2 model namespace contains an unexpected package initializer"
        )
    _reject_lrasd_shadow_code(model_root, "__init__", allowed_name=None)
    for module_name in ("Classifier", "Encoder", "Model"):
        _reject_lrasd_shadow_code(
            model_root,
            module_name,
            allowed_name=f"{module_name}.py",
        )
    for bytecode_root in (root / "__pycache__", root / "model" / "__pycache__"):
        if bytecode_root.exists() and any(bytecode_root.glob("*.py[co]")):
            raise ContractViolation("LR-ASD v2 source closure contains executable bytecode")
    manifest: list[dict[str, Any]] = []
    for relative_path in LRASD_V2_EXECUTED_SOURCE_FILES:
        digest, size = sha256_file(root / relative_path, f"LR-ASD v2 source {relative_path}")
        manifest.append({"bytes": size, "path": relative_path, "sha256": digest})
    return sha256_bytes(canonical_json_bytes(manifest)), manifest


def _reject_lrasd_shadow_code(
    parent: Path,
    stem: str,
    *,
    allowed_name: str | None,
    allow_directory: Path | None = None,
) -> None:
    for candidate in parent.iterdir():
        if candidate == allow_directory or candidate.name == allowed_name:
            continue
        if candidate.name == stem and candidate.is_dir():
            raise ContractViolation(
                f"LR-ASD v2 source contains shadowing {stem} package"
            )
        if not candidate.name.startswith(f"{stem}."):
            continue
        if candidate.suffix in {".py", ".pyc", ".pyo", ".so"}:
            raise ContractViolation(
                f"LR-ASD v2 source contains executable shadow for {stem}"
            )


class V2RawScoreLedger:
    """Build a closed component-plus-mean ledger for two-view ASD scores."""

    @staticmethod
    def build(
        admitted_frames_by_track: Mapping[str, Sequence[int]],
        canonical_scores_by_track: Mapping[str, Sequence[float]],
        mirrored_scores_by_track: Mapping[str, Sequence[float]],
        mean_scores_by_track: Mapping[str, Sequence[float]],
    ) -> list[dict[str, Any]]:
        expected_tracks = set(admitted_frames_by_track)
        if any(
            set(scores) != expected_tracks
            for scores in (
                canonical_scores_by_track,
                mirrored_scores_by_track,
                mean_scores_by_track,
            )
        ):
            raise ContractViolation("v2 score tracks must exactly match supplied face tracks")
        result: list[dict[str, Any]] = []
        for track_id in sorted(expected_tracks):
            result.append(
                V2RawScoreLedger._build_track(
                    track_id,
                    admitted_frames_by_track[track_id],
                    canonical_scores_by_track[track_id],
                    mirrored_scores_by_track[track_id],
                    mean_scores_by_track[track_id],
                )
            )
        return result

    @staticmethod
    def _build_track(
        track_id: str,
        frames_value: Sequence[int],
        canonical_value: Sequence[float],
        mirrored_value: Sequence[float],
        means_value: Sequence[float],
    ) -> dict[str, Any]:
        frames = list(frames_value)
        canonical = [float(value) for value in canonical_value]
        mirrored = [float(value) for value in mirrored_value]
        means = [float(value) for value in means_value]
        if not frames or frames != list(range(frames[0], frames[0] + len(frames))):
            raise ContractViolation("v2 admitted track frames must be non-empty and contiguous")
        if not (len(frames) == len(canonical) == len(mirrored) == len(means)):
            raise ContractViolation(f"v2 raw score count mismatch for {track_id}")
        samples: list[dict[str, Any]] = []
        for frame, canonical_score, mirrored_score, supplied_mean in zip(
            frames, canonical, mirrored, means, strict=True
        ):
            expected_mean = math.fsum((canonical_score, mirrored_score)) / 2.0
            if not all(math.isfinite(value) for value in (canonical_score, mirrored_score, supplied_mean)):
                raise ContractViolation("v2 LR-ASD score ledger contains a non-finite value")
            if supplied_mean != expected_mean:
                raise ContractViolation("v2 rawSpeakingScore is not the exact two-view mean")
            samples.append(
                {
                    "frameIndex": frame,
                    "pts": {"denominator": VIDEO_FRAMES_PER_SECOND, "numerator": frame},
                    "rawSpeakingScore": expected_mean,
                    "rawViewLogits": {
                        "canonical": canonical_score,
                        "horizontalMirror": mirrored_score,
                    },
                }
            )
        return {"samples": samples, "trackId": track_id}


def success_envelope_v2(
    *,
    identities: Mapping[str, str],
    clocks: Mapping[str, Any],
    supplied_tracks: Mapping[str, Any],
    tracks: Sequence[Mapping[str, Any]],
    score_ledger: Sequence[Mapping[str, Any]],
    outputs: Mapping[str, Any],
    measurements: Mapping[str, Any],
) -> dict[str, Any]:
    required = {
        "clockIdentity",
        "modelIdentity",
        "observationIdentity",
        "runIdentity",
        "runtimeIdentity",
        "trackIdentity",
    }
    if set(identities) != required:
        raise ContractViolation("v2 identities must exactly bind clock, track, model, observation, runtime, and run")
    for name, identity in identities.items():
        validate_content_identity(identity, f"v2 success identity {name}")
    result = {
        "authority": AUTHORITY,
        "clocks": dict(clocks),
        "cropAuthority": CROP_AUTHORITY,
        "identities": dict(identities),
        "measurements": dict(measurements),
        "outputs": dict(outputs),
        "rawScoreSemantics": (
            "Uncalibrated LR-ASD class-1 logits are independently averaged across "
            "ordered contexts [1,2,3,4,5,6] for canonical and horizontal-mirror "
            "112x112 crops; rawSpeakingScore is their arithmetic math.fsum mean. "
            "These values are not probabilities, confidence, speaker identity, "
            "crop decisions, or production authority."
        ),
        "schemaVersion": V2_OBSERVATION_SCHEMA_VERSION,
        "scoreLedger": list(score_ledger),
        "suppliedTracks": dict(supplied_tracks),
        "tracks": list(tracks),
    }
    assert_no_decision_authority(result)
    return result
