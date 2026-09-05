#!/usr/bin/env python3
"""Detector-free, supplied-track LR-ASD v2 diagnostic runtime.

This additive entry point leaves the receipt-bound v1 runtime byte-identical.
It scores authenticated canonical face geometry in original and horizontally
mirrored crop views and still grants no crop or production authority.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence

sys.dont_write_bytecode = True

from active_speaker_contracts import (  # noqa: E402
    AUDIO_CHANNELS,
    AUDIO_SAMPLE_RATE_HZ,
    AUDIO_SAMPLES_PER_VIDEO_FRAME,
    AUTHORITY,
    CROP_AUTHORITY,
    ContractViolation,
    SourceInterval,
    VIDEO_FRAMES_PER_SECOND,
    canonical_json_bytes,
    content_identity,
    require_file_hash,
    sha256_file,
    validate_content_identity,
    validate_git_revision,
    validate_sha256,
)
from active_speaker_media import TrackGeometry  # noqa: E402
from active_speaker_media_v2 import SuppliedTrackMediaProcessor  # noqa: E402
from active_speaker_model import LRASD_CONTEXT_SECONDS  # noqa: E402
from active_speaker_model_v2 import MirrorInvariantLrasdModelRunner  # noqa: E402
from active_speaker_supplied_tracks import (  # noqa: E402
    LRASD_V2_EXECUTED_SOURCE_FILES,
    SUPPLIED_TRACK_SCHEMA_VERSION,
    V2_FAILURE_SCHEMA_VERSION,
    V2_RUNTIME_VERSION,
    SuppliedTrackLimits,
    SuppliedTrackManifest,
    V1ObservationReceipt,
    V2RawScoreLedger,
    load_v1_observation_receipt,
    load_supplied_track_manifest,
    lrasd_v2_source_identity,
    success_envelope_v2,
    validate_base_observation_lineage,
    validate_geometry_lineage,
)


RUNTIME_V2_CLOSURE_FILES = (
    "Dockerfile.v2",
    "LR-ASD-LICENSE.txt",
    "active_speaker_contracts.py",
    "active_speaker_media.py",
    "active_speaker_media_v2.py",
    "active_speaker_model.py",
    "active_speaker_model_v2.py",
    "active_speaker_runtime_v2.py",
    "active_speaker_supplied_tracks.py",
    "requirements.lock.txt",
)
LRASD_LICENSE_SHA256 = "1ea9714e15424fb28d551675751172ba635b45c9d137a203a9988937e8215931"
BASE_IMAGE_ID_ENVIRONMENT = "STARFORGE_ACTIVE_SPEAKER_BASE_IMAGE_ID"
GIT_TIMEOUT_SECONDS = 30
MFCC_LOOKAHEAD_SAMPLES = 160


def _view_policy() -> dict[str, Any]:
    return {
        "cropStage": "canonical-112x112-grayscale-model-crop",
        "mirrorAxis": "width",
        "perViewContextAggregation": "ordered-[1,2,3,4,5,6]-math-fsum-mean-v1",
        "viewAggregation": "per-frame-math-fsum-arithmetic-mean-v1",
        "viewExecution": "sequential-single-view-v1",
        "views": ["CANONICAL", "HORIZONTAL_MIRROR"],
    }


def _preprocessing_policy_v2() -> dict[str, Any]:
    return {
        "audio": {
            "channels": AUDIO_CHANNELS,
            "decoderTail": "trim-to-selected-stream-presentation-duration-before-origin-alignment-v1",
            "originAlignment": "selected-stream-start-relative-to-video-frame-zero-v1",
            "sampleRateHz": AUDIO_SAMPLE_RATE_HZ,
            "sampleType": "signed-16-bit-little-endian",
            "timelineValidation": "packet-coverage-and-decoded-sample-clock-v1",
            "trackMfccLookaheadSamples": MFCC_LOOKAHEAD_SAMPLES,
            "trackMfccLookaheadTail": "zero-pad-only-at-canonical-audio-end-v1",
        },
        "faceInput": {
            "color": "grayscale",
            "geometrySource": "authenticated-dense-supplied-track-manifest-v2",
            "horizontalBoundsRounding": "floor-left-ceil-right-mirror-equivariant-v1",
            "interpolation": "none-supplied-boxes-are-dense",
            "modelHeight": 112,
            "modelWidth": 112,
            "smoothing": "edge-padded-sliding-median-center-size-13-v1",
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
        "viewPolicy": _view_policy(),
    }


class StageTimer:
    def __init__(self) -> None:
        self.started = time.perf_counter()
        self.stages_ms: dict[str, float] = {}

    def measure(self, name: str, action: Callable[[], Any]) -> Any:
        before = time.perf_counter()
        result = action()
        self.stages_ms[name] = round((time.perf_counter() - before) * 1_000, 3)
        return result

    def finish(self) -> dict[str, Any]:
        return {
            "stageMilliseconds": dict(self.stages_ms),
            "totalMilliseconds": round((time.perf_counter() - self.started) * 1_000, 3),
        }


def _absolute_existing_path(value: str, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise ContractViolation(f"{label} must be an absolute path")
    try:
        return path.resolve(strict=True)
    except OSError as error:
        raise ContractViolation(f"{label} does not resolve exactly: {path}") from error


def _absolute_json_input_path(value: str, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise ContractViolation(f"{label} must be an absolute path")
    try:
        parent = path.parent.resolve(strict=True)
    except OSError as error:
        raise ContractViolation(f"{label} parent does not resolve") from error
    candidate = parent / path.name
    if not candidate.exists() and not candidate.is_symlink():
        raise ContractViolation(f"{label} does not exist")
    return candidate


def _absolute_manifest_path(value: str) -> Path:
    return _absolute_json_input_path(value, "supplied-track manifest")


def _absolute_new_directory(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise ContractViolation("output directory must be an absolute path")
    if path.exists() or path.is_symlink():
        raise ContractViolation(f"output directory already exists: {path}")
    try:
        parent = path.parent.resolve(strict=True)
    except OSError as error:
        raise ContractViolation("output directory parent does not resolve exactly") from error
    return parent / path.name


def _run_git(command: Sequence[str], label: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as error:
        raise ContractViolation(f"{label} exceeded {GIT_TIMEOUT_SECONDS}s") from error
    except OSError as error:
        raise ContractViolation(f"cannot execute {label}: {error}") from error


def _verify_lrasd_v2_git_revision(root: Path, revision: str) -> None:
    exact_revision = validate_git_revision(revision)
    resolved = _run_git(
        [
            "git",
            "-c",
            f"safe.directory={root}",
            "-C",
            str(root),
            "rev-parse",
            "--verify",
            "HEAD^{commit}",
        ],
        "LR-ASD v2 revision check",
    )
    if resolved.returncode != 0 or resolved.stdout.strip() != exact_revision:
        raise ContractViolation("LR-ASD v2 checkout does not match the requested revision")
    clean = _run_git(
        [
            "git",
            "-c",
            f"safe.directory={root}",
            "-C",
            str(root),
            "diff",
            "--quiet",
            exact_revision,
            "--",
            *LRASD_V2_EXECUTED_SOURCE_FILES,
        ],
        "LR-ASD v2 source closure check",
    )
    if clean.returncode == 1:
        raise ContractViolation("executed LR-ASD v2 source differs from its Git revision")
    if clean.returncode != 0:
        raise ContractViolation("git could not verify the LR-ASD v2 source closure")


def _runtime_identity_v2(base_image_id: str) -> tuple[str, list[dict[str, Any]]]:
    validated_base = validate_content_identity(base_image_id, "Audio-Worker base image ID")
    root = Path(__file__).resolve(strict=True).parent
    manifest: list[dict[str, Any]] = []
    for relative_path in RUNTIME_V2_CLOSURE_FILES:
        digest, size = sha256_file(root / relative_path, f"v2 runtime source {relative_path}")
        manifest.append({"bytes": size, "path": relative_path, "sha256": digest})
    license_record = next(item for item in manifest if item["path"] == "LR-ASD-LICENSE.txt")
    if license_record["sha256"] != LRASD_LICENSE_SHA256:
        raise ContractViolation("bundled LR-ASD MIT license differs from pinned upstream")
    projection = {
        "audioWorkerBaseImageId": validated_base,
        "runtimeClosure": manifest,
        "runtimeVersion": V2_RUNTIME_VERSION,
    }
    return content_identity(projection), manifest


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = canonical_json_bytes(value) + b"\n"
    descriptor = -1
    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            os.fchmod(handle.fileno(), 0o400)
        os.link(temporary_path, path, follow_symlinks=False)
        directory_descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except OSError as error:
        raise ContractViolation(f"cannot write no-clobber v2 JSON output {path}: {error}") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


def _write_failure_receipt(output_directory: Path, error: Exception) -> None:
    _write_new_json(
        output_directory / "failure.json",
        {
            "authority": AUTHORITY,
            "cropAuthority": CROP_AUTHORITY,
            "error": {"message": str(error), "type": type(error).__name__},
            "reasonClass": "SYSTEM_FAILURE",
            "runtimeVersion": V2_RUNTIME_VERSION,
            "schemaVersion": V2_FAILURE_SCHEMA_VERSION,
        },
    )


def _hash_output(path: Path, relative_path: str) -> dict[str, Any]:
    digest, size = sha256_file(path, f"v2 output {relative_path}")
    return {"bytes": size, "path": relative_path, "sha256": digest}


def _track_audio_samples_v2(audio_samples: Any, track: TrackGeometry, numpy: Any) -> Any:
    start_sample = track.frame_indexes[0] * AUDIO_SAMPLES_PER_VIDEO_FRAME
    presentation_end = (track.frame_indexes[-1] + 1) * AUDIO_SAMPLES_PER_VIDEO_FRAME
    if start_sample < 0 or presentation_end > int(audio_samples.shape[0]):
        raise ContractViolation(f"track {track.track_id} exceeds canonical audio clock")
    desired_end = presentation_end + MFCC_LOOKAHEAD_SAMPLES
    samples = audio_samples[start_sample : min(desired_end, int(audio_samples.shape[0]))]
    expected = len(track.frame_indexes) * AUDIO_SAMPLES_PER_VIDEO_FRAME + MFCC_LOOKAHEAD_SAMPLES
    missing = expected - int(samples.shape[0])
    if missing < 0:
        raise ContractViolation(f"track {track.track_id} audio preflight over-read")
    if missing:
        samples = numpy.pad(samples, (0, missing), mode="constant")
    if getattr(samples, "ndim", None) != 1 or int(samples.shape[0]) != expected:
        raise ContractViolation(f"track {track.track_id} audio lookahead is incomplete")
    return samples


def _limits_from_args(args: argparse.Namespace) -> SuppliedTrackLimits:
    return SuppliedTrackLimits(
        maximum_manifest_bytes=args.maximum_manifest_bytes,
        maximum_input_bytes=args.maximum_input_bytes,
        maximum_checkpoint_bytes=args.maximum_checkpoint_bytes,
        maximum_frames=args.maximum_frames,
        maximum_frame_pixels=args.maximum_frame_pixels,
        maximum_tracks=args.maximum_tracks,
        maximum_track_frames=args.maximum_track_frames,
    )


def _validate_early_bindings(
    manifest: SuppliedTrackManifest,
    *,
    input_sha256: str,
    input_bytes: int,
    source_interval: SourceInterval,
    source_video_bytes: int,
    video_stream_index: int,
    audio_stream_index: int,
) -> None:
    clock = manifest.clock
    expected_input = {"bytes": input_bytes, "sha256": input_sha256}
    if clock["preparedInput"] != expected_input:
        raise ContractViolation("supplied tracks do not bind the exact prepared input")
    expected_source = {
        **source_interval.as_json(),
        "originalSourceVideoBytes": source_video_bytes,
    }
    if clock["sourceInterval"] != expected_source:
        raise ContractViolation("supplied tracks do not bind the exact source interval")
    streams = clock["inputStreams"]
    if streams["videoStreamIndex"] != video_stream_index:
        raise ContractViolation("supplied tracks bind a different video stream")
    if streams["audioStreamIndex"] != audio_stream_index:
        raise ContractViolation("supplied tracks bind a different audio stream")


def _build_clock_projection(
    *,
    manifest: SuppliedTrackManifest,
    streams: Any,
    input_sha256: str,
    input_bytes: int,
    source_interval: SourceInterval,
    source_video_bytes: int,
    audio_sample_count: int,
) -> dict[str, Any]:
    return {
        "audio": {
            "channels": AUDIO_CHANNELS,
            "sampleCount": audio_sample_count,
            "sampleRateHz": AUDIO_SAMPLE_RATE_HZ,
        },
        "inputStreams": {
            "audioStreamIndex": streams.audio_stream_index,
            "origins": streams.clock_origin_json(),
            "videoStreamIndex": streams.video_stream_index,
        },
        "preparedInput": {"bytes": input_bytes, "sha256": input_sha256},
        "shots": [shot.as_json() for shot in manifest.shots],
        "sourceInterval": {
            **source_interval.as_json(),
            "originalSourceVideoBytes": source_video_bytes,
        },
        "video": {
            "frameCount": manifest.frame_count,
            "frameRate": {"denominator": 1, "numerator": VIDEO_FRAMES_PER_SECOND},
            "height": manifest.height,
            "width": manifest.width,
        },
    }


def _authenticate_model_inputs(
    *,
    lrasd_root: Path,
    revision: str,
    expected_source_sha: str,
    checkpoint: Path,
    checkpoint_sha256: str,
    maximum_checkpoint_bytes: int,
) -> tuple[list[dict[str, Any]], int]:
    _verify_lrasd_v2_git_revision(lrasd_root, revision)
    source_sha, source_manifest = lrasd_v2_source_identity(lrasd_root)
    if source_sha != expected_source_sha:
        raise ContractViolation(
            f"LR-ASD v2 source closure mismatch: expected {expected_source_sha}, received {source_sha}"
        )
    checkpoint_bytes = require_file_hash(checkpoint, checkpoint_sha256, "LR-ASD checkpoint")
    if checkpoint_bytes > maximum_checkpoint_bytes:
        raise ContractViolation("LR-ASD checkpoint exceeds v2 preflight byte limit")
    return source_manifest, checkpoint_bytes


def _reauthenticate_media_inputs(
    *,
    input_video: Path,
    input_sha256: str,
    input_bytes: int,
    source_video: Path,
    source_sha256: str,
    source_bytes: int,
) -> None:
    post_input_bytes = require_file_hash(input_video, input_sha256, "input video")
    post_source_bytes = require_file_hash(
        source_video,
        source_sha256,
        "original source video",
    )
    if post_input_bytes != input_bytes or post_source_bytes != source_bytes:
        raise ContractViolation("v2 media input byte lengths changed during execution")


def _validate_authenticated_geometry_lineage(
    manifest: SuppliedTrackManifest,
    lineage_source_manifest: SuppliedTrackManifest | None,
    base_observation: V1ObservationReceipt,
) -> SuppliedTrackManifest:
    """Close supplied geometry through one exact, authenticated v1 observation."""

    validate_geometry_lineage(manifest, lineage_source_manifest)
    if manifest.producer["geometryLineage"]["kind"] == "BASE_OBSERVED":
        base_manifest = manifest
    else:
        if lineage_source_manifest is None:
            raise ContractViolation(
                "derived supplied tracks are missing their base lineage manifest"
            )
        base_manifest = lineage_source_manifest
    validate_base_observation_lineage(base_manifest, base_observation)
    return base_manifest


def _inspect_source(args: argparse.Namespace) -> int:
    root = _absolute_existing_path(args.lrasd_root, "LR-ASD root")
    if not root.is_dir():
        raise ContractViolation("LR-ASD root is not a directory")
    _verify_lrasd_v2_git_revision(root, args.lrasd_revision)
    identity, manifest = lrasd_v2_source_identity(root)
    response = {
        "executedFiles": manifest,
        "lrasdRevision": args.lrasd_revision,
        "lrasdSourceSha256": identity,
        "runtimeVersion": V2_RUNTIME_VERSION,
    }
    sys.stdout.buffer.write(canonical_json_bytes(response) + b"\n")
    return 0


def _score_tracks(
    *,
    model: MirrorInvariantLrasdModelRunner,
    audio_samples: Any,
    geometry: Sequence[TrackGeometry],
    crops_by_track: Mapping[str, Any],
) -> tuple[dict[str, list[float]], dict[str, list[float]], dict[str, list[float]]]:
    canonical_scores: dict[str, list[float]] = {}
    mirrored_scores: dict[str, list[float]] = {}
    mean_scores: dict[str, list[float]] = {}
    for track in geometry:
        samples = _track_audio_samples_v2(audio_samples, track, model.numpy)
        audio_feature, visual_feature = model.prepare_features(
            audio_samples=samples,
            visual_feature=crops_by_track[track.track_id],
            sample_rate_hz=AUDIO_SAMPLE_RATE_HZ,
        )
        if int(visual_feature.shape[0]) != len(track.frame_indexes):
            raise ContractViolation("v2 MFCC alignment must preserve every supplied track frame")
        scores = model.score_track_two_view(
            audio_feature=audio_feature,
            visual_feature=visual_feature,
        )
        canonical_scores[track.track_id] = list(scores.canonical)
        mirrored_scores[track.track_id] = list(scores.horizontal_mirror)
        mean_scores[track.track_id] = list(scores.mean)
    return canonical_scores, mirrored_scores, mean_scores


def _run(args: argparse.Namespace) -> int:
    timer = StageTimer()
    output_created = False
    output_directory: Path | None = None
    try:
        lrasd_root = _absolute_existing_path(args.lrasd_root, "LR-ASD root")
        checkpoint = _absolute_existing_path(args.checkpoint, "LR-ASD checkpoint")
        track_manifest_path = _absolute_manifest_path(args.supplied_tracks)
        base_observation_path = _absolute_json_input_path(
            args.base_observation_result,
            "base v1 observation result",
        )
        input_video = _absolute_existing_path(args.input_video, "input video")
        source_video = _absolute_existing_path(args.source_video, "original source video")
        output_directory = _absolute_new_directory(args.output_dir)
        limits = _limits_from_args(args)
        base_image_id = validate_content_identity(args.base_image_id, "Audio-Worker base image ID")
        baked_base = os.environ.get(BASE_IMAGE_ID_ENVIRONMENT)
        if baked_base != base_image_id:
            raise ContractViolation(
                f"Audio-Worker base image ID mismatch: expected baked {baked_base}, received {base_image_id}"
            )
        runtime_identity, runtime_manifest = timer.measure(
            "initialRuntimeClosure", lambda: _runtime_identity_v2(base_image_id)
        )
        source_interval = SourceInterval(
            args.source_video_sha256,
            args.source_interval_start_us,
            args.source_interval_end_us,
        )
        expected_source_sha = validate_sha256(
            args.lrasd_source_sha256, "LR-ASD v2 source closure SHA-256"
        )
        source_manifest, checkpoint_bytes = timer.measure(
            "initialModelClosure",
            lambda: _authenticate_model_inputs(
                lrasd_root=lrasd_root,
                revision=args.lrasd_revision,
                expected_source_sha=expected_source_sha,
                checkpoint=checkpoint,
                checkpoint_sha256=args.checkpoint_sha256,
                maximum_checkpoint_bytes=limits.maximum_checkpoint_bytes,
            ),
        )
        base_observation = timer.measure(
            "baseObservationAuthentication",
            lambda: load_v1_observation_receipt(
                base_observation_path,
                expected_sha256=args.base_observation_result_sha256,
                maximum_bytes=limits.maximum_manifest_bytes,
            ),
        )
        manifest = timer.measure(
            "suppliedTrackAuthentication",
            lambda: load_supplied_track_manifest(
                track_manifest_path,
                expected_sha256=args.supplied_tracks_sha256,
                limits=limits,
            ),
        )
        lineage_path_supplied = args.lineage_source_tracks is not None
        lineage_hash_supplied = args.lineage_source_tracks_sha256 is not None
        if lineage_path_supplied != lineage_hash_supplied:
            raise ContractViolation(
                "lineage source manifest path and SHA-256 must be supplied together"
            )
        lineage_source_path = (
            _absolute_manifest_path(args.lineage_source_tracks)
            if lineage_path_supplied
            else None
        )
        lineage_source_manifest = (
            timer.measure(
                "lineageSourceAuthentication",
                lambda: load_supplied_track_manifest(
                    lineage_source_path,
                    expected_sha256=args.lineage_source_tracks_sha256,
                    limits=limits,
                ),
            )
            if lineage_source_path is not None
            else None
        )
        _validate_authenticated_geometry_lineage(
            manifest,
            lineage_source_manifest,
            base_observation,
        )
        input_bytes = require_file_hash(input_video, args.input_sha256, "input video")
        if input_bytes > limits.maximum_input_bytes:
            raise ContractViolation("input video exceeds v2 preflight byte limit")
        source_video_bytes = require_file_hash(
            source_video, source_interval.source_video_sha256, "original source video"
        )
        _validate_early_bindings(
            manifest,
            input_sha256=args.input_sha256,
            input_bytes=input_bytes,
            source_interval=source_interval,
            source_video_bytes=source_video_bytes,
            video_stream_index=args.video_stream_index,
            audio_stream_index=args.audio_stream_index,
        )

        os.mkdir(output_directory, mode=0o700)
        os.chmod(output_directory, 0o700)
        output_created = True
        media = SuppliedTrackMediaProcessor(
            ffmpeg=args.ffmpeg,
            ffprobe=args.ffprobe,
            maximum_frames=limits.maximum_frames,
        )
        streams = timer.measure(
            "inputProbe",
            lambda: media.validate_input_streams(
                input_video,
                video_stream_index=args.video_stream_index,
                audio_stream_index=args.audio_stream_index,
            ),
        )
        canonical_video = output_directory / "canonical-25fps.mkv"
        timer.measure(
            "videoNormalization",
            lambda: media.normalize_video(
                input_video,
                canonical_video,
                video_stream_index=streams.video_stream_index,
            ),
        )
        canonical_validation = timer.measure(
            "canonicalVideoValidation",
            lambda: media.inspect_canonical_video(
                canonical_video,
                expected_width=manifest.width,
                expected_height=manifest.height,
                expected_frame_count=manifest.frame_count,
            ),
        )
        canonical_audio = output_directory / "canonical-16khz-mono.wav"
        audio_sample_count = timer.measure(
            "audioNormalization",
            lambda: media.normalize_audio(
                input_video,
                canonical_audio,
                audio_stream_index=streams.audio_stream_index,
                audio_presentation_samples=streams.audio_presentation_samples,
                audio_offset_samples_from_video_frame_zero=(
                    streams.audio_offset_samples_from_video_frame_zero
                ),
                frame_count=manifest.frame_count,
            ),
        )
        clock_projection = _build_clock_projection(
            manifest=manifest,
            streams=streams,
            input_sha256=args.input_sha256,
            input_bytes=input_bytes,
            source_interval=source_interval,
            source_video_bytes=source_video_bytes,
            audio_sample_count=audio_sample_count,
        )
        if canonical_json_bytes(clock_projection) != canonical_json_bytes(manifest.clock):
            raise ContractViolation("runtime media clock does not match supplied-track clock")
        if content_identity(clock_projection) != manifest.clock_identity:
            raise ContractViolation("runtime clock identity does not match supplied tracks")

        geometry = timer.measure("suppliedTrackGeometry", lambda: media.geometry_from_manifest(manifest))
        crops_by_track = timer.measure(
            "faceCropExtraction",
            lambda: media.extract_face_crops(canonical_video, geometry) if geometry else {},
        )
        audio_samples = media.read_audio(canonical_audio)
        model = timer.measure(
            "strictModelLoad",
            lambda: MirrorInvariantLrasdModelRunner(
                lrasd_root=lrasd_root,
                checkpoint=checkpoint,
                device=args.device,
            ),
        )
        canonical_scores, mirrored_scores, mean_scores = timer.measure(
            "lrasdTwoViewInference",
            lambda: _score_tracks(
                model=model,
                audio_samples=audio_samples,
                geometry=geometry,
                crops_by_track=crops_by_track,
            ),
        )

        post_manifest = timer.measure(
            "finalSuppliedTrackAuthentication",
            lambda: load_supplied_track_manifest(
                track_manifest_path,
                expected_sha256=args.supplied_tracks_sha256,
                limits=limits,
            ),
        )
        if post_manifest != manifest:
            raise ContractViolation("supplied-track manifest changed during v2 execution")
        if lineage_source_path is not None:
            post_lineage_source = timer.measure(
                "finalLineageSourceAuthentication",
                lambda: load_supplied_track_manifest(
                    lineage_source_path,
                    expected_sha256=args.lineage_source_tracks_sha256,
                    limits=limits,
                ),
            )
            if post_lineage_source != lineage_source_manifest:
                raise ContractViolation("lineage source manifest changed during v2 execution")
        else:
            post_lineage_source = None
        post_base_observation = timer.measure(
            "finalBaseObservationAuthentication",
            lambda: load_v1_observation_receipt(
                base_observation_path,
                expected_sha256=args.base_observation_result_sha256,
                maximum_bytes=limits.maximum_manifest_bytes,
            ),
        )
        if post_base_observation != base_observation:
            raise ContractViolation("base v1 observation changed during v2 execution")
        _validate_authenticated_geometry_lineage(
            post_manifest,
            post_lineage_source,
            post_base_observation,
        )
        post_source_manifest, post_checkpoint_bytes = timer.measure(
            "finalModelClosure",
            lambda: _authenticate_model_inputs(
                lrasd_root=lrasd_root,
                revision=args.lrasd_revision,
                expected_source_sha=expected_source_sha,
                checkpoint=checkpoint,
                checkpoint_sha256=args.checkpoint_sha256,
                maximum_checkpoint_bytes=limits.maximum_checkpoint_bytes,
            ),
        )
        if post_source_manifest != source_manifest or post_checkpoint_bytes != checkpoint_bytes:
            raise ContractViolation("LR-ASD v2 model closure changed during execution")
        timer.measure(
            "finalMediaInputAuthentication",
            lambda: _reauthenticate_media_inputs(
                input_video=input_video,
                input_sha256=args.input_sha256,
                input_bytes=input_bytes,
                source_video=source_video,
                source_sha256=source_interval.source_video_sha256,
                source_bytes=source_video_bytes,
            ),
        )

        admitted_frames = {track.track_id: track.frame_indexes for track in geometry}
        score_ledger = V2RawScoreLedger.build(
            admitted_frames,
            canonical_scores,
            mirrored_scores,
            mean_scores,
        )
        track_records = [track.as_json() for track in geometry]
        review_video = output_directory / "annotated-review.mp4"
        timer.measure(
            "annotatedReviewRender",
            lambda: media.render_annotated_review(
                canonical_video=canonical_video,
                canonical_audio=canonical_audio,
                output_video=review_video,
                width=manifest.width,
                height=manifest.height,
                geometry=geometry,
                scores_by_track=mean_scores,
            ),
        )
        output_validation = timer.measure(
            "annotatedReviewValidation",
            lambda: media.validate_annotated_output(
                review_video,
                expected_width=manifest.width,
                expected_height=manifest.height,
                expected_video_frames=manifest.frame_count,
                expected_audio_samples=audio_sample_count,
            ),
        )
        final_runtime_identity, final_runtime_manifest = timer.measure(
            "finalRuntimeClosure", lambda: _runtime_identity_v2(base_image_id)
        )
        if final_runtime_identity != runtime_identity or final_runtime_manifest != runtime_manifest:
            raise ContractViolation("active-speaker v2 runtime closure changed during execution")

        preprocessing_policy = _preprocessing_policy_v2()
        model_projection = {
            "checkpoint": {"bytes": checkpoint_bytes, "sha256": args.checkpoint_sha256},
            "contextSeconds": list(LRASD_CONTEXT_SECONDS),
            "device": str(model.device),
            "lrasdRevision": args.lrasd_revision,
            "lrasdSource": source_manifest,
            "lrasdSourceSha256": expected_source_sha,
            "preprocessingPolicy": preprocessing_policy,
            "stateLoad": {"strict": True, "weightsOnly": True},
            "viewPolicy": _view_policy(),
        }
        model_identity = content_identity(model_projection)
        observation_projection = {
            "clockIdentity": manifest.clock_identity,
            "modelIdentity": model_identity,
            "scoreLedger": score_ledger,
            "trackIdentity": manifest.content_identity,
            "tracks": track_records,
        }
        observation_identity = content_identity(observation_projection)
        output_records = {
            "annotatedReview": {
                **_hash_output(review_video, review_video.name),
                "validatedClock": output_validation.as_json(),
            },
            "canonicalAudio": _hash_output(canonical_audio, canonical_audio.name),
            "canonicalVideo": _hash_output(canonical_video, canonical_video.name),
        }
        run_identity = content_identity(
            {
                "clockIdentity": manifest.clock_identity,
                "modelIdentity": model_identity,
                "observationIdentity": observation_identity,
                "outputs": output_records,
                "runtimeIdentity": runtime_identity,
                "trackIdentity": manifest.content_identity,
            }
        )
        measurements = timer.finish()
        measurements.update(
            {
                "scoredTrackCount": len(geometry),
                "scoredTrackFrameCount": manifest.track_frame_count,
                "suppliedTrackCount": len(manifest.tracks),
                "suppliedTrackFrameCount": manifest.track_frame_count,
            }
        )
        supplied_track_record = manifest.receipt_record()
        supplied_track_record["baseObservation"] = base_observation.lineage_record()
        if lineage_source_manifest is not None:
            supplied_track_record["lineageSourceManifest"] = {
                "bytes": lineage_source_manifest.file_bytes,
                "contentIdentity": lineage_source_manifest.content_identity,
                "sha256": lineage_source_manifest.file_sha256,
            }
        receipt = success_envelope_v2(
            identities={
                "clockIdentity": manifest.clock_identity,
                "modelIdentity": model_identity,
                "observationIdentity": observation_identity,
                "runIdentity": run_identity,
                "runtimeIdentity": runtime_identity,
                "trackIdentity": manifest.content_identity,
            },
            clocks=clock_projection,
            supplied_tracks=supplied_track_record,
            tracks=track_records,
            score_ledger=score_ledger,
            outputs=output_records,
            measurements={
                **measurements,
                "input": {"bytes": input_bytes, "sha256": args.input_sha256},
                "model": model_projection,
                "preflightLimits": limits.as_json(),
                "runtime": {
                    "audioWorkerBaseImageId": base_image_id,
                    "baseAudioWorkerBuildSha": os.environ.get(
                        "AUDIO_WORKER_BUILD_SHA", "UNSET_LOCAL_RUNTIME"
                    ),
                    "dependencies": {
                        **media.dependency_versions(),
                        **model.dependency_versions(),
                        "python": platform.python_version(),
                        "pythonSpeechFeatures": "0.6",
                    },
                    "runtimeClosure": runtime_manifest,
                    "runtimeVersion": V2_RUNTIME_VERSION,
                    "tools": media.tool_versions(),
                },
                "suppliedTrackProducer": dict(manifest.producer),
                "validatedCanonicalVideo": canonical_validation.as_json(),
            },
        )
        result_path = output_directory / "result.json"
        _write_new_json(result_path, receipt)
        print(str(result_path))
        return 0
    except Exception as error:
        if output_created and output_directory is not None:
            try:
                _write_failure_receipt(output_directory, error)
            except Exception as failure_error:
                print(f"active-speaker v2 failure receipt also failed: {failure_error}", file=sys.stderr)
        raise


def _add_common_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--lrasd-root", required=True)
    parser.add_argument("--lrasd-revision", required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run detector-free supplied-track LR-ASD v2 diagnostics only."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect_parser = subparsers.add_parser(
        "source-identity-v2", help="print the exact v2 LR-ASD source closure"
    )
    _add_common_source_arguments(inspect_parser)
    inspect_parser.set_defaults(handler=_inspect_source)

    run_parser = subparsers.add_parser(
        "run-supplied-v2", help="run one no-clobber detector-free v2 diagnostic"
    )
    run_parser.add_argument("--base-image-id", required=True)
    _add_common_source_arguments(run_parser)
    run_parser.add_argument("--lrasd-source-sha256", required=True)
    run_parser.add_argument("--checkpoint", required=True)
    run_parser.add_argument("--checkpoint-sha256", required=True)
    run_parser.add_argument("--supplied-tracks", required=True)
    run_parser.add_argument("--supplied-tracks-sha256", required=True)
    run_parser.add_argument("--base-observation-result", required=True)
    run_parser.add_argument("--base-observation-result-sha256", required=True)
    run_parser.add_argument("--lineage-source-tracks")
    run_parser.add_argument("--lineage-source-tracks-sha256")
    run_parser.add_argument("--input-video", required=True)
    run_parser.add_argument("--input-sha256", required=True)
    run_parser.add_argument("--source-video", required=True)
    run_parser.add_argument("--source-video-sha256", required=True)
    run_parser.add_argument("--source-interval-start-us", required=True, type=int)
    run_parser.add_argument("--source-interval-end-us", required=True, type=int)
    run_parser.add_argument("--video-stream-index", required=True, type=int)
    run_parser.add_argument("--audio-stream-index", required=True, type=int)
    run_parser.add_argument("--output-dir", required=True)
    run_parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    run_parser.add_argument("--ffmpeg", default="ffmpeg")
    run_parser.add_argument("--ffprobe", default="ffprobe")
    run_parser.add_argument("--maximum-manifest-bytes", type=int, default=32 * 1024 * 1024)
    run_parser.add_argument("--maximum-input-bytes", type=int, default=2 * 1024 * 1024 * 1024)
    run_parser.add_argument("--maximum-checkpoint-bytes", type=int, default=64 * 1024 * 1024)
    run_parser.add_argument("--maximum-frames", type=int, default=4_500)
    run_parser.add_argument("--maximum-frame-pixels", type=int, default=3_840 * 2_160)
    run_parser.add_argument("--maximum-tracks", type=int, default=256)
    run_parser.add_argument("--maximum-track-frames", type=int, default=100_000)
    run_parser.set_defaults(handler=_run)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except ContractViolation as error:
        print(f"ACTIVE_SPEAKER_V2_CONTRACT_FAILURE: {error}", file=sys.stderr)
        return 2
    except Exception as error:
        print(
            f"ACTIVE_SPEAKER_V2_RUNTIME_FAILURE: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
