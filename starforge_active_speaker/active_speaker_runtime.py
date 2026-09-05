#!/usr/bin/env python3
"""Isolated local LR-ASD diagnostic runtime for Starforge Visual Director.

The successful output is an observation ledger and annotated review video. It
never emits a speaker decision or crop instruction and grants no authority to
production or any renderer.
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

from active_speaker_contracts import (
    AUDIO_CHANNELS,
    AUDIO_SAMPLE_RATE_HZ,
    AUDIO_SAMPLES_PER_VIDEO_FRAME,
    AUTHORITY,
    CROP_AUTHORITY,
    ContractViolation,
    DeterministicShotTracker,
    LRASD_EXECUTED_SOURCE_FILES,
    RawScoreLedger,
    SCHEMA_VERSION,
    SourceInterval,
    VIDEO_FRAMES_PER_SECOND,
    canonical_json_bytes,
    content_identity,
    lrasd_source_identity,
    require_file_hash,
    sha256_file,
    success_envelope,
    validate_content_identity,
    validate_git_revision,
    validate_sha256,
)
from active_speaker_media import MediaProcessor, TrackGeometry, shot_ranges
from active_speaker_model import LRASD_CONTEXT_SECONDS, LrasdModelRunner


RUNTIME_VERSION = "starforge-active-speaker-local-v1"
RUNTIME_CLOSURE_FILES = (
    "Dockerfile",
    "LR-ASD-LICENSE.txt",
    "active_speaker_contracts.py",
    "active_speaker_media.py",
    "active_speaker_model.py",
    "active_speaker_runtime.py",
    "requirements.lock.txt",
)
LRASD_LICENSE_SHA256 = "1ea9714e15424fb28d551675751172ba635b45c9d137a203a9988937e8215931"
BASE_IMAGE_ID_ENVIRONMENT = "STARFORGE_ACTIVE_SPEAKER_BASE_IMAGE_ID"


def _preprocessing_policy() -> dict[str, Any]:
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
            "frameRate": {
                "denominator": 1,
                "numerator": VIDEO_FRAMES_PER_SECOND,
            },
            "frameZero": "first-decoded-selected-video-frame",
            "normalization": "setpts-start-fps-near-setpts-frame-index-v1",
            "timelineValidation": "packet-coverage-and-decoded-cfr-clock-v1",
        },
    }


class StageTimer:
    def __init__(self) -> None:
        self.started = time.perf_counter()
        self.stages_ms: dict[str, float] = {}

    def measure(self, name: str, action: Callable[[], Any]) -> Any:
        before = time.perf_counter()
        result = action()
        elapsed_ms = (time.perf_counter() - before) * 1_000
        self.stages_ms[name] = round(elapsed_ms, 3)
        return result

    def finish(self) -> dict[str, Any]:
        return {
            "stageMilliseconds": dict(self.stages_ms),
            "totalMilliseconds": round(
                (time.perf_counter() - self.started) * 1_000,
                3,
            ),
        }


def _absolute_existing_path(value: str, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise ContractViolation(f"{label} must be an absolute path")
    try:
        return path.resolve(strict=True)
    except OSError as error:
        raise ContractViolation(f"{label} does not resolve exactly: {path}") from error


def _absolute_new_directory(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise ContractViolation("output directory must be an absolute path")
    if path.exists() or path.is_symlink():
        raise ContractViolation(f"output directory already exists: {path}")
    try:
        parent = path.parent.resolve(strict=True)
    except OSError as error:
        raise ContractViolation(
            f"output directory parent does not resolve exactly: {path.parent}"
        ) from error
    return parent / path.name


def _verify_lrasd_git_revision(root: Path, revision: str) -> None:
    exact_revision = validate_git_revision(revision)
    try:
        resolved = subprocess.run(
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
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        raise ContractViolation(f"cannot execute git for LR-ASD closure: {error}") from error
    if resolved.returncode != 0:
        raise ContractViolation("LR-ASD root is not an exact readable Git checkout")
    actual_revision = resolved.stdout.strip()
    if actual_revision != exact_revision:
        raise ContractViolation(
            f"LR-ASD revision mismatch: expected {exact_revision}, received {actual_revision}"
        )
    clean = subprocess.run(
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
            *LRASD_EXECUTED_SOURCE_FILES,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if clean.returncode != 0:
        if clean.returncode == 1:
            raise ContractViolation("executed LR-ASD source differs from its Git revision")
        raise ContractViolation("git could not verify the executed LR-ASD source closure")


def _runtime_identity(base_image_id: str) -> tuple[str, list[dict[str, Any]]]:
    validated_base_image_id = validate_content_identity(
        base_image_id,
        "Audio-Worker base image ID",
    )
    root = Path(__file__).resolve(strict=True).parent
    manifest: list[dict[str, Any]] = []
    for relative_path in RUNTIME_CLOSURE_FILES:
        digest, size = sha256_file(root / relative_path, f"runtime source {relative_path}")
        manifest.append(
            {
                "bytes": size,
                "path": relative_path,
                "sha256": digest,
            }
        )
    license_record = next(
        item for item in manifest if item["path"] == "LR-ASD-LICENSE.txt"
    )
    if license_record["sha256"] != LRASD_LICENSE_SHA256:
        raise ContractViolation("bundled LR-ASD MIT license differs from the pinned upstream text")
    projection = {
        "audioWorkerBaseImageId": validated_base_image_id,
        "runtimeClosure": manifest,
        "runtimeVersion": RUNTIME_VERSION,
    }
    return content_identity(projection), manifest


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = canonical_json_bytes(value) + b"\n"
    descriptor = -1
    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
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
        raise ContractViolation(f"cannot write no-clobber JSON output {path}: {error}") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


def _write_failure_receipt(output_directory: Path, error: Exception) -> None:
    failure = {
        "authority": AUTHORITY,
        "cropAuthority": CROP_AUTHORITY,
        "error": {
            "message": str(error),
            "type": type(error).__name__,
        },
        "runtimeVersion": RUNTIME_VERSION,
        "schemaVersion": "starforge-active-speaker-failure-v1",
    }
    _write_new_json(output_directory / "failure.json", failure)


def _hash_output(path: Path, relative_path: str) -> dict[str, Any]:
    digest, size = sha256_file(path, f"output {relative_path}")
    return {
        "bytes": size,
        "path": relative_path,
        "sha256": digest,
    }


def _track_audio_samples(audio_samples: Any, track: TrackGeometry) -> Any:
    start_sample = track.frame_indexes[0] * AUDIO_SAMPLES_PER_VIDEO_FRAME
    end_sample = (track.frame_indexes[-1] + 1) * AUDIO_SAMPLES_PER_VIDEO_FRAME
    if start_sample < 0 or end_sample > int(audio_samples.shape[0]):
        raise ContractViolation(f"track {track.track_id} exceeds canonical audio clock")
    samples = audio_samples[start_sample:end_sample]
    expected = len(track.frame_indexes) * AUDIO_SAMPLES_PER_VIDEO_FRAME
    if int(samples.shape[0]) != expected:
        raise ContractViolation(f"track {track.track_id} audio sample count is incomplete")
    return samples


def _inspect_source(args: argparse.Namespace) -> int:
    root = _absolute_existing_path(args.lrasd_root, "LR-ASD root")
    if not root.is_dir():
        raise ContractViolation("LR-ASD root is not a directory")
    _verify_lrasd_git_revision(root, args.lrasd_revision)
    identity, manifest = lrasd_source_identity(root)
    response = {
        "executedFiles": manifest,
        "lrasdRevision": args.lrasd_revision,
        "lrasdSourceSha256": identity,
    }
    sys.stdout.buffer.write(canonical_json_bytes(response) + b"\n")
    return 0


def _run(args: argparse.Namespace) -> int:
    timer = StageTimer()
    output_created = False
    output_directory: Path | None = None
    try:
        lrasd_root = _absolute_existing_path(args.lrasd_root, "LR-ASD root")
        checkpoint = _absolute_existing_path(args.checkpoint, "LR-ASD checkpoint")
        yunet = _absolute_existing_path(args.yunet, "YuNet model")
        input_video = _absolute_existing_path(args.input_video, "input video")
        source_video = _absolute_existing_path(args.source_video, "original source video")
        output_directory = _absolute_new_directory(args.output_dir)
        base_image_id = validate_content_identity(
            args.base_image_id,
            "Audio-Worker base image ID",
        )
        baked_base_image_id = os.environ.get(BASE_IMAGE_ID_ENVIRONMENT)
        if baked_base_image_id is None:
            raise ContractViolation(
                f"runtime environment is missing {BASE_IMAGE_ID_ENVIRONMENT}"
            )
        validate_content_identity(
            baked_base_image_id,
            "baked Audio-Worker base image ID",
        )
        if baked_base_image_id != base_image_id:
            raise ContractViolation(
                "Audio-Worker base image ID mismatch: "
                f"expected baked {baked_base_image_id}, received {base_image_id}"
            )
        source_interval = SourceInterval(
            source_video_sha256=args.source_video_sha256,
            start_microseconds=args.source_interval_start_us,
            end_microseconds=args.source_interval_end_us,
        )

        expected_source_sha = validate_sha256(
            args.lrasd_source_sha256,
            "LR-ASD source closure SHA-256",
        )
        _verify_lrasd_git_revision(lrasd_root, args.lrasd_revision)
        actual_source_sha, source_manifest = lrasd_source_identity(lrasd_root)
        if actual_source_sha != expected_source_sha:
            raise ContractViolation(
                "LR-ASD source closure mismatch: "
                f"expected {expected_source_sha}, received {actual_source_sha}"
            )
        checkpoint_bytes = require_file_hash(
            checkpoint,
            args.checkpoint_sha256,
            "LR-ASD checkpoint",
        )
        yunet_bytes = require_file_hash(yunet, args.yunet_sha256, "YuNet model")
        input_bytes = require_file_hash(
            input_video,
            args.input_sha256,
            "input video",
        )
        source_video_bytes = require_file_hash(
            source_video,
            source_interval.source_video_sha256,
            "original source video",
        )

        os.mkdir(output_directory, mode=0o700)
        os.chmod(output_directory, 0o700)
        output_created = True

        media = MediaProcessor(
            ffmpeg=args.ffmpeg,
            ffprobe=args.ffprobe,
            maximum_frames=args.maximum_frames,
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
        detection_pass = timer.measure(
            "shotAndFaceDetection",
            lambda: media.detect_faces_and_shots(
                canonical_video,
                yunet,
                shot_cut_threshold=args.shot_cut_threshold,
                face_score_threshold=args.face_score_threshold,
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
                frame_count=detection_pass.frame_count,
            ),
        )
        canonical_duration_microseconds = (
            detection_pass.frame_count * 1_000_000 // VIDEO_FRAMES_PER_SECOND
        )
        if (
            abs(source_interval.duration_microseconds - canonical_duration_microseconds)
            > 1_000_000 // VIDEO_FRAMES_PER_SECOND
        ):
            raise ContractViolation(
                "source interval duration differs from canonical video by more than one frame"
            )

        tracker = DeterministicShotTracker(
            minimum_iou=args.minimum_track_iou,
            maximum_gap_frames=args.maximum_track_gap_frames,
            minimum_detection_frames=args.minimum_track_detection_frames,
        )
        face_tracks = timer.measure(
            "faceTracking",
            lambda: tracker.track(detection_pass.detections),
        )
        geometry = timer.measure(
            "trackGeometry",
            lambda: media.build_track_geometry(face_tracks),
        )
        crops_by_track = timer.measure(
            "faceCropExtraction",
            lambda: media.extract_face_crops(canonical_video, geometry),
        )
        audio_samples = media.read_audio(canonical_audio)
        model = timer.measure(
            "strictModelLoad",
            lambda: LrasdModelRunner(
                lrasd_root=lrasd_root,
                checkpoint=checkpoint,
                device=args.device,
            ),
        )

        admitted_geometry: list[TrackGeometry] = []
        scores_by_track: dict[str, list[float]] = {}

        def score_all_tracks() -> None:
            for track in geometry:
                track_samples = _track_audio_samples(audio_samples, track)
                audio_feature, visual_feature = model.prepare_features(
                    audio_samples=track_samples,
                    visual_feature=crops_by_track[track.track_id],
                    sample_rate_hz=AUDIO_SAMPLE_RATE_HZ,
                )
                admitted = track.admitted_prefix(int(visual_feature.shape[0]))
                scores = model.score_track(
                    audio_feature=audio_feature,
                    visual_feature=visual_feature,
                )
                admitted_geometry.append(admitted)
                scores_by_track[track.track_id] = scores

        timer.measure("lrasdInference", score_all_tracks)
        admitted_frames_by_track = {
            track.track_id: track.frame_indexes for track in admitted_geometry
        }
        score_ledger = RawScoreLedger.build(
            admitted_frames_by_track,
            scores_by_track,
        )
        track_records = [track.as_json() for track in admitted_geometry]

        review_video = output_directory / "annotated-review.mp4"
        timer.measure(
            "annotatedReviewRender",
            lambda: media.render_annotated_review(
                canonical_video=canonical_video,
                canonical_audio=canonical_audio,
                output_video=review_video,
                width=detection_pass.width,
                height=detection_pass.height,
                geometry=admitted_geometry,
                scores_by_track=scores_by_track,
            ),
        )
        output_validation = timer.measure(
            "annotatedReviewValidation",
            lambda: media.validate_annotated_output(
                review_video,
                expected_width=detection_pass.width,
                expected_height=detection_pass.height,
                expected_video_frames=detection_pass.frame_count,
                expected_audio_samples=audio_sample_count,
            ),
        )

        runtime_identity, runtime_manifest = _runtime_identity(base_image_id)
        tool_versions = media.tool_versions()
        dependency_versions = {
            **media.dependency_versions(),
            **model.dependency_versions(),
            "python": platform.python_version(),
            "pythonSpeechFeatures": "0.6",
        }
        preprocessing_policy = _preprocessing_policy()
        model_projection = {
            "checkpoint": {
                "bytes": checkpoint_bytes,
                "sha256": args.checkpoint_sha256,
            },
            "contextSeconds": list(LRASD_CONTEXT_SECONDS),
            "device": str(model.device),
            "lrasdRevision": args.lrasd_revision,
            "lrasdSource": source_manifest,
            "lrasdSourceSha256": actual_source_sha,
            "preprocessingPolicy": preprocessing_policy,
            "stateLoad": {
                "strict": True,
                "weightsOnly": True,
            },
            "yunet": {
                "bytes": yunet_bytes,
                "sha256": args.yunet_sha256,
            },
        }
        model_identity = content_identity(model_projection)
        clock_projection = {
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
            "preparedInput": {
                "bytes": input_bytes,
                "sha256": args.input_sha256,
            },
            "shots": shot_ranges(detection_pass.shot_by_frame),
            "sourceInterval": {
                **source_interval.as_json(),
                "originalSourceVideoBytes": source_video_bytes,
            },
            "video": {
                "frameCount": detection_pass.frame_count,
                "frameRate": {
                    "denominator": 1,
                    "numerator": VIDEO_FRAMES_PER_SECOND,
                },
                "height": detection_pass.height,
                "width": detection_pass.width,
            },
        }
        clock_identity = content_identity(clock_projection)
        tracking_policy = {
            "faceScoreThreshold": args.face_score_threshold,
            "maximumGapFrames": args.maximum_track_gap_frames,
            "minimumDetectionFrames": args.minimum_track_detection_frames,
            "minimumIou": args.minimum_track_iou,
            "shotCutThreshold": args.shot_cut_threshold,
            "shotDetection": "64x36-gray-mean-absolute-difference-v1",
            "tracker": "deterministic-greedy-shot-bounded-iou-v1",
        }
        observation_projection = {
            "clockIdentity": clock_identity,
            "modelIdentity": model_identity,
            "scoreLedger": score_ledger,
            "trackingPolicy": tracking_policy,
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
                "clockIdentity": clock_identity,
                "modelIdentity": model_identity,
                "observationIdentity": observation_identity,
                "outputs": output_records,
                "runtimeIdentity": runtime_identity,
            }
        )
        measurements = timer.finish()
        measurements["detectedFaceCount"] = len(detection_pass.detections)
        measurements["detectedTrackCount"] = len(face_tracks)
        measurements["scoredTrackCount"] = len(admitted_geometry)

        receipt = success_envelope(
            identities={
                "clockIdentity": clock_identity,
                "modelIdentity": model_identity,
                "observationIdentity": observation_identity,
                "runIdentity": run_identity,
                "runtimeIdentity": runtime_identity,
            },
            clocks=clock_projection,
            tracks=track_records,
            score_ledger=score_ledger,
            outputs=output_records,
            measurements={
                **measurements,
                "input": {
                    "bytes": input_bytes,
                    "sha256": args.input_sha256,
                },
                "model": model_projection,
                "runtime": {
                    "audioWorkerBaseImageId": base_image_id,
                    "baseAudioWorkerBuildSha": os.environ.get(
                        "AUDIO_WORKER_BUILD_SHA", "UNSET_LOCAL_RUNTIME"
                    ),
                    "dependencies": dependency_versions,
                    "runtimeClosure": runtime_manifest,
                    "runtimeVersion": RUNTIME_VERSION,
                    "tools": tool_versions,
                },
                "trackingPolicy": tracking_policy,
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
                print(
                    f"active-speaker failure receipt also failed: {failure_error}",
                    file=sys.stderr,
                )
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run isolated LR-ASD diagnostics. Outputs are raw observations and "
            "never carry crop or production authority."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser(
        "source-identity",
        help="print the exact executed LR-ASD source-closure SHA-256",
    )
    inspect_parser.add_argument("--lrasd-root", required=True)
    inspect_parser.add_argument("--lrasd-revision", required=True)
    inspect_parser.set_defaults(handler=_inspect_source)

    run_parser = subparsers.add_parser("run", help="run one no-clobber local diagnostic")
    run_parser.add_argument("--base-image-id", required=True)
    run_parser.add_argument("--lrasd-root", required=True)
    run_parser.add_argument("--lrasd-revision", required=True)
    run_parser.add_argument("--lrasd-source-sha256", required=True)
    run_parser.add_argument("--checkpoint", required=True)
    run_parser.add_argument("--checkpoint-sha256", required=True)
    run_parser.add_argument("--yunet", required=True)
    run_parser.add_argument("--yunet-sha256", required=True)
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
    run_parser.add_argument("--maximum-frames", type=int, default=4_500)
    run_parser.add_argument("--shot-cut-threshold", type=float, default=32.0)
    run_parser.add_argument("--face-score-threshold", type=float, default=0.7)
    run_parser.add_argument("--minimum-track-iou", type=float, default=0.5)
    run_parser.add_argument("--maximum-track-gap-frames", type=int, default=15)
    run_parser.add_argument("--minimum-track-detection-frames", type=int, default=11)
    run_parser.set_defaults(handler=_run)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except ContractViolation as error:
        print(f"ACTIVE_SPEAKER_CONTRACT_FAILURE: {error}", file=sys.stderr)
        return 2
    except Exception as error:
        print(
            f"ACTIVE_SPEAKER_RUNTIME_FAILURE: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
