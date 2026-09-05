from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from active_speaker_contracts import (
    LRASD_EXECUTED_SOURCE_FILES,
    canonical_json_bytes,
    content_identity,
    sha256_bytes,
)
from active_speaker_supplied_tracks import (
    LRASD_LICENSE_SHA256,
    SUPPLIED_TRACK_SCHEMA_VERSION,
    V1_RUNTIME_CLOSURE_FILES,
    _v1_preprocessing_policy,
)


def _fraction(numerator: int, denominator: int = 1) -> dict[str, int]:
    return {"denominator": denominator, "numerator": numerator}


def _default_v1_identities() -> dict[str, str]:
    return {
        "clockIdentity": f"sha256:{'a' * 64}",
        "modelIdentity": f"sha256:{'b' * 64}",
        "observationIdentity": f"sha256:{'c' * 64}",
        "runIdentity": f"sha256:{'d' * 64}",
        "runtimeIdentity": f"sha256:{'4' * 64}",
    }


def default_source_observation_record() -> dict[str, Any]:
    return {
        "bytes": 98_765,
        "identities": _default_v1_identities(),
        "schemaVersion": "starforge-active-speaker-observation-v1",
        "sha256": "9" * 64,
    }


def build_base_manifest(
    *,
    frame_count: int = 4,
    input_sha256: str = "1" * 64,
    include_track: bool = True,
    source_observation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source_duration = frame_count * 40_000
    clock = {
        "audio": {"channels": 1, "sampleCount": frame_count * 640, "sampleRateHz": 16_000},
        "inputStreams": {
            "audioStreamIndex": 1,
            "origins": {
                "audioOffsetFromVideoFrameZero": {
                    "roundedSamples": 0,
                    "roundingErrorSeconds": _fraction(0),
                    "seconds": _fraction(0),
                },
                "audioPresentationDuration": {
                    "durationTs": frame_count * 640,
                    "roundedSamplesAtCanonicalRate": frame_count * 640,
                    "roundingErrorSeconds": _fraction(0),
                    "seconds": _fraction(frame_count, 25),
                    "source": "stream-duration-ts",
                },
                "audioStream": {
                    "seconds": _fraction(0),
                    "startPts": 0,
                    "streamIndex": 1,
                    "timeBase": _fraction(1, 16_000),
                },
                "timelineValidation": {"audio": {}, "video": {}},
                "videoStream": {
                    "seconds": _fraction(0),
                    "startPts": 0,
                    "streamIndex": 0,
                    "timeBase": _fraction(1, 25),
                },
            },
            "videoStreamIndex": 0,
        },
        "preparedInput": {"bytes": 12_345, "sha256": input_sha256},
        "shots": [
            {"endFrameExclusive": frame_count, "shotIndex": 0, "startFrameInclusive": 0}
        ],
        "sourceInterval": {
            "endMicrosecondsExclusive": source_duration,
            "originalSourceVideoBytes": 54_321,
            "originalSourceVideoSha256": "3" * 64,
            "startMicrosecondsInclusive": 0,
        },
        "video": {
            "frameCount": frame_count,
            "frameRate": {"denominator": 1, "numerator": 25},
            "height": 60,
            "width": 100,
        },
    }
    tracks = []
    if include_track:
        tracks.append(
            {
                "frames": [
                    {
                        "faceBox": {"x1": 10 + frame, "x2": 30 + frame, "y1": 5, "y2": 25},
                        "frameIndex": frame,
                        "isDetectorObservation": frame in (0, frame_count - 1),
                        "pts": _fraction(frame, 25),
                    }
                    for frame in range(frame_count)
                ],
                "shotIndex": 0,
                "trackId": "shot-0000-track-0000",
            }
        )
    root = {
        "clock": clock,
        "clockIdentity": content_identity(clock),
        "producer": {
            "detector": {
                "evidenceRole": "DIRECT_OBSERVATION",
                "fallbackPolicy": "NONE",
                "kind": "YuNet-2023mar",
                "modelSha256": "6" * 64,
                "scoreThreshold": 0.7,
            },
            "geometryLineage": {
                "inputSha256": input_sha256,
                "kind": "BASE_OBSERVED",
                "sourceObservation": deepcopy(
                    source_observation or default_source_observation_record()
                ),
            },
            "kind": "starforge-canonical-face-tracks-v1",
            "processedFrames": [
                {"frameIndex": frame, "pts": _fraction(frame, 25)}
                for frame in range(frame_count)
            ],
            "runtimeIdentity": f"sha256:{'4' * 64}",
            "sourceClosureSha256": "5" * 64,
            "trackingPolicy": {
                "faceScoreThreshold": 0.7,
                "maximumGapFrames": 15,
                "minimumDetectionFrames": 1,
                "minimumIou": 0.5,
                "shotCutThreshold": 32.0,
                "shotDetection": "64x36-gray-mean-absolute-difference-v1",
                "tracker": "deterministic-greedy-shot-bounded-iou-v1",
            },
        },
        "schemaVersion": SUPPLIED_TRACK_SCHEMA_VERSION,
        "status": "COMPLETE",
        "tracks": tracks,
    }
    root["contentIdentity"] = content_identity(root)
    return root


def build_mirrored_manifest(
    source: dict[str, Any],
    *,
    source_file_sha256: str,
    derived_input_sha256: str = "2" * 64,
) -> dict[str, Any]:
    result = deepcopy(source)
    result.pop("contentIdentity")
    width = int(result["clock"]["video"]["width"])
    source_input_sha = str(source["clock"]["preparedInput"]["sha256"])
    result["clock"]["preparedInput"]["sha256"] = derived_input_sha256
    result["clock"]["preparedInput"]["bytes"] = 23_456
    result["clockIdentity"] = content_identity(result["clock"])
    result["producer"]["kind"] = "starforge-horizontal-mirror-face-tracks-v1"
    result["producer"]["detector"]["evidenceRole"] = (
        "AUTHENTICATED_SOURCE_MANIFEST"
    )
    result["producer"]["runtimeIdentity"] = f"sha256:{'7' * 64}"
    result["producer"]["sourceClosureSha256"] = "8" * 64
    result["producer"]["geometryLineage"] = {
        "derivedInputSha256": derived_input_sha256,
        "kind": "HORIZONTAL_MIRROR_DERIVED",
        "sourceInputSha256": source_input_sha,
        "sourceManifestContentIdentity": source["contentIdentity"],
        "sourceManifestSha256": source_file_sha256,
        "sourceObservation": deepcopy(
            source["producer"]["geometryLineage"]["sourceObservation"]
        ),
        "transform": {
            "kind": "HORIZONTAL_MIRROR_PIXEL_BOX_V1",
            "topology": "preserve-track-frame-shot-pts-observation-v1",
            "x1": "width-minus-source-x2",
            "x2": "width-minus-source-x1",
            "y": "unchanged",
        },
    }
    for target_track, source_track in zip(result["tracks"], source["tracks"], strict=True):
        for target_frame, source_frame in zip(
            target_track["frames"], source_track["frames"], strict=True
        ):
            source_box = source_frame["faceBox"]
            target_frame["faceBox"] = {
                "x1": width - source_box["x2"],
                "x2": width - source_box["x1"],
                "y1": source_box["y1"],
                "y2": source_box["y2"],
            }
    result["contentIdentity"] = content_identity(result)
    return result


def build_v1_observation_receipt(manifest: dict[str, Any]) -> dict[str, Any]:
    lrasd_source = [
        {
            "bytes": 100 + index,
            "path": path,
            "sha256": f"{index + 1:x}" * 64,
        }
        for index, path in enumerate(LRASD_EXECUTED_SOURCE_FILES)
    ]
    model = {
        "checkpoint": {"bytes": 3_426_337, "sha256": "e" * 64},
        "contextSeconds": [1, 2, 3, 4, 5, 6],
        "device": "cpu",
        "lrasdRevision": "1" * 40,
        "lrasdSource": lrasd_source,
        "lrasdSourceSha256": sha256_bytes(canonical_json_bytes(lrasd_source)),
        "preprocessingPolicy": _v1_preprocessing_policy(),
        "stateLoad": {"strict": True, "weightsOnly": True},
        "yunet": {"bytes": 232_589, "sha256": "6" * 64},
    }
    runtime_closure = [
        {
            "bytes": 200 + index,
            "path": path,
            "sha256": (
                LRASD_LICENSE_SHA256
                if path == "LR-ASD-LICENSE.txt"
                else f"{index + 1:x}" * 64
            ),
        }
        for index, path in enumerate(V1_RUNTIME_CLOSURE_FILES)
    ]
    runtime = {
        "audioWorkerBaseImageId": f"sha256:{'1' * 64}",
        "baseAudioWorkerBuildSha": "2" * 40,
        "dependencies": {"python": "3.10.12"},
        "runtimeClosure": runtime_closure,
        "runtimeVersion": "starforge-active-speaker-local-v1",
        "tools": {"ffmpeg": "fixture", "ffprobe": "fixture"},
    }
    tracks = deepcopy(manifest["tracks"])
    score_ledger = [
        {
            "samples": [
                {
                    "frameIndex": frame["frameIndex"],
                    "pts": deepcopy(frame["pts"]),
                    "rawSpeakingScore": 0.25,
                }
                for frame in track["frames"]
            ],
            "trackId": track["trackId"],
        }
        for track in tracks
    ]
    outputs = {
        "annotatedReview": {
            "bytes": 300,
            "path": "annotated-review.mp4",
            "sha256": "a" * 64,
            "validatedClock": {},
        },
        "canonicalAudio": {
            "bytes": 200,
            "path": "canonical-16khz-mono.wav",
            "sha256": "b" * 64,
        },
        "canonicalVideo": {
            "bytes": 250,
            "path": "canonical-25fps.mkv",
            "sha256": "c" * 64,
        },
    }
    clock_identity = content_identity(manifest["clock"])
    model_identity = content_identity(model)
    runtime_identity = content_identity(
        {
            "audioWorkerBaseImageId": runtime["audioWorkerBaseImageId"],
            "runtimeClosure": runtime_closure,
            "runtimeVersion": runtime["runtimeVersion"],
        }
    )
    observation_identity = content_identity(
        {
            "clockIdentity": clock_identity,
            "modelIdentity": model_identity,
            "scoreLedger": score_ledger,
            "trackingPolicy": manifest["producer"]["trackingPolicy"],
            "tracks": tracks,
        }
    )
    run_identity = content_identity(
        {
            "clockIdentity": clock_identity,
            "modelIdentity": model_identity,
            "observationIdentity": observation_identity,
            "outputs": outputs,
            "runtimeIdentity": runtime_identity,
        }
    )
    return {
        "authority": "DIAGNOSTIC_OBSERVATION_ONLY",
        "clocks": deepcopy(manifest["clock"]),
        "cropAuthority": "NONE",
        "identities": {
            "clockIdentity": clock_identity,
            "modelIdentity": model_identity,
            "observationIdentity": observation_identity,
            "runIdentity": run_identity,
            "runtimeIdentity": runtime_identity,
        },
        "measurements": {
            "detectedFaceCount": sum(len(track["frames"]) for track in tracks),
            "detectedTrackCount": len(tracks),
            "input": deepcopy(manifest["clock"]["preparedInput"]),
            "model": model,
            "runtime": runtime,
            "scoredTrackCount": len(tracks),
            "stageMilliseconds": {},
            "totalMilliseconds": 1.0,
            "trackingPolicy": deepcopy(manifest["producer"]["trackingPolicy"]),
        },
        "outputs": outputs,
        "rawScoreSemantics": "Fixture v1 raw logits only.",
        "schemaVersion": "starforge-active-speaker-observation-v1",
        "scoreLedger": score_ledger,
        "tracks": tracks,
    }


def bind_manifest_to_v1_receipt(
    manifest: dict[str, Any],
    receipt: dict[str, Any],
    *,
    receipt_sha256: str,
    receipt_bytes: int,
) -> dict[str, Any]:
    result = deepcopy(manifest)
    result.pop("contentIdentity")
    result["producer"]["geometryLineage"]["sourceObservation"] = {
        "bytes": receipt_bytes,
        "identities": deepcopy(receipt["identities"]),
        "schemaVersion": receipt["schemaVersion"],
        "sha256": receipt_sha256,
    }
    result["producer"]["runtimeIdentity"] = receipt["identities"]["runtimeIdentity"]
    result["producer"]["sourceClosureSha256"] = sha256_bytes(
        canonical_json_bytes(receipt["measurements"]["runtime"]["runtimeClosure"])
    )
    result["contentIdentity"] = content_identity(result)
    return result


def write_manifest(path: Path, manifest: dict[str, Any]) -> str:
    payload = canonical_json_bytes(manifest) + b"\n"
    path.write_bytes(payload)
    return sha256_bytes(payload)
