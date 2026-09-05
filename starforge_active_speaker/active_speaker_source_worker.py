"""Complete-source throughput measurement, not a framing/quality authority.

One resident, authenticated AVA model processes a bounded batch of whole raw
videos. All face tracks are scored; no diarization, Gemini, or review rendering
is invoked. The frozen v1/v2 runtimes and their receipt formats are untouched.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import tempfile
import time
from dataclasses import replace
from typing import Any

from active_speaker_contracts import (
    ContractViolation, DeterministicShotTracker,
    content_identity, require_file_hash, sha256_file,
)
from active_speaker_media import MediaProcessor
from active_speaker_media_v2 import SuppliedTrackMediaProcessor, edge_padded_median_13
from active_speaker_model_v2 import MirrorInvariantLrasdModelRunner
from active_speaker_runpod_handler import (
    BASE_IMAGE_ID, CHECKPOINTS, EXPECTED_RUNTIME_IDENTITY, LRASD_ROOT,
    LRASD_REVISION, LRASD_SOURCE_SHA256, _artifact_request, _download_once,
    _exact_integer, _strict_mapping,
)
from active_speaker_runtime_v2 import (
    _authenticate_model_inputs, _runtime_identity_v2, _score_tracks,
)

SCHEMA = "starforge-active-speaker-complete-source-batch-v1"
YUNET_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
YUNET_PATH = Path("/opt/starforge-active-speaker/yunet.onnx")
CHUNK_FRAMES = 3_000
MAX_SOURCE_US = 3_600_000_000
MAX_BATCH_US = 7_200_000_000
POLICY = {
    "analysisWidthMaximum": 640, "analysisFps": 25,
    "chunkFrames": CHUNK_FRAMES, "chunkBoundaryTrackReset": True,
    "checkpoint": "AVA", "views": 2, "contextsSeconds": [1, 2, 3, 4, 5, 6],
    "yunetSha256": YUNET_SHA256, "faceScoreThreshold": 0.7,
    "shotCutThreshold": 32, "trackMinimumIou": 0.5,
    "trackMaximumGapFrames": 15, "trackMinimumDetectionFrames": 11,
    "cropGeometry": "v2-edge-padded-median-13-mirror-equivariant",
    "routing": "ALL_VALID_FACE_TRACKS_NO_SPEECH_OR_CLIP_FILTER",
    "reviewRender": False, "diarizationCalls": 0, "geminiCalls": 0,
}


class SourceBatchRequest:
    def __init__(self, value: Any, allowed_hosts: frozenset[str]):
        root = _strict_mapping(value, {"schemaVersion", "sources", "deadlineSeconds"}, "input")
        if root["schemaVersion"] != SCHEMA:
            raise ContractViolation("complete-source request schema differs")
        self.deadline_seconds = _exact_integer(
            root["deadlineSeconds"], "deadlineSeconds", minimum=1, maximum=2400)
        sources = root["sources"]
        if not isinstance(sources, list) or not 1 <= len(sources) <= 3:
            raise ContractViolation("a batch requires one to three complete sources")
        self.sources = []
        seen: set[str] = set()
        total_us = 0
        for index, value in enumerate(sources):
            row = _strict_mapping(value, {"artifact", "durationUs", "videoStreamIndex", "audioStreamIndex"}, "source")
            artifact = _artifact_request(row["artifact"], name=f"source-{index}",
                maximum_size=2 * 1024**3, suffix=".mp4", allowed_hosts=allowed_hosts)
            if artifact.sha256 in seen:
                raise ContractViolation("duplicate source cannot inflate the raw-hour denominator")
            seen.add(artifact.sha256)
            duration = _exact_integer(row["durationUs"], "durationUs", minimum=1, maximum=MAX_SOURCE_US)
            video = _exact_integer(row["videoStreamIndex"], "videoStreamIndex", minimum=0, maximum=32)
            audio = _exact_integer(row["audioStreamIndex"], "audioStreamIndex", minimum=0, maximum=32)
            if video == audio:
                raise ContractViolation("video and audio stream indexes must differ")
            total_us += duration
            self.sources.append((artifact, duration, video, audio))
        if total_us > MAX_BATCH_US:
            raise ContractViolation("batch exceeds two raw input hours")


class SourceMeasurements:
    def __init__(self):
        self.stage_ms: dict[str, float] = {}

    def measure(self, name, action):
        started = time.monotonic()
        print(json.dumps({"event": "stage-start", "stage": name}), flush=True)
        try:
            return action()
        finally:
            elapsed = (time.monotonic() - started) * 1000
            self.stage_ms[name] = self.stage_ms.get(name, 0) + elapsed
            print(json.dumps({"event": "stage-end", "stage": name, "elapsedMs": elapsed}), flush=True)


class CompleteSourceProcessor:
    def __init__(self, model, yunet_path: Path):
        self.model = model
        self.yunet_path = yunet_path
        self.media = MediaProcessor(ffmpeg="ffmpeg", ffprobe="ffprobe", maximum_frames=CHUNK_FRAMES)
        self.crops = SuppliedTrackMediaProcessor(ffmpeg="ffmpeg", ffprobe="ffprobe", maximum_frames=CHUNK_FRAMES)

    def normalize(self, source: Path, directory: Path, stream_index: int):
        # Decode the entire selected stream once, without source cuts. FFV1
        # segments bound crop memory while preserving every canonical frame.
        self.media._run([
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-xerror", "-n",
            "-threads", "4", "-i", str(source), "-map", f"0:{stream_index}", "-an",
            "-vf", "setpts=PTS-STARTPTS,scale=w='min(640,iw)':h=-2,fps=25:round=near,setpts=N/(25*TB)",
            "-filter_threads", "2", "-vsync", "cfr", "-c:v", "ffv1", "-level", "3",
            "-threads", "4", "-g", "1", "-f", "segment", "-segment_time", "120",
            "-reset_timestamps", "1", str(directory / "chunk-%04d.mkv"),
        ], "whole-source canonical normalization")
        chunks = sorted(directory.glob("chunk-*.mkv"))
        if not chunks or len(chunks) > 31:
            raise ContractViolation("whole-source normalization returned invalid chunk count")
        return chunks

    def geometry(self, tracks):
        # Interpolation and topology come from the existing tracker. Replace
        # only v1's zero-padded medians with the exact frozen v2 edge policy.
        result = []
        for track in self.media.build_track_geometry(tracks):
            boxes = track.face_boxes
            result.append(replace(track,
                crop_center_x=edge_padded_median_13([(b.x1 + b.x2) / 2 for b in boxes]),
                crop_center_y=edge_padded_median_13([(b.y1 + b.y2) / 2 for b in boxes]),
                crop_half_size=edge_padded_median_13([max(b.x2 - b.x1, b.y2 - b.y1) / 2 for b in boxes])))
        return tuple(result)

    @staticmethod
    def validate_coverage(chunks, expected_duration_us):
        if not chunks:
            raise ContractViolation("complete source has no processed chunks")
        position = 0
        for index, chunk in enumerate(chunks):
            count = chunk["frameCount"]
            if chunk["startFrame"] != position or not 0 < count <= CHUNK_FRAMES:
                raise ContractViolation("source coverage has a gap, overlap, or invalid chunk")
            if index < len(chunks) - 1 and count != CHUNK_FRAMES:
                raise ContractViolation("non-final chunk is incomplete")
            position += count
        if abs(position * 40_000 - expected_duration_us) > 80_000:
            raise ContractViolation("complete decoded coverage differs from raw source duration")
        return position

    def run(self, source, directory, duration_us, video_index, audio_index):
        timer = SourceMeasurements()
        streams = timer.measure("rawSourceClockValidation", lambda: self.media.validate_input_streams(
            source, video_stream_index=video_index, audio_stream_index=audio_index))
        chunks = timer.measure("rawDecodeResizeNormalization", lambda: self.normalize(source, directory, video_index))
        # Count using the exact encoded frame counts, then assert against the
        # independently probed raw duration. Never use track seconds as denominator.
        counts = []
        for chunk in chunks:
            probe = self.media._run(["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0",
                "-show_entries", "stream=nb_read_packets", "-of", "json", str(chunk)], "canonical packet count")
            counts.append(int(json.loads(probe.stdout)["streams"][0]["nb_read_packets"]))
        frame_count = sum(counts)
        audio_path = directory / "canonical.wav"
        timer.measure("audioNormalization", lambda: self.media.normalize_audio(
            source, audio_path, audio_stream_index=audio_index,
            audio_presentation_samples=streams.audio_presentation_samples,
            audio_offset_samples_from_video_frame_zero=streams.audio_offset_samples_from_video_frame_zero,
            frame_count=frame_count))
        audio = self.media.read_audio(audio_path)
        rows = []
        position = 0
        for chunk, expected_count in zip(chunks, counts, strict=True):
            row = self.process_chunk(chunk, audio, position, timer)
            if row["frameCount"] != expected_count:
                raise ContractViolation("detection did not decode every canonical frame")
            rows.append(row)
            position += row["frameCount"]
            chunk.unlink()  # This job's disposable canonical segment only.
        self.validate_coverage(rows, duration_us)
        return {"rawDurationUs": duration_us, "canonicalFrames": frame_count,
            "chunks": rows, "measurementsMs": timer.stage_ms,
            "faceTrackFrames": sum(row["faceTrackFrames"] for row in rows),
            "scoredTracks": sum(row["scoredTracks"] for row in rows)}

    def process_chunk(self, chunk, audio, position, timer):
        detections = timer.measure("faceAndShotDetection", lambda: self.media.detect_faces_and_shots(
            chunk, self.yunet_path, shot_cut_threshold=32, face_score_threshold=0.7))
        tracker = DeterministicShotTracker(minimum_iou=0.5, maximum_gap_frames=15, minimum_detection_frames=11)
        tracks = timer.measure("faceTracking", lambda: tracker.track(detections.detections))
        geometry = timer.measure("trackGeometry", lambda: self.geometry(tracks))
        crops = timer.measure("faceCropExtraction", lambda: self.crops.extract_face_crops(chunk, geometry) if geometry else {})
        chunk_audio = audio[position * 640:(position + detections.frame_count) * 640]
        self.model.torch.cuda.synchronize() if str(self.model.device) == "cuda" else None
        scores = timer.measure("twoViewScoring", lambda: _score_tracks(
            model=self.model, audio_samples=chunk_audio, geometry=geometry, crops_by_track=crops))
        self.model.torch.cuda.synchronize() if str(self.model.device) == "cuda" else None
        # Materialize and hash every score, not a mocked/empty GPU exercise.
        return {"startFrame": position, "frameCount": detections.frame_count,
            "detectedFaces": len(detections.detections), "scoredTracks": len(geometry),
            "faceTrackFrames": sum(len(track.frame_indexes) for track in geometry),
            "scoreLedgerIdentity": content_identity(scores),
            "geometryIdentity": content_identity([track.as_json() for track in geometry])}


class ResidentSourceWorker:
    def __init__(self, *, device="cuda", yunet_path=YUNET_PATH):
        started = time.monotonic()
        identity, _ = _runtime_identity_v2(BASE_IMAGE_ID)
        if identity != EXPECTED_RUNTIME_IDENTITY:
            raise ContractViolation("frozen v2 dependency closure changed")
        checkpoint = Path(CHECKPOINTS["AVA"]["path"])
        _authenticate_model_inputs(lrasd_root=LRASD_ROOT, revision=LRASD_REVISION,
            expected_source_sha=LRASD_SOURCE_SHA256, checkpoint=checkpoint,
            checkpoint_sha256=CHECKPOINTS["AVA"]["sha256"], maximum_checkpoint_bytes=256 * 1024**2)
        require_file_hash(yunet_path, YUNET_SHA256, "YuNet model")
        self.model = MirrorInvariantLrasdModelRunner(lrasd_root=LRASD_ROOT, checkpoint=checkpoint, device=device)
        self.processor = CompleteSourceProcessor(self.model, yunet_path)
        self.initialization_ms = (time.monotonic() - started) * 1000
        self.completed_sources = 0
        self.allowed_hosts = frozenset(os.environ.get("STARFORGE_ARTIFACT_HOSTS", "").lower().split(","))
        self.worker_session = f"{os.environ.get('RUNPOD_POD_ID', 'local')}:{os.getpid()}"

    @staticmethod
    def deadline(signum, frame):
        del signum, frame
        raise TimeoutError("complete-source batch deadline exceeded")

    def handler(self, event):
        request = SourceBatchRequest(event.get("input"), self.allowed_hosts)
        started = time.monotonic()
        signal.signal(signal.SIGALRM, self.deadline)
        signal.alarm(request.deadline_seconds)
        results = []
        try:
            for artifact, duration, video, audio in request.sources:
                source_started = time.monotonic()
                with tempfile.TemporaryDirectory(prefix="starforge-source-") as temporary:
                    directory = Path(temporary)
                    downloaded = _download_once(artifact, directory, min(180, request.deadline_seconds))
                    result = self.processor.run(downloaded.path, directory, duration, video, audio)
                    result.update({"sourceSha256": artifact.sha256, "sourceBytes": artifact.size,
                        "downloadMs": downloaded.elapsed_ms,
                        "sourceWallMs": (time.monotonic() - source_started) * 1000})
                    results.append(result)
                    self.completed_sources += 1
                    print(json.dumps({"event": "source-complete", "index": len(results) - 1,
                        "rawDurationUs": duration, "sourceWallMs": result["sourceWallMs"]}), flush=True)
            response = {"schemaVersion": SCHEMA, "status": "COMPLETE", "cropAuthority": "NONE",
                "qualityAuthority": "NONE", "costAuthority": "MEASUREMENTS_NOT_PROVIDER_INVOICE",
                "policy": POLICY, "policyIdentity": content_identity(POLICY),
                "workerSession": self.worker_session, "modelLoadCount": 1,
                "modelInitializationMs": self.initialization_ms, "device": str(self.model.device),
                "gpuName": self.model.torch.cuda.get_device_name(0) if str(self.model.device) == "cuda" else None,
                "workerCommit": os.environ.get("STARFORGE_ACTIVE_SPEAKER_RELEASE_SHA", "unknown"),
                "workerSourceSha256": sha256_file(Path(__file__), "source worker")[0],
                "frozenDependencyIdentity": EXPECTED_RUNTIME_IDENTITY,
                "sources": results, "rawDurationUs": sum(row["rawDurationUs"] for row in results),
                "batchWallMs": (time.monotonic() - started) * 1000}
            response["contentIdentity"] = content_identity(response)
            return response
        except Exception as error:
            # Retain completed workload on failure but never label a partial
            # batch COMPLETE or normalize it over unprocessed source hours.
            return {"schemaVersion": SCHEMA, "status": "FAILED", "errorType": type(error).__name__,
                "errorMessage": str(error)[:1000],
                "completedSources": results, "batchWallMs": (time.monotonic() - started) * 1000}
        finally:
            signal.alarm(0)


if __name__ == "__main__":
    import runpod
    worker = ResidentSourceWorker()
    runpod.serverless.start({"handler": worker.handler})
