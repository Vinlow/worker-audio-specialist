"""Opt-in throughput profile; the baseline and frozen scoring stay unchanged."""
import json
import multiprocessing
import os
from pathlib import Path
import time

from active_speaker_contracts import ContractViolation, DeterministicShotTracker, content_identity, sha256_file
from active_speaker_runtime_v2 import _score_tracks
from active_speaker_source_media import SingleDecodeMediaProcessor
from active_speaker_source_worker import CHUNK_FRAMES, CompleteSourceProcessor, ResidentSourceWorker, SourceMeasurements


class ChunkPreparation:
    @staticmethod
    def run(chunk, yunet_path):
        # spawn, not fork: CPU processes never inherit a live CUDA context.
        processor = CompleteSourceProcessor(None, yunet_path)
        timer = SourceMeasurements()
        detection = timer.measure("faceAndShotDetection", lambda: processor.media.detect_faces_and_shots(
            chunk, yunet_path, shot_cut_threshold=32, face_score_threshold=0.7))
        tracker = DeterministicShotTracker(minimum_iou=0.5, maximum_gap_frames=15, minimum_detection_frames=11)
        tracks = timer.measure("faceTracking", lambda: tracker.track(detection.detections))
        geometry = timer.measure("trackGeometry", lambda: processor.geometry(tracks))
        crops = timer.measure("faceCropExtraction", lambda:
            processor.crops.extract_face_crops(chunk, geometry) if geometry else {})
        return detection.frame_count, len(detection.detections), geometry, crops, timer.stage_ms


class FastSourceProcessor(CompleteSourceProcessor):
    def __init__(self, model, yunet_path, *, decoder="nvdec", workers=4):
        super().__init__(model, yunet_path)
        if type(workers) is not int or not 1 <= workers <= 4:
            raise ContractViolation("CPU preparation workers must be in [1,4]")
        self.workers = workers
        self.media = SingleDecodeMediaProcessor(decoder=decoder, ffmpeg="ffmpeg", ffprobe="ffprobe",
            maximum_frames=CHUNK_FRAMES)
        self.progress = None

    def run(self, source, directory, duration_us, video_index, audio_index):
        started = time.monotonic()
        timer = SourceMeasurements()
        self.progress = {"rawDurationUs": duration_us, "chunks": [], "status": "PREPARING",
            "costAuthority": "PARTIAL_WORKLOAD_NOT_NORMAL_RAW_HOUR_COST"}
        chunks = timer.measure("singleRawDecodeResize", lambda: self.media.prepare(source, directory, video_index))
        streams = timer.measure("rawSourceClockValidation", lambda: self.media.validate_input_streams(
            source, video_stream_index=video_index, audio_stream_index=audio_index))
        counts = []
        for chunk in chunks:
            probe = self.media._run(["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0",
                "-show_entries", "stream=nb_read_packets", "-of", "json", str(chunk)], "canonical packet count")
            counts.append(int(json.loads(probe.stdout)["streams"][0]["nb_read_packets"]))
        audio_path = directory / "canonical.wav"
        timer.measure("audioNormalization", lambda: self.media.normalize_audio(
            source, audio_path, audio_stream_index=audio_index,
            audio_presentation_samples=streams.audio_presentation_samples,
            audio_offset_samples_from_video_frame_zero=streams.audio_offset_samples_from_video_frame_zero,
            frame_count=sum(counts)))
        audio = self.media.read_audio(audio_path)
        rows = self.process_chunks(chunks, counts, audio, timer, started)
        self.validate_coverage(rows, duration_us)
        self.progress["status"] = "COMPLETE"
        return {"rawDurationUs": duration_us, "canonicalFrames": sum(counts), "chunks": rows,
            "measurementsMs": timer.stage_ms, "mediaPreparation": self.media.evidence,
            "cpuPreparationWorkers": self.workers,
            "timeToFirstChunkMs": rows[0]["completedAfterMs"],
            "stageTimingSemantics": "CPU_STAGE_SUMS_OVERLAP; SOURCE_WALL_IS_ELAPSED",
            "faceTrackFrames": sum(row["faceTrackFrames"] for row in rows),
            "scoredTracks": sum(row["scoredTracks"] for row in rows)}

    def process_chunks(self, chunks, counts, audio, timer, started):
        pool = multiprocessing.get_context("spawn").Pool(self.workers)
        pending = {}
        position = 0
        rows = self.progress["chunks"]
        self.progress["status"] = "SCORING"
        try:
            for index in range(min(self.workers, len(chunks))):
                pending[index] = pool.apply_async(ChunkPreparation.run, (chunks[index], self.yunet_path))
            for index, chunk in enumerate(chunks):
                prepared = pending.pop(index).get()
                next_index = index + self.workers
                if next_index < len(chunks):
                    pending[next_index] = pool.apply_async(ChunkPreparation.run, (chunks[next_index], self.yunet_path))
                row = self.score_prepared(prepared, audio, position, timer)
                if row["frameCount"] != counts[index]:
                    raise ContractViolation("detection did not decode every canonical frame")
                row["completedAfterMs"] = (time.monotonic() - started) * 1000
                rows.append(row)
                position += row["frameCount"]
                print(json.dumps({"event": "chunk-complete", **row}), flush=True)
                chunk.unlink()
            return rows
        finally:
            # Public pool API terminates children on deadline, including pending
            # work; no orphan GPU work or executor wait beyond the job watchdog.
            pool.terminate()
            pool.join()

    def score_prepared(self, prepared, audio, position, timer):
        frame_count, detected_faces, geometry, crops, stage_ms = prepared
        for name, elapsed in stage_ms.items():
            timer.stage_ms[name] = timer.stage_ms.get(name, 0) + elapsed
        chunk_audio = audio[position * 640:(position + frame_count) * 640]
        scores = timer.measure("twoViewScoring", lambda: self.score(chunk_audio, geometry, crops))
        return {"startFrame": position, "frameCount": frame_count, "detectedFaces": detected_faces,
            "scoredTracks": len(geometry), "faceTrackFrames": sum(len(track.frame_indexes) for track in geometry),
            "scoreLedgerIdentity": content_identity(scores),
            "geometryIdentity": content_identity([track.as_json() for track in geometry])}

    def score(self, audio, geometry, crops):
        if str(self.model.device) == "cuda":
            self.model.torch.cuda.synchronize()
        scores = _score_tracks(model=self.model, audio_samples=audio, geometry=geometry, crops_by_track=crops)
        if str(self.model.device) == "cuda":
            self.model.torch.cuda.synchronize()
        return scores


class FastResidentSourceWorker(ResidentSourceWorker):
    def __init__(self):
        super().__init__()
        self.processor = FastSourceProcessor(self.model, self.processor.yunet_path)

    def handler(self, event):
        self.processor.progress = None
        response = super().handler(event)
        response["executionProfile"] = "single-decode-nvdec-parallel-cpu-v1"
        response["executionSourceIdentities"] = {name: sha256_file(Path(__file__).with_name(name), name)[0]
            for name in ["active_speaker_source_fast_worker.py", "active_speaker_source_media.py"]}
        if response["status"] == "FAILED":
            response["partialSource"] = self.processor.progress
        response.pop("contentIdentity", None)
        response["contentIdentity"] = content_identity(response)
        return response


if __name__ == "__main__":
    import runpod
    worker = FastResidentSourceWorker()
    runpod.serverless.start({"handler": worker.handler})
