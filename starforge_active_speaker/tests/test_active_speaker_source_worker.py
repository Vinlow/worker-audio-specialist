"""Whole-source workload/denominator gates; no provider calls."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from active_speaker_contracts import ContractViolation
from active_speaker_source_worker import (
    CHUNK_FRAMES, CompleteSourceProcessor, POLICY, SCHEMA, SourceBatchRequest,
)


class SourceContractsTest(unittest.TestCase):
    def request(self):
        return {"schemaVersion": SCHEMA, "deadlineSeconds": 2400, "sources": [{
            "artifact": {"bytes": 100, "sha256": "a" * 64, "url": "https://fixture.example/video.mp4"},
            "durationUs": 3_600_000_000, "videoStreamIndex": 0, "audioStreamIndex": 1}]}

    def test_whole_hour_is_raw_duration_not_track_duration(self):
        request = SourceBatchRequest(self.request(), frozenset({"fixture.example"}))
        self.assertEqual(request.sources[0][1], 3_600_000_000)

    def test_duplicate_source_cannot_inflate_denominator(self):
        value = self.request()
        value["sources"].append(copy.deepcopy(value["sources"][0]))
        with self.assertRaisesRegex(ContractViolation, "duplicate"):
            SourceBatchRequest(value, frozenset({"fixture.example"}))

    def test_oversized_batch_rejected(self):
        value = self.request()
        for digest in ["b", "c"]:
            row = copy.deepcopy(value["sources"][0])
            row["artifact"]["sha256"] = digest * 64
            value["sources"].append(row)
        with self.assertRaisesRegex(ContractViolation, "two raw"):
            SourceBatchRequest(value, frozenset({"fixture.example"}))

    def test_request_remains_strict_and_host_bounded(self):
        for key, item in [("deadlineSeconds", 2401), ("deadlineSeconds", True), ("schemaVersion", "wrong")]:
            value = self.request()
            value[key] = item
            with self.assertRaises(Exception):
                SourceBatchRequest(value, frozenset({"fixture.example"}))
        with self.assertRaises(Exception):
            SourceBatchRequest(self.request(), frozenset({"different.example"}))

    def test_contiguous_whole_hour_coverage(self):
        rows = [{"startFrame": index * CHUNK_FRAMES, "frameCount": CHUNK_FRAMES} for index in range(30)]
        self.assertEqual(CompleteSourceProcessor.validate_coverage(rows, 3_600_000_000), 90_000)

    def test_coverage_gaps_and_incomplete_denominator_fail(self):
        for rows, duration in [([], 10), ([{"startFrame": 1, "frameCount": 25}], 1_000_000),
                ([{"startFrame": 0, "frameCount": 25}], 3_600_000_000),
                ([{"startFrame": 0, "frameCount": 25}, {"startFrame": 25, "frameCount": 25}], 2_000_000)]:
            with self.assertRaises(ContractViolation):
                CompleteSourceProcessor.validate_coverage(rows, duration)

    def test_no_hidden_paid_or_diagnostic_stages(self):
        self.assertFalse(POLICY["reviewRender"])
        self.assertEqual(POLICY["diarizationCalls"], 0)
        self.assertEqual(POLICY["geminiCalls"], 0)
        self.assertEqual(POLICY["views"], 2)
        self.assertEqual(POLICY["contextsSeconds"], [1, 2, 3, 4, 5, 6])

    @unittest.skipUnless(importlib.util.find_spec("cv2"), "image-resident media dependencies required")
    def test_actual_ffmpeg_chunk_boundary_preserves_every_frame(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.mp4"
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=size=32x32:rate=25",
                "-t", "121.04", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(source)], check=True)
            processor = CompleteSourceProcessor(None, root / "unused.onnx")
            chunks = processor.normalize(source, root, 0)
            counts = []
            for chunk in chunks:
                result = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                    "-show_entries", "stream=nb_read_frames", "-of", "json", str(chunk)], check=True, capture_output=True)
                counts.append(int(json.loads(result.stdout)["streams"][0]["nb_read_frames"]))
            self.assertEqual(counts, [3000, 26])


if __name__ == "__main__":
    unittest.main()
