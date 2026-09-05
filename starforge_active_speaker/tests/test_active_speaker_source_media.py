"""Real FFmpeg equivalence and negative controls for single raw decoding."""
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from active_speaker_contracts import ContractViolation
from active_speaker_media import MediaProcessor
from active_speaker_source_media import DecodedClock, SingleDecodeMediaProcessor
from active_speaker_source_worker import CompleteSourceProcessor
from active_speaker_source_fast_worker import FastSourceProcessor


class SingleDecodeTest(unittest.TestCase):
    def test_real_parallel_chunks_match_sequential_and_remain_ordered(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.mp4"
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                "color=size=64x64:rate=25:duration=121.04", "-f", "lavfi", "-i",
                "sine=sample_rate=16000:duration=121.04", "-c:v", "libx264", "-threads", "2",
                "-c:a", "aac", str(source)], check=True)
            baseline, candidate = root / "baseline", root / "candidate"
            baseline.mkdir()
            candidate.mkdir()
            model = SimpleNamespace(device="cpu")  # No faces: no model inference is permitted/needed.
            yunet = Path("/opt/starforge-active-speaker/yunet.onnx")
            expected = CompleteSourceProcessor(model, yunet).run(source, baseline, 121_040_000, 0, 1)
            actual = FastSourceProcessor(model, yunet, decoder="cpu", workers=2).run(
                source, candidate, 121_040_000, 0, 1)
            for row in actual["chunks"]:
                row.pop("completedAfterMs")
            self.assertEqual(actual["chunks"], expected["chunks"])
            self.assertEqual(actual["canonicalFrames"], 3026)

    def test_exact_pts_retained_without_float_rounding(self):
        clock = DecodedClock(0, "1/24000")
        clock.consume("[Parsed_showinfo_0] config in time_base: 1/24000, frame_rate: 24000/1001")
        clock.consume("[Parsed_showinfo_0] n: 0 pts: 79080001 pts_time:3295 fmt:yuv420p")
        self.assertEqual(Fraction(clock.frames[0]["best_effort_timestamp_time"]), Fraction(79080001, 24000))

    def test_missing_pts_reset_and_time_base_changes_fail(self):
        for bad in ["n: 0 pts: NOPTS pts_time:NOPTS", "n: 1 pts: 0 pts_time:0",
                "config in time_base: 1/1000, frame_rate: 25/1"]:
            clock = DecodedClock(0, "1/25")
            clock.consume("[Parsed_showinfo_0] config in time_base: 1/25, frame_rate: 25/1")
            with self.assertRaises(ContractViolation):
                clock.consume("[Parsed_showinfo_0] " + bad)

    def test_hardware_unsupported_format_has_no_silent_cpu_fallback(self):
        media = SingleDecodeMediaProcessor(decoder="nvdec", ffmpeg="ffmpeg", ffprobe="ffprobe",
            maximum_frames=3000, load_analysis_dependencies=False)
        with self.assertRaises(ContractViolation):
            media.decoder_options({"codec_name": "av1", "pix_fmt": "yuv420p10le"}, 0)

    def test_actual_clock_and_pixels_equal_two_pass_baseline(self):
        for fps, offset, duration in [("50", "0", "2"), ("24000/1001", "0.5", "2.002"),
                ("25", "0", "121.04")]:
            with self.subTest(fps=fps, offset=offset), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source = root / "source.mp4"
                subprocess.run(["ffmpeg", "-v", "error", "-itsoffset", offset,
                    "-f", "lavfi", "-i", f"testsrc2=size=96x64:rate={fps}:duration={duration}",
                    "-f", "lavfi", "-i", "sine=frequency=1000:sample_rate=16000:duration=" + duration,
                    "-c:v", "libx264", "-threads", "2", "-pix_fmt", "yuv420p", "-c:a", "aac",
                    "-vsync", "passthrough", str(source)], check=True)
                reference = MediaProcessor(ffmpeg="ffmpeg", ffprobe="ffprobe", maximum_frames=3000)
                expected_streams = reference.validate_input_streams(source, video_stream_index=0, audio_stream_index=1)
                raw = reference._probe_decoded_frames(source, stream_index=0, media_type="video")
                baseline, candidate = root / "baseline", root / "candidate"
                baseline.mkdir()
                candidate.mkdir()
                chunks = CompleteSourceProcessor(None, root / "unused").normalize(source, baseline, 0)
                media = SingleDecodeMediaProcessor(ffmpeg="ffmpeg", ffprobe="ffprobe", maximum_frames=3000)
                actual_chunks = media.prepare(source, candidate, 0)
                self.assertEqual([r["best_effort_timestamp"] for r in media.clock.frames],
                    [r["best_effort_timestamp"] for r in raw])
                actual_streams = media.validate_input_streams(source, video_stream_index=0, audio_stream_index=1)
                self.assertEqual(actual_streams, expected_streams)
                self.assertEqual(len(chunks), len(actual_chunks))
                for expected, actual in zip(chunks, actual_chunks):
                    command = ["ffmpeg", "-v", "error", "-i"]
                    suffix = ["-map", "0:v:0", "-f", "framemd5", "-"]
                    self.assertEqual(subprocess.check_output(command + [str(expected)] + suffix),
                        subprocess.check_output(command + [str(actual)] + suffix))
                # Damaged decoded evidence must still be rejected by the frozen
                # validator, even though canonical pixels have already been made.
                media.clock.frames[1]["best_effort_timestamp"] += 100
                with self.assertRaises(ContractViolation):
                    media.validate_input_streams(source, video_stream_index=0, audio_stream_index=1)


if __name__ == "__main__":
    unittest.main()
