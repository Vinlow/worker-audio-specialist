from __future__ import annotations

from array import array
from fractions import Fraction
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import wave


MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))

from active_speaker_contracts import ContractViolation  # noqa: E402
from active_speaker_media import MediaProcessor, StreamOrigin  # noqa: E402


FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")


@unittest.skipUnless(FFMPEG and FFPROBE, "FFmpeg and ffprobe are required")
class SyntheticStreamOriginRegressionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        assert FFMPEG is not None
        assert FFPROBE is not None
        self.media = MediaProcessor(
            ffmpeg=FFMPEG,
            ffprobe=FFPROBE,
            maximum_frames=100,
            load_analysis_dependencies=False,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def run_ffmpeg(self, *arguments: str) -> None:
        result = subprocess.run(
            [FFMPEG, "-nostdin", "-hide_banner", "-loglevel", "error", *arguments],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def make_offset_fixture(self, name: str, *, audio_late: bool) -> Path:
        output = self.root / f"{name}.mkv"
        video = "testsrc2=size=160x90:rate=50:duration=" + ("2" if audio_late else "1.5")
        audio = (
            "aevalsrc=if(between(n\\,9600\\,9759)\\,0.8\\,0):"
            "s=16000:d=" + ("1.5" if audio_late else "2")
        )
        inputs = (
            ["-f", "lavfi", "-i", video, "-itsoffset", "0.5", "-f", "lavfi", "-i", audio]
            if audio_late
            else ["-itsoffset", "0.5", "-f", "lavfi", "-i", video, "-f", "lavfi", "-i", audio]
        )
        self.run_ffmpeg(
            *inputs,
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "ffv1",
            "-c:a",
            "pcm_s16le",
            "-shortest",
            str(output),
        )
        return output

    def normalized_impulse_index(self, fixture: Path) -> tuple[int, int, dict[str, object]]:
        streams = self.media.validate_input_streams(
            fixture,
            video_stream_index=0,
            audio_stream_index=1,
        )
        canonical_video = self.root / f"{fixture.stem}-video.mkv"
        self.media.normalize_video(fixture, canonical_video, video_stream_index=0)
        probe = subprocess.run(
            [
                FFPROBE,
                "-v",
                "error",
                "-count_frames",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_read_frames,start_time,r_frame_rate",
                "-of",
                "json",
                str(canonical_video),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        video_stream = json.loads(probe.stdout)["streams"][0]
        self.assertEqual(video_stream["start_time"], "0.000000")
        self.assertEqual(video_stream["r_frame_rate"], "25/1")
        frame_count = int(video_stream["nb_read_frames"])

        canonical_audio = self.root / f"{fixture.stem}-audio.wav"
        self.media.normalize_audio(
            fixture,
            canonical_audio,
            audio_stream_index=1,
            audio_presentation_samples=streams.audio_presentation_samples,
            audio_offset_samples_from_video_frame_zero=(
                streams.audio_offset_samples_from_video_frame_zero
            ),
            frame_count=frame_count,
        )
        with wave.open(str(canonical_audio), "rb") as handle:
            values = array("h")
            values.frombytes(handle.readframes(handle.getnframes()))
        nonzero = [index for index, value in enumerate(values) if abs(value) > 1_000]
        self.assertTrue(nonzero)
        return (
            streams.audio_offset_samples_from_video_frame_zero,
            nonzero[0],
            streams.clock_origin_json(),
        )

    def test_real_ffmpeg_preserves_positive_and_negative_500ms_origins(self) -> None:
        audio_late = self.make_offset_fixture("audio-late", audio_late=True)
        audio_early = self.make_offset_fixture("audio-early", audio_late=False)

        late_offset, late_impulse, late_origin = self.normalized_impulse_index(audio_late)
        early_offset, early_impulse, early_origin = self.normalized_impulse_index(audio_early)

        self.assertEqual(late_offset, 8_000)
        self.assertEqual(early_offset, -8_000)
        # The source impulse begins at local audio sample 9,600. Relative to
        # video frame zero it must land at 17,600 or 1,600 respectively.
        self.assertLessEqual(abs(late_impulse - 17_600), 16)
        self.assertLessEqual(abs(early_impulse - 1_600), 16)
        self.assertEqual(
            late_origin["audioOffsetFromVideoFrameZero"]["seconds"],
            {"denominator": 2, "numerator": 1},
        )
        self.assertEqual(
            early_origin["audioOffsetFromVideoFrameZero"]["seconds"],
            {"denominator": 2, "numerator": -1},
        )
        self.assertEqual(
            late_origin["audioPresentationDuration"]["source"],
            "packet-timeline",
        )
        self.assertEqual(
            early_origin["audioPresentationDuration"]["source"],
            "packet-timeline",
        )

    def test_aac_decoder_tail_is_trimmed_before_negative_origin_alignment(self) -> None:
        fixture = self.root / "aac-audio-early.mp4"
        self.run_ffmpeg(
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x90:rate=50:duration=1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=997:sample_rate=16000:duration=1",
            "-filter:v",
            "setpts=PTS+0.04/TB",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-vsync",
            "0",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            str(fixture),
        )
        fixture_probe = subprocess.run(
            [
                FFPROBE,
                "-v",
                "error",
                "-show_entries",
                "stream=index,start_pts,start_time",
                "-of",
                "json",
                str(fixture),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        fixture_streams = {
            stream["index"]: stream
            for stream in json.loads(fixture_probe.stdout)["streams"]
        }
        self.assertEqual(fixture_streams[0]["start_time"], "0.040000")
        self.assertEqual(fixture_streams[1]["start_time"], "0.000000")
        streams = self.media.validate_input_streams(
            fixture,
            video_stream_index=0,
            audio_stream_index=1,
        )
        self.assertEqual(streams.audio_offset_samples_from_video_frame_zero, -640)
        self.assertEqual(streams.audio_presentation_duration_ts, 16_000)
        self.assertEqual(streams.audio_presentation_duration_seconds, Fraction(1, 1))
        self.assertEqual(streams.audio_presentation_samples, 16_000)
        self.assertEqual(streams.audio_presentation_rounding_error_seconds, 0)

        decoded = subprocess.run(
            [
                FFMPEG,
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(fixture),
                "-map",
                "0:1",
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                "-f",
                "s16le",
                "-",
            ],
            check=True,
            capture_output=True,
        )
        decoded_sample_count = len(decoded.stdout) // 2
        self.assertGreater(decoded_sample_count, streams.audio_presentation_samples)

        canonical_video = self.root / "aac-audio-early-video.mkv"
        self.media.normalize_video(fixture, canonical_video, video_stream_index=0)
        video_probe = subprocess.run(
            [
                FFPROBE,
                "-v",
                "error",
                "-count_frames",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_read_frames",
                "-of",
                "json",
                str(canonical_video),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        frame_count = int(json.loads(video_probe.stdout)["streams"][0]["nb_read_frames"])
        self.assertEqual(frame_count, 25)

        canonical_audio = self.root / "aac-audio-early.wav"
        self.media.normalize_audio(
            fixture,
            canonical_audio,
            audio_stream_index=1,
            audio_presentation_samples=streams.audio_presentation_samples,
            audio_offset_samples_from_video_frame_zero=(
                streams.audio_offset_samples_from_video_frame_zero
            ),
            frame_count=frame_count,
        )
        with wave.open(str(canonical_audio), "rb") as handle:
            values = array("h")
            values.frombytes(handle.readframes(handle.getnframes()))
        unavailable_tail_samples = 640
        self.assertTrue(any(values[-1_280:-unavailable_tail_samples]))
        self.assertEqual(
            list(values[-unavailable_tail_samples:]),
            [0] * unavailable_tail_samples,
        )
        duration_receipt = streams.clock_origin_json()["audioPresentationDuration"]
        self.assertEqual(duration_receipt["durationTs"], 16_000)
        self.assertEqual(duration_receipt["roundedSamplesAtCanonicalRate"], 16_000)

    def test_midstream_audio_gap_fails_closed_before_normalization(self) -> None:
        fixture = self.root / "audio-gap.mp4"
        self.run_ffmpeg(
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x90:rate=25:duration=2",
            "-f",
            "lavfi",
            "-i",
            (
                "aevalsrc=if(between(n\\,19200\\,19359)\\,0.8\\,0):"
                "s=16000:d=2"
            ),
            "-filter_complex",
            "[1:a]asetpts=PTS+gte(T\\,1)*0.5/TB[a]",
            "-map",
            "0:v:0",
            "-map",
            "[a]",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            str(fixture),
        )
        decoded_probe = subprocess.run(
            [
                FFPROBE,
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_frames",
                "-show_entries",
                "frame=best_effort_timestamp_time",
                "-of",
                "json",
                str(fixture),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        times = [
            Fraction(frame["best_effort_timestamp_time"])
            for frame in json.loads(decoded_probe.stdout)["frames"]
        ]
        self.assertGreater(max(right - left for left, right in zip(times, times[1:])), Fraction(1, 2))
        with self.assertRaisesRegex(
            ContractViolation,
            "selected audio decoded frame timeline is discontinuous",
        ):
            self.media.validate_input_streams(
                fixture,
                video_stream_index=0,
                audio_stream_index=1,
            )

    def test_midstream_video_gap_fails_closed_before_normalization(self) -> None:
        fixture = self.root / "video-gap.mp4"
        self.run_ffmpeg(
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x90:rate=25:duration=2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=997:sample_rate=16000:duration=2.5",
            "-filter_complex",
            "[0:v]setpts=PTS+gte(T\\,1)*0.5/TB[v]",
            "-map",
            "[v]",
            "-map",
            "1:a:0",
            "-vsync",
            "0",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            str(fixture),
        )
        decoded_probe = subprocess.run(
            [
                FFPROBE,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_frames",
                "-show_entries",
                "frame=best_effort_timestamp_time",
                "-of",
                "json",
                str(fixture),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        times = [
            Fraction(frame["best_effort_timestamp_time"])
            for frame in json.loads(decoded_probe.stdout)["frames"]
        ]
        self.assertGreater(max(right - left for left, right in zip(times, times[1:])), Fraction(1, 2))
        with self.assertRaisesRegex(
            ContractViolation,
            "selected video (packet|decoded frame) timeline",
        ):
            self.media.validate_input_streams(
                fixture,
                video_stream_index=0,
                audio_stream_index=1,
            )

    def test_declared_origin_disagreement_with_first_decoded_frame_fails(self) -> None:
        self.media._run = lambda command, label: subprocess.CompletedProcess(  # type: ignore[method-assign]
            command,
            0,
            json.dumps(
                {
                    "frames": [
                        {
                            "best_effort_timestamp": 500,
                            "best_effort_timestamp_time": "0.500000",
                            "media_type": "video",
                            "stream_index": 0,
                        }
                    ]
                }
            ),
            "",
        )
        with self.assertRaisesRegex(ContractViolation, "differs from its first decoded frame"):
            self.media._first_decoded_frame_origin(
                self.root / "unused.mp4",
                stream_index=0,
                expected_media_type="video",
                stream_origin=StreamOrigin(0, 0, Fraction(1, 1_000)),
            )

    def test_annotated_review_requires_exact_decoded_media_contract(self) -> None:
        review = self.root / "review.mp4"
        self.run_ffmpeg(
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x90:rate=25:duration=1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=1000:sample_rate=16000:duration=1",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-shortest",
            str(review),
        )
        validation = self.media.validate_annotated_output(
            review,
            expected_width=160,
            expected_height=90,
            expected_video_frames=25,
            expected_audio_samples=16_000,
        )
        self.assertEqual(validation.video_decoded_frames, 25)
        self.assertGreaterEqual(validation.audio_decoded_samples, 16_000)
        with self.assertRaisesRegex(ContractViolation, "frame count mismatch"):
            self.media.validate_annotated_output(
                review,
                expected_width=160,
                expected_height=90,
                expected_video_frames=24,
                expected_audio_samples=16_000,
            )

        video_only = self.root / "video-only.mp4"
        self.run_ffmpeg(
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x90:rate=25:duration=1",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video_only),
        )
        with self.assertRaisesRegex(ContractViolation, "exactly one video and one audio"):
            self.media.validate_annotated_output(
                video_only,
                expected_width=160,
                expected_height=90,
                expected_video_frames=25,
                expected_audio_samples=16_000,
            )


if __name__ == "__main__":
    unittest.main()
