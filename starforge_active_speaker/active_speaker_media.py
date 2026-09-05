"""Bounded, explicit-stream media processing for the active-speaker lab."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence
import wave

from active_speaker_contracts import (
    AUDIO_CHANNELS,
    AUDIO_SAMPLE_RATE_HZ,
    AUDIO_SAMPLES_PER_VIDEO_FRAME,
    ContractViolation,
    FaceBox,
    FaceDetection,
    FaceTrack,
    VIDEO_FRAMES_PER_SECOND,
)


@dataclass(frozen=True)
class StreamOrigin:
    stream_index: int
    start_pts: int
    time_base: Fraction

    @property
    def seconds(self) -> Fraction:
        return self.start_pts * self.time_base

    def as_json(self) -> dict[str, Any]:
        return {
            "seconds": {
                "denominator": self.seconds.denominator,
                "numerator": self.seconds.numerator,
            },
            "startPts": self.start_pts,
            "streamIndex": self.stream_index,
            "timeBase": {
                "denominator": self.time_base.denominator,
                "numerator": self.time_base.numerator,
            },
        }


@dataclass(frozen=True)
class PacketTimeline:
    stream_index: int
    packet_count: int
    intervals: tuple[tuple[int, int], ...]
    earliest_pts: int
    latest_end_pts: int


@dataclass(frozen=True)
class InputStreams:
    video_stream_index: int
    audio_stream_index: int
    video_origin: StreamOrigin
    audio_origin: StreamOrigin
    audio_presentation_duration_ts: int
    audio_presentation_duration_seconds: Fraction
    audio_presentation_samples: int
    audio_presentation_rounding_error_seconds: Fraction
    audio_presentation_source: str
    audio_offset_samples_from_video_frame_zero: int
    audio_offset_rounding_error_seconds: Fraction
    audio_timeline_validation: Mapping[str, Any]
    video_timeline_validation: Mapping[str, Any]
    probe: Mapping[str, Any]

    def clock_origin_json(self) -> dict[str, Any]:
        offset_seconds = self.audio_origin.seconds - self.video_origin.seconds
        return {
            "audioPresentationDuration": {
                "durationTs": self.audio_presentation_duration_ts,
                "roundedSamplesAtCanonicalRate": self.audio_presentation_samples,
                "roundingErrorSeconds": {
                    "denominator": (
                        self.audio_presentation_rounding_error_seconds.denominator
                    ),
                    "numerator": self.audio_presentation_rounding_error_seconds.numerator,
                },
                "seconds": {
                    "denominator": self.audio_presentation_duration_seconds.denominator,
                    "numerator": self.audio_presentation_duration_seconds.numerator,
                },
                "source": self.audio_presentation_source,
            },
            "audioOffsetFromVideoFrameZero": {
                "roundedSamples": self.audio_offset_samples_from_video_frame_zero,
                "roundingErrorSeconds": {
                    "denominator": self.audio_offset_rounding_error_seconds.denominator,
                    "numerator": self.audio_offset_rounding_error_seconds.numerator,
                },
                "seconds": {
                    "denominator": offset_seconds.denominator,
                    "numerator": offset_seconds.numerator,
                },
            },
            "audioStream": self.audio_origin.as_json(),
            "timelineValidation": {
                "audio": dict(self.audio_timeline_validation),
                "video": dict(self.video_timeline_validation),
            },
            "videoStream": self.video_origin.as_json(),
        }


@dataclass(frozen=True)
class DetectionPass:
    width: int
    height: int
    frame_count: int
    shot_by_frame: tuple[int, ...]
    detections: tuple[FaceDetection, ...]


@dataclass(frozen=True)
class OutputMediaValidation:
    audio_decoded_samples: int
    audio_first_pts_seconds: Fraction
    video_decoded_frames: int
    video_first_pts_seconds: Fraction

    def as_json(self) -> dict[str, Any]:
        return {
            "audio": {
                "decodedSamples": self.audio_decoded_samples,
                "firstPtsSeconds": {
                    "denominator": self.audio_first_pts_seconds.denominator,
                    "numerator": self.audio_first_pts_seconds.numerator,
                },
            },
            "maximumClockErrorMilliseconds": 1,
            "video": {
                "decodedFrames": self.video_decoded_frames,
                "firstPtsSeconds": {
                    "denominator": self.video_first_pts_seconds.denominator,
                    "numerator": self.video_first_pts_seconds.numerator,
                },
            },
        }


@dataclass(frozen=True)
class TrackGeometry:
    track_id: str
    shot_index: int
    frame_indexes: tuple[int, ...]
    face_boxes: tuple[FaceBox, ...]
    crop_center_x: tuple[float, ...]
    crop_center_y: tuple[float, ...]
    crop_half_size: tuple[float, ...]
    observed_detection_frames: tuple[int, ...]

    def admitted_prefix(self, frame_count: int) -> "TrackGeometry":
        if frame_count < 1 or frame_count > len(self.frame_indexes):
            raise ContractViolation("invalid admitted track prefix length")
        return TrackGeometry(
            track_id=self.track_id,
            shot_index=self.shot_index,
            frame_indexes=self.frame_indexes[:frame_count],
            face_boxes=self.face_boxes[:frame_count],
            crop_center_x=self.crop_center_x[:frame_count],
            crop_center_y=self.crop_center_y[:frame_count],
            crop_half_size=self.crop_half_size[:frame_count],
            observed_detection_frames=tuple(
                frame
                for frame in self.observed_detection_frames
                if frame <= self.frame_indexes[frame_count - 1]
            ),
        )

    def as_json(self) -> dict[str, Any]:
        frames = []
        observed = set(self.observed_detection_frames)
        for frame_index, box in zip(
            self.frame_indexes,
            self.face_boxes,
            strict=True,
        ):
            frames.append(
                {
                    "faceBox": box.as_json(),
                    "frameIndex": frame_index,
                    "isDetectorObservation": frame_index in observed,
                    "pts": {
                        "denominator": VIDEO_FRAMES_PER_SECOND,
                        "numerator": frame_index,
                    },
                }
            )
        return {
            "frames": frames,
            "shotIndex": self.shot_index,
            "trackId": self.track_id,
        }


class MediaProcessor:
    def __init__(
        self,
        *,
        ffmpeg: str,
        ffprobe: str,
        maximum_frames: int,
        load_analysis_dependencies: bool = True,
    ) -> None:
        if maximum_frames < 1:
            raise ContractViolation("maximum frames must be positive")
        self.ffmpeg = ffmpeg
        self.ffprobe = ffprobe
        self.maximum_frames = maximum_frames
        if not load_analysis_dependencies:
            return
        try:
            import cv2
            import numpy
            from scipy import signal
            from scipy.io import wavfile
        except ImportError as error:
            raise ContractViolation(
                f"media runtime dependency is unavailable: {error.name}"
            ) from error
        self.cv2 = cv2
        self.numpy = numpy
        self.signal = signal
        self.wavfile = wavfile
        cv2.setNumThreads(1)
        cv2.setRNGSeed(0)

    @staticmethod
    def _parse_stream_origin(stream: Mapping[str, Any], label: str) -> StreamOrigin:
        stream_index = stream.get("index")
        start_pts = stream.get("start_pts")
        time_base_text = stream.get("time_base")
        start_time_text = stream.get("start_time")
        if not isinstance(stream_index, int):
            raise ContractViolation(f"{label} has no integer stream index")
        if not isinstance(start_pts, int):
            raise ContractViolation(f"{label} has no exact integer start_pts")
        if not isinstance(time_base_text, str):
            raise ContractViolation(f"{label} has no exact time_base")
        if not isinstance(start_time_text, str):
            raise ContractViolation(f"{label} has no exact start_time")
        try:
            time_base = Fraction(time_base_text)
            reported_start = Fraction(start_time_text)
        except (ValueError, ZeroDivisionError) as error:
            raise ContractViolation(f"{label} has malformed origin metadata") from error
        if time_base <= 0:
            raise ContractViolation(f"{label} time_base must be positive")
        origin = StreamOrigin(stream_index, start_pts, time_base)
        if abs(origin.seconds - reported_start) > Fraction(1, 1_000):
            raise ContractViolation(
                f"{label} start_pts and start_time disagree by more than 1ms"
            )
        return origin

    @staticmethod
    def _round_fraction(value: Fraction) -> int:
        if value >= 0:
            return (2 * value.numerator + value.denominator) // (2 * value.denominator)
        positive = -value
        return -(
            (2 * positive.numerator + positive.denominator)
            // (2 * positive.denominator)
        )

    @staticmethod
    def _fraction_json(value: Fraction) -> dict[str, int]:
        return {
            "denominator": value.denominator,
            "numerator": value.numerator,
        }

    def _probe_packet_timeline(
        self,
        input_video: Path,
        *,
        stream_origin: StreamOrigin,
        media_type: str,
    ) -> PacketTimeline:
        result = self._run(
            [
                self.ffprobe,
                "-v",
                "error",
                "-select_streams",
                str(stream_origin.stream_index),
                "-show_packets",
                "-show_entries",
                "packet=stream_index,pts,pts_time,duration,duration_time",
                "-of",
                "json",
                str(input_video),
            ],
            f"selected {media_type} packet timeline probe",
        )
        try:
            packets = json.loads(result.stdout).get("packets")
        except json.JSONDecodeError as error:
            raise ContractViolation(
                f"selected {media_type} packet timeline probe returned invalid JSON"
            ) from error
        if not isinstance(packets, list) or not packets:
            raise ContractViolation(f"selected {media_type} stream has no packet timeline")
        intervals: list[tuple[int, int]] = []
        tolerance = Fraction(1, 1_000)
        for packet in packets:
            if not isinstance(packet, dict):
                raise ContractViolation(
                    f"selected {media_type} packet timeline is malformed"
                )
            if packet.get("stream_index") != stream_origin.stream_index:
                raise ContractViolation(
                    f"selected {media_type} packet timeline contains a different stream"
                )
            pts = packet.get("pts")
            duration = packet.get("duration")
            pts_time_text = packet.get("pts_time")
            duration_time_text = packet.get("duration_time")
            if (
                not isinstance(pts, int)
                or isinstance(pts, bool)
                or not isinstance(duration, int)
                or isinstance(duration, bool)
                or duration <= 0
                or not isinstance(pts_time_text, str)
                or not isinstance(duration_time_text, str)
            ):
                raise ContractViolation(
                    f"selected {media_type} packet has no exact positive timestamp and duration"
                )
            try:
                pts_time = Fraction(pts_time_text)
                duration_time = Fraction(duration_time_text)
            except (ValueError, ZeroDivisionError) as error:
                raise ContractViolation(
                    f"selected {media_type} packet has malformed timeline metadata"
                ) from error
            if (
                abs(pts * stream_origin.time_base - pts_time) > tolerance
                or abs(duration * stream_origin.time_base - duration_time) > tolerance
            ):
                raise ContractViolation(
                    f"selected {media_type} packet timestamp representations disagree"
                )
            intervals.append((pts, duration))
        intervals.sort(key=lambda interval: (interval[0], interval[1]))
        earliest_pts = intervals[0][0]
        latest_end_pts = max(pts + duration for pts, duration in intervals)
        return PacketTimeline(
            stream_index=stream_origin.stream_index,
            packet_count=len(intervals),
            intervals=tuple(intervals),
            earliest_pts=earliest_pts,
            latest_end_pts=latest_end_pts,
        )

    @classmethod
    def _resolve_presentation_duration(
        cls,
        stream: Mapping[str, Any],
        *,
        label: str,
        origin: StreamOrigin,
        packet_timeline: PacketTimeline,
    ) -> tuple[int, Fraction, str]:
        duration_ts = stream.get("duration_ts")
        duration_text = stream.get("duration")
        tolerance = Fraction(1, 1_000)
        if duration_ts is None:
            if (
                abs(
                    (packet_timeline.earliest_pts - origin.start_pts)
                    * origin.time_base
                )
                > tolerance
            ):
                raise ContractViolation(
                    f"{label} has no exact presentation-duration boundary"
                )
            duration_ts = packet_timeline.latest_end_pts - origin.start_pts
            source = "packet-timeline"
        else:
            if not isinstance(duration_ts, int) or isinstance(duration_ts, bool):
                raise ContractViolation(f"{label} duration_ts is not an exact integer")
            source = "stream-duration-ts"
        duration_seconds = duration_ts * origin.time_base
        if duration_seconds <= 0:
            raise ContractViolation(f"{label} duration must be positive")
        if duration_text is not None:
            if not isinstance(duration_text, str):
                raise ContractViolation(f"{label} duration is not an exact decimal")
            try:
                reported_duration = Fraction(duration_text)
            except (ValueError, ZeroDivisionError) as error:
                raise ContractViolation(
                    f"{label} has malformed duration metadata"
                ) from error
            if reported_duration <= 0:
                raise ContractViolation(f"{label} duration must be positive")
            if abs(duration_seconds - reported_duration) > tolerance:
                raise ContractViolation(
                    f"{label} duration_ts and duration disagree by more than 1ms"
                )
        return duration_ts, duration_seconds, source

    @classmethod
    def _canonical_audio_presentation_duration(
        cls,
        *,
        duration_ts: int,
        duration_seconds: Fraction,
        source: str,
    ) -> tuple[int, Fraction, int, Fraction, str]:
        exact_samples = duration_seconds * AUDIO_SAMPLE_RATE_HZ
        rounded_samples = cls._round_fraction(exact_samples)
        rounding_error_seconds = abs(
            Fraction(rounded_samples, AUDIO_SAMPLE_RATE_HZ) - duration_seconds
        )
        if rounded_samples < 1:
            raise ContractViolation(
                "selected audio stream presentation duration is less than one canonical sample"
            )
        if rounding_error_seconds > Fraction(1, 1_000):
            raise ContractViolation(
                "selected audio stream duration cannot be represented within 1ms at 16kHz"
            )
        return (
            duration_ts,
            duration_seconds,
            rounded_samples,
            rounding_error_seconds,
            source,
        )

    @classmethod
    def _validate_packet_timeline(
        cls,
        packet_timeline: PacketTimeline,
        *,
        media_type: str,
        origin: StreamOrigin,
        presentation_duration_ts: int,
        duration_source: str,
    ) -> dict[str, Any]:
        presentation_start = origin.start_pts
        presentation_end = presentation_start + presentation_duration_ts
        covered_end = presentation_start
        presentation_packet_count = 0
        maximum_error = Fraction(0, 1)
        tolerance = Fraction(1, 1_000)
        for pts, duration in packet_timeline.intervals:
            packet_end = pts + duration
            if packet_end <= presentation_start or pts >= presentation_end:
                continue
            clipped_start = max(pts, presentation_start)
            clipped_end = min(packet_end, presentation_end)
            continuity_error = (clipped_start - covered_end) * origin.time_base
            maximum_error = max(maximum_error, abs(continuity_error))
            if abs(continuity_error) > tolerance:
                direction = "gap" if continuity_error > 0 else "overlap"
                raise ContractViolation(
                    f"selected {media_type} packet timeline has a {direction} greater than 1ms"
                )
            covered_end = max(covered_end, clipped_end)
            presentation_packet_count += 1
        trailing_error = (presentation_end - covered_end) * origin.time_base
        maximum_error = max(maximum_error, abs(trailing_error))
        if presentation_packet_count < 1 or abs(trailing_error) > tolerance:
            raise ContractViolation(
                f"selected {media_type} packet timeline does not cover its presentation duration"
            )
        return {
            "durationSource": duration_source,
            "maximumContinuityErrorSeconds": cls._fraction_json(maximum_error),
            "packetCount": packet_timeline.packet_count,
            "presentationDurationTs": presentation_duration_ts,
            "presentationPacketCount": presentation_packet_count,
            "presentationOrder": "pts-ascending",
        }

    def dependency_versions(self) -> dict[str, str]:
        return {
            "opencv": str(self.cv2.__version__),
            "scipy": self._module_version("scipy"),
        }

    @staticmethod
    def _module_version(module_name: str) -> str:
        module = __import__(module_name)
        return str(getattr(module, "__version__", "UNKNOWN"))

    @staticmethod
    def _run(command: Sequence[str], label: str) -> subprocess.CompletedProcess[str]:
        try:
            result = subprocess.run(
                list(command),
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError as error:
            raise ContractViolation(f"failed to execute {label}: {error}") from error
        if result.returncode != 0:
            stderr = result.stderr.strip()
            if len(stderr) > 4_096:
                stderr = stderr[-4_096:]
            raise ContractViolation(
                f"{label} exited {result.returncode}: {stderr or 'no stderr'}"
            )
        return result

    @staticmethod
    def _run_bytes(
        command: Sequence[str],
        label: str,
    ) -> subprocess.CompletedProcess[bytes]:
        try:
            result = subprocess.run(
                list(command),
                check=False,
                capture_output=True,
            )
        except OSError as error:
            raise ContractViolation(f"failed to execute {label}: {error}") from error
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace").strip()
            if len(stderr) > 4_096:
                stderr = stderr[-4_096:]
            raise ContractViolation(
                f"{label} exited {result.returncode}: {stderr or 'no stderr'}"
            )
        return result

    def tool_versions(self) -> dict[str, str]:
        ffmpeg = self._run([self.ffmpeg, "-version"], "ffmpeg version")
        ffprobe = self._run([self.ffprobe, "-version"], "ffprobe version")
        return {
            "ffmpeg": ffmpeg.stdout.splitlines()[0].strip(),
            "ffprobe": ffprobe.stdout.splitlines()[0].strip(),
        }

    def _first_decoded_frame_origin(
        self,
        input_video: Path,
        *,
        stream_index: int,
        expected_media_type: str,
        stream_origin: StreamOrigin,
    ) -> StreamOrigin:
        result = self._run(
            [
                self.ffprobe,
                "-v",
                "error",
                "-select_streams",
                str(stream_index),
                "-read_intervals",
                "%+#64",
                "-show_frames",
                "-show_entries",
                "frame=stream_index,media_type,best_effort_timestamp,best_effort_timestamp_time",
                "-of",
                "json",
                str(input_video),
            ],
            f"first decoded {expected_media_type} frame probe",
        )
        try:
            frames = json.loads(result.stdout).get("frames")
        except json.JSONDecodeError as error:
            raise ContractViolation("first decoded frame probe returned invalid JSON") from error
        if not isinstance(frames, list):
            raise ContractViolation("first decoded frame probe returned no frame list")
        matching = [
            frame
            for frame in frames
            if isinstance(frame, dict)
            and frame.get("stream_index") == stream_index
            and frame.get("media_type") == expected_media_type
        ]
        if not matching:
            raise ContractViolation(
                f"selected {expected_media_type} stream yielded no decoded frame in 64 packets"
            )
        first = matching[0]
        first_pts = first.get("best_effort_timestamp")
        first_time = first.get("best_effort_timestamp_time")
        if not isinstance(first_pts, int) or not isinstance(first_time, str):
            raise ContractViolation(
                f"first decoded {expected_media_type} frame has no exact timestamp"
            )
        decoded_origin = StreamOrigin(stream_index, first_pts, stream_origin.time_base)
        try:
            reported_time = Fraction(first_time)
        except (ValueError, ZeroDivisionError) as error:
            raise ContractViolation(
                f"first decoded {expected_media_type} timestamp is malformed"
            ) from error
        tolerance = Fraction(1, 1_000)
        if abs(decoded_origin.seconds - reported_time) > tolerance:
            raise ContractViolation(
                f"first decoded {expected_media_type} timestamp representations disagree"
            )
        if abs(decoded_origin.seconds - stream_origin.seconds) > tolerance:
            raise ContractViolation(
                f"selected {expected_media_type} stream origin differs from its first decoded frame"
            )
        return decoded_origin

    def _probe_decoded_frames(
        self,
        input_video: Path,
        *,
        stream_index: int,
        media_type: str,
    ) -> list[Mapping[str, Any]]:
        result = self._run(
            [
                self.ffprobe,
                "-v",
                "error",
                "-select_streams",
                str(stream_index),
                "-show_frames",
                "-show_entries",
                (
                    "frame=stream_index,media_type,best_effort_timestamp,"
                    "best_effort_timestamp_time,nb_samples"
                ),
                "-of",
                "json",
                str(input_video),
            ],
            f"selected {media_type} decoded frame timeline probe",
        )
        try:
            frames = json.loads(result.stdout).get("frames")
        except json.JSONDecodeError as error:
            raise ContractViolation(
                f"selected {media_type} decoded frame timeline returned invalid JSON"
            ) from error
        if not isinstance(frames, list) or not frames:
            raise ContractViolation(f"selected {media_type} stream has no decoded timeline")
        matching: list[Mapping[str, Any]] = []
        for frame in frames:
            if (
                not isinstance(frame, dict)
                or frame.get("stream_index") != stream_index
                or frame.get("media_type") != media_type
            ):
                raise ContractViolation(
                    f"selected {media_type} decoded timeline contains a different stream"
                )
            matching.append(frame)
        return matching

    @staticmethod
    def _decoded_frame_seconds(
        frame: Mapping[str, Any],
        *,
        media_type: str,
        time_base: Fraction,
    ) -> Fraction:
        pts = frame.get("best_effort_timestamp")
        pts_time_text = frame.get("best_effort_timestamp_time")
        if (
            not isinstance(pts, int)
            or isinstance(pts, bool)
            or not isinstance(pts_time_text, str)
        ):
            raise ContractViolation(
                f"selected {media_type} decoded frame has no exact timestamp"
            )
        try:
            reported_time = Fraction(pts_time_text)
        except (ValueError, ZeroDivisionError) as error:
            raise ContractViolation(
                f"selected {media_type} decoded frame timestamp is malformed"
            ) from error
        exact_time = pts * time_base
        if abs(exact_time - reported_time) > Fraction(1, 1_000):
            raise ContractViolation(
                f"selected {media_type} decoded frame timestamp representations disagree"
            )
        return exact_time

    def _validate_audio_decoded_timeline(
        self,
        input_video: Path,
        *,
        audio_origin: StreamOrigin,
        sample_rate_hz: int,
        presentation_duration_seconds: Fraction,
    ) -> dict[str, Any]:
        frames = self._probe_decoded_frames(
            input_video,
            stream_index=audio_origin.stream_index,
            media_type="audio",
        )
        decoded_samples = 0
        maximum_error = Fraction(0, 1)
        tolerance = Fraction(1, 1_000)
        for frame in frames:
            actual_time = self._decoded_frame_seconds(
                frame,
                media_type="audio",
                time_base=audio_origin.time_base,
            )
            expected_time = audio_origin.seconds + Fraction(
                decoded_samples,
                sample_rate_hz,
            )
            clock_error = abs(actual_time - expected_time)
            maximum_error = max(maximum_error, clock_error)
            if clock_error > tolerance:
                raise ContractViolation(
                    "selected audio decoded frame timeline is discontinuous"
                )
            sample_count = frame.get("nb_samples")
            if (
                not isinstance(sample_count, int)
                or isinstance(sample_count, bool)
                or sample_count < 1
            ):
                raise ContractViolation(
                    "selected audio decoded frame has no positive integer sample count"
                )
            decoded_samples += sample_count
        decoded_end = audio_origin.seconds + Fraction(decoded_samples, sample_rate_hz)
        presentation_end = audio_origin.seconds + presentation_duration_seconds
        if decoded_end + tolerance < presentation_end:
            raise ContractViolation(
                "selected audio decoded frame timeline does not cover its presentation duration"
            )
        return {
            "basis": "best-effort-pts-vs-cumulative-decoded-samples",
            "decodedEndSeconds": self._fraction_json(decoded_end),
            "decodedFrameCount": len(frames),
            "decodedSampleCount": decoded_samples,
            "maximumClockErrorSeconds": self._fraction_json(maximum_error),
            "sampleRateHz": sample_rate_hz,
        }

    def _validate_video_decoded_timeline(
        self,
        input_video: Path,
        *,
        video_origin: StreamOrigin,
        frame_rate: Fraction,
        presentation_duration_seconds: Fraction,
    ) -> dict[str, Any]:
        frames = self._probe_decoded_frames(
            input_video,
            stream_index=video_origin.stream_index,
            media_type="video",
        )
        maximum_error = Fraction(0, 1)
        tolerance = Fraction(1, 1_000)
        for frame_index, frame in enumerate(frames):
            actual_time = self._decoded_frame_seconds(
                frame,
                media_type="video",
                time_base=video_origin.time_base,
            )
            expected_time = video_origin.seconds + Fraction(frame_index, 1) / frame_rate
            clock_error = abs(actual_time - expected_time)
            maximum_error = max(maximum_error, clock_error)
            if clock_error > tolerance:
                raise ContractViolation(
                    "selected video decoded frame timeline is discontinuous"
                )
        decoded_duration = Fraction(len(frames), 1) / frame_rate
        if abs(decoded_duration - presentation_duration_seconds) > tolerance:
            raise ContractViolation(
                "selected video decoded frame timeline duration is inconsistent"
            )
        decoded_end = video_origin.seconds + decoded_duration
        return {
            "basis": "best-effort-pts-vs-cumulative-nominal-frames",
            "decodedEndSeconds": self._fraction_json(decoded_end),
            "decodedFrameCount": len(frames),
            "maximumClockErrorSeconds": self._fraction_json(maximum_error),
            "nominalFrameRate": self._fraction_json(frame_rate),
        }

    def validate_input_streams(
        self,
        input_video: Path,
        *,
        video_stream_index: int,
        audio_stream_index: int,
    ) -> InputStreams:
        if video_stream_index < 0 or audio_stream_index < 0:
            raise ContractViolation("explicit stream indexes must be non-negative")
        if video_stream_index == audio_stream_index:
            raise ContractViolation("video and audio stream indexes must differ")
        result = self._run(
            [
                self.ffprobe,
                "-v",
                "error",
                "-show_entries",
                (
                    "format=format_name,duration:"
                    "stream=index,codec_name,codec_type,start_pts,start_time,time_base,"
                    "duration_ts,duration,sample_rate,r_frame_rate"
                ),
                "-of",
                "json",
                str(input_video),
            ],
            "input ffprobe",
        )
        try:
            probe = json.loads(result.stdout)
        except json.JSONDecodeError as error:
            raise ContractViolation("ffprobe returned invalid JSON") from error
        streams = probe.get("streams")
        if not isinstance(streams, list):
            raise ContractViolation("ffprobe did not return a stream list")
        by_index = {
            stream.get("index"): stream
            for stream in streams
            if isinstance(stream, dict) and isinstance(stream.get("index"), int)
        }
        video = by_index.get(video_stream_index)
        audio = by_index.get(audio_stream_index)
        if video is None or video.get("codec_type") != "video":
            raise ContractViolation(
                f"stream {video_stream_index} is not the requested video stream"
            )
        if audio is None or audio.get("codec_type") != "audio":
            raise ContractViolation(
                f"stream {audio_stream_index} is not the requested audio stream"
            )
        declared_video_origin = self._parse_stream_origin(video, "selected video stream")
        declared_audio_origin = self._parse_stream_origin(audio, "selected audio stream")
        audio_sample_rate_text = audio.get("sample_rate")
        video_frame_rate_text = video.get("r_frame_rate")
        if not isinstance(audio_sample_rate_text, str):
            raise ContractViolation("selected audio stream has no exact sample rate")
        if not isinstance(video_frame_rate_text, str):
            raise ContractViolation("selected video stream has no exact nominal frame rate")
        try:
            audio_sample_rate_hz = int(audio_sample_rate_text)
            video_frame_rate = Fraction(video_frame_rate_text)
        except (ValueError, ZeroDivisionError) as error:
            raise ContractViolation("selected stream rate metadata is malformed") from error
        if audio_sample_rate_hz < 1 or video_frame_rate <= 0:
            raise ContractViolation("selected stream rates must be positive")
        audio_packet_timeline = self._probe_packet_timeline(
            input_video,
            stream_origin=declared_audio_origin,
            media_type="audio",
        )
        video_packet_timeline = self._probe_packet_timeline(
            input_video,
            stream_origin=declared_video_origin,
            media_type="video",
        )
        (
            resolved_audio_duration_ts,
            resolved_audio_duration_seconds,
            resolved_audio_duration_source,
        ) = self._resolve_presentation_duration(
            audio,
            label="selected audio stream",
            origin=declared_audio_origin,
            packet_timeline=audio_packet_timeline,
        )
        (
            audio_presentation_duration_ts,
            audio_presentation_duration_seconds,
            audio_presentation_samples,
            audio_presentation_rounding_error_seconds,
            audio_presentation_source,
        ) = self._canonical_audio_presentation_duration(
            duration_ts=resolved_audio_duration_ts,
            duration_seconds=resolved_audio_duration_seconds,
            source=resolved_audio_duration_source,
        )
        (
            video_presentation_duration_ts,
            video_presentation_duration_seconds,
            video_presentation_duration_source,
        ) = self._resolve_presentation_duration(
            video,
            label="selected video stream",
            origin=declared_video_origin,
            packet_timeline=video_packet_timeline,
        )
        video_origin = self._first_decoded_frame_origin(
            input_video,
            stream_index=video_stream_index,
            expected_media_type="video",
            stream_origin=declared_video_origin,
        )
        audio_origin = self._first_decoded_frame_origin(
            input_video,
            stream_index=audio_stream_index,
            expected_media_type="audio",
            stream_origin=declared_audio_origin,
        )
        audio_packet_validation = self._validate_packet_timeline(
            audio_packet_timeline,
            media_type="audio",
            origin=audio_origin,
            presentation_duration_ts=audio_presentation_duration_ts,
            duration_source=audio_presentation_source,
        )
        video_packet_validation = self._validate_packet_timeline(
            video_packet_timeline,
            media_type="video",
            origin=video_origin,
            presentation_duration_ts=video_presentation_duration_ts,
            duration_source=video_presentation_duration_source,
        )
        audio_decoded_validation = self._validate_audio_decoded_timeline(
            input_video,
            audio_origin=audio_origin,
            sample_rate_hz=audio_sample_rate_hz,
            presentation_duration_seconds=audio_presentation_duration_seconds,
        )
        video_decoded_validation = self._validate_video_decoded_timeline(
            input_video,
            video_origin=video_origin,
            frame_rate=video_frame_rate,
            presentation_duration_seconds=video_presentation_duration_seconds,
        )
        exact_offset_samples = (
            audio_origin.seconds - video_origin.seconds
        ) * AUDIO_SAMPLE_RATE_HZ
        rounded_offset_samples = self._round_fraction(exact_offset_samples)
        rounding_error_seconds = abs(
            Fraction(rounded_offset_samples, AUDIO_SAMPLE_RATE_HZ)
            - (audio_origin.seconds - video_origin.seconds)
        )
        if rounding_error_seconds > Fraction(1, 1_000):
            raise ContractViolation(
                "selected stream origins cannot be represented within 1ms at 16kHz"
            )
        return InputStreams(
            video_stream_index=video_stream_index,
            audio_stream_index=audio_stream_index,
            video_origin=video_origin,
            audio_origin=audio_origin,
            audio_presentation_duration_ts=audio_presentation_duration_ts,
            audio_presentation_duration_seconds=audio_presentation_duration_seconds,
            audio_presentation_samples=audio_presentation_samples,
            audio_presentation_rounding_error_seconds=(
                audio_presentation_rounding_error_seconds
            ),
            audio_presentation_source=audio_presentation_source,
            audio_offset_samples_from_video_frame_zero=rounded_offset_samples,
            audio_offset_rounding_error_seconds=rounding_error_seconds,
            audio_timeline_validation={
                "decodedFrames": audio_decoded_validation,
                "packets": audio_packet_validation,
            },
            video_timeline_validation={
                "decodedFrames": video_decoded_validation,
                "packets": video_packet_validation,
            },
            probe=probe,
        )

    def normalize_video(
        self,
        input_video: Path,
        output_video: Path,
        *,
        video_stream_index: int,
    ) -> None:
        self._run(
            [
                self.ffmpeg,
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-xerror",
                "-n",
                "-fflags",
                "+genpts",
                "-i",
                str(input_video),
                "-map",
                f"0:{video_stream_index}",
                "-an",
                "-vf",
                (
                    "setpts=PTS-STARTPTS,"
                    f"fps={VIDEO_FRAMES_PER_SECOND}:round=near,"
                    f"setpts=N/({VIDEO_FRAMES_PER_SECOND}*TB)"
                ),
                "-vsync",
                "cfr",
                "-c:v",
                "ffv1",
                "-level",
                "3",
                str(output_video),
            ],
            "25fps video normalization",
        )

    def detect_faces_and_shots(
        self,
        canonical_video: Path,
        yunet_model: Path,
        *,
        shot_cut_threshold: float,
        face_score_threshold: float,
    ) -> DetectionPass:
        if not math.isfinite(shot_cut_threshold) or shot_cut_threshold <= 0:
            raise ContractViolation("shot cut threshold must be positive and finite")
        if not math.isfinite(face_score_threshold) or not 0 < face_score_threshold <= 1:
            raise ContractViolation("face score threshold must be within (0, 1]")

        cv2 = self.cv2
        numpy = self.numpy
        capture = cv2.VideoCapture(str(canonical_video))
        if not capture.isOpened():
            raise ContractViolation("OpenCV could not open normalized video")

        detector = None
        width = 0
        height = 0
        frame_index = 0
        shot_index = 0
        previous_probe = None
        shot_by_frame: list[int] = []
        detections: list[FaceDetection] = []
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_index >= self.maximum_frames:
                    raise ContractViolation(
                        f"normalized video exceeds maximum {self.maximum_frames} frames"
                    )
                if getattr(frame, "ndim", None) != 3 or frame.shape[2] != 3:
                    raise ContractViolation("normalized frame is not BGR24")
                current_height, current_width = frame.shape[:2]
                if frame_index == 0:
                    width = int(current_width)
                    height = int(current_height)
                    if width < 2 or height < 2:
                        raise ContractViolation("normalized video dimensions are invalid")
                    detector = cv2.FaceDetectorYN.create(
                        str(yunet_model),
                        "",
                        (width, height),
                        face_score_threshold,
                        0.3,
                        5_000,
                    )
                elif current_width != width or current_height != height:
                    raise ContractViolation("normalized video dimensions changed mid-stream")

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                probe = cv2.resize(gray, (64, 36), interpolation=cv2.INTER_AREA)
                if previous_probe is not None:
                    difference = float(numpy.mean(cv2.absdiff(probe, previous_probe)))
                    if not math.isfinite(difference):
                        raise ContractViolation("shot difference score is non-finite")
                    if difference >= shot_cut_threshold:
                        shot_index += 1
                previous_probe = probe
                shot_by_frame.append(shot_index)

                assert detector is not None
                detector.setInputSize((width, height))
                _, faces = detector.detect(frame)
                if faces is not None:
                    ordered_faces = sorted(
                        faces.tolist(),
                        key=lambda face: (
                            float(face[0]),
                            float(face[1]),
                            float(face[2]),
                            float(face[3]),
                            -float(face[-1]),
                        ),
                    )
                    for face in ordered_faces:
                        x, y, face_width, face_height = map(float, face[:4])
                        score = float(face[-1])
                        x1 = max(0.0, x)
                        y1 = max(0.0, y)
                        x2 = min(float(width), x + face_width)
                        y2 = min(float(height), y + face_height)
                        if x2 <= x1 or y2 <= y1:
                            continue
                        detections.append(
                            FaceDetection(
                                frame_index=frame_index,
                                shot_index=shot_index,
                                box=FaceBox(x1, y1, x2, y2),
                                detection_score=score,
                            )
                        )
                frame_index += 1
        finally:
            capture.release()

        if frame_index < 1:
            raise ContractViolation("normalized video contains no decodable frames")
        return DetectionPass(
            width=width,
            height=height,
            frame_count=frame_index,
            shot_by_frame=tuple(shot_by_frame),
            detections=tuple(detections),
        )

    def normalize_audio(
        self,
        input_video: Path,
        output_audio: Path,
        *,
        audio_stream_index: int,
        audio_presentation_samples: int,
        audio_offset_samples_from_video_frame_zero: int,
        frame_count: int,
    ) -> int:
        if audio_presentation_samples < 1:
            raise ContractViolation("audio presentation sample count must be positive")
        expected_samples = frame_count * AUDIO_SAMPLES_PER_VIDEO_FRAME
        origin_adjustment = (
            f"adelay={audio_offset_samples_from_video_frame_zero}S:all=1"
            if audio_offset_samples_from_video_frame_zero >= 0
            else f"atrim=start_sample={-audio_offset_samples_from_video_frame_zero}"
        )
        audio_filter = ",".join(
            (
                f"aresample={AUDIO_SAMPLE_RATE_HZ}:async=0",
                f"aformat=sample_rates={AUDIO_SAMPLE_RATE_HZ}:channel_layouts=mono",
                f"atrim=end_sample={audio_presentation_samples}",
                origin_adjustment,
                "asetpts=PTS-STARTPTS",
                f"apad=whole_len={expected_samples}",
                f"atrim=end_sample={expected_samples}",
                "asetpts=N/SR/TB",
            )
        )
        self._run(
            [
                self.ffmpeg,
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-xerror",
                "-n",
                "-i",
                str(input_video),
                "-map",
                f"0:{audio_stream_index}",
                "-vn",
                "-af",
                audio_filter,
                "-ac",
                str(AUDIO_CHANNELS),
                "-ar",
                str(AUDIO_SAMPLE_RATE_HZ),
                "-c:a",
                "pcm_s16le",
                str(output_audio),
            ],
            "16kHz mono audio normalization",
        )
        with wave.open(str(output_audio), "rb") as handle:
            actual = {
                "channels": handle.getnchannels(),
                "sampleRate": handle.getframerate(),
                "sampleWidth": handle.getsampwidth(),
                "samples": handle.getnframes(),
            }
        expected = {
            "channels": AUDIO_CHANNELS,
            "sampleRate": AUDIO_SAMPLE_RATE_HZ,
            "sampleWidth": 2,
            "samples": expected_samples,
        }
        if actual != expected:
            raise ContractViolation(
                f"normalized audio contract mismatch: expected {expected}, received {actual}"
            )
        return expected_samples

    def read_audio(self, canonical_audio: Path) -> Any:
        sample_rate, samples = self.wavfile.read(str(canonical_audio))
        if sample_rate != AUDIO_SAMPLE_RATE_HZ:
            raise ContractViolation("canonical audio sample rate drifted after validation")
        if getattr(samples, "ndim", None) != 1:
            raise ContractViolation("canonical audio is not mono")
        return samples

    def build_track_geometry(self, tracks: Sequence[FaceTrack]) -> tuple[TrackGeometry, ...]:
        numpy = self.numpy
        result: list[TrackGeometry] = []
        for track in tracks:
            observed_frames = numpy.asarray(
                [item.frame_index for item in track.detections],
                dtype=numpy.int64,
            )
            target_frames = numpy.arange(
                int(observed_frames[0]),
                int(observed_frames[-1]) + 1,
                dtype=numpy.int64,
            )
            source_boxes = numpy.asarray(
                [
                    [item.box.x1, item.box.y1, item.box.x2, item.box.y2]
                    for item in track.detections
                ],
                dtype=numpy.float64,
            )
            interpolated = numpy.stack(
                [
                    numpy.interp(target_frames, observed_frames, source_boxes[:, index])
                    for index in range(4)
                ],
                axis=1,
            )
            center_x = (interpolated[:, 0] + interpolated[:, 2]) / 2.0
            center_y = (interpolated[:, 1] + interpolated[:, 3]) / 2.0
            half_size = numpy.maximum(
                interpolated[:, 2] - interpolated[:, 0],
                interpolated[:, 3] - interpolated[:, 1],
            ) / 2.0
            if len(target_frames) >= 13:
                center_x = self.signal.medfilt(center_x, kernel_size=13)
                center_y = self.signal.medfilt(center_y, kernel_size=13)
                half_size = self.signal.medfilt(half_size, kernel_size=13)
            boxes = tuple(
                FaceBox(*[float(value) for value in row]) for row in interpolated.tolist()
            )
            result.append(
                TrackGeometry(
                    track_id=track.track_id,
                    shot_index=track.shot_index,
                    frame_indexes=tuple(int(value) for value in target_frames.tolist()),
                    face_boxes=boxes,
                    crop_center_x=tuple(float(value) for value in center_x.tolist()),
                    crop_center_y=tuple(float(value) for value in center_y.tolist()),
                    crop_half_size=tuple(float(value) for value in half_size.tolist()),
                    observed_detection_frames=tuple(
                        int(value) for value in observed_frames.tolist()
                    ),
                )
            )
        return tuple(result)

    def extract_face_crops(
        self,
        canonical_video: Path,
        geometry: Sequence[TrackGeometry],
    ) -> dict[str, Any]:
        cv2 = self.cv2
        numpy = self.numpy
        needed: dict[int, list[tuple[TrackGeometry, int]]] = {}
        result_lists: dict[str, list[Any]] = {}
        for track in geometry:
            result_lists[track.track_id] = []
            for local_index, frame_index in enumerate(track.frame_indexes):
                needed.setdefault(frame_index, []).append((track, local_index))

        capture = cv2.VideoCapture(str(canonical_video))
        if not capture.isOpened():
            raise ContractViolation("OpenCV could not reopen normalized video for crops")
        frame_index = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                for track, local_index in needed.get(frame_index, []):
                    result_lists[track.track_id].append(
                        self._official_face_crop(frame, track, local_index)
                    )
                frame_index += 1
        finally:
            capture.release()

        arrays: dict[str, Any] = {}
        for track in geometry:
            crops = result_lists[track.track_id]
            if len(crops) != len(track.frame_indexes):
                raise ContractViolation(
                    f"face crop count mismatch for {track.track_id}: "
                    f"expected {len(track.frame_indexes)}, received {len(crops)}"
                )
            arrays[track.track_id] = numpy.stack(crops, axis=0)
        return arrays

    def _official_face_crop(
        self,
        frame: Any,
        track: TrackGeometry,
        local_index: int,
    ) -> Any:
        cv2 = self.cv2
        numpy = self.numpy
        crop_scale = 0.40
        half_size = track.crop_half_size[local_index]
        center_x = track.crop_center_x[local_index]
        center_y = track.crop_center_y[local_index]
        if not all(math.isfinite(value) for value in (half_size, center_x, center_y)):
            raise ContractViolation("face crop geometry is non-finite")
        if half_size <= 0:
            raise ContractViolation("face crop half-size must be positive")
        border = max(1, int(half_size * (1 + 2 * crop_scale)))
        padded = numpy.pad(
            frame,
            ((border, border), (border, border), (0, 0)),
            "constant",
            constant_values=110,
        )
        x = center_x + border
        y = center_y + border
        y1 = int(y - half_size)
        y2 = int(y + half_size * (1 + 2 * crop_scale))
        x1 = int(x - half_size * (1 + crop_scale))
        x2 = int(x + half_size * (1 + crop_scale))
        face = padded[y1:y2, x1:x2]
        if face.size == 0:
            raise ContractViolation("face crop is empty")
        face_224 = cv2.resize(face, (224, 224), interpolation=cv2.INTER_LINEAR)
        grayscale = cv2.cvtColor(face_224, cv2.COLOR_BGR2GRAY)
        model_crop = grayscale[56:168, 56:168]
        if model_crop.shape != (112, 112):
            raise ContractViolation("LR-ASD model crop is not exactly 112x112")
        return model_crop

    def render_annotated_review(
        self,
        *,
        canonical_video: Path,
        canonical_audio: Path,
        output_video: Path,
        width: int,
        height: int,
        geometry: Sequence[TrackGeometry],
        scores_by_track: Mapping[str, Sequence[float]],
    ) -> None:
        cv2 = self.cv2
        by_frame: dict[int, list[tuple[str, FaceBox, float]]] = {}
        for track in geometry:
            scores = scores_by_track.get(track.track_id)
            if scores is None or len(scores) != len(track.frame_indexes):
                raise ContractViolation(
                    f"annotated review score ledger mismatch for {track.track_id}"
                )
            for frame_index, box, score in zip(
                track.frame_indexes,
                track.face_boxes,
                scores,
                strict=True,
            ):
                by_frame.setdefault(frame_index, []).append(
                    (track.track_id, box, float(score))
                )

        command = [
            self.ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-xerror",
            "-n",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            str(VIDEO_FRAMES_PER_SECOND),
            "-i",
            "pipe:0",
            "-i",
            str(canonical_audio),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-movflags",
            "+faststart",
            "-shortest",
            str(output_video),
        ]
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except OSError as error:
            raise ContractViolation(f"failed to launch annotated review ffmpeg: {error}") from error
        if process.stdin is None or process.stderr is None:
            process.kill()
            raise ContractViolation("annotated review ffmpeg pipes are unavailable")

        capture = cv2.VideoCapture(str(canonical_video))
        if not capture.isOpened():
            process.kill()
            raise ContractViolation("OpenCV could not reopen normalized video for review")
        frame_index = 0
        write_error: Exception | None = None
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                cv2.putText(
                    frame,
                    "DIAGNOSTIC LR-ASD OBSERVATION - NO CROP AUTHORITY",
                    (24, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 220, 255),
                    2,
                    cv2.LINE_AA,
                )
                for track_id, box, score in sorted(by_frame.get(frame_index, [])):
                    start = (int(round(box.x1)), int(round(box.y1)))
                    end = (int(round(box.x2)), int(round(box.y2)))
                    cv2.rectangle(frame, start, end, (255, 220, 0), 3)
                    cv2.putText(
                        frame,
                        f"{track_id} raw={score:+.3f}",
                        (start[0], max(64, start[1] - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 220, 0),
                        2,
                        cv2.LINE_AA,
                    )
                process.stdin.write(frame.tobytes())
                frame_index += 1
        except (BrokenPipeError, OSError) as error:
            write_error = error
        finally:
            capture.release()
            try:
                process.stdin.close()
            except OSError:
                pass
        stderr = process.stderr.read().decode("utf-8", errors="replace")
        return_code = process.wait()
        if write_error is not None or return_code != 0:
            if len(stderr) > 4_096:
                stderr = stderr[-4_096:]
            raise ContractViolation(
                "annotated review render failed: "
                f"writeError={write_error}, exit={return_code}, stderr={stderr.strip()}"
            )

    def validate_annotated_output(
        self,
        output_video: Path,
        *,
        expected_width: int,
        expected_height: int,
        expected_video_frames: int,
        expected_audio_samples: int,
    ) -> OutputMediaValidation:
        """Decode and validate the published review's bounded A/V clock contract."""

        if expected_width < 1 or expected_height < 1:
            raise ContractViolation("annotated review expected dimensions are invalid")
        if expected_video_frames < 1 or expected_audio_samples < 1:
            raise ContractViolation("annotated review expected clocks are empty")
        probe_result = self._run(
            [
                self.ffprobe,
                "-v",
                "error",
                "-count_frames",
                "-show_entries",
                (
                    "stream=index,codec_name,codec_type,width,height,r_frame_rate,"
                    "avg_frame_rate,time_base,start_pts,start_time,duration_ts,duration,"
                    "sample_rate,channels,nb_read_frames"
                ),
                "-of",
                "json",
                str(output_video),
            ],
            "annotated review stream probe",
        )
        try:
            probe = json.loads(probe_result.stdout)
        except json.JSONDecodeError as error:
            raise ContractViolation("annotated review ffprobe returned invalid JSON") from error
        streams = probe.get("streams")
        if not isinstance(streams, list) or len(streams) != 2:
            raise ContractViolation("annotated review must contain exactly one video and one audio stream")
        video_streams = [
            item for item in streams if isinstance(item, dict) and item.get("codec_type") == "video"
        ]
        audio_streams = [
            item for item in streams if isinstance(item, dict) and item.get("codec_type") == "audio"
        ]
        if len(video_streams) != 1 or len(audio_streams) != 1:
            raise ContractViolation("annotated review stream types are not exactly video plus audio")
        video_stream = video_streams[0]
        audio_stream = audio_streams[0]
        if video_stream.get("codec_name") != "h264":
            raise ContractViolation("annotated review video codec must be H.264")
        if audio_stream.get("codec_name") != "aac":
            raise ContractViolation("annotated review audio codec must be AAC")
        if (video_stream.get("width"), video_stream.get("height")) != (
            expected_width,
            expected_height,
        ):
            raise ContractViolation("annotated review dimensions do not match canonical video")
        if (
            video_stream.get("r_frame_rate") != f"{VIDEO_FRAMES_PER_SECOND}/1"
            or video_stream.get("avg_frame_rate") != f"{VIDEO_FRAMES_PER_SECOND}/1"
        ):
            raise ContractViolation("annotated review frame rate is not exactly 25fps")
        try:
            probed_video_frames = int(video_stream.get("nb_read_frames", ""))
        except (TypeError, ValueError) as error:
            raise ContractViolation("annotated review has no decoded video frame count") from error
        if probed_video_frames != expected_video_frames:
            raise ContractViolation(
                "annotated review decoded video frame count mismatch: "
                f"expected {expected_video_frames}, received {probed_video_frames}"
            )
        if audio_stream.get("sample_rate") != str(AUDIO_SAMPLE_RATE_HZ):
            raise ContractViolation("annotated review audio sample rate is not exactly 16kHz")
        if audio_stream.get("channels") != AUDIO_CHANNELS:
            raise ContractViolation("annotated review audio is not exactly mono")

        frame_result = self._run(
            [
                self.ffprobe,
                "-v",
                "error",
                "-show_frames",
                "-show_entries",
                "frame=media_type,best_effort_timestamp_time,nb_samples",
                "-of",
                "json",
                str(output_video),
            ],
            "annotated review decoded clock probe",
        )
        try:
            decoded_frames = json.loads(frame_result.stdout).get("frames")
        except json.JSONDecodeError as error:
            raise ContractViolation("annotated review clock probe returned invalid JSON") from error
        if not isinstance(decoded_frames, list):
            raise ContractViolation("annotated review clock probe returned no frames")
        video_frames = [
            item
            for item in decoded_frames
            if isinstance(item, dict) and item.get("media_type") == "video"
        ]
        audio_frames = [
            item
            for item in decoded_frames
            if isinstance(item, dict) and item.get("media_type") == "audio"
        ]
        if len(video_frames) != expected_video_frames:
            raise ContractViolation("annotated review frame ledger is incomplete")
        if not audio_frames:
            raise ContractViolation("annotated review audio frame ledger is empty")

        tolerance = Fraction(1, 1_000)
        video_first_pts = Fraction(0)
        for frame_index, frame in enumerate(video_frames):
            try:
                actual_pts = Fraction(str(frame["best_effort_timestamp_time"]))
            except (KeyError, ValueError, ZeroDivisionError) as error:
                raise ContractViolation("annotated review video frame has malformed PTS") from error
            expected_pts = Fraction(frame_index, VIDEO_FRAMES_PER_SECOND)
            if abs(actual_pts - expected_pts) > tolerance:
                raise ContractViolation(
                    f"annotated review video clock drifted at frame {frame_index}"
                )
            if frame_index == 0:
                video_first_pts = actual_pts

        decoded_audio_samples = 0
        audio_first_pts = Fraction(0)
        for frame_index, frame in enumerate(audio_frames):
            try:
                actual_pts = Fraction(str(frame["best_effort_timestamp_time"]))
                frame_samples = int(frame["nb_samples"])
            except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
                raise ContractViolation("annotated review audio frame has malformed clock") from error
            if frame_samples < 1:
                raise ContractViolation("annotated review audio frame has no decoded samples")
            expected_pts = Fraction(decoded_audio_samples, AUDIO_SAMPLE_RATE_HZ)
            if abs(actual_pts - expected_pts) > tolerance:
                raise ContractViolation(
                    f"annotated review audio clock drifted at frame {frame_index}"
                )
            if frame_index == 0:
                audio_first_pts = actual_pts
            decoded_audio_samples += frame_samples

        # AAC frames are 1024 samples and may expose one padded tail frame after
        # decoding. The MP4 stream duration remains the authoritative exact end.
        if not (
            expected_audio_samples
            <= decoded_audio_samples
            < expected_audio_samples + 1_024
        ):
            raise ContractViolation(
                "annotated review decoded audio coverage is outside one AAC tail frame"
            )
        try:
            audio_time_base = Fraction(str(audio_stream["time_base"]))
            audio_duration_samples = int(audio_stream["duration_ts"])
            declared_audio_duration = audio_duration_samples * audio_time_base
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
            raise ContractViolation("annotated review audio stream duration is malformed") from error
        expected_audio_duration = Fraction(expected_audio_samples, AUDIO_SAMPLE_RATE_HZ)
        if abs(declared_audio_duration - expected_audio_duration) > tolerance:
            raise ContractViolation("annotated review declared audio end drifted by more than 1ms")

        decoded_audio = self._run_bytes(
            [
                self.ffmpeg,
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-xerror",
                "-i",
                str(output_video),
                "-map",
                "0:a:0",
                "-ac",
                str(AUDIO_CHANNELS),
                "-ar",
                str(AUDIO_SAMPLE_RATE_HZ),
                "-c:a",
                "pcm_s16le",
                "-f",
                "s16le",
                "pipe:1",
            ],
            "annotated review audio decode",
        )
        if len(decoded_audio.stdout) % 2 != 0:
            raise ContractViolation("annotated review decoded PCM byte count is odd")
        if len(decoded_audio.stdout) // 2 != decoded_audio_samples:
            raise ContractViolation("annotated review decoded audio count disagrees with frame ledger")
        self._run(
            [
                self.ffmpeg,
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-xerror",
                "-i",
                str(output_video),
                "-map",
                "0:v:0",
                "-f",
                "null",
                "-",
            ],
            "annotated review video decode",
        )
        return OutputMediaValidation(
            audio_decoded_samples=decoded_audio_samples,
            audio_first_pts_seconds=audio_first_pts,
            video_decoded_frames=len(video_frames),
            video_first_pts_seconds=video_first_pts,
        )


def shot_ranges(shot_by_frame: Sequence[int]) -> list[dict[str, int]]:
    if not shot_by_frame:
        raise ContractViolation("shot ledger cannot be empty")
    result: list[dict[str, int]] = []
    start = 0
    current = shot_by_frame[0]
    for frame_index, shot_index in enumerate(shot_by_frame[1:], start=1):
        if shot_index != current:
            if shot_index != current + 1:
                raise ContractViolation("shot indexes must increase contiguously")
            result.append(
                {
                    "endFrameExclusive": frame_index,
                    "shotIndex": current,
                    "startFrameInclusive": start,
                }
            )
            current = shot_index
            start = frame_index
    result.append(
        {
            "endFrameExclusive": len(shot_by_frame),
            "shotIndex": current,
            "startFrameInclusive": start,
        }
    )
    return result
