"""Detector-free media helpers for supplied-track active-speaker v2."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import statistics
from typing import Any, Sequence

from active_speaker_contracts import ContractViolation, FaceBox
from active_speaker_media import MediaProcessor, TrackGeometry
from active_speaker_supplied_tracks import SuppliedTrackManifest


def edge_padded_median_13(values: Sequence[float]) -> tuple[float, ...]:
    """Return an odd-window median whose edges preserve affine mirror symmetry."""

    source = tuple(float(value) for value in values)
    if not source:
        raise ContractViolation("v2 crop smoothing input must not be empty")
    radius = 6
    padded = (source[0],) * radius + source + (source[-1],) * radius
    return tuple(
        float(statistics.median(padded[index : index + 2 * radius + 1]))
        for index in range(len(source))
    )


@dataclass(frozen=True)
class CanonicalVideoValidation:
    width: int
    height: int
    frame_count: int

    def as_json(self) -> dict[str, int]:
        return {
            "decodedFrameCount": self.frame_count,
            "height": self.height,
            "width": self.width,
        }


class SuppliedTrackMediaProcessor(MediaProcessor):
    """Consumes authenticated dense geometry and has no detection capability."""

    def detect_faces_and_shots(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise ContractViolation("face detection and tracking are forbidden in v2")

    def inspect_canonical_video(
        self,
        canonical_video: Path,
        *,
        expected_width: int,
        expected_height: int,
        expected_frame_count: int,
    ) -> CanonicalVideoValidation:
        cv2 = self.cv2
        capture = cv2.VideoCapture(str(canonical_video))
        if not capture.isOpened():
            raise ContractViolation("OpenCV could not open normalized v2 video")
        width = 0
        height = 0
        frame_count = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_count >= self.maximum_frames:
                    raise ContractViolation(
                        f"normalized v2 video exceeds maximum {self.maximum_frames} frames"
                    )
                if getattr(frame, "ndim", None) != 3 or frame.shape[2] != 3:
                    raise ContractViolation("normalized v2 frame is not BGR24")
                current_height, current_width = frame.shape[:2]
                if frame_count == 0:
                    width = int(current_width)
                    height = int(current_height)
                elif current_width != width or current_height != height:
                    raise ContractViolation("normalized v2 video dimensions changed mid-stream")
                frame_count += 1
        finally:
            capture.release()
        received = (width, height, frame_count)
        expected = (expected_width, expected_height, expected_frame_count)
        if frame_count < 1 or received != expected:
            raise ContractViolation(
                "normalized v2 video disagrees with supplied-track clock: "
                f"expected {expected}, received {received}"
            )
        return CanonicalVideoValidation(width, height, frame_count)

    def geometry_from_manifest(
        self,
        manifest: SuppliedTrackManifest,
    ) -> tuple[TrackGeometry, ...]:
        numpy = self.numpy
        result: list[TrackGeometry] = []
        for supplied_track in manifest.tracks:
            boxes_array = numpy.asarray(
                [
                    [
                        frame.face_box.x1,
                        frame.face_box.y1,
                        frame.face_box.x2,
                        frame.face_box.y2,
                    ]
                    for frame in supplied_track.frames
                ],
                dtype=numpy.float64,
            )
            center_x = (boxes_array[:, 0] + boxes_array[:, 2]) / 2.0
            center_y = (boxes_array[:, 1] + boxes_array[:, 3]) / 2.0
            half_size = numpy.maximum(
                boxes_array[:, 2] - boxes_array[:, 0],
                boxes_array[:, 3] - boxes_array[:, 1],
            ) / 2.0
            center_x = numpy.asarray(
                edge_padded_median_13(center_x.tolist()), dtype=numpy.float64
            )
            center_y = numpy.asarray(
                edge_padded_median_13(center_y.tolist()), dtype=numpy.float64
            )
            half_size = numpy.asarray(
                edge_padded_median_13(half_size.tolist()), dtype=numpy.float64
            )
            result.append(
                TrackGeometry(
                    track_id=supplied_track.track_id,
                    shot_index=supplied_track.shot_index,
                    frame_indexes=tuple(
                        frame.frame_index for frame in supplied_track.frames
                    ),
                    face_boxes=tuple(
                        FaceBox(
                            frame.face_box.x1,
                            frame.face_box.y1,
                            frame.face_box.x2,
                            frame.face_box.y2,
                        )
                        for frame in supplied_track.frames
                    ),
                    crop_center_x=tuple(float(value) for value in center_x.tolist()),
                    crop_center_y=tuple(float(value) for value in center_y.tolist()),
                    crop_half_size=tuple(float(value) for value in half_size.tolist()),
                    observed_detection_frames=tuple(
                        frame.frame_index
                        for frame in supplied_track.frames
                        if frame.is_detector_observation
                    ),
                )
            )
        return tuple(result)

    def _official_face_crop(
        self,
        frame: Any,
        track: TrackGeometry,
        local_index: int,
    ) -> Any:
        """Apply the frozen LR-ASD crop with mirror-equivariant x bounds."""

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
        x1 = math.floor(x - half_size * (1 + crop_scale))
        x2 = math.ceil(x + half_size * (1 + crop_scale))
        face = padded[y1:y2, x1:x2]
        if face.size == 0:
            raise ContractViolation("face crop is empty")
        face_224 = cv2.resize(face, (224, 224), interpolation=cv2.INTER_LINEAR)
        grayscale = cv2.cvtColor(face_224, cv2.COLOR_BGR2GRAY)
        model_crop = grayscale[56:168, 56:168]
        if model_crop.shape != (112, 112):
            raise ContractViolation("LR-ASD model crop is not exactly 112x112")
        return model_crop
