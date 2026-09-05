"""Frozen, offline parity checks against official LR-ASD preprocessing.

The independent reference below transcribes the preprocessing in LR-ASD commit
1b6dcd2d8fc2895683de6508ec6294ec47d388ca without importing that checkout:

- Columbia_test.py SHA-256
  65e4cdfb762b85fdf5a985754c30f64497d934e71f059ca1b33f754244d2db62
  (`crop_video` and `evaluate_network`)
- dataLoader.py SHA-256
  e9213416a08dc294ca2b5e41b05aa951fb425fc811dd1ac81c652959e4276609
  (`load_audio` and `load_visual`)

Golden hashes pin the synthetic inputs and reference outputs. No source checkout,
checkpoint, media file, or network access is used by these tests.
"""

from __future__ import annotations

from contextlib import nullcontext
import hashlib
import math
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest


MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))

try:
    import cv2
    import numpy
    import python_speech_features
    from scipy import interpolate, signal

    PARITY_DEPENDENCIES_AVAILABLE = True
except ImportError:
    PARITY_DEPENDENCIES_AVAILABLE = False

from active_speaker_contracts import (  # noqa: E402
    FaceBox,
    FaceDetection,
    FaceTrack,
)
from active_speaker_media import MediaProcessor, TrackGeometry  # noqa: E402
from active_speaker_model import LRASD_CONTEXT_SECONDS, LrasdModelRunner  # noqa: E402


OFFICIAL_COMMIT = "1b6dcd2d8fc2895683de6508ec6294ec47d388ca"
SYNTHETIC_AUDIO_SHA256 = "bed777b98ed78bef708dc2373e289614afe27e346a2762cb0c0285574f0d44c3"
SYNTHETIC_VISUAL_SHA256 = "d1b6e860a3c8ac1a523afc9aa3aa7f5b5507ff6f37c9d379a2f0d2c8bcb93c7c"
SYNTHETIC_FRAME_SHA256 = "605e54438b9515979d8c40f926f83d77a3e30a7101df626232febe336d83b1d9"
REFERENCE_MFCC_ROUNDED_9_SHA256 = (
    "36682990038d886a227cdfe619c59b7b07fa201cdaea61d1fafdf55f26a14b96"
)
REFERENCE_ADMITTED_VISUAL_SHA256 = (
    "10e00bce700f61563463dbe41f4e40830e13eb95fd4b5fa274c2f3046f5701c2"
)
REFERENCE_FACE_CROP_SHA256 = (
    "3eb9a3b355686eda84f53f5d4b437efd7fb8c4b5293da777c74bc6cf30f0542c"
)
MFCC_ABSOLUTE_TOLERANCE = 1e-12
GEOMETRY_ABSOLUTE_TOLERANCE = 1e-12


def _sha256_array(value: object, dtype: str | None = None) -> str:
    array = numpy.ascontiguousarray(value, dtype=dtype)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _synthetic_audio() -> object:
    indexes = numpy.arange(16_800, dtype=numpy.int64)
    return (((indexes * 97 + 13) % 30_001) - 15_000).astype(numpy.int16)


def _synthetic_visual() -> object:
    pixels = numpy.arange(112 * 112, dtype=numpy.uint32).reshape(112, 112)
    return numpy.stack(
        [((pixels * 3 + frame * 17) % 256).astype(numpy.uint8) for frame in range(27)]
    )


def _synthetic_bgr_frame() -> object:
    rows = numpy.arange(180, dtype=numpy.uint32)[:, None, None]
    columns = numpy.arange(320, dtype=numpy.uint32)[None, :, None]
    channels = numpy.arange(3, dtype=numpy.uint32)[None, None, :]
    return ((rows * 3 + columns * 5 + channels * 37) % 256).astype(numpy.uint8)


def _official_reference_features(audio: object, visual: object) -> tuple[object, object]:
    """Literal `evaluate_network` feature and admission equations."""

    audio_feature = python_speech_features.mfcc(
        audio,
        16_000,
        numcep=13,
        winlen=0.025,
        winstep=0.010,
    )
    length = min(
        (audio_feature.shape[0] - audio_feature.shape[0] % 4) / 100,
        visual.shape[0],
    )
    return (
        audio_feature[: int(round(length * 100)), :],
        visual[: int(round(length * 25)), :, :],
    )


def _official_reference_crop(
    frame: object,
    *,
    center_x: float,
    center_y: float,
    half_size: float,
) -> object:
    """Literal `crop_video` followed by `evaluate_network` visual conversion."""

    crop_scale = 0.40
    border = int(half_size * (1 + 2 * crop_scale))
    padded = numpy.pad(
        frame,
        ((border, border), (border, border), (0, 0)),
        "constant",
        constant_values=(110, 110),
    )
    middle_y = center_y + border
    middle_x = center_x + border
    face = padded[
        int(middle_y - half_size) : int(middle_y + half_size * (1 + 2 * crop_scale)),
        int(middle_x - half_size * (1 + crop_scale)) : int(
            middle_x + half_size * (1 + crop_scale)
        ),
    ]
    face = cv2.resize(face, (224, 224))
    grayscale = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    grayscale = cv2.resize(grayscale, (224, 224))
    return grayscale[56:168, 56:168]


def _official_reference_geometry(track: FaceTrack) -> tuple[object, object, object, object]:
    """Literal interpolation and 13-frame median smoothing from `crop_video`."""

    observed_frames = numpy.asarray(
        [detection.frame_index for detection in track.detections],
        dtype=numpy.int64,
    )
    target_frames = numpy.arange(observed_frames[0], observed_frames[-1] + 1)
    boxes = numpy.asarray(
        [
            [detection.box.x1, detection.box.y1, detection.box.x2, detection.box.y2]
            for detection in track.detections
        ],
        dtype=numpy.float64,
    )
    interpolated_boxes = numpy.stack(
        [
            interpolate.interp1d(observed_frames, boxes[:, coordinate])(target_frames)
            for coordinate in range(4)
        ],
        axis=1,
    )
    half_size = numpy.maximum(
        interpolated_boxes[:, 3] - interpolated_boxes[:, 1],
        interpolated_boxes[:, 2] - interpolated_boxes[:, 0],
    ) / 2
    center_y = (interpolated_boxes[:, 1] + interpolated_boxes[:, 3]) / 2
    center_x = (interpolated_boxes[:, 0] + interpolated_boxes[:, 2]) / 2
    return (
        interpolated_boxes,
        signal.medfilt(center_x, kernel_size=13),
        signal.medfilt(center_y, kernel_size=13),
        signal.medfilt(half_size, kernel_size=13),
    )


class _CapturedTensor:
    def __init__(self, value: object) -> None:
        self.value = numpy.asarray(value, dtype=numpy.float32)
        self.shape = self.value.shape

    def unsqueeze(self, axis: int) -> "_CapturedTensor":
        return _CapturedTensor(numpy.expand_dims(self.value, axis=axis))


class _Embedding:
    def __init__(self, frame_count: int) -> None:
        self.shape = (1, frame_count, 1)


class _Fused:
    def __init__(self, frame_count: int) -> None:
        self.frame_count = frame_count

    def squeeze(self, axis: int) -> "_Fused":
        if axis != 1:
            raise AssertionError(f"unexpected squeeze axis {axis}")
        return self


class _ScoreVector:
    def __init__(self, frame_count: int) -> None:
        self.values = [float(index) for index in range(frame_count)]

    def detach(self) -> "_ScoreVector":
        return self

    def to(self, device: str, *, dtype: object) -> "_ScoreVector":
        if device != "cpu" or dtype != "float64":
            raise AssertionError("unexpected score conversion")
        return self

    def tolist(self) -> list[float]:
        return self.values


class _Logits:
    def __init__(self, frame_count: int) -> None:
        self.frame_count = frame_count

    def __getitem__(self, key: object) -> _ScoreVector:
        if key != (slice(None), 1):
            raise AssertionError(f"unexpected logit selection {key}")
        return _ScoreVector(self.frame_count)


class _CapturingModel:
    def __init__(self) -> None:
        self.windows: list[tuple[str, object]] = []

    def forward_audio_frontend(self, value: _CapturedTensor) -> _Embedding:
        self.windows.append(("audio", value.value.copy()))
        return _Embedding(value.shape[1] // 4)

    def forward_visual_frontend(self, value: _CapturedTensor) -> _Embedding:
        self.windows.append(("visual", value.value.copy()))
        return _Embedding(value.shape[1])

    def forward_audio_visual_backend(
        self,
        audio: _Embedding,
        visual: _Embedding,
    ) -> _Fused:
        if audio.shape[1] != visual.shape[1]:
            raise AssertionError("captured fake embeddings drifted")
        return _Fused(audio.shape[1])


class _FakeTorch:
    float32 = "float32"
    float64 = "float64"

    @staticmethod
    def inference_mode() -> object:
        return nullcontext()

    @staticmethod
    def as_tensor(value: object, *, dtype: object, device: object) -> _CapturedTensor:
        if dtype != "float32" or device != "cpu":
            raise AssertionError("unexpected model input conversion")
        return _CapturedTensor(value)


class _FakeLoss:
    @staticmethod
    def FC(value: _Fused) -> _Logits:
        return _Logits(value.frame_count)


@unittest.skipUnless(
    PARITY_DEPENDENCIES_AVAILABLE,
    "pinned NumPy, SciPy, OpenCV, and python-speech-features are required",
)
class OfficialPreprocessingParityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.processor = MediaProcessor(
            ffmpeg="ffmpeg",
            ffprobe="ffprobe",
            maximum_frames=100,
        )

    def test_mfcc_admission_matches_frozen_official_reference(self) -> None:
        audio = _synthetic_audio()
        visual = _synthetic_visual()
        self.assertEqual(_sha256_array(audio, "<i2"), SYNTHETIC_AUDIO_SHA256)
        self.assertEqual(_sha256_array(visual, "u1"), SYNTHETIC_VISUAL_SHA256)

        reference_audio, reference_visual = _official_reference_features(audio, visual)
        self.assertEqual(reference_audio.shape, (104, 13))
        self.assertEqual(reference_visual.shape, (26, 112, 112))
        self.assertEqual(
            _sha256_array(numpy.round(reference_audio, 9), "<f8"),
            REFERENCE_MFCC_ROUNDED_9_SHA256,
        )
        self.assertEqual(
            _sha256_array(reference_visual, "u1"),
            REFERENCE_ADMITTED_VISUAL_SHA256,
        )

        runner = LrasdModelRunner.__new__(LrasdModelRunner)
        adapter_audio, adapter_visual = runner.prepare_features(
            audio_samples=audio,
            visual_feature=visual,
            sample_rate_hz=16_000,
        )
        numpy.testing.assert_allclose(
            adapter_audio,
            reference_audio,
            rtol=0,
            atol=MFCC_ABSOLUTE_TOLERANCE,
        )
        numpy.testing.assert_array_equal(adapter_visual, reference_visual)

    def test_geometry_and_face_crop_match_frozen_official_reference(self) -> None:
        track = FaceTrack(
            track_id="reference-track",
            shot_index=0,
            detections=tuple(
                FaceDetection(
                    frame_index=frame_index,
                    shot_index=0,
                    box=FaceBox(
                        10 + frame_index,
                        20 + frame_index / 2,
                        80 + frame_index * 1.5,
                        100 + frame_index,
                    ),
                    detection_score=0.99,
                )
                for frame_index in (0, 7, 14)
            ),
        )
        geometry = self.processor.build_track_geometry((track,))[0]
        reference_boxes, reference_x, reference_y, reference_size = (
            _official_reference_geometry(track)
        )
        adapter_boxes = numpy.asarray(
            [
                [box.x1, box.y1, box.x2, box.y2]
                for box in geometry.face_boxes
            ]
        )
        numpy.testing.assert_allclose(
            adapter_boxes,
            reference_boxes,
            rtol=0,
            atol=GEOMETRY_ABSOLUTE_TOLERANCE,
        )
        numpy.testing.assert_allclose(
            geometry.crop_center_x,
            reference_x,
            rtol=0,
            atol=GEOMETRY_ABSOLUTE_TOLERANCE,
        )
        numpy.testing.assert_allclose(
            geometry.crop_center_y,
            reference_y,
            rtol=0,
            atol=GEOMETRY_ABSOLUTE_TOLERANCE,
        )
        numpy.testing.assert_allclose(
            geometry.crop_half_size,
            reference_size,
            rtol=0,
            atol=GEOMETRY_ABSOLUTE_TOLERANCE,
        )

        frame = _synthetic_bgr_frame()
        self.assertEqual(_sha256_array(frame, "u1"), SYNTHETIC_FRAME_SHA256)
        reference_crop = _official_reference_crop(
            frame,
            center_x=12.5,
            center_y=20.25,
            half_size=37.25,
        )
        self.assertEqual(
            _sha256_array(reference_crop, "u1"),
            REFERENCE_FACE_CROP_SHA256,
        )
        manual_geometry = TrackGeometry(
            track_id="manual-reference",
            shot_index=0,
            frame_indexes=(0,),
            face_boxes=(FaceBox(0, 0, 1, 1),),
            crop_center_x=(12.5,),
            crop_center_y=(20.25,),
            crop_half_size=(37.25,),
            observed_detection_frames=(0,),
        )
        adapter_crop = self.processor._official_face_crop(frame, manual_geometry, 0)
        numpy.testing.assert_array_equal(adapter_crop, reference_crop)

    def test_context_windows_match_official_duration_slicing(self) -> None:
        frame_count = 31
        audio = numpy.arange(frame_count * 4 * 13, dtype=numpy.float64).reshape(-1, 13)
        visual = numpy.arange(
            frame_count * 112 * 112,
            dtype=numpy.uint32,
        ).reshape(frame_count, 112, 112)
        visual = (visual % 256).astype(numpy.uint8)

        capturing_model = _CapturingModel()
        runner = LrasdModelRunner.__new__(LrasdModelRunner)
        runner.numpy = numpy
        runner.torch = _FakeTorch()
        runner.device = "cpu"
        runner.closure = SimpleNamespace(model=capturing_model, lossAV=_FakeLoss())
        scores = runner.score_track(audio_feature=audio, visual_feature=visual)
        self.assertEqual(len(scores), frame_count)

        reference_windows: list[tuple[str, object]] = []
        official_length_seconds = frame_count / 25
        for duration in LRASD_CONTEXT_SECONDS:
            batch_size = int(math.ceil(official_length_seconds / duration))
            for batch_index in range(batch_size):
                reference_windows.append(
                    (
                        "audio",
                        audio[
                            batch_index * duration * 100 : (batch_index + 1)
                            * duration
                            * 100,
                            :,
                        ][None, ...].astype(numpy.float32),
                    )
                )
                reference_windows.append(
                    (
                        "visual",
                        visual[
                            batch_index * duration * 25 : (batch_index + 1)
                            * duration
                            * 25,
                            :,
                            :,
                        ][None, ...].astype(numpy.float32),
                    )
                )
        self.assertEqual(len(capturing_model.windows), len(reference_windows))
        for adapter_window, reference_window in zip(
            capturing_model.windows,
            reference_windows,
            strict=True,
        ):
            self.assertEqual(adapter_window[0], reference_window[0])
            numpy.testing.assert_array_equal(adapter_window[1], reference_window[1])


if __name__ == "__main__":
    unittest.main()
