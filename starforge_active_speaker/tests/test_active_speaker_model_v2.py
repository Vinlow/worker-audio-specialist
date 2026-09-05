from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest


MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))

from active_speaker_contracts import ContractViolation  # noqa: E402
from active_speaker_model_v2 import (  # noqa: E402
    MirrorInvariantLrasdModelRunner,
    average_two_view_scores,
)


class FakeFlags:
    def __init__(self, contiguous: bool) -> None:
        self.contiguous = contiguous

    def __getitem__(self, key: str) -> bool:
        if key != "C_CONTIGUOUS":
            raise KeyError(key)
        return self.contiguous


class FakeVisual:
    ndim = 3
    shape = (2, 112, 112)

    def __init__(self, orientation: str, *, contiguous: bool = True) -> None:
        self.orientation = orientation
        self.flags = FakeFlags(contiguous)

    def __getitem__(self, key: object) -> "FakeVisual":
        if key != (slice(None), slice(None), slice(None, None, -1)):
            raise AssertionError(f"unexpected fake tensor slice: {key}")
        flipped = "mirror" if self.orientation == "canonical" else "canonical"
        return FakeVisual(flipped, contiguous=False)


class FakeNumpy:
    @staticmethod
    def ascontiguousarray(value: FakeVisual) -> FakeVisual:
        return FakeVisual(value.orientation, contiguous=True)


def fake_runner(score_action):
    runner = object.__new__(MirrorInvariantLrasdModelRunner)
    runner.numpy = FakeNumpy()
    runner.score_track = score_action
    return runner


class MirrorInvariantModelTests(unittest.TestCase):
    def test_exact_mean_rejects_bad_shape_and_nonfinite_components(self) -> None:
        self.assertEqual(average_two_view_scores([1.0, -1.0], [3.0, 1.0]), (2.0, 0.0))
        for canonical, mirrored in (([], []), ([1.0], [1.0, 2.0]), ([math.nan], [1.0])):
            with self.subTest(canonical=canonical, mirrored=mirrored):
                with self.assertRaises(ContractViolation):
                    average_two_view_scores(canonical, mirrored)

    def test_views_are_sequential_use_identical_audio_and_mirror_is_contiguous(self) -> None:
        audio = object()
        calls: list[tuple[object, str, bool]] = []

        def score_track(*, audio_feature, visual_feature):
            calls.append(
                (
                    audio_feature,
                    visual_feature.orientation,
                    visual_feature.flags["C_CONTIGUOUS"],
                )
            )
            return [1.0, 3.0] if visual_feature.orientation == "canonical" else [-1.0, 5.0]

        runner = fake_runner(score_track)
        source = FakeVisual("canonical")
        result = MirrorInvariantLrasdModelRunner.score_track_two_view(
            runner,
            audio_feature=audio,
            visual_feature=source,
        )
        self.assertEqual(
            calls,
            [(audio, "canonical", True), (audio, "mirror", True)],
        )
        self.assertEqual(result.canonical, (1.0, 3.0))
        self.assertEqual(result.horizontal_mirror, (-1.0, 5.0))
        self.assertEqual(result.mean, (0.0, 4.0))
        self.assertEqual(source.orientation, "canonical")

    def test_exact_flip_swaps_components_and_preserves_aggregate(self) -> None:
        def score_track(*, audio_feature, visual_feature):
            del audio_feature
            return [1.0, 3.0] if visual_feature.orientation == "canonical" else [-1.0, 5.0]

        runner = fake_runner(score_track)
        canonical = FakeVisual("canonical")
        mirrored = FakeNumpy.ascontiguousarray(canonical[:, :, ::-1])
        first = MirrorInvariantLrasdModelRunner.score_track_two_view(
            runner, audio_feature=object(), visual_feature=canonical
        )
        second = MirrorInvariantLrasdModelRunner.score_track_two_view(
            runner, audio_feature=object(), visual_feature=mirrored
        )
        self.assertEqual(first.canonical, second.horizontal_mirror)
        self.assertEqual(first.horizontal_mirror, second.canonical)
        self.assertEqual(first.mean, second.mean)
        double_mirror = FakeNumpy.ascontiguousarray(mirrored[:, :, ::-1])
        self.assertEqual(double_mirror.orientation, canonical.orientation)

    def test_either_view_failure_or_bad_count_poison_the_result(self) -> None:
        calls = 0

        def failing_score(*, audio_feature, visual_feature):
            nonlocal calls
            del audio_feature
            calls += 1
            if visual_feature.orientation == "mirror":
                raise ContractViolation("mirror failure")
            return [1.0, 2.0]

        runner = fake_runner(failing_score)
        with self.assertRaisesRegex(ContractViolation, "mirror failure"):
            MirrorInvariantLrasdModelRunner.score_track_two_view(
                runner,
                audio_feature=object(),
                visual_feature=FakeVisual("canonical"),
            )
        self.assertEqual(calls, 2)

        def bad_count(*, audio_feature, visual_feature):
            del audio_feature
            return [1.0, 2.0] if visual_feature.orientation == "canonical" else [1.0]

        with self.assertRaisesRegex(ContractViolation, "same non-zero length"):
            MirrorInvariantLrasdModelRunner.score_track_two_view(
                fake_runner(bad_count),
                audio_feature=object(),
                visual_feature=FakeVisual("canonical"),
            )

    def test_invalid_visual_contract_fails_before_any_view_runs(self) -> None:
        calls = 0

        def score(*, audio_feature, visual_feature):
            nonlocal calls
            del audio_feature, visual_feature
            calls += 1
            return [1.0]

        runner = fake_runner(score)
        malformed = FakeVisual("canonical")
        malformed.shape = (2, 111, 112)
        with self.assertRaisesRegex(ContractViolation, "112x112"):
            MirrorInvariantLrasdModelRunner.score_track_two_view(
                runner,
                audio_feature=object(),
                visual_feature=malformed,
            )
        self.assertEqual(calls, 0)


if __name__ == "__main__":
    unittest.main()
