from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import unittest


MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))

from active_speaker_contracts import ContractViolation  # noqa: E402
from active_speaker_model import validate_checkpoint_state_dict  # noqa: E402


@dataclass(frozen=True)
class FakeTensor:
    shape: tuple[int, ...]


class StrictCheckpointContractTests(unittest.TestCase):
    @staticmethod
    def validate(expected: object, loaded: object) -> None:
        validate_checkpoint_state_dict(
            expected_state=expected,
            loaded_state=loaded,
            is_tensor=lambda value: isinstance(value, FakeTensor),
        )

    def test_exact_keys_and_shapes_pass(self) -> None:
        state = {
            "model.weight": FakeTensor((4, 3)),
            "model.bias": FakeTensor((4,)),
        }
        self.validate(state, dict(reversed(tuple(state.items()))))

    def test_missing_and_unexpected_keys_fail_closed(self) -> None:
        expected = {"expected": FakeTensor((1,))}
        for loaded in (
            {"unexpected": FakeTensor((1,))},
            {
                "expected": FakeTensor((1,)),
                "unexpected": FakeTensor((1,)),
            },
        ):
            with self.subTest(loaded=loaded):
                with self.assertRaisesRegex(ContractViolation, "key closure mismatch"):
                    self.validate(expected, loaded)

    def test_shape_mismatch_fails_before_framework_load(self) -> None:
        with self.assertRaisesRegex(ContractViolation, "tensor shape mismatch"):
            self.validate(
                {"weight": FakeTensor((2, 3))},
                {"weight": FakeTensor((3, 2))},
            )

    def test_non_dictionary_empty_non_string_and_non_tensor_fail(self) -> None:
        fixtures = (
            (None, "direct state dictionary"),
            ({}, "empty"),
            ({1: FakeTensor((1,))}, "keys must all be strings"),
            ({"weight": object()}, "values must all be tensors"),
        )
        for loaded, message in fixtures:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ContractViolation, message):
                    self.validate({"weight": FakeTensor((1,))}, loaded)


if __name__ == "__main__":
    unittest.main()
