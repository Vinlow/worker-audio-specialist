import ast
import unittest
from pathlib import Path


PREDICT_PATH = Path(__file__).with_name("predict.py")


def predictor_method(method_name):
    module = ast.parse(PREDICT_PATH.read_text(encoding="utf-8"))
    predictor = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "Predictor"
    )
    return next(
        node
        for node in predictor.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )


class PredictorParakeetRoutingTest(unittest.TestCase):
    def test_predict_declares_backward_compatible_whisper_default(self):
        method = predictor_method("predict")
        positional_names = [argument.arg for argument in method.args.args]
        backend_index = positional_names.index("asr_backend")
        first_default_index = len(positional_names) - len(method.args.defaults)
        backend_default = method.args.defaults[
            backend_index - first_default_index
        ]

        self.assertIsInstance(backend_default, ast.Constant)
        self.assertEqual(backend_default.value, "whisper")

    def test_explicit_parakeet_branch_calls_sidecar_before_whisper_load(self):
        method = predictor_method("predict")
        statements = method.body
        parakeet_branch_index = next(
            index
            for index, statement in enumerate(statements)
            if isinstance(statement, ast.If)
            and "asr_backend" in ast.unparse(statement.test)
            and "parakeet" in ast.unparse(statement.test)
        )
        whisper_model_validation_index = next(
            index
            for index, statement in enumerate(statements)
            if isinstance(statement, ast.If)
            and "model_name not in AVAILABLE_MODELS"
            in ast.unparse(statement.test)
        )
        parakeet_branch = statements[parakeet_branch_index]
        sidecar_calls = [
            node
            for node in ast.walk(parakeet_branch)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "transcribe"
            and "parakeet_transcriber" in ast.unparse(node.func.value)
        ]

        self.assertLess(
            parakeet_branch_index,
            whisper_model_validation_index,
        )
        self.assertEqual(len(sidecar_calls), 1)

    def test_parakeet_branch_guards_unsupported_enrichments(self):
        method_text = ast.unparse(predictor_method("predict"))

        for feature in (
            "translate",
            "clap_queries",
            "force_align",
            "diarize",
            "enable_vad",
        ):
            self.assertIn(
                f"incompatible_features.append('{feature}')",
                method_text,
            )


if __name__ == "__main__":
    unittest.main()
