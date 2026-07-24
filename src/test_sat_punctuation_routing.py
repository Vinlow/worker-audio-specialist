import ast
import unittest
from pathlib import Path


HANDLER_PATH = Path(__file__).with_name("rp_handler.py")
PREDICT_PATH = Path(__file__).with_name("predict.py")


def function_from_file(path, function_name):
    module = ast.parse(path.read_text(encoding="utf-8"))
    return next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == function_name
    )


class SaTPunctuationRoutingTest(unittest.TestCase):
    def test_probe_is_explicit_and_returns_before_audio_requirement(self):
        method = function_from_file(HANDLER_PATH, "run_whisper_job")
        method_text = ast.unparse(method)
        probe_call = method_text.index("MODEL.predict_punctuation_window")
        audio_requirement = method_text.index(
            "Must provide either audio or audio_base64"
        )
        self.assertLess(probe_call, audio_requirement)
        self.assertIn(
            "job_input.pop('sat_punctuation_probe', None)",
            method_text,
        )

    def test_probe_is_not_part_of_default_whisper_predict_path(self):
        module = ast.parse(PREDICT_PATH.read_text(encoding="utf-8"))
        predictor = next(
            node
            for node in module.body
            if isinstance(node, ast.ClassDef)
            and node.name == "Predictor"
        )
        predict_method = next(
            node
            for node in predictor.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "predict"
        )
        self.assertNotIn(
            "sat_punctuator",
            ast.unparse(predict_method),
        )


if __name__ == "__main__":
    unittest.main()
