import ast
import unittest
from pathlib import Path


HANDLER_PATH = Path(__file__).with_name("rp_handler.py")
SCHEMA_PATH = Path(__file__).with_name("rp_schema.py")


def predict_keywords(function_name):
    module = ast.parse(HANDLER_PATH.read_text(encoding="utf-8"))
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "predict"
    ]
    return [
        {keyword.arg for keyword in call.keywords if keyword.arg is not None}
        for call in calls
    ]


class HandlerParakeetRoutingTest(unittest.TestCase):
    def test_schema_defaults_to_whisper(self):
        module = ast.parse(SCHEMA_PATH.read_text(encoding="utf-8"))
        assignment = next(
            node
            for node in module.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "INPUT_VALIDATIONS"
                for target in node.targets
            )
        )
        backend_value = next(
            value
            for key, value in zip(
                assignment.value.keys,
                assignment.value.values,
            )
            if isinstance(key, ast.Constant)
            and key.value == "asr_backend"
        )
        default_value = next(
            value
            for key, value in zip(
                backend_value.keys,
                backend_value.values,
            )
            if isinstance(key, ast.Constant)
            and key.value == "default"
        )
        self.assertIsInstance(default_value, ast.Constant)
        self.assertEqual(default_value.value, "whisper")

    def test_classic_and_final_forward_explicit_backend(self):
        classic = predict_keywords("run_whisper_job")
        final = predict_keywords("run_final_span_stream_job")

        self.assertEqual(len(classic), 1)
        self.assertEqual(len(final), 1)
        self.assertIn("asr_backend", classic[0])
        self.assertIn("asr_backend", final[0])

    def test_draft_path_remains_whisper_only(self):
        draft = predict_keywords("run_draft_span_stream_job")

        self.assertEqual(len(draft), 1)
        self.assertNotIn("asr_backend", draft[0])


if __name__ == "__main__":
    unittest.main()
