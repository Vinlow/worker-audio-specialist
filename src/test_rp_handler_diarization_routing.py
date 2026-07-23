import ast
import unittest
from pathlib import Path


HANDLER_PATH = Path(__file__).with_name("rp_handler.py")


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


class FinalSpanDiarizationRoutingTests(unittest.TestCase):
    def test_final_span_stream_forwards_opt_in_diarization(self):
        keyword_sets = predict_keywords("run_final_span_stream_job")
        self.assertEqual(len(keyword_sets), 1)
        self.assertTrue(
            {
                "diarize",
                "diarize_min_speakers",
                "diarize_max_speakers",
            }.issubset(keyword_sets[0])
        )

    def test_draft_stream_keeps_diarization_off_its_latency_path(self):
        keyword_sets = predict_keywords("run_draft_span_stream_job")
        self.assertEqual(len(keyword_sets), 1)
        self.assertNotIn("diarize", keyword_sets[0])


if __name__ == "__main__":
    unittest.main()
