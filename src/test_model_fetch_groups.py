import ast
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).parent.parent
FETCH_MODELS_PATH = REPOSITORY_ROOT / "builder" / "fetch_models.py"
DOCKERFILE_PATH = REPOSITORY_ROOT / "Dockerfile"


class ModelFetchGroupsTest(unittest.TestCase):
    def test_fetch_script_has_explicit_registry_safe_groups(self):
        module = ast.parse(FETCH_MODELS_PATH.read_text(encoding="utf-8"))
        groups = next(
            node
            for node in module.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "MODEL_GROUPS"
                for target in node.targets
            )
        )
        group_text = ast.unparse(groups.value)
        self.assertIn("'all'", group_text)
        self.assertIn("*WHISPER_MODEL_GROUPS", group_text)
        self.assertIn("'clap-alignment'", group_text)
        self.assertIn("'experimental'", group_text)
        self.assertIn("'diarization'", group_text)

        function_names = {
            node.name for node in module.body if isinstance(node, ast.FunctionDef)
        }
        self.assertTrue(
            {
                "download_whisper_models",
                "download_clap_alignment_models",
                "download_experimental_models",
                "download_diarization_models",
            }.issubset(function_names)
        )

        whisper_groups = next(
            node
            for node in module.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "WHISPER_MODEL_GROUPS"
                for target in node.targets
            )
        )
        self.assertEqual(
            ast.literal_eval(whisper_groups.value),
            {
                "whisper-standard": ("base", "small", "medium"),
                "whisper-large-v3": ("large-v3",),
                "whisper-turbo": ("turbo",),
            },
        )

    def test_diarization_group_fails_closed_without_token(self):
        module = ast.parse(FETCH_MODELS_PATH.read_text(encoding="utf-8"))
        function = next(
            node
            for node in module.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "download_diarization_models"
        )
        text = ast.unparse(function)
        self.assertIn("if not token", text)
        self.assertIn("raise RuntimeError", text)

    def test_dockerfile_layers_public_and_gated_models_separately(self):
        dockerfile = DOCKERFILE_PATH.read_text(encoding="utf-8")
        public_groups = (
            "whisper-standard",
            "whisper-large-v3",
            "whisper-turbo",
            "clap-alignment",
            "experimental",
        )
        diarization = "python /fetch_models.py --group diarization"

        for group in public_groups:
            command = f"RUN python /fetch_models.py --group {group}"
            self.assertEqual(dockerfile.count(command), 1)
        self.assertEqual(dockerfile.count(diarization), 1)
        public_positions = [
            dockerfile.index(f"RUN python /fetch_models.py --group {group}")
            for group in public_groups
        ]
        self.assertEqual(public_positions, sorted(public_positions))
        self.assertLess(public_positions[-1], dockerfile.index(diarization))
        self.assertNotIn("python /fetch_models.py &&", dockerfile)

        secret_run = dockerfile.index(
            "RUN --mount=type=secret,id=hf_token,required=false"
        )
        self.assertGreater(dockerfile.index(diarization), secret_run)
        self.assertGreater(secret_run, public_positions[-1])


if __name__ == "__main__":
    unittest.main()
