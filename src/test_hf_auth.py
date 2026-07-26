import unittest
from unittest.mock import Mock, patch

from hf_auth import install_legacy_use_auth_token_compat


class HuggingFaceAuthCompatTest(unittest.TestCase):
    def test_translates_legacy_use_auth_token_to_token(self):
        download = Mock(return_value="/cache/model.bin")

        with patch("huggingface_hub.hf_hub_download", download):
            compatible = install_legacy_use_auth_token_compat()
            result = compatible(
                "pyannote/segmentation-3.0",
                "pytorch_model.bin",
                use_auth_token="secret",
                revision="main",
            )

        self.assertEqual(result, "/cache/model.bin")
        download.assert_called_once_with(
            "pyannote/segmentation-3.0",
            "pytorch_model.bin",
            token="secret",
            revision="main",
        )

    def test_rejects_conflicting_legacy_and_current_tokens(self):
        download = Mock()

        with patch("huggingface_hub.hf_hub_download", download):
            compatible = install_legacy_use_auth_token_compat()
            with self.assertRaisesRegex(
                ValueError,
                "CONFLICTING_HUGGINGFACE_AUTH_TOKENS",
            ):
                compatible(
                    "pyannote/segmentation-3.0",
                    "pytorch_model.bin",
                    use_auth_token="legacy",
                    token="current",
                )

        download.assert_not_called()

    def test_install_is_idempotent(self):
        def download(*_args, **_kwargs):
            return "ok"

        with patch("huggingface_hub.hf_hub_download", download):
            first = install_legacy_use_auth_token_compat()
            second = install_legacy_use_auth_token_compat()

        self.assertIs(first, second)


if __name__ == "__main__":
    unittest.main()
