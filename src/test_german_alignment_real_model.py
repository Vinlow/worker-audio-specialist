"""Offline image/model compatibility, separate from real German speech quality."""
import os
import unittest


@unittest.skipUnless(os.environ.get("AUDIO_WORKER_REAL_MODEL_SMOKE") == "1",
                     "requires the exact built image with baked models")
class GermanAlignmentOfflineSmoke(unittest.TestCase):
    def test_baked_language_models_load_and_emit_with_network_disabled(self):
        # Import Predictor first to exercise its required skops/Transformers
        # ordering. The signal checks actual acoustic inference, not recognition
        # or boundary quality; creator speech canaries follow test deployment.
        from predict import Predictor
        import torch

        self.assertEqual(os.environ.get("HF_HUB_OFFLINE"), "1")
        self.assertEqual(os.environ.get("TRANSFORMERS_OFFLINE"), "1")
        predictor = Predictor()
        waveform = torch.sin(torch.arange(32000, dtype=torch.float32) * .0864)[None, :] * .1
        for aligner, language in ((predictor.aligner, "en"),
                                  (predictor.german_aligner, "de")):
            with self.subTest(language=language):
                aligner.setup(device="cpu")
                self.assertTrue(aligner.supports_language(language))
                self.assertFalse(aligner.supports_language("fr"))
                with torch.inference_mode():
                    emissions, _ = aligner.model(waveform)
                self.assertEqual(emissions.shape[0], 1)
                self.assertGreater(emissions.shape[1], 50)
                self.assertEqual(emissions.shape[2], len(aligner.labels))
                self.assertTrue(torch.isfinite(emissions).all())
                self.assertEqual(aligner.sample_rate, 16000)
        self.assertNotEqual(predictor.aligner.model_id, predictor.german_aligner.model_id)


if __name__ == "__main__":
    unittest.main()
