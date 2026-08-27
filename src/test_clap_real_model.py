import math
import os
import struct
import tempfile
import unittest
import wave
from unittest.mock import patch


@unittest.skipUnless(
    os.environ.get("AUDIO_WORKER_REAL_MODEL_SMOKE") == "1",
    "set AUDIO_WORKER_REAL_MODEL_SMOKE=1 inside the built image",
)
class RealClapModelSmokeTest(unittest.TestCase):
    def test_baked_clap_loads_and_scores_on_cpu_without_network_or_token(self):
        import torch

        from clap_scorer import CLAP_RESPONSE_SCHEMA, ClapScorer
        from model_manifest import CLAP_MODEL_ID, CLAP_MODEL_REVISION

        sample_rate = 48_000
        duration_seconds = 1.25
        frame_count = int(sample_rate * duration_seconds)
        audio_path = None
        # Make the device intent explicit even when this smoke runs on a GPU
        # host. Offline mode must come from the image itself rather than this
        # test, otherwise the production warmup path is not being exercised.
        test_environment = {"CUDA_VISIBLE_DEVICES": ""}
        token_names = (
            "HF_TOKEN",
            "HUGGINGFACE_TOKEN",
            "HUGGING_FACE_HUB_TOKEN",
            "HUGGINGFACE_HUB_TOKEN",
        )

        try:
            with tempfile.NamedTemporaryFile(
                suffix=".wav",
                delete=False,
            ) as temp_file:
                audio_path = temp_file.name
            with wave.open(audio_path, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                frames = bytearray()
                for index in range(frame_count):
                    # A deterministic two-tone signal avoids checking in a
                    # binary fixture while exercising decoding and inference.
                    frequency = 220 if index < sample_rate else 440
                    sample = int(
                        0.15
                        * 32767
                        * math.sin(2 * math.pi * frequency * index / sample_rate)
                    )
                    frames.extend(struct.pack("<h", sample))
                wav_file.writeframes(frames)

            self.assertEqual(os.environ.get("HF_HUB_OFFLINE"), "1")
            self.assertEqual(os.environ.get("TRANSFORMERS_OFFLINE"), "1")
            with patch.dict(os.environ, test_environment):
                for token_name in token_names:
                    os.environ.pop(token_name, None)
                # Patch only device selection. Model loading, preprocessing,
                # embedding, and scoring all use the real runtime classes.
                with patch.object(torch.cuda, "is_available", return_value=False):
                    result = ClapScorer().score(
                        audio_path,
                        {
                            "tone": "a steady musical tone",
                            "speech": "a person speaking clearly",
                        },
                    )

            self.assertEqual(result["schema_version"], CLAP_RESPONSE_SCHEMA)
            self.assertEqual(result["status"], "COMPLETED", result)
            self.assertIsNone(result["error_code"], result)
            self.assertIsNone(result["error"], result)
            self.assertFalse(result["retryable"], result)
            self.assertEqual(result["device"], "cpu")
            self.assertEqual(result["model"], CLAP_MODEL_ID)
            self.assertEqual(result["model_revision"], CLAP_MODEL_REVISION)

            self.assertEqual(result["duration"], duration_seconds)
            self.assertEqual(result["windowSize"], 1.0)
            self.assertEqual(result["window_count"], 2)
            self.assertEqual(result["final_window_valid_seconds"], 0.25)
            self.assertEqual(
                math.ceil(result["duration"] / result["windowSize"]),
                result["window_count"],
            )
            self.assertEqual(set(result["scores"]), {"tone", "speech"})
            for scores in result["scores"].values():
                self.assertEqual(len(scores), result["window_count"])
                self.assertTrue(all(math.isfinite(score) for score in scores))
                self.assertTrue(all(0.0 <= score <= 1.0 for score in scores))

            preprocessing = result["preprocessing"]
            self.assertEqual(
                preprocessing["audio_processor_padding_argument"],
                "repeatpad",
            )
            self.assertEqual(
                preprocessing["checkpoint_padding_strategy"],
                "repeatpad",
            )
            self.assertEqual(
                preprocessing["final_window_padding"],
                "zero_to_one_second",
            )
            semantics = result["score_semantics"]
            self.assertEqual(semantics["source"], "cosine_similarity")
            self.assertEqual(
                semantics["calibration"],
                "affine_cosine_not_probability",
            )
            self.assertEqual(
                semantics["transform"],
                "clip((cosine + 1) / 2, 0, 1)",
            )
        finally:
            if audio_path:
                try:
                    os.unlink(audio_path)
                except OSError:
                    pass


if __name__ == "__main__":
    unittest.main()
