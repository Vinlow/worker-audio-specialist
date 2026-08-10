import math
import os
import struct
import tempfile
import unittest
import wave


@unittest.skipUnless(
    os.environ.get("AUDIO_WORKER_REAL_MODEL_SMOKE") == "1",
    "set AUDIO_WORKER_REAL_MODEL_SMOKE=1 inside the built image",
)
class RealDiarizerModelSmokeTest(unittest.TestCase):
    def test_baked_default_pipeline_loads_and_runs_without_network_or_token(self):
        from diarizer import DEFAULT_MODEL_ID, SpeakerDiarizer

        sample_rate = 16_000
        duration_sec = 8
        audio_path = None
        try:
            self.assertEqual(os.environ.get("HF_HUB_OFFLINE"), "1")
            self.assertEqual(os.environ.get("TRANSFORMERS_OFFLINE"), "1")
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
                for index in range(sample_rate * duration_sec):
                    # Deterministic two-tone carrier exercises the full audio
                    # decoder/inference API without adding a binary fixture.
                    frequency = 220 if index < sample_rate * 4 else 440
                    sample = int(
                        0.15
                        * 32767
                        * math.sin(2 * math.pi * frequency * index / sample_rate)
                    )
                    frames.extend(struct.pack("<h", sample))
                wav_file.writeframes(frames)

            # The workflow runs this container with --network none and passes
            # no token. Setup can succeed only from the exact baked snapshots.
            for token_name in (
                "HF_TOKEN",
                "HUGGINGFACE_TOKEN",
                "HUGGING_FACE_HUB_TOKEN",
                "HUGGINGFACE_HUB_TOKEN",
            ):
                os.environ.pop(token_name, None)

            diarizer = SpeakerDiarizer()
            diarizer.setup("cpu")
            self.assertIsNotNone(diarizer.pipeline)
            self.assertEqual(diarizer.model_id, DEFAULT_MODEL_ID)

            result = diarizer.diarize(
                audio_path,
                [{"word": "test", "start": 0.5, "end": 1.0}],
            )
            self.assertEqual(result["schema_version"], "w2l-speaker-diarization-v1")
            self.assertIn(result["status"], {"COMPLETED", "FAILED"})
            self.assertEqual(result["model_load_policy"], "BAKED_CACHE_ONLY")
            self.assertIn(
                result.get("error_code"),
                {None, "DIARIZATION_EMPTY_OUTPUT"},
                result,
            )
        finally:
            if audio_path:
                try:
                    os.unlink(audio_path)
                except OSError:
                    pass


if __name__ == "__main__":
    unittest.main()
