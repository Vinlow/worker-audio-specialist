import io
import os
import wave

import numpy as np
from locust import HttpUser, task
import base64


def generate_test_audio(duration_ms):
    """Create deterministic PCM WAV without the undeclared pydub dependency."""
    sample_rate = 16_000
    sample_count = int(sample_rate * duration_ms / 1000.0)
    time_axis = np.arange(sample_count, dtype=np.float32) / sample_rate
    samples = (0.15 * np.sin(2 * np.pi * 440 * time_axis) * 32767).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(samples.tobytes())
    base64_audio = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return base64_audio

class ApiUser(HttpUser):
    @task
    def send_audio_request(self):
        headers = {
            'Content-Type': 'application/json',
        }
        api_key = os.environ.get("RUNPOD_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        audio_data = generate_test_audio(5000)
        
        data = {
            "input": {
                "audio_base64": audio_data,
                "model": "turbo",
                "word_timestamps": True,
                "clap_queries": {
                    "tone": "a steady electronic tone",
                    "speech": "a person speaking",
                },
            }
        }
        endpoint_id = os.environ.get("RUNPOD_ENDPOINT_ID", "dx99xymo20v3o9")
        self.client.post(f"/v2/{endpoint_id}/runsync", json=data, headers=headers)
