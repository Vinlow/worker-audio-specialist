from faster_whisper.utils import download_model

# ── Whisper Models ──────────────────────────────────────────────────
# Only pre-download models we actually use in production.
# Other models are still available via AVAILABLE_MODELS in predict.py
# and will be downloaded on first request (with a cold-start penalty).
whisper_models = [
    "large-v3",
    "turbo",
]

for model_name in whisper_models:
    print(f"Downloading Whisper model: {model_name}...")
    download_model(model_name, cache_dir=None)
    print(f"Finished downloading {model_name}.")

# ── CLAP Model ──────────────────────────────────────────────────────
# Pre-download CLAP for audio-text similarity scoring.
# ~1.5 GB, loaded on first request but cached in the Docker image.
CLAP_MODEL_ID = "laion/larger_clap_music_and_speech"
print(f"Downloading CLAP model: {CLAP_MODEL_ID}...")

from transformers import ClapModel, ClapProcessor
ClapProcessor.from_pretrained(CLAP_MODEL_ID)
ClapModel.from_pretrained(CLAP_MODEL_ID)
print(f"Finished downloading CLAP model.")

print("All models downloaded.")
