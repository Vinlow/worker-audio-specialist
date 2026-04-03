"""
Model downloader for Audio Specialist worker.

Downloads Whisper and CLAP models to a shared cache directory.
When using a RunPod network volume, models are downloaded once
and reused by all workers on the same volume.

Usage:
  - During Docker build: not run (keeps image small)
  - At container startup: called by start.sh before the handler
"""
import os
import sys

# Network volume path — set via RunPod env or default
MODEL_DIR = os.environ.get("MODEL_DIR", "/runpod-volume/models")
HF_CACHE = os.path.join(MODEL_DIR, "huggingface")

# Point HuggingFace libraries at the network volume cache
os.environ["HF_HOME"] = HF_CACHE
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE

# ── Whisper Models ──────────────────────────────────────────────────
WHISPER_MODELS = ["large-v3", "turbo"]
WHISPER_CACHE = os.path.join(MODEL_DIR, "whisper")


def download_whisper():
    from faster_whisper.utils import download_model

    os.makedirs(WHISPER_CACHE, exist_ok=True)
    for model_name in WHISPER_MODELS:
        model_path = os.path.join(WHISPER_CACHE, model_name)
        if os.path.isdir(model_path) and os.listdir(model_path):
            print(f"[models] Whisper {model_name} already cached, skipping.")
            continue
        print(f"[models] Downloading Whisper model: {model_name}...")
        download_model(model_name, cache_dir=WHISPER_CACHE)
        print(f"[models] Whisper {model_name} downloaded.")


# ── CLAP Model ──────────────────────────────────────────────────────
CLAP_MODEL_ID = "laion/larger_clap_music_and_speech"


def download_clap():
    from transformers import ClapModel, ClapProcessor

    # Check if already cached (HF cache structure: models--org--name)
    cache_marker = os.path.join(HF_CACHE, "hub", f"models--{CLAP_MODEL_ID.replace('/', '--')}")
    if os.path.isdir(cache_marker):
        print(f"[models] CLAP model already cached, skipping.")
        return
    print(f"[models] Downloading CLAP model: {CLAP_MODEL_ID}...")
    ClapProcessor.from_pretrained(CLAP_MODEL_ID)
    ClapModel.from_pretrained(CLAP_MODEL_ID)
    print(f"[models] CLAP model downloaded.")


if __name__ == "__main__":
    print(f"[models] Cache directory: {MODEL_DIR}")
    os.makedirs(MODEL_DIR, exist_ok=True)
    download_whisper()
    download_clap()
    print("[models] All models ready.")
