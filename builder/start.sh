#!/bin/bash
# Startup script: download models to network volume (or local fallback), then start handler.
# Models are cached — subsequent starts on the same volume skip the download.

set -e

# If no network volume is mounted, fall back to local cache
if [ ! -d "/runpod-volume" ]; then
    echo "[start] No network volume found, using local model cache."
    export MODEL_DIR="/models"
fi

echo "[start] Model cache: $MODEL_DIR"
echo "[start] Ensuring models are downloaded..."
python -u /fetch_models.py

echo "[start] Starting worker handler..."
exec python -u /rp_handler.py
