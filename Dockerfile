# faster-whisper turbo needs cudnn >= 9
# see https://github.com/runpod-workers/worker-faster_whisper/pull/44
FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

# Remove any third-party apt sources to avoid issues with expiring keys.
RUN rm -f /etc/apt/sources.list.d/*.list

# Set shell and noninteractive environment variables
SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL=/bin/bash

# Set working directory
WORKDIR /

# Fix stale Ubuntu mirrors in the NVIDIA base image
RUN sed -i 's|http://archive.ubuntu.com|http://us.archive.ubuntu.com|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com|http://us.archive.ubuntu.com|g' /etc/apt/sources.list

# Update and install system packages (combined to reduce layers)
RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt-get install --yes --no-install-recommends \
        sudo ca-certificates git wget curl bash \
        libgl1 libx11-6 software-properties-common \
        ffmpeg build-essential libsndfile1 \
        python3.10 python3.10-dev python3.10-venv python3-pip && \
    ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA support (must come before transformers/CLAP)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install Python dependencies
COPY builder/requirements.txt /requirements.txt
RUN pip install --no-cache-dir huggingface_hub[hf_xet] && \
    pip install --no-cache-dir -r /requirements.txt

# Bake in only the turbo model (small, needed for RunPod test).
# large-v3 and CLAP are loaded from network volume at runtime.
RUN python -c "\
from faster_whisper.utils import download_model; \
import os; \
os.makedirs('/models/whisper', exist_ok=True); \
download_model('turbo', cache_dir='/models/whisper'); \
print('turbo model baked in.')"

# Copy startup script and model fetcher
COPY builder/fetch_models.py /fetch_models.py
COPY builder/start.sh /start.sh
RUN chmod +x /start.sh

# Copy handler and other code
COPY src .

# test input that will be used when the container runs outside of runpod
COPY test_input.json .

# Default model cache — overridden by start.sh when network volume exists
ENV MODEL_DIR=/models

# Start: ensure models are cached, then run handler
CMD ["/start.sh"]
