# syntax=docker/dockerfile:1.7

# faster-whisper turbo needs cudnn >= 9
# see https://github.com/runpod-workers/worker-faster_whisper/pull/44
FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04@sha256:fa44193567d1908f7ca1f3abf8623ce9c63bc8cba7bcfdb32702eb04d326f7a8

# Remove any third-party apt sources to avoid issues with expiring keys.
RUN rm -f /etc/apt/sources.list.d/*.list

# Set shell and noninteractive environment variables
SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL=/bin/bash
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV HF_HOME=/root/.cache/huggingface
ENV HF_HUB_CACHE=/root/.cache/huggingface/hub
ENV PYANNOTE_CACHE=/root/.cache/huggingface/hub
ENV HF_HUB_DISABLE_TELEMETRY=1
ENV PYANNOTE_METRICS_ENABLED=0
ENV TOKENIZERS_PARALLELISM=false
# CLAP is used on the normal final-tier path. Deserialize it concurrently with
# RunPod registration so the first paid request joins a single-flight warmup
# instead of paying the full model transfer after dispatch.
ENV AUDIO_WORKER_PRELOAD=clap

# Set working directory
WORKDIR /

# Fix stale Ubuntu mirrors in the NVIDIA base image
RUN sed -i 's|http://archive.ubuntu.com|http://us.archive.ubuntu.com|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com|http://us.archive.ubuntu.com|g' /etc/apt/sources.list

# Update and install system packages (combined to reduce layers)
RUN apt-get update -y && \
    apt-get install --yes --no-install-recommends \
        sudo ca-certificates git wget curl bash \
        libgl1 libx11-6 software-properties-common \
        ffmpeg build-essential libsndfile1 \
        python3.10 python3.10-dev python3.10-venv python3-pip python3-cairo && \
    ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA (needed for CLAP GPU scoring + wav2vec2 forced alignment).
# torchaudio is now required — used by aligner.py for the WAV2VEC2_ASR_LARGE_LV60K_960H
# pipeline that re-times Whisper word_timestamps with sub-50ms accuracy.
#
# The opt-in diarization sidecar intentionally uses pyannote.audio 3.3.2 +
# speaker-diarization-3.1 first. That pair supports this exact torch/torchaudio
# stack, so the experiment does not turn into a Whisper/alignment dependency
# migration. Community-1 requires torch >=2.8 and is a separate future
# challenger, not part of this protected spike.
#
# 2026-05-23: pinned to torch==2.7.1 / torchaudio==2.7.1 on cu128 wheels.
# Unpinned cu124 worked through ~2026-05-21, then RunPod silently started
# routing "AMPERE_24" jobs to NVIDIA RTX PRO 6000 Blackwell MIG slices
# (sm_120, 2025 architecture). The cu124 wheel does not ship compiled
# torchaudio kernels for sm_120 → "no kernel image is available for execution
# on the device" inside torchaudio.pipelines._wav2vec2.utils.layer_norm (CUDA
# kernel runtime mismatch). cu128 wheels (2.7.1+) ship sm_120 kernels and
# retain sm_86/sm_89/sm_90 → covers every GPU RunPod might assign.
# See web2labs/docs/project/relaunch/sessions/ for the diagnosis.
RUN pip install --no-cache-dir pip==26.2.1 && \
    pip install --no-cache-dir torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128

# Install Python dependencies
COPY builder/requirements.txt /requirements.txt
COPY builder/constraints.txt /constraints.txt
COPY builder/verify_dependency_lock.py /verify_dependency_lock.py
RUN pip install --no-cache-dir --constraint /constraints.txt -r /requirements.txt && \
    pip check && \
    python /verify_dependency_lock.py /constraints.txt && \
    rm /verify_dependency_lock.py

# Pre-download all models into the image (no network volume needed)
COPY src/model_manifest.py /model_manifest.py
COPY builder/fetch_models.py /fetch_models.py
ARG GATED_MODELS_AVAILABLE=true
RUN --mount=type=secret,id=hf_token,required=false \
    echo "Gated model build cache enabled: ${GATED_MODELS_AVAILABLE}" && \
    if [[ "${GATED_MODELS_AVAILABLE}" == "true" && ! -s /run/secrets/hf_token ]]; then \
        echo "GATED_MODELS_AVAILABLE=true but hf_token BuildKit secret is missing" >&2; \
        exit 1; \
    fi && \
    if [[ -s /run/secrets/hf_token ]]; then \
        export HF_TOKEN="$(</run/secrets/hf_token)"; \
    fi && \
    python /fetch_models.py && \
    rm /fetch_models.py

# Runtime is strictly image-resident. Transformers 5.x may otherwise launch a
# background safetensors conversion lookup even when a loader itself passes
# local_files_only=True.
ARG AUDIO_WORKER_BUILD_SHA=unknown
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV AUDIO_WORKER_BUILD_SHA=${AUDIO_WORKER_BUILD_SHA}
LABEL org.opencontainers.image.revision=${AUDIO_WORKER_BUILD_SHA}

# Retain the source used to build the baked model set so the runtime image's
# contract tests can audit import order without affecting the expensive model
# download layer above.
COPY builder/fetch_models.py /builder/fetch_models.py

# Copy handler and other code
COPY src .

# test input that will be used when the container runs outside of runpod
COPY test_input.json .

# Set default command
CMD ["python", "-u", "/rp_handler.py"]
