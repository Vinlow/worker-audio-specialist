import gc
import hashlib
import inspect
import os
from pathlib import Path

# wtpsplit 2.2.1 deliberately initializes skops before Transformers. Skops
# enumerates trusted types at import time; reversing the order can force
# Transformers' unrelated lazy vision modules (and optional torchvision) into
# this text-only image. Keep this as the first third-party import.
import skops.io as _skops_io  # noqa: F401

from faster_whisper.utils import download_model
from huggingface_hub import snapshot_download
from model_manifest import (
    CLAP_MODEL_ID,
    CLAP_MODEL_REVISION,
    PYANNOTE_SNAPSHOTS,
    WAV2VEC2_CHECKPOINT_FILENAME,
    WAV2VEC2_CHECKPOINT_SHA256,
    WHISPER_MODEL_REVISIONS,
)


def get_hf_token():
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or ""
    ).strip()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


HF_TOKEN = get_hf_token()
if HF_TOKEN:
    os.environ.setdefault("HF_TOKEN", HF_TOKEN)
    os.environ.setdefault("HUGGINGFACE_TOKEN", HF_TOKEN)
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", HF_TOKEN)
    print("Hugging Face token detected; authenticated model downloads enabled.")
else:
    print("No Hugging Face token detected; using anonymous model downloads.")


def kwargs_for_hf_callable(callable_obj):
    if not HF_TOKEN:
        return {}
    try:
        params = inspect.signature(callable_obj).parameters
    except (TypeError, ValueError):
        return {}
    if "token" in params or any(param.kind == param.VAR_KEYWORD for param in params.values()):
        return {"token": HF_TOKEN}
    if "use_auth_token" in params:
        return {"use_auth_token": HF_TOKEN}
    return {}


def download_whisper_model(model_name, revision):
    kwargs = {
        "cache_dir": None,
        "revision": revision,
        **kwargs_for_hf_callable(download_model),
    }
    try:
        return download_model(model_name, **kwargs)
    except TypeError as exc:
        if "token" not in str(exc) and "use_auth_token" not in str(exc):
            raise
        return download_model(model_name, cache_dir=None, revision=revision)

# ── Whisper Models ──────────────────────────────────────────────────
# Pre-download every accepted model. ``predict.py`` derives its allowlist from
# this same manifest and reopens only the exact baked revision offline. Do not
# add an API-visible model anywhere else: request-time Hub access makes that
# path mutable and adds 30-60s cold latency. ``medium`` is the Web2Labs
# fallback, which fires exactly when a job already failed once — the worst
# possible moment to discover a missing artifact.
for model_name, revision in WHISPER_MODEL_REVISIONS.items():
    print(f"Downloading Whisper model: {model_name}@{revision}...")
    download_whisper_model(model_name, revision)
    print(f"Finished downloading {model_name}@{revision}.")

# ── CLAP Model ──────────────────────────────────────────────────────
# ~1.5 GB, pre-downloaded for zero cold-start on CLAP scoring requests.
print(f"Downloading CLAP model: {CLAP_MODEL_ID}@{CLAP_MODEL_REVISION}...")

from transformers import ClapModel, ClapProcessor
snapshot_download(
    repo_id=CLAP_MODEL_ID,
    revision=CLAP_MODEL_REVISION,
    **kwargs_for_hf_callable(snapshot_download),
)
CLAP_KWARGS = {
    "revision": CLAP_MODEL_REVISION,
    "local_files_only": True,
    **kwargs_for_hf_callable(ClapModel.from_pretrained),
}
ClapProcessor.from_pretrained(CLAP_MODEL_ID, **CLAP_KWARGS)
ClapModel.from_pretrained(CLAP_MODEL_ID, **CLAP_KWARGS)
print(f"Finished downloading CLAP model.")


# ── pyannote diarization artifacts (when a build secret is available) ───
# The token is mounted as a BuildKit secret and is never baked into the image.
# Runtime is deliberately cache-only, so a deployable diarization image must
# include all exact files here.

if HF_TOKEN:
    for repo_id, snapshot in PYANNOTE_SNAPSHOTS.items():
        expected_revision = snapshot["revision"]
        allow_patterns = list(snapshot["allow_patterns"])
        print(f"Downloading diarization artifact: {repo_id}@{expected_revision}...")
        snapshot_path = snapshot_download(
            repo_id=repo_id,
            # Runtime rewrites every legacy pyannote 3.1 dependency to this
            # immutable revision, so the image cache must use the same identity.
            revision=expected_revision,
            allow_patterns=allow_patterns,
            **kwargs_for_hf_callable(snapshot_download),
        )
        resolved_revision = Path(snapshot_path).name
        if resolved_revision != expected_revision:
            raise RuntimeError(
                f"{repo_id} snapshot mismatch: expected {expected_revision}, "
                f"resolved {resolved_revision}"
            )
        print(f"Finished downloading {repo_id}@{resolved_revision}.")
else:
    print(
        "No Hugging Face token at build time; skipping gated pyannote cache. "
        "The optional diarization path will fail closed in offline mode."
    )

# ── Parakeet TDT Experimental ASR Model ─────────────────────────────
# The model is optional at request time but immutable and image-resident when
# selected.  Exclude the duplicate NeMo checkpoint: the worker uses the
# Transformers safetensors path only.
PARAKEET_MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"
PARAKEET_MODEL_REVISION = "7c35754d166cca382ad1e53e68b01e7c575f3a1d"
print(
    "Downloading Parakeet model: "
    f"{PARAKEET_MODEL_ID}@{PARAKEET_MODEL_REVISION}..."
)
snapshot_download(
    repo_id=PARAKEET_MODEL_ID,
    revision=PARAKEET_MODEL_REVISION,
    allow_patterns=["*.json", "*.safetensors"],
    **kwargs_for_hf_callable(snapshot_download),
)

# Construct both processor and model during the build.  This is deliberately
# more than a cache download: it proves that the pinned Transformers runtime
# can deserialize this exact model before RunPod promotes the image.
import torch
from transformers import AutoModelForTDT, AutoProcessor

PARAKEET_PROCESSOR_KWARGS = {
    "revision": PARAKEET_MODEL_REVISION,
    "local_files_only": True,
    **kwargs_for_hf_callable(AutoProcessor.from_pretrained),
}
PARAKEET_MODEL_KWARGS = {
    "revision": PARAKEET_MODEL_REVISION,
    "local_files_only": True,
    **kwargs_for_hf_callable(AutoModelForTDT.from_pretrained),
}
_parakeet_processor = AutoProcessor.from_pretrained(
    PARAKEET_MODEL_ID,
    **PARAKEET_PROCESSOR_KWARGS,
)
_parakeet_model = AutoModelForTDT.from_pretrained(
    PARAKEET_MODEL_ID,
    dtype=torch.float16,
    **PARAKEET_MODEL_KWARGS,
)
del _parakeet_model
del _parakeet_processor
gc.collect()
print("Finished downloading and validating Parakeet model.")

# ── SaT Experimental Punctuation Window Model ──────────────────────
# This is a request-explicit diagnostic probe for Starforge. It consumes one
# already-tokenized, source-bound XLM-R window and returns terminal-boundary
# probabilities. It never rewrites Whisper words or geometry.
SAT_MODEL_ID = "segment-any-text/sat-3l-sm"
SAT_MODEL_REVISION = "137da054051ad9f1eac42025f758db4ac9f22535"
SAT_TOKENIZER_ID = "FacebookAI/xlm-roberta-base"
SAT_TOKENIZER_REVISION = "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089"
print(
    "Downloading SaT punctuation model: "
    f"{SAT_MODEL_ID}@{SAT_MODEL_REVISION}..."
)
sat_model_snapshot = snapshot_download(
    repo_id=SAT_MODEL_ID,
    revision=SAT_MODEL_REVISION,
    allow_patterns=["config.json", "model.safetensors"],
    **kwargs_for_hf_callable(snapshot_download),
)
sat_tokenizer_snapshot = snapshot_download(
    repo_id=SAT_TOKENIZER_ID,
    revision=SAT_TOKENIZER_REVISION,
    allow_patterns=[
        "config.json",
        "sentencepiece.bpe.model",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ],
    **kwargs_for_hf_callable(snapshot_download),
)

# Construction is the compatibility gate: wtpsplit must deserialize the
# pinned checkpoint under this image's exact Transformers runtime.
from wtpsplit import SaT
_sat_punctuator = SaT(
    sat_model_snapshot,
    tokenizer_name_or_path=sat_tokenizer_snapshot,
    from_pretrained_kwargs={"local_files_only": True},
)
del _sat_punctuator
gc.collect()
print("Finished downloading and validating SaT punctuation model.")

# ── Wav2Vec2 Forced Alignment Model ──────────────────────────────────
# ~1.2 GB, pre-downloaded for zero cold-start on word-level forced alignment.
# Used when input has `force_align: true`. Re-times Whisper word_timestamps
# from ~100-300ms accuracy (Whisper cross-attention) to ~30-50ms (CTC forced
# alignment against actual audio). English-only (librispeech-trained).
print("Downloading wav2vec2 alignment model: WAV2VEC2_ASR_LARGE_LV60K_960H...")
from torchaudio.pipelines import WAV2VEC2_ASR_LARGE_LV60K_960H as W2V_BUNDLE
_ = W2V_BUNDLE.get_model()  # downloads + caches the .pth into ~/.cache/torch/hub/checkpoints
wav2vec_checkpoint = (
    Path(torch.hub.get_dir())
    / "checkpoints"
    / WAV2VEC2_CHECKPOINT_FILENAME
)
wav2vec_digest = sha256_file(wav2vec_checkpoint)
if wav2vec_digest != WAV2VEC2_CHECKPOINT_SHA256:
    raise RuntimeError(
        "wav2vec2 checkpoint digest mismatch: "
        f"expected {WAV2VEC2_CHECKPOINT_SHA256}, resolved {wav2vec_digest}"
    )
print(f"Verified wav2vec2 checkpoint sha256: {wav2vec_digest}")
print("Finished downloading wav2vec2 alignment model.")

print("All models downloaded.")
