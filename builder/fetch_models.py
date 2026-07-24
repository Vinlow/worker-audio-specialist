import gc
import inspect
import os

from faster_whisper.utils import download_model
from huggingface_hub import snapshot_download


def get_hf_token():
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or ""
    ).strip()


HF_TOKEN = get_hf_token()
if HF_TOKEN:
    os.environ.setdefault("HF_TOKEN", HF_TOKEN)
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


def download_whisper_model(model_name):
    kwargs = {"cache_dir": None, **kwargs_for_hf_callable(download_model)}
    try:
        return download_model(model_name, **kwargs)
    except TypeError as exc:
        if "token" not in str(exc) and "use_auth_token" not in str(exc):
            raise
        return download_model(model_name, cache_dir=None)

# ── Whisper Models ──────────────────────────────────────────────────
# Pre-download every model production actually requests.
# Other models in AVAILABLE_MODELS (predict.py) download on first request —
# do NOT let a model land in a production code path without adding it here:
# a request-time download makes that path depend on HuggingFace availability
# and adds 30-60s cold latency. `medium` is the web2labs fallback model,
# which fires exactly when a job already failed once — the worst possible
# moment to be downloading from the network.
whisper_models = [
    "large-v3",  # web2labs primary transcription model (transcribeStream)
    "medium",    # web2labs fallback + tools QUALITY_PRESET + static transcribe()
    "small",     # tools FAST_PRESET (tools.whisper.service.ts)
    "turbo",     # RunPod hub CI test (.runpod/tests.json)
]

for model_name in whisper_models:
    print(f"Downloading Whisper model: {model_name}...")
    download_whisper_model(model_name)
    print(f"Finished downloading {model_name}.")

# ── CLAP Model ──────────────────────────────────────────────────────
# ~1.5 GB, pre-downloaded for zero cold-start on CLAP scoring requests.
CLAP_MODEL_ID = "laion/larger_clap_music_and_speech"
print(f"Downloading CLAP model: {CLAP_MODEL_ID}...")

from transformers import ClapModel, ClapProcessor
CLAP_KWARGS = kwargs_for_hf_callable(ClapModel.from_pretrained)
ClapProcessor.from_pretrained(CLAP_MODEL_ID, **CLAP_KWARGS)
ClapModel.from_pretrained(CLAP_MODEL_ID, **CLAP_KWARGS)
print(f"Finished downloading CLAP model.")

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

# ── Wav2Vec2 Forced Alignment Model ──────────────────────────────────
# ~1.2 GB, pre-downloaded for zero cold-start on word-level forced alignment.
# Used when input has `force_align: true`. Re-times Whisper word_timestamps
# from ~100-300ms accuracy (Whisper cross-attention) to ~30-50ms (CTC forced
# alignment against actual audio). English-only (librispeech-trained).
print("Downloading wav2vec2 alignment model: WAV2VEC2_ASR_LARGE_LV60K_960H...")
from torchaudio.pipelines import WAV2VEC2_ASR_LARGE_LV60K_960H as W2V_BUNDLE
_ = W2V_BUNDLE.get_model()  # downloads + caches the .pth into ~/.cache/torch/hub/checkpoints
print("Finished downloading wav2vec2 alignment model.")

print("All models downloaded.")
