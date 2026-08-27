import argparse
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


WHISPER_MODEL_GROUPS = {
    "whisper-standard": ("base", "small", "medium"),
    "whisper-large-v3": ("large-v3",),
    "whisper-turbo": ("turbo",),
}
MODEL_GROUPS = (
    "all",
    *WHISPER_MODEL_GROUPS,
    "clap-alignment",
    "experimental",
    "diarization",
)


def get_hf_token():
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or ""
    ).strip()


def configure_hf_token():
    token = get_hf_token()
    if token:
        os.environ.setdefault("HF_TOKEN", token)
        os.environ.setdefault("HUGGINGFACE_TOKEN", token)
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)
        print("Hugging Face token detected; authenticated model downloads enabled.")
    else:
        print("No Hugging Face token detected; using anonymous model downloads.")
    return token


def kwargs_for_hf_callable(callable_obj, token):
    if not token:
        return {}
    try:
        params = inspect.signature(callable_obj).parameters
    except (TypeError, ValueError):
        return {}
    if "token" in params or any(
        param.kind == param.VAR_KEYWORD for param in params.values()
    ):
        return {"token": token}
    if "use_auth_token" in params:
        return {"use_auth_token": token}
    return {}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_whisper_model(model_name, revision, token):
    kwargs = {
        "cache_dir": None,
        "revision": revision,
        **kwargs_for_hf_callable(download_model, token),
    }
    try:
        return download_model(model_name, **kwargs)
    except TypeError as exc:
        if "token" not in str(exc) and "use_auth_token" not in str(exc):
            raise
        return download_model(model_name, cache_dir=None, revision=revision)


def download_whisper_models(token, model_names=None):
    # Pre-download every accepted model. ``predict.py`` derives its allowlist
    # from this same manifest and reopens only the exact baked revision offline.
    selected_names = model_names or tuple(WHISPER_MODEL_REVISIONS)
    for model_name in selected_names:
        revision = WHISPER_MODEL_REVISIONS[model_name]
        print(f"Downloading Whisper model: {model_name}@{revision}...")
        download_whisper_model(model_name, revision, token)
        print(f"Finished downloading {model_name}@{revision}.")


def download_clap_model(token):
    # Construct both artifacts during the build so a dependency drift cannot
    # leave a superficially complete but unreadable cache in the image.
    from transformers import ClapModel, ClapProcessor

    print(f"Downloading CLAP model: {CLAP_MODEL_ID}@{CLAP_MODEL_REVISION}...")
    snapshot_download(
        repo_id=CLAP_MODEL_ID,
        revision=CLAP_MODEL_REVISION,
        **kwargs_for_hf_callable(snapshot_download, token),
    )
    clap_kwargs = {
        "revision": CLAP_MODEL_REVISION,
        "local_files_only": True,
        **kwargs_for_hf_callable(ClapModel.from_pretrained, token),
    }
    ClapProcessor.from_pretrained(CLAP_MODEL_ID, **clap_kwargs)
    ClapModel.from_pretrained(CLAP_MODEL_ID, **clap_kwargs)
    print("Finished downloading CLAP model.")


def download_diarization_models(token):
    # These repositories are gated. The token is supplied only to this build
    # layer through a BuildKit secret and is never persisted in the image.
    if not token:
        raise RuntimeError(
            "The diarization model group requires an HF token build secret."
        )
    for repo_id, snapshot in PYANNOTE_SNAPSHOTS.items():
        expected_revision = snapshot["revision"]
        allow_patterns = list(snapshot["allow_patterns"])
        print(f"Downloading diarization artifact: {repo_id}@{expected_revision}...")
        snapshot_path = snapshot_download(
            repo_id=repo_id,
            revision=expected_revision,
            allow_patterns=allow_patterns,
            **kwargs_for_hf_callable(snapshot_download, token),
        )
        resolved_revision = Path(snapshot_path).name
        if resolved_revision != expected_revision:
            raise RuntimeError(
                f"{repo_id} snapshot mismatch: expected {expected_revision}, "
                f"resolved {resolved_revision}"
            )
        print(f"Finished downloading {repo_id}@{resolved_revision}.")


def download_parakeet_model(token):
    import torch
    from transformers import AutoModelForTDT, AutoProcessor

    model_id = "nvidia/parakeet-tdt-0.6b-v3"
    revision = "7c35754d166cca382ad1e53e68b01e7c575f3a1d"
    print(f"Downloading Parakeet model: {model_id}@{revision}...")
    snapshot_download(
        repo_id=model_id,
        revision=revision,
        allow_patterns=["*.json", "*.safetensors"],
        **kwargs_for_hf_callable(snapshot_download, token),
    )
    processor_kwargs = {
        "revision": revision,
        "local_files_only": True,
        **kwargs_for_hf_callable(AutoProcessor.from_pretrained, token),
    }
    model_kwargs = {
        "revision": revision,
        "local_files_only": True,
        **kwargs_for_hf_callable(AutoModelForTDT.from_pretrained, token),
    }
    processor = AutoProcessor.from_pretrained(model_id, **processor_kwargs)
    model = AutoModelForTDT.from_pretrained(
        model_id,
        dtype=torch.float16,
        **model_kwargs,
    )
    del model
    del processor
    gc.collect()
    print("Finished downloading and validating Parakeet model.")


def download_sat_model(token):
    # SaT is a request-explicit punctuation diagnostic. Construction is the
    # compatibility gate for wtpsplit and the pinned Transformers runtime.
    from wtpsplit import SaT

    model_id = "segment-any-text/sat-3l-sm"
    revision = "137da054051ad9f1eac42025f758db4ac9f22535"
    tokenizer_id = "FacebookAI/xlm-roberta-base"
    tokenizer_revision = "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089"
    print(f"Downloading SaT punctuation model: {model_id}@{revision}...")
    model_snapshot = snapshot_download(
        repo_id=model_id,
        revision=revision,
        allow_patterns=["config.json", "model.safetensors"],
        **kwargs_for_hf_callable(snapshot_download, token),
    )
    tokenizer_snapshot = snapshot_download(
        repo_id=tokenizer_id,
        revision=tokenizer_revision,
        allow_patterns=[
            "config.json",
            "sentencepiece.bpe.model",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ],
        **kwargs_for_hf_callable(snapshot_download, token),
    )
    punctuator = SaT(
        model_snapshot,
        tokenizer_name_or_path=tokenizer_snapshot,
        from_pretrained_kwargs={"local_files_only": True},
    )
    del punctuator
    gc.collect()
    print("Finished downloading and validating SaT punctuation model.")


def download_wav2vec_model():
    import torch
    from torchaudio.pipelines import WAV2VEC2_ASR_LARGE_LV60K_960H as bundle

    print("Downloading wav2vec2 alignment model: WAV2VEC2_ASR_LARGE_LV60K_960H...")
    _ = bundle.get_model()
    checkpoint = (
        Path(torch.hub.get_dir())
        / "checkpoints"
        / WAV2VEC2_CHECKPOINT_FILENAME
    )
    digest = sha256_file(checkpoint)
    if digest != WAV2VEC2_CHECKPOINT_SHA256:
        raise RuntimeError(
            "wav2vec2 checkpoint digest mismatch: "
            f"expected {WAV2VEC2_CHECKPOINT_SHA256}, resolved {digest}"
        )
    print(f"Verified wav2vec2 checkpoint sha256: {digest}")
    print("Finished downloading wav2vec2 alignment model.")


def download_clap_alignment_models(token):
    download_clap_model(token)
    download_wav2vec_model()


def download_experimental_models(token):
    download_parakeet_model(token)
    download_sat_model(token)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download one independently layered Audio Worker model group."
    )
    parser.add_argument(
        "--group",
        choices=MODEL_GROUPS,
        default="all",
        help="Model group to download (default: all).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    token = configure_hf_token()

    if args.group == "all":
        download_whisper_models(token)
        download_clap_alignment_models(token)
        download_experimental_models(token)
        if token:
            download_diarization_models(token)
        else:
            print(
                "No Hugging Face token at build time; skipping gated pyannote cache. "
                "The optional diarization path will fail closed in offline mode."
            )
    elif args.group in WHISPER_MODEL_GROUPS:
        download_whisper_models(token, WHISPER_MODEL_GROUPS[args.group])
    elif args.group == "clap-alignment":
        download_clap_alignment_models(token)
    elif args.group == "experimental":
        download_experimental_models(token)
    elif args.group == "diarization":
        download_diarization_models(token)

    print(f"Model group downloaded: {args.group}.")


if __name__ == "__main__":
    main()
