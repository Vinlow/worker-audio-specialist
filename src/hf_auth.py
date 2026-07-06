"""Hugging Face auth helpers.

The token is supplied by the RunPod endpoint as an environment variable. Keep it
in-process only: never call `huggingface_hub.login()` here, because that writes
a token file into the container filesystem.
"""

import os


def get_hf_token():
    """Return the configured Hugging Face token without logging it."""
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or ""
    ).strip()


def normalize_hf_token_env():
    """Populate common HF token aliases so downstream libraries can discover it."""
    token = get_hf_token()
    if not token:
        return ""
    os.environ.setdefault("HF_TOKEN", token)
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)
    return token


def hf_from_pretrained_kwargs():
    """Keyword args for Transformers `from_pretrained` calls."""
    token = normalize_hf_token_env()
    return {"token": token} if token else {}
