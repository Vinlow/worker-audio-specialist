"""Hugging Face auth helpers.

The token is supplied by the RunPod endpoint as an environment variable. Keep it
in-process only: never call `huggingface_hub.login()` here, because that writes
a token file into the container filesystem.
"""

import functools
import inspect
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


def install_legacy_use_auth_token_compat():
    """Translate pyannote 3.x's legacy download keyword without pinning HF Hub.

    pyannote.audio 3.3.2 imports ``hf_hub_download`` by value and still calls
    it with ``use_auth_token``. Current huggingface_hub accepts ``token``
    instead. Install this adapter before importing pyannote so every copied
    reference receives the compatible callable.
    """
    import huggingface_hub

    current = huggingface_hub.hf_hub_download
    if (
        "use_auth_token" in inspect.signature(current).parameters
        or getattr(current, "_w2l_legacy_auth_compat", False) is True
    ):
        return current

    @functools.wraps(current)
    def compatible(*args, **kwargs):
        legacy_token = kwargs.pop("use_auth_token", None)
        explicit_token = kwargs.get("token")
        if (
            legacy_token is not None
            and explicit_token is not None
            and explicit_token != legacy_token
        ):
            raise ValueError("CONFLICTING_HUGGINGFACE_AUTH_TOKENS")
        if legacy_token is not None:
            kwargs["token"] = legacy_token
        return current(*args, **kwargs)

    compatible._w2l_legacy_auth_compat = True
    huggingface_hub.hf_hub_download = compatible
    return compatible
