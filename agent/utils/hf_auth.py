"""Hugging Face authentication helpers."""

import os


def get_hf_token() -> str | None:
    """Return the local HF token, preferring HF_TOKEN over login cache."""
    token = os.environ.get("HF_TOKEN")
    if token:
        return token

    try:
        from huggingface_hub import get_token

        token = get_token()
    except Exception:
        return None

    return token or None
