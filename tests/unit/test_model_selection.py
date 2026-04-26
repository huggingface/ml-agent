"""Tests for backend model-id acceptance."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_agent_routes():
    root = Path(__file__).parent.parent.parent
    backend = root / "backend"
    sys.path.insert(0, str(backend))
    spec = importlib.util.spec_from_file_location(
        "backend_agent_routes",
        backend / "routes" / "agent.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["backend_agent_routes"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_backend_accepts_any_google_ai_studio_model_id():
    mod = _load_agent_routes()

    assert mod._is_known_or_direct_model("google/gemini-3.1-pro-preview")
    assert mod._is_known_or_direct_model("google/gemini-2.5-flash-preview-09-2025")
    assert mod._is_known_or_direct_model("google/gemini-custom-preview-12-2026")
    assert mod._is_known_or_direct_model("google/gemma-3-27b-it")


def test_backend_accepts_any_vertex_ai_gemini_or_gemma_model_id():
    mod = _load_agent_routes()

    assert mod._is_known_or_direct_model("google-geap/gemini-3-flash-preview")
    assert mod._is_known_or_direct_model("google-geap/gemini-2.5-pro")
    assert mod._is_known_or_direct_model("google-geap/gemini-custom-preview-12-2026")
    assert mod._is_known_or_direct_model("google-geap/gemma-3-27b-it")


def test_backend_still_rejects_unknown_and_non_google_direct_ids():
    mod = _load_agent_routes()

    assert not mod._is_known_or_direct_model("not-a-provider-model")
    assert not mod._is_known_or_direct_model("openai/not-in-available-list")
    assert not mod._is_known_or_direct_model("anthropic/not-in-available-list")
    assert not mod._is_known_or_direct_model("MiniMaxAI/Some-New-Model")
    assert not mod._is_known_or_direct_model("some-org/some-hf-router-model")
    assert not mod._is_known_or_direct_model("huggingface/some-org/some-model")
    assert not mod._is_known_or_direct_model("gemini/gemini-3.1-pro-preview")
    assert not mod._is_known_or_direct_model("vertex_ai/gemini-3-flash-preview")
    assert not mod._is_known_or_direct_model("vertex_ai/text-bison")
    assert not mod._is_known_or_direct_model("google/deep-research-pro-preview-12-2025")
    assert not mod._is_known_or_direct_model("google/deep-research-max-preview-04-2026")
    assert not mod._is_known_or_direct_model("google/imagen-4.0-generate-001")
    assert not mod._is_known_or_direct_model("google/")
    assert not mod._is_known_or_direct_model("google-geap/")
    assert not mod._is_known_or_direct_model(123)
