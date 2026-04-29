from agent.core.hf_tokens import resolve_hf_request_token
from agent.core.llm_params import (
    UnsupportedEffortError,
    _resolve_hf_router_token,
    _resolve_llm_params,
)


def test_openai_xhigh_effort_is_forwarded():
    params = _resolve_llm_params(
        "openai/gpt-5.5",
        reasoning_effort="xhigh",
        strict=True,
    )

    assert params["model"] == "openai/gpt-5.5"
    assert params["reasoning_effort"] == "xhigh"


def test_openai_max_effort_is_still_rejected():
    try:
        _resolve_llm_params(
            "openai/gpt-5.4",
            reasoning_effort="max",
            strict=True,
        )
    except UnsupportedEffortError as exc:
        assert "OpenAI doesn't accept effort='max'" in str(exc)
    else:
        raise AssertionError("Expected UnsupportedEffortError for max effort")


def test_tokenrouter_params_use_openai_compatible_endpoint(monkeypatch):
    monkeypatch.setenv("TOKENROUTER_API_KEY", " tokenrouter-key ")
    monkeypatch.delenv("TOKENROUTER_API_BASE", raising=False)

    params = _resolve_llm_params("tokenrouter/auto:balance")

    assert params == {
        "model": "openai/auto:balance",
        "api_base": "https://api.tokenrouter.io/v1",
        "api_key": "tokenrouter-key",
    }


def test_tokenrouter_api_base_can_be_overridden(monkeypatch):
    monkeypatch.setenv("TOKENROUTER_API_KEY", "tokenrouter-key")
    monkeypatch.setenv("TOKENROUTER_API_BASE", " https://proxy.example.test/v1/ ")

    params = _resolve_llm_params("tokenrouter/openai:gpt-4o")

    assert params["model"] == "openai/openai:gpt-4o"
    assert params["api_base"] == "https://proxy.example.test/v1"
    assert params["api_key"] == "tokenrouter-key"


def test_tokenrouter_effort_is_rejected_in_strict_probe_mode():
    try:
        _resolve_llm_params(
            "tokenrouter/auto:balance",
            reasoning_effort="high",
            strict=True,
        )
    except UnsupportedEffortError as exc:
        assert "TokenRouter doesn't accept effort='high'" in str(exc)
    else:
        raise AssertionError("Expected UnsupportedEffortError for TokenRouter effort")


def test_hf_router_token_prefers_inference_token(monkeypatch):
    monkeypatch.setenv("INFERENCE_TOKEN", " inference-token ")
    monkeypatch.setenv("HF_TOKEN", "hf-token")

    assert _resolve_hf_router_token("session-token") == "inference-token"


def test_hf_router_token_prefers_session_over_hf_cache(monkeypatch):
    monkeypatch.delenv("INFERENCE_TOKEN", raising=False)
    monkeypatch.setenv("HF_TOKEN", "hf-token")

    assert _resolve_hf_router_token(" session-token ") == "session-token"


def test_hf_router_token_uses_hf_token_env_via_huggingface_hub(monkeypatch):
    monkeypatch.delenv("INFERENCE_TOKEN", raising=False)
    monkeypatch.setenv("HF_TOKEN", " hf-token ")

    assert _resolve_hf_router_token(None) == "hf-token"


def test_hf_router_token_uses_huggingface_hub_cache(monkeypatch):
    import huggingface_hub

    monkeypatch.delenv("INFERENCE_TOKEN", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(huggingface_hub, "get_token", lambda: "cached-token")

    assert _resolve_hf_router_token(None) == "cached-token"


def test_hf_router_token_swallows_huggingface_hub_errors(monkeypatch):
    import huggingface_hub

    def fail():
        raise RuntimeError("cache unavailable")

    monkeypatch.delenv("INFERENCE_TOKEN", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(huggingface_hub, "get_token", fail)

    assert _resolve_hf_router_token(None) is None


def test_hf_router_params_set_bill_to_only_for_inference_token(monkeypatch):
    monkeypatch.setenv("INFERENCE_TOKEN", "inference-token")
    monkeypatch.setenv("HF_BILL_TO", "test-org")

    params = _resolve_llm_params("moonshotai/Kimi-K2.6")

    assert params["api_key"] == "inference-token"
    assert params["extra_headers"] == {"X-HF-Bill-To": "test-org"}


def test_hf_request_token_keeps_browser_user_precedence(monkeypatch):
    class Request:
        headers = {"Authorization": "Bearer browser-token"}
        cookies = {"hf_access_token": "cookie-token"}

    monkeypatch.setenv("HF_TOKEN", "server-token")

    assert resolve_hf_request_token(Request()) == "browser-token"


def test_hf_request_token_does_not_use_cached_login(monkeypatch):
    import huggingface_hub

    class Request:
        headers = {}
        cookies = {}

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(huggingface_hub, "get_token", lambda: "cached-token")

    assert resolve_hf_request_token(Request()) is None
