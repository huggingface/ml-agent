"""No-op telemetry — replaces `agent.core.telemetry` for the vendored tools.

The standalone CLI uses telemetry to emit Events for the session JSONL trail.
The MCP server has no session, and Claude Code's transcript captures tool
input/output natively, so we drop telemetry calls on the floor.

Every coroutine in the real telemetry module returns None or a small dict;
we mirror that. Synchronous helpers are no-ops.
"""

from __future__ import annotations

from typing import Any


async def record_llm_call(*_a: Any, **_kw: Any) -> dict:
    return {}


async def record_hf_job_submit(*_a: Any, **_kw: Any) -> None:
    return None


async def record_hf_job_complete(*_a: Any, **_kw: Any) -> None:
    return None


async def record_sandbox_create(*_a: Any, **_kw: Any) -> None:
    return None


async def record_sandbox_destroy(*_a: Any, **_kw: Any) -> None:
    return None


async def record_feedback(*_a: Any, **_kw: Any) -> None:
    return None


def extract_usage(*_a: Any, **_kw: Any) -> dict:
    return {}


class HeartbeatSaver:
    @staticmethod
    def maybe_fire(_session: Any) -> None:
        return None
