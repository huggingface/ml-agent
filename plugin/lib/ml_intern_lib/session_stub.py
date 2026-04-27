"""
Stub `Event` and a minimal session-like object so vendored tools that were
written against the standalone CLI's Session/Event types can run inside the
Claude Code MCP server (where there is no Session — Claude Code is the loop).

Tools call `session.send_event(Event(...))` for telemetry and `session.hf_token`
for token access. The MCP server doesn't surface those events to the user
(Claude Code's tool-output channel does that already), so we drop them on
the floor and read the token from the environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Event:
    """Minimal stand-in for agent.core.session.Event.

    The real Event has more fields, but the only attributes vendored tools
    construct are `event_type` and `data`.
    """
    event_type: str = ""
    data: dict[str, Any] = field(default_factory=dict)


class StubSession:
    """Drop-in for the Session object the standalone CLI passes into tool handlers.

    Implements just enough surface for `jobs_tool` and `sandbox_tool`:
      - `send_event(...)`  → swallowed
      - `hf_token`         → from env
      - `_running_job_ids` → in-memory set, used by jobs_tool to track concurrent jobs
    """

    def __init__(self) -> None:
        self.hf_token: str | None = (
            os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
            or None
        )
        self._running_job_ids: set[str] = set()
        # Some tools touch session._sandbox_created_at via telemetry; provide
        # the attribute so attribute access doesn't AttributeError.
        self._sandbox_created_at: float | None = None

    async def send_event(self, _event: Any) -> None:
        # MCP server has no event channel — tool output is what Claude Code shows.
        return None

    # Some call sites use `getattr(session, "config", None)` for things like
    # `session.config.yolo_mode`. Provide a None-shaped config; tools must
    # handle the missing case (they do — checked grep before vendoring).
    config = None
