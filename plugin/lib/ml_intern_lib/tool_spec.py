"""Minimal ToolSpec dataclass — replaces `agent.core.tools.ToolSpec` for the
vendored tool factories (sandbox_tool.get_sandbox_tools, local_tools.get_local_tools).

Same shape as the original; we just don't drag the rest of the agent.core.tools
module (ToolRouter, MCP client, etc.) along with it — those concepts don't
exist inside an MCP-server frontend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Optional[Callable[[dict[str, Any]], Awaitable[tuple[str, bool]]]] = None
