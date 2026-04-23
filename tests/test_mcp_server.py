"""Tests for agent.mcp_server — FastMCP wrapper around ToolSpecs."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, patch

from agent.core.tools import ToolSpec, create_builtin_tools
from agent.mcp_server import EXCLUDED, build_server


def _expected_registered_names() -> set[str]:
    return {t.name for t in create_builtin_tools() if t.name not in EXCLUDED}


def test_build_server_registers_every_toolspec() -> None:
    server = build_server()
    registered = set(server.list_tool_names())
    assert registered == _expected_registered_names()


def test_build_server_preserves_tool_schema() -> None:
    server = build_server()
    schemas = server.tool_schemas()
    specs_by_name = {t.name: t for t in create_builtin_tools() if t.name not in EXCLUDED}

    assert set(schemas.keys()) == set(specs_by_name.keys())
    for name, spec in specs_by_name.items():
        registered = schemas[name]
        assert registered["description"] == spec.description
        assert registered["parameters"] == spec.parameters


def test_plan_tool_is_excluded() -> None:
    assert {"plan_tool", "bash", "read", "write", "edit"} <= EXCLUDED
    server = build_server()
    registered = set(server.list_tool_names())
    for excluded in {"plan_tool", "bash", "read", "write", "edit"}:
        assert excluded not in registered


def test_tool_invocation_dispatches_to_handler() -> None:
    mock_handler = AsyncMock(return_value=("result-output", True))
    fake_spec = ToolSpec(
        name="research",
        description="mocked research",
        parameters={"type": "object", "properties": {"query": {"type": "string"}}},
        handler=mock_handler,
    )
    real_tools = create_builtin_tools()
    patched_tools = [fake_spec if t.name == "research" else t for t in real_tools]

    with patch("agent.mcp_server.create_builtin_tools", return_value=patched_tools):
        server = build_server()
        result: Any = asyncio.run(server.invoke("research", {"query": "hello"}))

    mock_handler.assert_awaited_once_with({"query": "hello"})
    assert result == ("result-output", True)
