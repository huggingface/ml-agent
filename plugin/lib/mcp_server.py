"""
ml-intern MCP server, plugin edition.

Same shape as packages/mcp_server/server.py in the upstream repo, but imports
from the vendored `ml_intern_lib` (under plugin/lib/) so the plugin is
self-contained — users don't need the full ml-intern repo.

The `research` and `plan_tool` tools are intentionally NOT exposed:
  research → replaced by the plugin's research subagent (agents/research.md)
  plan_tool → replaced by Claude Code's built-in TodoWrite

Run via the plugin's `.mcp.json`. Not intended to be invoked manually.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any, Awaitable, Callable

# When launched by Claude Code's plugin loader, ${CLAUDE_PLUGIN_ROOT} points at
# the installed plugin directory and `command: uv` runs us inside our own venv.
# Make sure the vendored ml_intern_lib is importable regardless of CWD.
_LIB = Path(__file__).resolve().parent
if str(_LIB) not in sys.path:
    sys.path.insert(0, str(_LIB))

from mcp import types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

from ml_intern_lib.tools.dataset_tools import (
    HF_INSPECT_DATASET_TOOL_SPEC,
    hf_inspect_dataset_handler,
)
from ml_intern_lib.tools.docs_tools import (
    EXPLORE_HF_DOCS_TOOL_SPEC,
    HF_DOCS_FETCH_TOOL_SPEC,
    explore_hf_docs_handler,
    hf_docs_fetch_handler,
)
from ml_intern_lib.tools.github_find_examples import (
    GITHUB_FIND_EXAMPLES_TOOL_SPEC,
    github_find_examples_handler,
)
from ml_intern_lib.tools.github_list_repos import (
    GITHUB_LIST_REPOS_TOOL_SPEC,
    github_list_repos_handler,
)
from ml_intern_lib.tools.github_read_file import (
    GITHUB_READ_FILE_TOOL_SPEC,
    github_read_file_handler,
)
from ml_intern_lib.tools.hf_repo_files_tool import (
    HF_REPO_FILES_TOOL_SPEC,
    hf_repo_files_handler,
)
from ml_intern_lib.tools.hf_repo_git_tool import (
    HF_REPO_GIT_TOOL_SPEC,
    hf_repo_git_handler,
)
from ml_intern_lib.tools.jobs_tool import HF_JOBS_TOOL_SPEC, hf_jobs_handler
from ml_intern_lib.tools.papers_tool import HF_PAPERS_TOOL_SPEC, hf_papers_handler
from ml_intern_lib.tools.sandbox_tool import get_sandbox_tools

logger = logging.getLogger(__name__)

_TOOL_SPECS: list[tuple[dict[str, Any], Callable[..., Awaitable[tuple[str, bool]]]]] = [
    (EXPLORE_HF_DOCS_TOOL_SPEC, explore_hf_docs_handler),
    (HF_DOCS_FETCH_TOOL_SPEC, hf_docs_fetch_handler),
    (HF_PAPERS_TOOL_SPEC, hf_papers_handler),
    (HF_INSPECT_DATASET_TOOL_SPEC, hf_inspect_dataset_handler),
    (HF_JOBS_TOOL_SPEC, hf_jobs_handler),
    (HF_REPO_FILES_TOOL_SPEC, hf_repo_files_handler),
    (HF_REPO_GIT_TOOL_SPEC, hf_repo_git_handler),
    (GITHUB_FIND_EXAMPLES_TOOL_SPEC, github_find_examples_handler),
    (GITHUB_LIST_REPOS_TOOL_SPEC, github_list_repos_handler),
    (GITHUB_READ_FILE_TOOL_SPEC, github_read_file_handler),
]

_REGISTRY: dict[str, tuple[types.Tool, Callable[..., Awaitable[tuple[str, bool]]]]] = {}


def _build_registry() -> None:
    for spec, handler in _TOOL_SPECS:
        tool = types.Tool(
            name=spec["name"],
            description=spec["description"],
            inputSchema=spec["parameters"],
        )
        _REGISTRY[spec["name"]] = (tool, handler)

    local_mode = os.environ.get("ML_INTERN_LOCAL_MODE", "").lower() in ("1", "true", "yes")
    if local_mode:
        from ml_intern_lib.tools.local_tools import get_local_tools
        sandbox_specs = get_local_tools()
    else:
        sandbox_specs = get_sandbox_tools()

    for tool_spec in sandbox_specs:
        tool = types.Tool(
            name=tool_spec.name,
            description=tool_spec.description,
            inputSchema=tool_spec.parameters,
        )
        _REGISTRY[tool_spec.name] = (tool, tool_spec.handler)


server: Server = Server("ml-intern-tools")


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [tool for tool, _ in _REGISTRY.values()]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
    entry = _REGISTRY.get(name)
    if entry is None:
        raise ValueError(f"Unknown tool: {name}")
    _tool, handler = entry
    output, ok = await handler(arguments or {})
    if not ok:
        raise RuntimeError(output)
    return [types.TextContent(type="text", text=output)]


async def _amain() -> None:
    _build_registry()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(_amain())
