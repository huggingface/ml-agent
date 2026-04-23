"""FastMCP server exposing ml-intern's built-in ToolSpecs as MCP tools."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

from fastmcp import FastMCP
from fastmcp.tools import FunctionTool

from agent.core.tools import ToolSpec, create_builtin_tools

logger = logging.getLogger(__name__)

EXCLUDED: frozenset[str] = frozenset({"plan_tool", "bash", "read", "write", "edit"})


@dataclass
class _Server:
    mcp: FastMCP
    specs: dict[str, ToolSpec]

    def list_tool_names(self) -> list[str]:
        return list(self.specs.keys())

    def tool_schemas(self) -> dict[str, dict[str, Any]]:
        return {
            name: {"description": spec.description, "parameters": spec.parameters}
            for name, spec in self.specs.items()
        }

    async def invoke(self, name: str, arguments: dict[str, Any]) -> Any:
        spec = self.specs[name]
        if spec.handler is None:
            raise RuntimeError(f"Tool '{name}' has no handler")
        return await spec.handler(arguments)

    def run(self) -> None:
        self.mcp.run()


def _make_handler(spec: ToolSpec):
    async def handler(**kwargs: Any) -> Any:
        if spec.handler is None:
            raise RuntimeError(f"Tool '{spec.name}' has no handler")
        return await spec.handler(kwargs)

    handler.__name__ = spec.name
    return handler


def build_server() -> _Server:
    mcp = FastMCP("ml-intern")
    specs: dict[str, ToolSpec] = {}
    for spec in create_builtin_tools():
        if spec.name in EXCLUDED:
            continue
        specs[spec.name] = spec
        mcp.add_tool(
            FunctionTool(
                name=spec.name,
                description=spec.description,
                parameters=spec.parameters,
                fn=_make_handler(spec),
            )
        )
    logger.info("ml-intern MCP server registered %d tools", len(specs))
    return _Server(mcp=mcp, specs=specs)


def main() -> None:
    level = os.environ.get("ML_INTERN_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    build_server().run()


if __name__ == "__main__":
    main()
