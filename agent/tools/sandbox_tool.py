"""
Sandbox tools — expose the Sandbox client as agent tools.

5 tools total:
  sandbox_create — explicit sandbox creation (requires approval)
  bash, read, write, edit — operations on the sandbox

If any operation tool is called without an active sandbox,
a cpu-basic sandbox is auto-created (no approval needed).
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from huggingface_hub import HfApi, SpaceHardware

from agent.core.session import Event
from agent.tools.sandbox_client import Sandbox

# ── Tool name mapping (short agent names → Sandbox client names) ──────


async def _ensure_sandbox(
    session: Any, hardware: str = "cpu-basic", **create_kwargs
) -> tuple[Sandbox | None, str | None]:
    """
    Ensure a sandbox exists on the session. Auto-creates with given hardware if needed.

    Returns:
        (sandbox, error_message) — one will be None.
    """
    if session and getattr(session, "sandbox", None):
        return session.sandbox, None

    if not session:
        return None, "No session available."

    token = os.environ.get("HF_TOKEN")
    if not token:
        return None, "HF_TOKEN environment variable not set. Cannot create sandbox."

    api = HfApi(token=token)
    user_info = api.whoami()
    owner = user_info.get("name", user_info.get("user", ""))
    if not owner:
        return None, "Could not determine HF username from token."

    await session.send_event(
        Event(
            event_type="tool_log",
            data={
                "tool": "sandbox",
                "log": f"Auto-creating sandbox for {owner} ({hardware})...",
            },
        )
    )

    kwargs = {"owner": owner, "hardware": hardware, "token": token, **create_kwargs}
    sb = await asyncio.to_thread(Sandbox.create, **kwargs)
    session.sandbox = sb

    await session.send_event(
        Event(
            event_type="tool_log",
            data={"tool": "sandbox", "log": f"Sandbox ready: {sb.space_id} ({sb.url})"},
        )
    )

    return sb, None


# ── sandbox_create tool ──────────────────────────────────────────────

SANDBOX_CREATE_TOOL_SPEC = {
    "name": "sandbox_create",
    "description": (
        "Create a persistent remote Linux sandbox on HF Spaces for interactive development.\n"
        "YOU MUST DO THIS BEFORE USING bash/read/write/edit tools.\n"
        "\n"
        "Spins up a new sandbox with a given hardware tier where you can run commands, read/write/edit files, "
        "install packages, and debug iteratively. The sandbox persists across tool calls within the session."
        "\n"
        "You can choose from the following hardware tiers (GPU is required for model development or other tasks that benefit from and utilize the GPU): "
        + ", ".join([e.value for e in SpaceHardware])
        + ".\n"
        "Use sandbox for: iterative development, debugging, multi-step workflows, testing code.\n"
        "Use hf_jobs instead for: one-shot batch runs, scheduled tasks, fire-and-forget training.\n"
    ),
    "parameters": {
        "type": "object",
        "required": [],
        "additionalProperties": False,
        "properties": {
            "hardware": {
                "type": "string",
                "enum": [e.value for e in SpaceHardware],
                "description": "Hardware tier for the sandbox (default: cpu-basic)",
            },
            "private": {
                "type": "boolean",
                "description": "If true, create a private Space",
            },
        },
    },
}


async def sandbox_create_handler(
    args: dict[str, Any], session: Any = None
) -> tuple[str, bool]:
    """Handle sandbox_create tool calls."""
    # If sandbox already exists, return its info
    if session and getattr(session, "sandbox", None):
        sb = session.sandbox
        return (
            f"Sandbox already active: {sb.space_id}\n"
            f"URL: {sb.url}\n"
            f"Use bash/read/write/edit to interact with it."
        ), True

    hardware = args.get("hardware", "cpu-basic")
    create_kwargs = {}
    if "private" in args:
        create_kwargs["private"] = args["private"]

    try:
        sb, error = await _ensure_sandbox(session, hardware=hardware, **create_kwargs)
    except Exception as e:
        return f"Failed to create sandbox: {e}", False

    if error:
        return error, False

    return (
        f"Sandbox created: {sb.space_id}\n"
        f"URL: {sb.url}\n"
        f"Hardware: {hardware}\n"
        f"Use bash/read/write/edit to interact with it."
    ), True


def _make_tool_handler(sandbox_tool_name: str):
    """Factory: create a handler for a sandbox operation tool."""

    async def handler(args: dict[str, Any], session: Any = None) -> tuple[str, bool]:
        # Auto-create sandbox if not present
        try:
            sb, error = await _ensure_sandbox(session)
        except Exception as e:
            return f"Failed to auto-create sandbox: {e}", False

        if error:
            return error, False

        try:
            result = await asyncio.to_thread(sb.call_tool, sandbox_tool_name, args)
            if result.success:
                return result.output or "(no output)", True
            else:
                error_msg = result.error or "Unknown error"
                output = result.output
                if output:
                    return f"{output}\n\nERROR: {error_msg}", False
                return f"ERROR: {error_msg}", False
        except Exception as e:
            return f"Sandbox operation failed: {e}", False

    return handler


def get_sandbox_tools():
    """Return all 5 sandbox ToolSpecs (sandbox_create + 4 operation tools)."""
    from agent.core.tools import ToolSpec

    tools = []

    # sandbox_create (explicit creation, requires approval)
    tools.append(
        ToolSpec(
            name=SANDBOX_CREATE_TOOL_SPEC["name"],
            description=SANDBOX_CREATE_TOOL_SPEC["description"],
            parameters=SANDBOX_CREATE_TOOL_SPEC["parameters"],
            handler=sandbox_create_handler,
        )
    )

    # Operation tools (auto-execute, no approval needed)
    for name in Sandbox.TOOLS.keys():
        spec = Sandbox.TOOLS[name]
        tools.append(
            ToolSpec(
                name=name,
                description=spec["description"],
                parameters=spec["parameters"],
                handler=_make_tool_handler(name),
            )
        )

    return tools
