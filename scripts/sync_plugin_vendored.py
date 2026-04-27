#!/usr/bin/env python3
"""Re-sync the plugin's vendored library from agent/tools and agent/core/redact.

Run from the repo root:

    uv run python scripts/sync_plugin_vendored.py

What it does:
  1. Copies agent/tools/*.py (minus research/plan/private) → plugin/lib/ml_intern_lib/tools/
  2. Copies agent/core/redact.py → plugin/lib/ml_intern_lib/redact.py
  3. Rewrites imports inside the copies:
       from agent.tools.X            → from ml_intern_lib.tools.X
       from agent.core.session import Event
                                      → from ml_intern_lib.session_stub import Event
       from agent.core.tools import ToolSpec
                                      → from ml_intern_lib.tool_spec import ToolSpec
       from agent.core import telemetry
                                      → from ml_intern_lib import telemetry_stub as telemetry

This is idempotent — running it twice produces the same output.

Diff the result to confirm before committing.
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_TOOLS = REPO_ROOT / "agent" / "tools"
SRC_REDACT = REPO_ROOT / "agent" / "core" / "redact.py"
DST_PKG = REPO_ROOT / "plugin" / "lib" / "ml_intern_lib"
DST_TOOLS = DST_PKG / "tools"

# Tools NOT vendored (replaced by Claude Code natives or disabled upstream).
SKIP = {"research_tool.py", "plan_tool.py", "private_hf_repo_tools.py"}


def rewrite(src: str) -> str:
    src = re.sub(r"\bfrom agent\.tools\.", "from ml_intern_lib.tools.", src)
    src = re.sub(
        r"\bfrom agent\.core\.session import Event\b",
        "from ml_intern_lib.session_stub import Event",
        src,
    )
    src = re.sub(
        r"\bfrom agent\.core\.tools import ToolSpec\b",
        "from ml_intern_lib.tool_spec import ToolSpec",
        src,
    )
    src = re.sub(
        r"\bfrom agent\.core import telemetry\b",
        "from ml_intern_lib import telemetry_stub as telemetry",
        src,
    )
    return src


def main() -> int:
    if not SRC_TOOLS.is_dir():
        print(f"FATAL: {SRC_TOOLS} not found — run from repo root", file=sys.stderr)
        return 1
    if not DST_TOOLS.is_dir():
        print(f"FATAL: {DST_TOOLS} not found — plugin layout missing?", file=sys.stderr)
        return 1

    # Tools
    copied = 0
    for src in SRC_TOOLS.glob("*.py"):
        if src.name in SKIP:
            continue
        dst = DST_TOOLS / src.name
        text = src.read_text()
        rewritten = rewrite(text)
        dst.write_text(rewritten)
        copied += 1
    print(f"Vendored {copied} tool files to {DST_TOOLS}")

    # Redact
    redact_src = SRC_REDACT.read_text()
    (DST_PKG / "redact.py").write_text(rewrite(redact_src))
    print(f"Vendored redact.py to {DST_PKG}/redact.py")

    # Sanity check: no remaining `from agent.` imports
    leftover = []
    for f in DST_PKG.rglob("*.py"):
        for i, line in enumerate(f.read_text().splitlines(), start=1):
            if re.search(r"\bfrom agent\.|\bimport agent\.", line):
                leftover.append(f"{f}:{i}: {line.strip()}")
    if leftover:
        print("WARNING — leftover agent.* imports:", file=sys.stderr)
        for line in leftover:
            print(f"  {line}", file=sys.stderr)
        return 2

    print("OK — no leftover agent.* imports")
    return 0


if __name__ == "__main__":
    sys.exit(main())
