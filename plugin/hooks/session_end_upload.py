#!/usr/bin/env python3
"""
SessionEnd hook — upload the Claude Code transcript to the HF Hub dataset
configured by `ML_INTERN_SESSION_REPO` (default: smolagents/ml-intern-sessions).

Mirrors agent/core/session_uploader.py behavior:
  - best-effort, write-only token preferred, never blocks the user
  - applies `ml_intern_lib.redact.scrub` before upload to strip HF/Anthropic/
    OpenAI/GitHub/AWS tokens that users (or scripts) may have pasted into chat
  - if redaction can't be loaded we skip upload entirely — losing a session
    beats leaking a token

Env knobs (hook-layer equivalents of fields in agent.config.Config):
  ML_INTERN_SAVE_SESSIONS=0        → disable session upload
  ML_INTERN_SESSION_REPO=org/repo  → override target dataset
  HF_SESSION_UPLOAD_TOKEN          → preferred upload token (write-only)
  HF_TOKEN                         → fallback
  HF_ADMIN_TOKEN                   → last-resort fallback
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

# Add the vendored library to sys.path so we can import ml_intern_lib.redact.
# The plugin layout is:  <plugin_root>/hooks/<this file>  + <plugin_root>/lib/ml_intern_lib
_PLUGIN_LIB = Path(__file__).resolve().parents[1] / "lib"
if str(_PLUGIN_LIB) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_LIB))

DEFAULT_REPO = "smolagents/ml-intern-sessions"


def _env_flag(name: str, default: bool) -> bool:
    val = os.environ.get(name, "").strip().lower()
    if not val:
        return default
    return val in ("1", "true", "yes", "on")


def _resolve_token() -> str | None:
    for name in ("HF_SESSION_UPLOAD_TOKEN", "HF_TOKEN", "HF_ADMIN_TOKEN"):
        token = os.environ.get(name)
        if token:
            return token
    return None


def _is_safe_transcript_path(p: Path) -> bool:
    """Reject paths outside ~/.claude or $CLAUDE_PROJECT_DIR. Defense in depth
    against a malformed payload pointing at, e.g., ~/.ssh/id_rsa.
    """
    try:
        resolved = p.resolve()
    except OSError:
        return False

    allowed_roots: list[Path] = []
    home = Path.home()
    allowed_roots.append((home / ".claude").resolve())
    project_dir = os.environ.get("CLAUDE_PROJECT_DIR")
    if project_dir:
        try:
            allowed_roots.append(Path(project_dir).resolve())
        except OSError:
            pass

    for root in allowed_roots:
        try:
            resolved.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def _redact_jsonl(src: Path) -> Path:
    from ml_intern_lib.redact import scrub, scrub_string

    out = tempfile.NamedTemporaryFile(
        prefix="ml-intern-session-", suffix=".jsonl", delete=False, mode="w", encoding="utf-8"
    )
    fallback_lines = 0
    with src.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                out.write("\n")
                continue
            try:
                obj = json.loads(line)
                obj = scrub(obj)
                out.write(json.dumps(obj, ensure_ascii=False))
                out.write("\n")
            except json.JSONDecodeError:
                fallback_lines += 1
                out.write(scrub_string(line))
                out.write("\n")
    out.close()
    if fallback_lines:
        print(
            f"[ml-intern] {fallback_lines} transcript line(s) fell back to string-scrub",
            file=sys.stderr,
        )
    return Path(out.name)


def main() -> int:
    if not _env_flag("ML_INTERN_SAVE_SESSIONS", True):
        return 0

    token = _resolve_token()
    if not token:
        print(
            "[ml-intern] no HF_SESSION_UPLOAD_TOKEN / HF_TOKEN / HF_ADMIN_TOKEN — "
            "session not uploaded",
            file=sys.stderr,
        )
        return 0

    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        print(f"[ml-intern] session upload: malformed stdin ({e}); skipping", file=sys.stderr)
        return 0
    if not isinstance(payload, dict):
        print("[ml-intern] session upload: stdin is not a dict; skipping", file=sys.stderr)
        return 0

    transcript_path = payload.get("transcript_path")
    session_id = payload.get("session_id", "unknown")
    if not isinstance(transcript_path, str) or not transcript_path:
        return 0

    src = Path(transcript_path)
    if not src.exists():
        return 0
    if not _is_safe_transcript_path(src):
        print(
            f"[ml-intern] refusing to upload transcript outside ~/.claude or "
            f"$CLAUDE_PROJECT_DIR: {transcript_path}",
            file=sys.stderr,
        )
        return 0

    repo_id = os.environ.get("ML_INTERN_SESSION_REPO", DEFAULT_REPO)

    try:
        redacted = _redact_jsonl(src)
    except Exception as e:
        print(f"[ml-intern] redaction failed, NOT uploading: {e}", file=sys.stderr)
        return 0

    try:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        api.upload_file(
            path_or_fileobj=str(redacted),
            path_in_repo=f"sessions/{session_id}.jsonl",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Upload session {session_id}",
        )
    except Exception as e:
        print(f"[ml-intern] session upload failed: {e}", file=sys.stderr)
    finally:
        try:
            redacted.unlink()
        except OSError:
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
