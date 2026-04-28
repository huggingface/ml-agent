#!/usr/bin/env python3
"""Run a lightweight contamination/model-use judge for a PostTrainBench task."""

import argparse
import os
from pathlib import Path

MAX_CHARS = 220_000
DEFAULT_MODEL = "gpt-5.1-codex"


def iter_files(task_dir: Path):
    skip_dirs = {
        "final_model",
        "templates",
        "evaluation_code",
        "__pycache__",
        ".git",
        "session_logs",
    }
    skip_names = {
        "evaluate.py",
        "timer.sh",
    }
    for path in sorted(task_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(task_dir)
        if any(part in skip_dirs for part in rel.parts):
            continue
        if path.name in skip_names:
            continue
        if path.stat().st_size > 2_000_000:
            continue
        yield path, rel


def collect_code(task_dir: Path) -> str:
    chunks = []
    remaining = MAX_CHARS
    for path, rel in iter_files(task_dir):
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        block = f"\n\n--- FILE: {rel} ---\n{content}"
        if len(block) > remaining:
            chunks.append(block[:remaining])
            break
        chunks.append(block)
        remaining -= len(block)
        if remaining <= 0:
            break
    return "".join(chunks)


def write_fallback(output_dir: Path, reason: str) -> None:
    (output_dir / "contamination_judgement.txt").write_text(
        f"judge not run: {reason}\n", encoding="utf-8"
    )
    (output_dir / "disallowed_model_judgement.txt").write_text(
        f"judge not run: {reason}\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default=os.environ.get("PTB_JUDGE_MODEL", DEFAULT_MODEL))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    task_dir = Path(args.task_dir)
    prompt = Path(args.prompt_file).read_text(encoding="utf-8")
    code = collect_code(task_dir)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        write_fallback(output_dir, "OPENAI_API_KEY is not set")
        return 0

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model=args.model,
            input=(
                f"{prompt}\n\n"
                "Inspect the following student-created files. Do not penalize "
                "the copied benchmark harness files that were excluded from this bundle.\n"
                f"{code}\n"
            ),
        )
        text = getattr(response, "output_text", "") or ""
    except Exception as exc:
        write_fallback(output_dir, f"{type(exc).__name__}: {exc}")
        return 0

    (output_dir / "judge_raw_response.txt").write_text(text, encoding="utf-8")

    lower = text.lower()
    if "contamination detected" in lower:
        contamination = "contamination detected"
    elif "no contamination detected" in lower:
        contamination = "no contamination detected"
    else:
        contamination = "judge unclear"

    if "disallowed use detected" in lower:
        model_use = "disallowed use detected"
    elif "only allowed use detected" in lower:
        model_use = "only allowed use detected"
    else:
        model_use = "judge unclear"

    (output_dir / "contamination_judgement.txt").write_text(
        contamination + "\n", encoding="utf-8"
    )
    (output_dir / "disallowed_model_judgement.txt").write_text(
        model_use + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
