#!/bin/bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  ML_INTERN_AGENT_MODEL=anthropic/claude-opus-4-6 \
    bash post_train_bench/submit_eval_set.sh smoke

  ML_INTERN_AGENT_MODEL=anthropic/claude-opus-4-6 \
    bash post_train_bench/submit_eval_set.sh full --dry-run

Modes:
  smoke  Submit one short validation job.
  full   Submit the full 4-model x 7-benchmark matrix. This is documented for manual use.

Options:
  --dry-run  Create metadata and matrix, print the sbatch command, do not submit.

Environment:
  ML_INTERN_AGENT_MODEL        Required intern model, used literally in runs/<model>/<run_id>.
  POST_TRAIN_BENCH_DIR         Default: scratch/PostTrainBench
  POST_TRAIN_BENCH_DOCKER_IMAGE
                               Default: registry.hpc-cluster-hopper.hpc.internal.huggingface.tech/library/posttrainbench:latest
  POST_TRAIN_BENCH_RUN_ID      Optional explicit run id.
EOF
}

MODE="${1:-}"
if [ -z "$MODE" ] || [ "$MODE" = "-h" ] || [ "$MODE" = "--help" ]; then
    usage
    exit 0
fi
shift || true

DRY_RUN=0
while [ "$#" -gt 0 ]; do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

if [ -z "${ML_INTERN_AGENT_MODEL:-}" ]; then
    echo "ML_INTERN_AGENT_MODEL is required" >&2
    exit 2
fi

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

PTB_DIR="${POST_TRAIN_BENCH_DIR:-scratch/PostTrainBench}"
if [ ! -d "$PTB_DIR/src/eval/tasks" ]; then
    echo "PostTrainBench repo not found at $PTB_DIR" >&2
    exit 2
fi
PTB_DIR="$(cd "$PTB_DIR" && pwd)"

SHORT_COMMIT="$(git rev-parse --short=12 HEAD)"
RUN_ID="${POST_TRAIN_BENCH_RUN_ID:-$(date -u +%Y-%m-%d_%H-%M)_${SHORT_COMMIT}}"
RUN_ROOT="${REPO_ROOT}/post_train_bench/runs/${ML_INTERN_AGENT_MODEL}/${RUN_ID}"

if [ -e "$RUN_ROOT" ]; then
    echo "Run directory already exists: $RUN_ROOT" >&2
    exit 2
fi

mkdir -p "$RUN_ROOT"/{slurm,results,artifacts,env}

MATRIX_FILE="$RUN_ROOT/matrix.jsonl"
case "$MODE" in
    smoke)
        python - "$MATRIX_FILE" <<'PY'
import json
import sys
from pathlib import Path

rows = [{"benchmark": "gsm8k", "model_to_train": "Qwen/Qwen3-1.7B-Base", "num_hours": 1}]
Path(sys.argv[1]).write_text("\n".join(json.dumps(row) for row in rows) + "\n")
PY
        ;;
    full)
        python - "$MATRIX_FILE" <<'PY'
import json
import sys
from pathlib import Path

models = [
    "google/gemma-3-4b-pt",
    "Qwen/Qwen3-4B-Base",
    "Qwen/Qwen3-1.7B-Base",
    "HuggingFaceTB/SmolLM3-3B-Base",
]
benchmarks = [
    "aime2025",
    "arenahardwriting",
    "bfcl",
    "gpqamain",
    "gsm8k",
    "humaneval",
    "healthbench",
]
rows = [
    {"benchmark": benchmark, "model_to_train": model, "num_hours": 10}
    for model in models
    for benchmark in benchmarks
]
Path(sys.argv[1]).write_text("\n".join(json.dumps(row) for row in rows) + "\n")
PY
        ;;
    *)
        echo "Unknown mode: $MODE" >&2
        usage >&2
        exit 2
        ;;
esac

MATRIX_COUNT="$(wc -l < "$MATRIX_FILE" | tr -d ' ')"
DOCKER_IMAGE="${POST_TRAIN_BENCH_DOCKER_IMAGE:-registry.hpc-cluster-hopper.hpc.internal.huggingface.tech/library/posttrainbench:latest}"
export RUN_ID MODE DOCKER_IMAGE PTB_DIR MATRIX_FILE MATRIX_COUNT

python - "$RUN_ROOT/run_metadata.json" <<'PY'
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

def git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, text=True, capture_output=True).stdout.strip()

status = git("status", "--short")
metadata = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "run_id": os.environ["RUN_ID"],
    "mode": os.environ["MODE"],
    "ml_intern_agent_model": os.environ["ML_INTERN_AGENT_MODEL"],
    "ml_intern_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
    "ml_intern_commit": git("rev-parse", "HEAD"),
    "ml_intern_short_commit": git("rev-parse", "--short=12", "HEAD"),
    "ml_intern_status_short": status,
    "dirty_worktree": bool(status),
    "docker_image": os.environ["DOCKER_IMAGE"],
    "post_train_bench_dir": os.environ["PTB_DIR"],
    "matrix_file": os.environ["MATRIX_FILE"],
    "matrix_count": int(os.environ["MATRIX_COUNT"]),
}
Path(sys.argv[1]).write_text(json.dumps(metadata, indent=2) + "\n")
PY

env | sort > "$RUN_ROOT/env/submit_env.txt"

SBATCH_CMD=(
    sbatch
    "--array=0-$((MATRIX_COUNT - 1))"
    "--export=ALL,RUN_ROOT=${RUN_ROOT},MATRIX_FILE=${MATRIX_FILE},PTB_DIR=${PTB_DIR},REPO_ROOT=${REPO_ROOT},POST_TRAIN_BENCH_DOCKER_IMAGE=${DOCKER_IMAGE},RUN_ID=${RUN_ID}"
    post_train_bench/launch.slurm
)

printf '%q ' "${SBATCH_CMD[@]}" > "$RUN_ROOT/sbatch_command.txt"
printf '\n' >> "$RUN_ROOT/sbatch_command.txt"

echo "Run root: $RUN_ROOT"
echo "Matrix rows: $MATRIX_COUNT"
echo "Command: $(cat "$RUN_ROOT/sbatch_command.txt")"

if [ "$DRY_RUN" -eq 1 ]; then
    echo "Dry run only; not submitting."
    exit 0
fi

"${SBATCH_CMD[@]}" | tee "$RUN_ROOT/sbatch_output.txt"
