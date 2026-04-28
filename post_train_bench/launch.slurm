#!/bin/bash
#SBATCH --job-name=ml-intern-ptb
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=hopper-prod
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=14:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

if [ -z "${RUN_ROOT:-}" ]; then
    echo "RUN_ROOT is required" >&2
    exit 2
fi
if [ -z "${MATRIX_FILE:-}" ]; then
    echo "MATRIX_FILE is required" >&2
    exit 2
fi
if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "launch.slurm must be submitted as an array job" >&2
    exit 2
fi

mkdir -p "${RUN_ROOT}/slurm"
exec >"${RUN_ROOT}/slurm/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
exec 2>"${RUN_ROOT}/slurm/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"

module load cuda/12.9 || true
set -x
cd "$REPO_ROOT"

readarray -t ROW < <(
    python - "$MATRIX_FILE" "$SLURM_ARRAY_TASK_ID" <<'PY'
import json
import sys
from pathlib import Path

matrix_path = Path(sys.argv[1])
task_id = int(sys.argv[2])
rows = [json.loads(line) for line in matrix_path.read_text().splitlines() if line.strip()]
row = rows[task_id]
print(row["benchmark"])
print(row["model_to_train"])
print(row["num_hours"])
PY
)

BENCHMARK="${ROW[0]}"
MODEL_TO_TRAIN="${ROW[1]}"
NUM_HOURS="${ROW[2]}"

bash post_train_bench/run_task_docker.sh \
    "${BENCHMARK}" \
    "${MODEL_TO_TRAIN}" \
    "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" \
    "${NUM_HOURS}"
