#!/bin/bash
set -euo pipefail

BENCHMARK="$1"
MODEL_TO_TRAIN="$2"
TASK_RUN_ID="$3"
NUM_HOURS="$4"

if [ -z "${RUN_ROOT:-}" ] || [ -z "${REPO_ROOT:-}" ] || [ -z "${PTB_DIR:-}" ]; then
    echo "RUN_ROOT, REPO_ROOT, and PTB_DIR must be exported" >&2
    exit 2
fi
if [ -z "${ML_INTERN_AGENT_MODEL:-}" ]; then
    echo "ML_INTERN_AGENT_MODEL must be exported" >&2
    exit 2
fi

DOCKER_IMAGE="${POST_TRAIN_BENCH_DOCKER_IMAGE:-registry.hpc-cluster-hopper.hpc.internal.huggingface.tech/library/posttrainbench:latest}"
HF_HOME_HOST="${HF_HOME:-$HOME/.cache/huggingface}"

safe_name() {
    python - "$1" <<'PY'
import sys
print(sys.argv[1].replace("/", "_").replace(":", "_").replace("[", "_").replace("]", "_"))
PY
}

MODEL_SAFE="$(safe_name "$MODEL_TO_TRAIN")"
AGENT_SAFE="$(safe_name "$ML_INTERN_AGENT_MODEL")"
METHOD_DIR="ml_intern_${AGENT_SAFE}_${NUM_HOURS}h"
EVAL_DIR="${RUN_ROOT}/results/${METHOD_DIR}/${BENCHMARK}_${MODEL_SAFE}_${TASK_RUN_ID}"
TMP_SUBDIR="/tmp/ml_intern_ptb_${BENCHMARK}_${MODEL_SAFE}_${TASK_RUN_ID}"
JOB_DIR="${TMP_SUBDIR}/job_dir"
JOB_TMP="${TMP_SUBDIR}/tmp"

rm -rf "$TMP_SUBDIR"
mkdir -p "$EVAL_DIR" "$JOB_DIR/task" "$JOB_TMP" "$HF_HOME_HOST"

exec > >(tee "$EVAL_DIR/output.log")
exec 2> >(tee "$EVAL_DIR/error.log" >&2)

echo "benchmark=$BENCHMARK"
echo "model_to_train=$MODEL_TO_TRAIN"
echo "agent_model=$ML_INTERN_AGENT_MODEL"
echo "task_run_id=$TASK_RUN_ID"
echo "num_hours=$NUM_HOURS"
echo "docker_image=$DOCKER_IMAGE"

cp "$PTB_DIR/src/eval/tasks/${BENCHMARK}/evaluate.py" "$JOB_DIR/task/"
if [ -d "$PTB_DIR/src/eval/tasks/${BENCHMARK}/evaluation_code" ]; then
    cp -r "$PTB_DIR/src/eval/tasks/${BENCHMARK}/evaluation_code" "$JOB_DIR/task/"
fi
cp -r "$PTB_DIR/src/eval/templates" "$JOB_DIR/task/"
if [ -d "$PTB_DIR/src/eval/tasks/${BENCHMARK}/task_context" ]; then
    cp -r "$PTB_DIR/src/eval/tasks/${BENCHMARK}/task_context/." "$JOB_DIR/task/"
fi

BENCHMARK_NAME="$(cat "$PTB_DIR/src/eval/tasks/${BENCHMARK}/benchmark.txt")"
PROMPT="$(
    cd "$PTB_DIR"
    POST_TRAIN_BENCH_PROMPT="${POST_TRAIN_BENCH_PROMPT:-prompt}" \
        python src/eval/general/get_prompt.py \
            --model-to-train "$MODEL_TO_TRAIN" \
            --benchmark-id "$BENCHMARK" \
            --num-hours "$NUM_HOURS" \
            --num-gpus 1 \
            --agent ml_intern
)"
printf '%s\n' "$PROMPT" > "$EVAL_DIR/prompt.txt"
export PROMPT

bash "$PTB_DIR/src/utils/create_timer.sh" "$NUM_HOURS" "$JOB_DIR/task/timer.sh"

CONTAINER_MOUNTS="${REPO_ROOT}:/ml-intern-src,${PTB_DIR}:/posttrainbench,${JOB_DIR}:/workspace,${JOB_TMP}:/tmp,${HF_HOME_HOST}:/hf-cache,${EVAL_DIR}:/result"
CONTAINER_ENV="HF_TOKEN,HUGGING_FACE_HUB_TOKEN,ANTHROPIC_API_KEY,OPENAI_API_KEY,GEMINI_API_KEY,INFERENCE_TOKEN,HF_BILL_TO,ML_INTERN_AGENT_MODEL,PROMPT"

run_in_container() {
    srun \
        --container-image="$DOCKER_IMAGE" \
        --container-mounts="$CONTAINER_MOUNTS" \
        --container-workdir=/workspace/task \
        --container-env="$CONTAINER_ENV" \
        "$@"
}

export HF_HOME=/hf-cache
SOLVE_OUT="$EVAL_DIR/solve_out.txt"

echo "================================"
echo "========= RUNNING TASK ========="
echo "================================"

START_TS="$(date --iso-8601=seconds)"
set +e
timeout --signal=TERM --kill-after=30s "$((NUM_HOURS * 60 + 5))m" \
    srun \
        --container-image="$DOCKER_IMAGE" \
        --container-mounts="$CONTAINER_MOUNTS" \
        --container-workdir=/workspace/task \
        --container-env="$CONTAINER_ENV" \
        bash -lc '
        set -euo pipefail
        export HF_HOME=/hf-cache
        export PYTHONNOUSERSITE=1
        cd /ml-intern-src
        uv pip install --system -e .
        cd /workspace/task
        ml-intern \
            --config /ml-intern-src/post_train_bench/ml_intern_posttrain_config.json \
            --model "$ML_INTERN_AGENT_MODEL" \
            --max-iterations -1 \
            "$PROMPT"
    ' > "$SOLVE_OUT" 2>&1
SOLVE_EXIT=$?
set -e
END_TS="$(date --iso-8601=seconds)"
python - "$START_TS" "$END_TS" "$EVAL_DIR/time_taken.txt" <<'PY'
import datetime as dt
import sys

start = dt.datetime.fromisoformat(sys.argv[1])
end = dt.datetime.fromisoformat(sys.argv[2])
seconds = int((end - start).total_seconds())
with open(sys.argv[3], "w") as f:
    f.write(f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}\n")
PY

echo "solve_exit=$SOLVE_EXIT"

if [ -d "$JOB_DIR/task/final_model" ]; then
    cp -r "$JOB_DIR/task/final_model" "$EVAL_DIR/final_model"
    rm -rf "$JOB_DIR/task/final_model"
fi

cp -r "$JOB_DIR/task" "$EVAL_DIR/task"

echo "========================================="
echo "=== RUNNING CONTAMINATION JUDGE ========"
echo "========================================="

JUDGE_PROMPT="$(
    cd "$PTB_DIR"
    python src/disallowed_usage_judge/get_judge_prompt.py \
        --benchmark "$BENCHMARK_NAME" \
        --model "$MODEL_TO_TRAIN"
)"
printf '%s\n' "$JUDGE_PROMPT" > "$EVAL_DIR/judge_prompt.txt"

set +e
run_in_container python /ml-intern-src/post_train_bench/run_judge.py \
    --task-dir /result/task \
    --prompt-file /result/judge_prompt.txt \
    --output-dir /result > "$EVAL_DIR/judge_output.txt" 2>&1
JUDGE_EXIT=$?
set -e
echo "judge_exit=$JUDGE_EXIT"

echo "================================"
echo "========= EVALUATING ==========="
echo "================================"

run_evaluation() {
    local max_tokens_arg="$1"
    local eval_num="$2"
    set +e
    run_in_container bash -lc "
        set -euo pipefail
        export HF_HOME=/hf-cache
        export PYTHONNOUSERSITE=1
        export VLLM_API_KEY=inspectai
        cd /posttrainbench/src/eval/tasks/${BENCHMARK}
        python evaluate.py \
            --model-path /result/final_model \
            --templates-dir ../../../../src/eval/templates \
            --limit -1 \
            ${max_tokens_arg} \
            --json-output-file /result/metrics.json
    " > "$EVAL_DIR/final_eval_${eval_num}.txt" 2>&1
    local status=$?
    set -e
    return "$status"
}

run_evaluation_with_retry() {
    local max_retries="$1"
    local max_tokens_arg="$2"
    local attempt
    for ((attempt=1; attempt<=max_retries; attempt++)); do
        if [ -f "$EVAL_DIR/metrics.json" ]; then
            return 0
        fi
        EVAL_COUNTER=$((EVAL_COUNTER + 1))
        echo "Evaluation attempt $EVAL_COUNTER (phase attempt $attempt of $max_retries)"
        run_evaluation "$max_tokens_arg" "$EVAL_COUNTER" || true
        if [ -f "$EVAL_DIR/metrics.json" ]; then
            return 0
        fi
    done
    return 1
}

EVAL_COUNTER=0
run_evaluation_with_retry 4 ""

case "$BENCHMARK" in
    aime2025|bfcl|gpqamain) MAX_TOKENS_ARG="--max-tokens 12000" ;;
    gsm8k|humaneval) MAX_TOKENS_ARG="--max-tokens 3000" ;;
    arenahardwriting|healthbench) MAX_TOKENS_ARG="--max-new-tokens 12288" ;;
    *) MAX_TOKENS_ARG="" ;;
esac
run_evaluation_with_retry 3 "$MAX_TOKENS_ARG"

case "$BENCHMARK" in
    aime2025|bfcl|gpqamain) MAX_TOKENS_ARG="--max-tokens 8000" ;;
    gsm8k|humaneval) MAX_TOKENS_ARG="--max-tokens 2000" ;;
    arenahardwriting|healthbench) MAX_TOKENS_ARG="--max-new-tokens 8192" ;;
    *) MAX_TOKENS_ARG="" ;;
esac
run_evaluation_with_retry 2 "$MAX_TOKENS_ARG"

python post_train_bench/collect_artifacts.py \
    --run-root "$RUN_ROOT" \
    --eval-dir "$EVAL_DIR" \
    --benchmark "$BENCHMARK" \
    --model-to-train "$MODEL_TO_TRAIN" \
    --task-run-id "$TASK_RUN_ID" \
    --method "$METHOD_DIR"

rm -rf "$TMP_SUBDIR"

if [ "$SOLVE_EXIT" -ne 0 ]; then
    exit "$SOLVE_EXIT"
fi
