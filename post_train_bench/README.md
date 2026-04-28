# PostTrainBench Evaluation

This directory contains the Slurm/Docker integration for evaluating `ml-intern`
on PostTrainBench with local H100 compute.

All run outputs are written under:

```bash
post_train_bench/runs/{ML_INTERN_AGENT_MODEL}/{RUN_ID}/
```

`ML_INTERN_AGENT_MODEL` is used literally as a path. For example,
`anthropic/claude-opus-4-6` writes under
`post_train_bench/runs/anthropic/claude-opus-4-6/...`.

`RUN_ID` is generated once per evaluation set as:

```text
YYYY-MM-DD_HH-MM_{short_commit}
```

## Prerequisites

- The PostTrainBench repo exists at `scratch/PostTrainBench`.
- Slurm with Pyxis container support is available.
- The current checkout contains the `ml-intern` commit you want to evaluate.
- Required tokens are exported:

```bash
export HF_TOKEN=hf_...
export ANTHROPIC_API_KEY=sk-ant-...   # or the provider key for ML_INTERN_AGENT_MODEL
export OPENAI_API_KEY=sk-...          # used by Arena/Health evals and optional judge
export ML_INTERN_AGENT_MODEL=anthropic/claude-opus-4-6
```

The default Docker image is:

```bash
registry.hpc-cluster-hopper.hpc.internal.huggingface.tech/library/posttrainbench:latest
```

Override it with:

```bash
export POST_TRAIN_BENCH_DOCKER_IMAGE=registry.../posttrainbench:your-tag
```

## Smoke Test

Submit one short GSM8K / Qwen3-1.7B job:

```bash
bash post_train_bench/submit_eval_set.sh smoke
```

To check paths and metadata without submitting:

```bash
bash post_train_bench/submit_eval_set.sh smoke --dry-run
```

Monitor with:

```bash
squeue -u "$USER"
tail -f post_train_bench/runs/${ML_INTERN_AGENT_MODEL}/*/slurm/*.out
```

After completion, inspect:

```bash
find post_train_bench/runs/${ML_INTERN_AGENT_MODEL} -maxdepth 4 -type f | sort
```

Important files:

- `run_metadata.json`: source commit, Docker image, matrix size, dirty status.
- `matrix.jsonl`: benchmark/model rows for the Slurm array.
- `results/.../solve_out.txt`: raw agent trace.
- `results/.../task/session_logs/*.json`: local `ml-intern` trajectory logs.
- `results/.../metrics.json`: per-run benchmark metrics.
- `artifacts/.../manifest.json`: checksums and copied artifact summary.

## Full Matrix

Do not run this until the smoke test succeeds. This command submits the full
4-model x 7-benchmark matrix with 10 agent hours per job:

```bash
bash post_train_bench/submit_eval_set.sh full
```

To inspect the generated full matrix without submitting:

```bash
bash post_train_bench/submit_eval_set.sh full --dry-run
```

## Rebuilding The Docker Image

The checked-in `post_train_bench/Dockerfile` mirrors the Dockerfile from the
`posttrain-bench` integration branch and pins the PostTrainBench-compatible ML
stack.

Build locally:

```bash
docker build -t registry.hpc-cluster-hopper.hpc.internal.huggingface.tech/library/posttrainbench:latest \
  -f post_train_bench/Dockerfile .
```

Push to the cluster registry:

```bash
docker push registry.hpc-cluster-hopper.hpc.internal.huggingface.tech/library/posttrainbench:latest
```

Use a custom tag when testing dependency changes:

```bash
docker build -t registry.hpc-cluster-hopper.hpc.internal.huggingface.tech/library/posttrainbench:ptb-test \
  -f post_train_bench/Dockerfile .
docker push registry.hpc-cluster-hopper.hpc.internal.huggingface.tech/library/posttrainbench:ptb-test
export POST_TRAIN_BENCH_DOCKER_IMAGE=registry.hpc-cluster-hopper.hpc.internal.huggingface.tech/library/posttrainbench:ptb-test
```

You do not need to rebuild the image just to evaluate a different `ml-intern`
commit. The Slurm job mounts the current checkout into the container and
installs it at runtime.

## Notes

- `post_train_bench/runs/` is ignored by Git.
- The run metadata records whether the source worktree was dirty at submission
  time. Commit intended changes before running official evaluations.
- The optional judge writes `judge not run: ...` if `OPENAI_API_KEY` is not set
  or the judge API call fails.
