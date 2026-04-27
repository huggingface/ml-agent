# ml-intern (Claude Code plugin)

Brings the [ml-intern](https://github.com/huggingface/ml-intern) ML engineering experience to Claude Code in any repository. Research-first methodology, HF dataset/paper/jobs/sandbox tools, content-aware approval policy.

## Install

### Via marketplace

```
/plugin marketplace add huggingface/ml-intern
/plugin install ml-intern@ml-intern
```

### Or directly from this repo

```
/plugin install <path-to-this-repo>/plugin
```

After install, restart Claude Code. The plugin will:

- Bootstrap a stdio MCP server (`ml-intern-tools`) the first time you use one of its tools — this triggers `uv sync` against the bundled `pyproject.toml` (~30s the first time, instant after).
- Register five slash commands: `/ml-intern`, `/research`, `/inspect-dataset`, `/finetune`, `/run-job`.
- Register a `research` subagent invoked via the Task tool.
- Wire three lifecycle hooks (SessionStart, PreToolUse, SessionEnd).

## Required environment

Set in your shell or `.env`:

```bash
HF_TOKEN=hf_...        # required — papers, datasets, jobs, repo tools, sessions upload
GITHUB_TOKEN=ghp_...   # required — github_find_examples, github_read_file, github_list_repos
```

The plugin's MCP server needs these in the Claude Code launching shell. See [Troubleshooting](#troubleshooting) if you see 401s.

## What it gives you

### Slash commands

| Command | Purpose |
|---|---|
| `/ml-intern <task>` | Default entrypoint — runs the full research → validate → implement workflow. |
| `/research <topic>` | Force a literature crawl via the research subagent (returns a ranked recipe table). |
| `/inspect-dataset <id>` | Audit a HF dataset (schema, splits, sample rows, training-method compatibility). |
| `/finetune <task>` | Strict end-to-end fine-tune (research → dataset audit → sandbox → pre-flight → submit ONE job). |
| `/run-job <description>` | Submit any HF Job with the pre-flight checklist (≥2h timeout, push_to_hub, Trackio). |

### MCP tools (10)

`hf_papers`, `hf_inspect_dataset`, `hf_jobs`, `hf_repo_files`, `hf_repo_git`, `explore_hf_docs`, `fetch_hf_docs`, `github_find_examples`, `github_list_repos`, `github_read_file`, plus sandbox `bash`/`read`/`write`/`edit`/`sandbox_create`.

These appear in Claude Code as `mcp__ml-intern__ml-intern-tools__<name>`.

### Approval policy (PreToolUse hook)

| Tool / op | Behavior |
|---|---|
| `hf_jobs` (run/uv) on **GPU hardware** | Always prompts |
| `hf_jobs` on CPU hardware | Prompts when `ML_INTERN_CONFIRM_CPU_JOBS=1` (default) |
| `hf_jobs` script with `from_pretrained` but no `push_to_hub` | Always prompts (warning surfaces) |
| `sandbox_create` | Always prompts |
| `hf_repo_files` `upload`/`delete` | Always prompts |
| `hf_repo_git` destructive ops | Always prompts |

The hook **fails safe** — malformed payloads force a prompt, never silent allow.

### Session redaction + upload (SessionEnd hook)

When `ML_INTERN_SAVE_SESSIONS=1` (default), transcripts upload to `smolagents/ml-intern-sessions` (override with `ML_INTERN_SESSION_REPO`). The bundled `redact.py` strips HF/Anthropic/OpenAI/GitHub/AWS tokens before upload. Refuses to upload paths outside `~/.claude/` or `$CLAUDE_PROJECT_DIR`.

### Dynamic context (SessionStart hook)

Injects:
- HF username from `HF_TOKEN` (so the agent uses your namespace for `hub_model_id`).
- "Local mode" banner when `ML_INTERN_LOCAL_MODE=1` (sandbox tools operate on the local fs).

## Environment knobs

| Var | Default | What it does |
|---|---|---|
| `ML_INTERN_YOLO` | `0` | Skip all approvals |
| `ML_INTERN_CONFIRM_CPU_JOBS` | `1` | Prompt for CPU jobs |
| `ML_INTERN_SAVE_SESSIONS` | `1` | Upload transcripts on session end |
| `ML_INTERN_SESSION_REPO` | `smolagents/ml-intern-sessions` | Target dataset for session uploads |
| `ML_INTERN_LOCAL_MODE` | `0` | Run sandbox tools on local fs (no remote sandbox) |
| `HF_SESSION_UPLOAD_TOKEN` | — | Preferred (write-only) token; falls back to `HF_TOKEN`, then `HF_ADMIN_TOKEN` |

## Troubleshooting

**Tool not found / `mcp__ml-intern__ml-intern-tools__...` errors.** The MCP server didn't start. Run `/mcp` to see status. If it's failing, check that `uv` is on `$PATH` and that the plugin's `pyproject.toml` could install — common cause is missing build tools or a network failure during the first `uv sync`. To diagnose, run the server manually:

```bash
cd <plugin-install-dir>
uv run python lib/mcp_server.py < /dev/null
```

**401 Unauthorized from `hf_papers` or `hf_jobs`.** `HF_TOKEN` not set in the shell that launched Claude Code. The plugin's `.mcp.json` substitutes `${HF_TOKEN}` from the launching environment.

**SessionStart shows `HF user: unknown (whoami HTTP error: ...)`.** Token rejected. Probably expired or scoped wrong. Generate a new one at <https://huggingface.co/settings/tokens>.

**Approval prompt every turn for `hf_papers`.** Static permissions list doesn't include the tool name. The plugin doesn't ship its own `permissions.allow` — those are project-level in your repo's `.claude/settings.json`. Add `mcp__ml-intern__ml-intern-tools__hf_papers` (and friends) to your project's allowlist if you want to skip prompts.

## Layout

```
plugin/
├── .claude-plugin/plugin.json     # manifest
├── CLAUDE.md                      # persona / methodology
├── .mcp.json                      # MCP servers (ml-intern-tools, hf-mcp-server)
├── pyproject.toml                 # bundled deps for the MCP server
├── commands/                      # 5 slash commands
├── agents/research.md             # research subagent
├── hooks/
│   ├── hooks.json
│   ├── pre_tool_use_approval.py   # content-aware approval
│   ├── session_start_context.py   # HF user + local-mode injection
│   └── session_end_upload.py      # redaction + HF dataset upload
└── lib/
    ├── mcp_server.py              # MCP frontend
    └── ml_intern_lib/             # vendored tools + redact (no agent.* deps)
```

## Updating

The vendored library under `lib/ml_intern_lib/` is a snapshot of `agent/tools/*` and `agent/core/redact.py` from the upstream ml-intern repo. When the upstream tools change, run:

```bash
make sync-vendored   # from the upstream ml-intern repo root
```

(Or do `cp -r agent/tools/* plugin/lib/ml_intern_lib/tools/` and re-run the import-rewrite step in the upstream `Makefile`.)

## License

Apache-2.0, same as upstream ml-intern.
