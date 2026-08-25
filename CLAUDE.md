# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A six-part teaching series on Databricks Foundation Model APIs. Each sample is
self-contained and deliberately minimal — sample N exists because sample N-1 hit a
wall (see the arc table in `README.md`). Brevity is a feature: prefer deleting code
over adding a case.

Each sample's `README.md` is part of the deliverable, not documentation about it.
They quote **real measured output** (timings, scores), so re-run and update those
numbers when behavior changes.

## Commands

There is no build and no test suite. Verification is running the sample against a
live workspace.

```bash
# once, from the repo root
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r 0N-sample/requirements.txt

# then, from inside the sample directory
cd 03-langgraph-agent && ../.venv/bin/python agent.py
cd 02-streamlit-chatbot && ../.venv/bin/streamlit run app.py   # also 04
cd 05-fastapi-server && ../.venv/bin/uvicorn server:app        # then: python load_test.py
cd 06-mlflow-evals && ../.venv/bin/python evaluate.py
```

Samples 2 and 4 are Streamlit; drive them headlessly with `streamlit.testing.v1.AppTest`
(`AppTest.from_file(path, default_timeout=120)` — the 3s default is too short for model
calls).

Env vars, all optional: `DATABRICKS_PROFILE` (default `genai-series`), `SERVING_ENDPOINT`,
`USER_ID` (sample 4), `MLFLOW_EXPERIMENT_NAME` (sample 6).

## Architecture

**The through-line:** every sample asks the Databricks CLI for a short-lived OAuth token
at runtime and points an OpenAI-compatible client at `{host}/serving-endpoints`. No
secrets in the repo.

```python
w = WorkspaceClient(profile=PROFILE)
token = w.config.authenticate()["Authorization"].removeprefix("Bearer ")
```

This is why `ChatOpenAI`/`OpenAI` is used with a Claude model — Databricks serving
endpoints speak the OpenAI protocol regardless of the model behind them. `ChatDatabricks`
from `databricks-langchain` would be more idiomatic but is **broken in this environment**
(importing it raises `cannot import name 'RequestContext' from 'mcp.shared.context'`).

**Framework progression:** samples 1–2 use the raw OpenAI SDK on purpose, to show the
thing the framework hides. Sample 3 introduces LangGraph, and **everything after it uses
LangGraph** — don't reintroduce hand-rolled tool loops.

Samples never import from each other. Each has its own `requirements.txt`.

## Gotchas that have cost real time here

- **Sample 6 requires `databricks-agents`.** Built-in judges live there. Without it every
  scorer fails with `No module named 'databricks.agents'` while `evaluate()` still prints
  success and returns empty metrics.
- **MLflow's eval harness calls `predict_fn` from ~10 threads.** Mint one token in
  `configure_mlflow()` and share it; otherwise each judge shells out to the Databricks CLI
  and the concurrent refreshes contend over the OS keyring (`cache update: exit status 45`).
- **`ToolCallCorrectness` finds tools via `span_type=SpanType.TOOL`.** Without that span
  type it sees an agent that never used a tool. `langchain.autolog()` emits them; disable
  it and `grounded_in_lookup` silently drops to 0.00.
- **Sample 6 stalls before scoring roughly one run in three**, stuck at `0/10`. Cause not
  understood; kill and re-run. `MLFLOW_GENAI_EVAL_SKIP_TRACE_VALIDATION` is already set and
  is *not* the fix for this.
- **LangChain wraps tool returns in a ToolMessage.** Span outputs look like
  `{"content": "<json>", "status": "success"}` — that `status` means the tool ran, not what
  it found. Parse `content`.
- **A LangGraph node written as `async def` cannot be run by the sync `graph.invoke()`**
  (`No synchronous function provided`).
- **Known pyright false positives, all runtime-verified:** `max_tokens` on `ChatOpenAI`
  (pydantic alias), `chunk.content` from `stream`/`astream` (loose union return type), and
  plain dicts passed as OpenAI `messages`/`tools` (strict TypedDicts).

Ruff config (line-length 100, import sorting) lives in the root `pyproject.toml`.
