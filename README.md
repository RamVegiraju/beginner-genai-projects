# Beginner GenAI Series

Six small, self-contained samples that take you from "how do I call an LLM?"
to "how do I serve and evaluate an agent?" — using Databricks Foundation Model
APIs.

Every sample is intentionally small enough to read in one sitting. No
framework magic before you've seen the thing it's hiding.

## The arc

Each sample exists because the previous one hit a wall.

| # | Sample | The wall it solves |
|---|---|---|
| 1 | [Talking to an LLM](01-first-llm-call/) | How do you call a model at all? Streaming vs non-streaming. |
| 2 | [Streamlit chatbot](02-streamlit-chatbot/) | One question isn't a conversation, and a script isn't a product. |
| 3 | [LangGraph agent](03-langgraph-agent/) | The model can't know today's weather. Give it tools. |
| 4 | [Memory](04-agent-memory/) | Close the tab and it forgets you. Persist the conversation, and distill what's worth keeping about the person. |
| 5 | [FastAPI server](05-fastapi-server/) | One user at a time doesn't scale. Serve many at once. |
| 6 | [MLflow evals](06-mlflow-evals/) | "It seems good" isn't a quality bar. Measure it. |

## Status

- **Sample 1 — built and tested** against a live workspace.
- **Sample 2 — built and tested** against a live workspace.
- **Sample 3 — built and tested** against a live workspace.
- **Sample 4 — built and tested** against a live workspace.
- **Sample 5 — built and tested** against a live workspace.
- **Sample 6 — built and tested** against a live workspace.

All six samples are complete.

## Setup

**→ [SETUP.md](SETUP.md)** — install the CLI, log in, verify, and run your
first sample. About five minutes, once for the whole series.

The short version, if you already have the Databricks CLI:

```bash
databricks auth login \
  --host https://<your-workspace>.cloud.databricks.com \
  --profile genai-series
export DATABRICKS_PROFILE=genai-series

uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r 01-first-llm-call/requirements.txt
cd 01-first-llm-call && ../.venv/bin/python invoke.py
```

Each sample has its own `requirements.txt` and its own README. Start with
[`01-first-llm-call/`](01-first-llm-call/).

## Quick validation

The smoke tests check configuration and pure helper behavior without calling a
model or needing workspace credentials:

```bash
uv pip install --python .venv/bin/python -r requirements-dev.txt
.venv/bin/ruff check .
.venv/bin/pytest -q
```

Running each sample is still the end-to-end test because model access,
authentication, tracing, and endpoint capabilities belong to the workspace.

## Configuration

The samples default to `databricks-claude-haiku-4-5`. To use another ready
chat endpoint, export the override directly or copy `.env.example` to `.env`
and load it into your shell:

```bash
set -a; source .env; set +a
```

| Variable | Required? | Meaning |
|---|---|---|
| `SERVING_ENDPOINT` | No; defaults to `databricks-claude-haiku-4-5` | A different ready chat endpoint |
| `DATABRICKS_PROFILE` | Yes, unless credentials come from elsewhere | A named CLI profile. Omit it only where the environment already supplies credentials. |
| `USER_ID` | No; defaults to `demo-user` | Who long-term memory belongs to (sample 4) |

If Haiku is unavailable in your workspace, [SETUP.md](SETUP.md) shows how to
choose another endpoint.

## Last validated

All six samples were exercised against an Azure Databricks workspace on
August 25, 2026, using Python 3.12.13. Key resolved versions were
Databricks SDK 0.133.0, OpenAI 3.2.0, LangGraph 1.2.11, FastAPI 0.141.1, and
MLflow 3.15.1. The requirements specify supported minimums rather than tying
the series to that one workspace or environment.

## A note on credentials

No API keys or tokens appear anywhere in this repo, by design. The samples ask
the Databricks CLI for a short-lived OAuth token at runtime, so there is
nothing to paste, leak, or rotate — and `.databrickscfg`, `.env`, and `*.db`
are all gitignored.

If you're used to pasting a personal access token into your code, read
[why we don't](SETUP.md#best-practice-how-auth-should-work).
