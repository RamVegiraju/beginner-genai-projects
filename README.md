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
| 2 | Streamlit chatbot | One question isn't a conversation, and a script isn't a product. |
| 3 | LangGraph agent | The model can't know today's weather. Give it tools. |
| 4 | Memory | Close the tab and it forgets you. Persist across sessions. |
| 5 | FastAPI server | One user at a time doesn't scale. Agents are a systems problem. |
| 6 | MLflow evals | "It seems good" isn't a quality bar. Measure it. |

## Status

- **Sample 1 — built and tested** against a live workspace.
- **Sample 2 — built and tested** against a live workspace.
- Samples 3–6 — scaffolded, not yet written.

## Setup

**→ [SETUP.md](SETUP.md)** — install the CLI, log in, verify, and run your
first sample. About five minutes, once for the whole series.

The short version, if you already have the Databricks CLI:

```bash
databricks auth login --host https://<your-workspace>.cloud.databricks.com
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r 01-first-llm-call/requirements.txt
cd 01-first-llm-call && ../.venv/bin/python invoke.py
```

Each sample has its own `requirements.txt` and its own README. Start with
[`01-first-llm-call/`](01-first-llm-call/).

## Configuration

Two optional environment variables, used by every sample. Copy `.env.example`
to `.env` if your setup differs from the defaults.

| Variable | Default | Meaning |
|---|---|---|
| `DATABRICKS_PROFILE` | `DEFAULT` | Which profile in `~/.databrickscfg`. [SETUP.md](SETUP.md) has you create a named profile — export this to point at it. |
| `SERVING_ENDPOINT` | `databricks-claude-haiku-4-5` | Which model to call |

## A note on credentials

No API keys or tokens appear anywhere in this repo, by design. The samples ask
the Databricks CLI for a short-lived OAuth token at runtime, so there is
nothing to paste, leak, or rotate — and `.databrickscfg`, `.env`, and `*.db`
are all gitignored.

If you're used to pasting a personal access token into your code, read
[why we don't](SETUP.md#best-practice-how-auth-should-work).
