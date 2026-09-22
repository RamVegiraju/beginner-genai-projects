# Beginner GenAI Series

Seven small, self-contained samples that take you from "how do I call an LLM?"
to "how do I serve and evaluate an agent?" — using Databricks Foundation Model
APIs.

Every sample is intentionally small enough to read in one sitting. No
framework magic before you've seen the thing it's hiding.

## Watch the series

Prefer to learn by following along? Watch the complete
[Applied AI Engineering Beginner Series on YouTube](https://www.youtube.com/playlist?list=PLOR4-dEcH4aA),
then use this repository as the code companion.

| Video | Code companion |
|---|---|
| [Applied AI Engineering Roadmap \| 2026](https://www.youtube.com/watch?v=79FlS1Dy0kI&list=PLOR4-dEcH4aA) | Start with the repository overview below. |
| [Your First LLM API Call in Python \| AI Engineering Project #1](https://www.youtube.com/watch?v=1wc4QVDD08s&list=PLOR4-dEcH4aA) | [01-first-llm-call](01-first-llm-call/) |
| [Open Source vs Closed Source LLMs Explained \| AI Engineering #2](https://www.youtube.com/watch?v=MIwta-jRMQs&list=PLOR4-dEcH4aA) | Background concepts used throughout the series. |
| [Build Your First AI Chatbot with Streamlit \| AI Engineering Project #3](https://www.youtube.com/watch?v=PQUxN9ErhzI&list=PLOR4-dEcH4aA) | [02-streamlit-chatbot](02-streamlit-chatbot/) |
| [What is an AI Agent & Agent Framework? \| AI Engineering Project #4](https://www.youtube.com/watch?v=uRDP5PUhNA4&list=PLOR4-dEcH4aA) | [03-langgraph-agent](03-langgraph-agent/) |
| [LangChain vs LangGraph vs Deep Agents Explained \| AI Engineering Project #5](https://www.youtube.com/watch?v=-sZHOqbh3hA&list=PLOR4-dEcH4aA) | [03-langgraph-agent](03-langgraph-agent/) |
| [Building Your First Agent with LangGraph \| AI Engineering Project #6](https://www.youtube.com/watch?v=rUfBxNpfYXw&list=PLOR4-dEcH4aA) | [03-langgraph-agent](03-langgraph-agent/) |
| [Building RAG Workflows \| AI Engineering Project #7](https://www.youtube.com/watch?v=d-hk9IpKZiI&list=PLOR4-dEcH4aA) | [04-rag](04-rag/) |

## The arc

Each sample exists because the previous one hit a wall.

| # | Sample | The wall it solves |
|---|---|---|
| 1 | [Talking to an LLM](01-first-llm-call/) | How do you call a model at all? Streaming vs non-streaming. |
| 2 | [Streamlit chatbot](02-streamlit-chatbot/) | One question isn't a conversation, and a script isn't a product. |
| 3 | [LangChain → LangGraph agent](03-langgraph-agent/) | Start with a tool-calling agent, then add a custom review loop. |
| 4 | [RAG](04-rag/) | A tool can fetch live facts, but the model still cannot know your private documents. Retrieve the relevant evidence. |
| 5 | [Memory](05-agent-memory/) | Close the tab and it forgets you. Persist the conversation, and distill what's worth keeping about the person. |
| 6 | [FastAPI server](06-fastapi-server/) | One user at a time doesn't scale. Serve many at once. |
| 7 | [MLflow evals](07-mlflow-evals/) | "It seems good" isn't a quality bar. Measure it. |

## Status

- **Sample 1 — built and tested** against a live workspace.
- **Sample 2 — built and tested** against a live workspace.
- **Sample 3 — built and tested** against a live workspace.
- **Sample 4 — built and tested** against a live workspace.
- **Sample 5 — built and tested** against a live workspace.
- **Sample 6 — built and tested** against a live workspace.
- **Sample 7 — built and tested** against a live workspace.

All seven samples are implemented.

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
| `EMBEDDING_ENDPOINT` | No; defaults to `databricks-gte-large-en` | A different ready embedding endpoint (sample 4) |
| `RAG_DOCUMENT` | No; defaults to the checked-in guide | A guide stored in a Unity Catalog Volume (sample 4) |
| `RAG_QUESTION` | No; has a built-in example question | A different question for the RAG guide (sample 4) |
| `DATABRICKS_PROFILE` | Yes, unless credentials come from elsewhere | A named CLI profile. Omit it only where the environment already supplies credentials. |
| `USER_ID` | No; defaults to `demo-user` | Who long-term memory belongs to (sample 5) |

If Haiku is unavailable in your workspace, [SETUP.md](SETUP.md) shows how to
choose another endpoint.

## Last validated

Samples 1–3 and 5–7 were exercised against an Azure Databricks workspace on
August 25, 2026. Sample 4 was exercised on September 22, 2026 using a guide in
an existing Unity Catalog Volume and local in-memory vector search. Python
3.12 was used throughout. Key resolved versions for the original samples were
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
