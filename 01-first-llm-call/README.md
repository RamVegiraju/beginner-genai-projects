# 1 — Talking to an LLM

Your first model call on Databricks. One script, two ways to get the answer
back: **non-streaming** and **streaming**.

## What this sample showcases

**1. Calling a model is just an API call.** Databricks serving endpoints are
OpenAI-compatible — you use the standard OpenAI SDK and point `base_url` at
your workspace.

**2. No API keys in your code.** Auth comes from your Databricks identity via
the CLI profile. Nothing to paste, nothing to leak, nothing to rotate.

**3. Non-streaming vs streaming.** The difference every chat app depends on.

## Setup

First time here? Do [SETUP.md](../SETUP.md) first — CLI, login, and Python
environment. Once that's done:

## Run

```bash
cd 01-first-llm-call
../.venv/bin/python invoke.py
```

## Non-streaming vs streaming

Same model, same question — only the delivery changes.

| | Non-streaming | Streaming |
|---|---|---|
| Code | default | `stream=True` |
| You get | one response object | an iterator of chunks |
| You see the answer | all at once, at the end | word by word, immediately |
| Total time | the same | the same |
| Use it when | something else consumes the output — a database write, another function, a batch job | a human is waiting and watching |

The key point: **streaming isn't faster.** The model takes just as long. What
changes is that the user stops staring at a blank screen. That's the entire
reason chat interfaces type at you.

## Configuration

Both are optional — the script falls back to these defaults.

```bash
export SERVING_ENDPOINT=databricks-claude-haiku-4-5
export DATABRICKS_PROFILE=genai-series
```

See what your workspace offers — and prefer this over any hard-coded name,
since models come and go:

```bash
databricks serving-endpoints list --profile genai-series
```

The default is `databricks-claude-haiku-4-5` — fast and cheap, which matters
while you're re-running the script and learning.

## Next

This sample asks one question and exits. To hold a conversation you need to
know that the API is **stateless** — that's [sample 2](../02-streamlit-chatbot/).
