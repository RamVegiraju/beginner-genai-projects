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

**First time?** Do [SETUP.md](../SETUP.md) first — install the CLI, log in,
and create the virtual environment. Five minutes, once for the whole series.

Then install this sample's dependencies, **from the repo root**:

```bash
uv pip install --python .venv/bin/python -r 01-first-llm-call/requirements.txt
```

## Run

```bash
export DATABRICKS_PROFILE=genai-series   # once per terminal
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

| Variable | Required? | Meaning |
|---|---|---|
| `DATABRICKS_PROFILE` | Yes, unless credentials come from elsewhere | The profile you logged in with in [SETUP.md](../SETUP.md) |
| `SERVING_ENDPOINT` | No — defaults to `databricks-claude-haiku-4-5` | Any other ready chat endpoint |

See what your workspace offers — and prefer this over any hard-coded name,
since models come and go:

```bash
databricks serving-endpoints list --profile genai-series
```

If Haiku isn't there, export any ready chat endpoint from that list:

```bash
export SERVING_ENDPOINT=<another-ready-chat-endpoint>
```

**When can you omit the profile?** When something else already supplies
credentials — `DATABRICKS_HOST` + `DATABRICKS_TOKEN`, a service principal's
`DATABRICKS_CLIENT_ID`/`DATABRICKS_CLIENT_SECRET`, or code running inside
Databricks. The SDK checks all of those before it reads `~/.databrickscfg`.
On a laptop with more than one profile, leaving it unset fails with
`Use --profile to specify which profile to use`.

## Next

This sample asks one question and exits. To hold a conversation you need to
know that the API is **stateless** — that's [sample 2](../02-streamlit-chatbot/).
