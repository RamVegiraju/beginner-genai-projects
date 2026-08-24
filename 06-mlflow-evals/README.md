# 6 — Evaluating with MLflow

> **Not built yet.**

You changed the prompt. Did it get better? This sample measures the answer
instead of guessing at it.

Will cover:

- **Tracing** — `mlflow.langchain.autolog()`, then open a trace and see every
  step of the agent: prompts, tool calls, tokens, latency
- **Eval sets** — a small hand-written set of questions and expected answers
- **Scoring** — `mlflow.genai.evaluate()` with built-in judges for correctness
  and groundedness
- Comparing two prompt versions and letting the scores decide
