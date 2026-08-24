# 6 — Evaluation with MLflow

> **Not built yet.** Design notes only.

**The wall:** you changed the prompt. Did it get better? "It seems good" is
not an answer, and it's how most GenAI projects quietly fail.

Planned:
- **Tracing first** — `mlflow.langchain.autolog()`, then open a trace and see
  every step of the sample 3 agent: prompts, tool calls, tokens, latency.
  Tracing is the debugging superpower; it should land before scoring does.
- **Then evals** — a small hand-written eval set (10–20 rows), scored with
  `mlflow.genai.evaluate()` and built-in judges (correctness, groundedness)
- Compare two prompt versions side by side and let the scores decide

**The point:** you cannot improve what you cannot measure — and you cannot
debug what you cannot see.
