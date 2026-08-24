# 6 — Evaluating with MLflow

You changed the prompt. Did it get better?

"It seems good" is how most GenAI projects quietly fail. This sample measures
the answer instead, using Databricks-managed MLflow.

## What this sample showcases

**1. Tracing is one decorator.** `@mlflow.trace` records every call — inputs,
outputs, latency, tokens — and you can open any of them in the Databricks UI.

**2. An eval set is just a list of questions and expected facts.** Hand-written,
small, and the most valuable thing in this folder.

**3. LLM judges do the scoring.** Built-in scorers read each answer and decide
whether it is correct, relevant, and follows your rules.

**4. Two prompts, same questions, real numbers.** That is the whole point.

## Setup

```bash
uv pip install --python .venv/bin/python -r 06-mlflow-evals/requirements.txt
```

`databricks-agents` is required, not optional — the built-in judges are
implemented there, and without it every scorer fails with
`No module named 'databricks.agents'`.

## Run

First, record a trace:

```bash
cd 06-mlflow-evals
../.venv/bin/python app.py
```

Open the experiment it prints in the Databricks UI. You will see the call, the
exact prompt that was sent, and the tokens it cost. **Do this before scoring
anything** — you cannot debug what you cannot see.

Then score both prompt versions:

```bash
../.venv/bin/python evaluate.py
```

```
==============================================
metric                          v1        v2
----------------------------------------------
concise/mean                  0.00      0.80
correctness/mean              0.80      0.80
relevance_to_query/mean       1.00      1.00
==============================================
```

The two prompts differ only in a few added instructions. Conciseness went from
0.00 to 0.80; correctness did not move. That is a more useful answer than "it
felt better" — the change helped one thing and left another alone.

## The question that shows why this matters

Both versions are asked something the policy does not cover: *"Can I change
the grind size on my subscription?"*

**v1** admits it does not know, and then invents guidance anyway:

> I don't have specific information about changing grind size... **Checking
> your account page** — you may be able to modify grind size settings directly

Nothing in the policy says that. **v2**:

> I don't know the answer to that question based on our policy information.
> Let me pass this to a human agent who can help.

One of those creates a support ticket. Scores make the difference visible
across every question at once, instead of one you happened to try by hand.

## The scorers

| Scorer | Asks | Needs ground truth? |
|---|---|---|
| `Correctness` | Does the answer contain the expected facts? | Yes — `expected_facts` |
| `RelevanceToQuery` | Does it address the question at all? | No |
| `Guidelines` | Does it follow a rule you wrote? | No |

`Guidelines` is the flexible one: give it a `name` and a plain-English rule.
It sees the `request` and the `response`, so write rules about those.

## Notes

- **Judges are LLM calls.** Every score costs a model call, so a 5-question
  eval set with 3 scorers is 15 judgements per run. Keep eval sets small and
  focused on what you actually care about.
- **The evaluation runs single-threaded on purpose.** By default MLflow scores
  10 rows at a time, and each judge asks the Databricks CLI for a token.
  Concurrent token refreshes collide on the OS keyring and scorers fail with
  `cache update: exit status 45`. See the top of `evaluate.py`.
- **Results live in your workspace**, not on disk. Everything is written to an
  MLflow experiment under your user folder.

## Next

That is the series. You can call a model, hold a conversation, give it tools,
give it memory, serve it to many people, and measure whether any of it works.
