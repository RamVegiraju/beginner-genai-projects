# 6 — Evaluating with MLflow

"It seems good" is not a quality bar. This sample measures a LangGraph agent
instead of guessing at it, using Databricks-managed MLflow.

The agent is customer support with one tool, `look_up_order`. Anything about a
specific order has to come from that tool — the model cannot know it otherwise.

## What this sample showcases

**1. Tracing is one decorator.** `@mlflow.trace` records every call — inputs,
outputs, latency, tokens, and each tool call — and you can open any of them in
the Databricks UI.

**2. Three kinds of scorer**, because they answer different questions and cost
different amounts:

| Scorer | Kind | Asks | Cost |
|---|---|---|---|
| `ToolCallCorrectness` | built-in | Did it call the right tool with the right arguments? | an LLM call |
| `grounded_in_lookup` | your Python | Does the reply contradict what the tool returned? | free |
| `helpfulness` | your LLM judge | Was this a good support reply, 1–5? | an LLM call |

**3. You do not always need labelled data.** None of these three need a
written-out correct answer, which matters — ground truth is the expensive part
of an eval set.

## Setup

**First time?** Do [SETUP.md](../SETUP.md) first — install the CLI, log in,
and create the virtual environment. Five minutes, once for the whole series.

Then install this sample's dependencies, **from the repo root**:

```bash
uv pip install --python .venv/bin/python -r 06-mlflow-evals/requirements.txt
```

`databricks-agents` is required, not optional: the built-in judges live there,
and without it every scorer fails with `No module named 'databricks.agents'`
while `evaluate()` still reports success.

## Run

First, record a trace:

```bash
cd 06-mlflow-evals
../.venv/bin/python app.py
```

Open the experiment it prints. You will see the model call, the tool call with
its arguments and result, and the tokens it cost. **Do this before scoring
anything** — you cannot debug what you cannot see.

Then score the agent:

```bash
../.venv/bin/python evaluate.py
```

```
Evaluation complete
grounded_in_lookup/mean: 1.00
helpfulness/mean: 4.75
tool_call_correctness/mean: 1.00
```

About 30 seconds for four questions. `helpfulness` moves between 4.50 and 4.75
across runs — it is an LLM judge, so it is not perfectly repeatable even at
temperature 0. On four rows one changed judgement is worth 0.25, so treat small
gaps as noise and add rows before trusting a close call.

## The three scorers

### Built-in: `ToolCallCorrectness`

Reads the trace, finds the tool calls, and judges whether they were the right
ones. It finds them by searching for spans of type `TOOL`, which is why the
tool in `app.py` is decorated:

```python
@tool
def look_up_order(order_id: str) -> dict[str, str]: ...
```

LangGraph's `ToolNode` emits the `TOOL` span for you. Without that span type
the scorer sees an agent that never used a tool.

### Your Python: `grounded_in_lookup`

No model call, no cost, same answer every time. If a check can be written this
way, write it this way and save the judges for things that need judgement.

The catch is that code scorers are *literal*. An earlier version required the
reply to repeat the tool's status verbatim, and failed this perfectly good
answer:

```
tool returned:  "preparing"
agent replied:  "Order A1003 will ship on September 1. It's currently being prepared."
```

So it checks for **contradiction** instead: the reply must not claim a status
the tool did not return. That survives paraphrase and still catches the failure
that matters — inventing a status, especially for an order that does not exist.

### Your LLM judge: `helpfulness`

`make_judge` takes plain instructions with `{{ inputs }}` and `{{ outputs }}`
placeholders, and a `feedback_value_type` — here `int`, for a 1–5 rating. Use
it for what code cannot decide: tone, clarity, whether the customer knows what
to do next. `inference_params={"temperature": 0}` keeps it as steady as an LLM
judge gets.

## Notes

- **Judges are LLM calls.** Four questions and two LLM scorers is eight
  judgements per run. Keep eval sets small and pointed.
- **The graph is compiled once** (`@functools.cache`). MLflow scores rows in
  several threads, and building a fresh client per call inside them is both
  wasteful and a source of flakiness.
- **One token is minted up front** in `configure_mlflow()` and shared. Without
  it every judge shells out to the Databricks CLI, and those concurrent
  refreshes race over the OS keyring (`cache update: exit status 45`). That is
  a property of browser-based login; a service principal has none of it — see
  [SETUP.md](../SETUP.md#anything-unattended-is-different).
- **Results live in your workspace**, not on disk, under an MLflow experiment
  in your user folder. Set `MLFLOW_EXPERIMENT_NAME` to change where.
- `ToolCallCorrectness` is marked **Experimental** in MLflow 3.15 and may change.

## Next

That is the series. You can call a model, hold a conversation, give it tools,
give it memory, serve it to many people, and measure whether any of it works.
