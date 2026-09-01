# 6 — Evaluating with MLflow

"It seems good" is not a quality bar. This sample measures a LangChain agent
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
of an eval set. The ten questions in `evaluate.py` are just questions.

## Setup

**First time?** Do [SETUP.md](../SETUP.md) first — install the CLI, log in,
and create the virtual environment. Five minutes, once for the whole series.

Both `langchain` and `databricks-agents` are required. MLflow's LangChain
autologging imports the top-level `langchain` package, while the built-in
Databricks judges use `databricks-agents`.

## Run

First, record a trace. Open a terminal at the repo root and paste the whole
block:

```bash
export DATABRICKS_PROFILE=genai-series   # once per terminal
uv pip install --python .venv/bin/python -r 06-mlflow-evals/requirements.txt
cd 06-mlflow-evals
../.venv/bin/python app.py
```

Open the experiment it prints. You will see the model call, the tool call with
its arguments and result, and the tokens it cost. **Do this before scoring
anything** — you cannot debug what you cannot see.

Then score the agent — same terminal, still inside `06-mlflow-evals`:

```bash
../.venv/bin/python evaluate.py
```

```
Evaluation complete
grounded_in_lookup/mean: 1.00
helpfulness/mean: 4.80
tool_call_correctness/mean: 1.00
```

About 15–30 seconds for ten questions. The two correctness scorers sit at 1.00;
`helpfulness` moves between 4.60 and 4.80 across runs, because it is an LLM
judge and is not perfectly repeatable even at temperature 0. Treat small gaps
as noise and add questions before trusting a close call.

## The three scorers

### Built-in: `ToolCallCorrectness`

Reads the trace, finds the tool calls, and judges whether they were the right
ones. It finds them by searching for spans of type `TOOL`, which is why the
tool in `app.py` is decorated:

```python
@tool
def look_up_order(order_id: str) -> dict[str, str]: ...
```

`create_agent()` runs on LangGraph internally, and MLflow's LangChain
autologging records its tool execution as a `TOOL` span. Without that span type
the scorer would see an agent that never used a tool.

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

- **Judges are LLM calls.** Ten questions and two LLM scorers is twenty
  judgements per run, on top of the agent's own calls. Keep eval sets pointed.
- **`MLFLOW_GENAI_EVAL_SKIP_TRACE_VALIDATION` is set** at the top of
  `evaluate.py`. MLflow otherwise calls your function once before scoring to
  check its signature, and that check can hang here. Anything it would catch
  surfaces on the first scored question anyway.
- **If a run sits at `0/10` for more than ~30 seconds, kill it and re-run.**
  It stalls before scoring starts, roughly one run in three, and the cause is
  not yet understood — most likely contention between this sample's
  `langchain.autolog()` and the tracing MLflow enables during evaluation.
  A completed run is always correct; a stalled one never starts.
- **The agent is created once** (`@functools.cache`). `create_agent()` returns
  a compiled LangGraph graph. MLflow scores rows in
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
