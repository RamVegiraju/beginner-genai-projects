# 6 — Evaluating with MLflow

You changed the prompt. Did it get better?

"It seems good" is how most GenAI projects quietly fail. This sample measures
the answer instead, using Databricks-managed MLflow.

The app is a support agent with one tool, `look_up_order`. Anything about a
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
| `grounded_in_lookup` | your code | Does the reply match what the tool returned? | free |
| `helpfulness` | your LLM judge | Was this a good support reply, 1–5? | an LLM call |

**3. You do not always need labelled data.** None of these three need a
written-out correct answer, which matters — ground truth is the expensive part
of an eval set.

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

Open the experiment it prints. You will see the model call, the tool call with
its arguments and result, and the tokens it cost. **Do this before scoring
anything** — you cannot debug what you cannot see.

Then score both prompt versions:

```bash
../.venv/bin/python evaluate.py
```

```
====================================================
metric                                v1        v2
----------------------------------------------------
grounded_in_lookup/mean             1.00      1.00
helpfulness/mean                    4.75      4.50
tool_call_correctness/mean          0.75      0.75
====================================================
```

**Your numbers will not match exactly.** Two of the three scorers are LLM
judges, so they are not deterministic. On four questions a single changed
judgement is worth 0.25, so treat small gaps as noise and add rows before
trusting a close call.

And read this table honestly: **v2 did not beat v1.** The stricter prompt
changed nothing measurable and scored slightly lower on helpfulness, because
it is terser. That is a real result, and knowing it before you ship beats
believing the change helped.

## The three scorers

### Built-in: `ToolCallCorrectness`

Reads the trace, finds the tool calls, and judges whether they were the right
ones. It finds them by searching for spans of type `TOOL`, which is why the
tool in `app.py` is decorated:

```python
@mlflow.trace(span_type=SpanType.TOOL)
def look_up_order(order_id: str) -> dict[str, str]: ...
```

Without that `span_type`, the scorer sees an agent that never used a tool.

### Your code: `grounded_in_lookup`

Plain Python. No model call, no cost, same answer every time. If a check can
be written this way, write it this way and save the judges for things that
need judgement.

Code scorers are *literal*, though. The first version of this one looked for
the string `"not found"` in the reply — and scored a perfectly good *"I'm
unable to find order B9999"* as a failure. It now checks the two cases
separately: a real status must be repeated, and a missing order must not have
a status invented for it.

### Your LLM judge: `helpfulness`

`make_judge` takes plain instructions with `{{ inputs }}` and `{{ outputs }}`
placeholders, and a `feedback_value_type` — here `int`, for a 1–5 rating.
Use this for what code cannot decide: tone, clarity, whether the customer
actually knows what to do next.

## A judge that is wrong, on purpose

`tool_call_correctness` sits at 0.75 for both versions. The failure is the
policy question, *"How long do I have to return something?"* — the agent
correctly answered from policy without touching the tool, and the judge marked
it wrong:

> the agent did not call any tools. The available tool, 'look_up_order', is
> relevant because...

Without ground truth this scorer treats "called nothing" as a miss. You can
pin it down by adding `expectations={"expected_tool_calls": [...]}` per row —
but note an **empty list is read as "no expectations given"**, so there is no
way to state "correctly called nothing".

Keep this one in mind. A score is where you start looking, not the answer.
Open the rationale, which is why the sample pairs LLM judges with a cheap
deterministic scorer that cannot be talked into anything.

`ToolCallCorrectness` is also marked **Experimental** in MLflow 3.15 and may
change.

## Notes

- **Judges are LLM calls.** Four questions, three scorers, two versions is 24
  judgements per run. Keep eval sets small and pointed.
- **One token is minted up front** in `configure_mlflow()` and shared. Without
  it every judge shells out to the Databricks CLI, and those concurrent
  refreshes race over the OS keyring (`cache update: exit status 45`). That is
  a property of browser-based login; a service principal has none of it, which
  is what you would use to run this on a schedule. See
  [SETUP.md](../SETUP.md#anything-unattended-is-different).
- **Results live in your workspace**, not on disk, under an MLflow experiment
  in your user folder.

## Next

That is the series. You can call a model, hold a conversation, give it tools,
give it memory, serve it to many people, and measure whether any of it works.
