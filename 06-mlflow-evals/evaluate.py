"""Score both versions of the agent on the same questions, then compare.

Three kinds of scorer, because they answer different questions and cost
different amounts:

  ToolCallCorrectness   built-in. Reads the trace and judges whether the agent
                        called the right tool with the right arguments.
  grounded_in_lookup    our own, plain Python. No LLM, no cost, deterministic.
  helpfulness           our own, an LLM judge, for the qualitative call that
                        code cannot make.

Run:  python evaluate.py
"""

import json

import mlflow
import mlflow.genai
from app import ORDERS, answer, configure_mlflow
from mlflow.entities import Feedback, SpanType
from mlflow.genai.judges import make_judge
from mlflow.genai.scorers import ToolCallCorrectness, scorer

# Just questions. None of these scorers need a labelled answer, which matters
# in practice -- ground truth is the expensive part of an eval set.
# (For expected answers, add "expectations" and use the Correctness scorer.)
EVAL_SET = [
    {"inputs": {"question": "Where is my order A1002?"}},
    {"inputs": {"question": "Has order A1001 arrived yet?"}},
    {"inputs": {"question": "I can't find my order B9999, what happened to it?"}},
    # No order ID: answering this well means NOT calling the tool.
    {"inputs": {"question": "How long do I have to return something?"}},
]


# Every status the fake order database can return. Derived from the app so the
# two cannot drift apart.
REAL_STATUSES = {order["status"] for order in ORDERS.values()}


@scorer
def grounded_in_lookup(outputs, trace) -> Feedback:
    """Check the reply against what the tool actually returned.

    Pure Python: no model call, no cost, same answer every time. If a scorer
    can be written this way, write it this way and save the LLM judges for
    things that genuinely need judgement.

    The catch with code scorers is that they are literal. This one has to know
    that a missing order comes back as "not found" while the agent will phrase
    it as "I'm unable to find that order" -- so the two cases are checked
    differently.
    """
    tool_spans = trace.search_spans(span_type=SpanType.TOOL)
    if not tool_spans:
        return Feedback(value=True, rationale="No lookup was made, so nothing to contradict.")

    # LangChain wraps a tool's return value in a ToolMessage, so the span's
    # outputs are {"content": "<json string>", "status": "success", ...}.
    # Note the trap: that outer "status" means the tool ran, not what it found.
    raw = tool_spans[0].outputs or {}
    payload = raw.get("content", raw) if isinstance(raw, dict) else raw
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            payload = {}

    status = payload.get("status", "") if isinstance(payload, dict) else ""
    reply = outputs.get("response", "").lower()

    if status not in REAL_STATUSES:
        # The order does not exist. A grounded reply must not invent a status.
        invented = sorted(s for s in REAL_STATUSES if s in reply)
        return Feedback(
            value=not invented,
            rationale=(
                f"Order was not found, but the reply claims {invented}."
                if invented
                else "Order was not found and the reply claims no status."
            ),
        )

    grounded = status in reply
    return Feedback(
        value=grounded,
        rationale=f"Tool returned {status!r}; the reply "
        + ("repeats it." if grounded else "never mentions it."),
    )


# The qualitative one. "Was this a good support reply?" is not something a
# regex can answer. {{ inputs }} and {{ outputs }} are filled in per row.
helpfulness = make_judge(
    name="helpfulness",
    instructions=(
        "Rate this customer support reply from 1 to 5.\n\n"
        "The customer asked: {{ inputs }}\n"
        "The agent replied: {{ outputs }}\n\n"
        "5 = specific, and the customer knows what happens next.\n"
        "3 = correct but generic.\n"
        "1 = vague, evasive, or unhelpful."
    ),
    feedback_value_type=int,
    model="databricks",
)

SCORERS = [ToolCallCorrectness(name="tool_call_correctness"), grounded_in_lookup, helpfulness]


def score(version: str) -> dict[str, float]:
    """Run the whole eval set through one prompt version.

    Args:
        version: Which prompt in app.PROMPTS to evaluate.

    Returns:
        The aggregate metrics, keyed like "helpfulness/mean".
    """
    with mlflow.start_run(run_name=f"prompt-{version}"):
        result = mlflow.genai.evaluate(
            data=EVAL_SET,
            # predict_fn is called with the "inputs" dict unpacked as kwargs,
            # so it receives question="..." and the version is bound here.
            predict_fn=lambda question: answer(question, version=version),
            scorers=SCORERS,
        )
    return result.metrics


if __name__ == "__main__":
    experiment = configure_mlflow()
    print(f"experiment: {experiment}\n")

    results = {version: score(version) for version in ("v1", "v2")}

    print("\n" + "=" * 52)
    print(f"{'metric':<30}{'v1':>10}{'v2':>10}")
    print("-" * 52)
    for metric in sorted({m for scores in results.values() for m in scores}):
        # A scorer that failed for one version has no metric for it, so format
        # defensively rather than crashing on the comparison.
        cells = "".join(
            f"{results[v][metric]:>10.2f}" if metric in results[v] else f"{'--':>10}"
            for v in ("v1", "v2")
        )
        print(f"{metric:<30}{cells}")
    print("=" * 52)
    print("\nOpen the experiment in Databricks to see every answer, every tool")
    print("call, and each judge's reasoning.")
