"""Run one MLflow evaluation using three complementary scorer types.

Run: python evaluate.py
"""

import json

import mlflow
import mlflow.genai
from app import ORDERS, answer, configure_mlflow
from mlflow.entities import Feedback, SpanType
from mlflow.genai.judges import make_judge
from mlflow.genai.scorers import ToolCallCorrectness, scorer

# Each question requires an order lookup. This keeps the built-in tool-call
# scorer focused on a clear, observable behavior.
EVAL_SET = [
    {"inputs": {"question": "Where is my order A1002?"}},
    {"inputs": {"question": "Has order A1001 arrived yet?"}},
    {"inputs": {"question": "I can't find my order B9999, what happened to it?"}},
    {"inputs": {"question": "When will order A1003 ship?"}},
]

REAL_STATUSES = {order["status"] for order in ORDERS.values()}


@scorer
def grounded_in_lookup(outputs, trace) -> Feedback:
    """Python scorer: check that the reply matches the lookup result."""
    tool_spans = trace.search_spans(span_type=SpanType.TOOL)
    if not tool_spans:
        return Feedback(value=False, rationale="The order lookup tool was not called.")

    raw = tool_spans[0].outputs or {}
    payload = raw.get("content", raw) if isinstance(raw, dict) else raw
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            payload = {}

    status = payload.get("status", "") if isinstance(payload, dict) else ""
    reply = outputs.get("response", "").lower()

    # Check the reply does not claim a status the tool did not return. Matching
    # the status verbatim would be too strict: the tool says "preparing" and a
    # good reply says "being prepared". Contradiction is what we can detect in
    # plain Python, and it is the failure that matters.
    contradictions = sorted(other for other in REAL_STATUSES - {status} if other in reply)
    if contradictions:
        return Feedback(
            value=False,
            rationale=f"Tool returned {status or 'no match'!r} but the reply claims {contradictions}.",
        )

    return Feedback(value=True, rationale=f"Reply is consistent with {status or 'no match'!r}.")


# LLM judge: use this for a quality question that plain Python cannot answer.
helpfulness = make_judge(
    name="helpfulness",
    instructions=(
        "Rate this customer-support reply from 1 to 5.\n\n"
        "Customer question: {{ inputs }}\n"
        "Agent reply: {{ outputs }}\n\n"
        "5 = specific, correct, and tells the customer what happens next.\n"
        "3 = correct but generic.\n"
        "1 = vague, evasive, or unhelpful."
    ),
    feedback_value_type=int,
    inference_params={"temperature": 0},
    model="databricks",
)

# Built-in MLflow judge: inspect the trace and judge tool choice and arguments.
SCORERS = [
    grounded_in_lookup,
    helpfulness,
    ToolCallCorrectness(name="tool_call_correctness"),
]


if __name__ == "__main__":
    experiment = configure_mlflow()
    print(f"experiment: {experiment}")

    with mlflow.start_run(run_name="support-agent-eval"):
        result = mlflow.genai.evaluate(
            data=EVAL_SET,
            predict_fn=answer,
            scorers=SCORERS,
        )

    print("\nEvaluation complete")
    for name, value in sorted(result.metrics.items()):
        print(f"{name}: {value:.2f}")
