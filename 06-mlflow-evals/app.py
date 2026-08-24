"""The app we are going to measure: a support agent with one tool.

It answers questions about orders. Anything about a specific order has to come
from the `look_up_order` tool -- the model has no way to know it otherwise.

Two versions differ only in the system prompt, so sample 6 can ask whether the
change actually helped.

Run this file directly to send one question through and record a trace.
"""

import functools
import json
import os

import mlflow
from databricks.sdk import WorkspaceClient
from mlflow.entities import SpanType
from mlflow.openai import autolog as trace_openai_calls
from openai import OpenAI

MODEL = os.environ.get("SERVING_ENDPOINT", "databricks-claude-haiku-4-5")
PROFILE = os.environ.get("DATABRICKS_PROFILE", "genai-series")

# Stands in for the database a real support agent would query.
ORDERS = {
    "A1001": {"status": "delivered", "delivered_on": "2026-08-19"},
    "A1002": {"status": "in transit", "expected_on": "2026-08-27"},
    "A1003": {"status": "preparing", "ships_on": "2026-09-01"},
}

POLICY = """Bean Box policy:
- Refunds: request within 30 days of delivery for a full refund.
- Shipping: free on orders over $40, otherwise $5.
- Orders ship on the first Tuesday of every month."""

PROMPTS = {
    "v1": f"You are a support agent for Bean Box.\n\n{POLICY}",
    "v2": f"""You are a support agent for Bean Box.

Never guess the status of an order. Always call look_up_order first, and
report exactly what it returns. If an order is not found, say so plainly.
Only answer policy questions from the policy below. Keep replies short.

{POLICY}""",
}

# What the model is told it can call. The description and parameter names are
# the model's only instructions for when and how to use it.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "look_up_order",
            "description": "Look up the current status of a customer order by its ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "Order ID, e.g. A1001"}
                },
                "required": ["order_id"],
            },
        },
    }
]


# span_type=TOOL is not decoration. MLflow's ToolCallCorrectness scorer finds
# tool calls by searching a trace for spans of this type -- without it, the
# scorer sees an agent that never used a tool.
@mlflow.trace(span_type=SpanType.TOOL)
def look_up_order(order_id: str) -> dict[str, str]:
    """Return the status of one order.

    Args:
        order_id: The ID the customer asked about.

    Returns:
        The order record, or a not-found marker.
    """
    return ORDERS.get(order_id, {"status": "not found"})


def configure_mlflow() -> str:
    """Point MLflow at Databricks-managed tracking and pick the experiment.

    Returns:
        The experiment name traces and evaluations are written to.
    """
    # Mint one OAuth token up front and hand it to every client MLflow builds
    # internally. Otherwise each judge shells out to the Databricks CLI for its
    # own token, and those refreshes collide on the OS keyring under the
    # evaluation harness's parallelism ("cache update: exit status 45").
    # The token is short-lived and never written to disk.
    workspace = WorkspaceClient(profile=PROFILE)
    os.environ["DATABRICKS_HOST"] = workspace.config.host
    os.environ["DATABRICKS_TOKEN"] = workspace.config.authenticate()["Authorization"].removeprefix(
        "Bearer "
    )

    mlflow.set_tracking_uri("databricks")

    user = workspace.current_user.me().user_name
    experiment = f"/Users/{user}/beginner-genai-evals"
    mlflow.set_experiment(experiment)
    trace_openai_calls()
    return experiment


@functools.cache
def _client() -> OpenAI:
    """Build the model client once, on first use."""
    workspace = WorkspaceClient(profile=PROFILE)
    token = workspace.config.authenticate()["Authorization"].removeprefix("Bearer ")
    return OpenAI(api_key=token, base_url=f"{workspace.config.host}/serving-endpoints")


@mlflow.trace
def answer(question: str, version: str = "v2") -> dict[str, str]:
    """Answer a customer question, calling the order tool if the model asks to.

    Args:
        question: What the customer asked.
        version: Which prompt in PROMPTS to use.

    Returns:
        A dict with the agent's reply under "response".
    """
    client = _client()
    messages = [
        {"role": "system", "content": PROMPTS[version]},
        {"role": "user", "content": question},
    ]

    first = client.chat.completions.create(
        model=MODEL, messages=messages, tools=TOOLS, max_tokens=400
    )
    reply = first.choices[0].message

    if not reply.tool_calls:
        return {"response": reply.content or ""}

    # The model asked for the tool. Run it and hand the result back so it can
    # write the real answer -- the same loop LangGraph ran for us in sample 3.
    messages.append(reply.model_dump(exclude_none=True))
    for call in reply.tool_calls:
        # tool_calls is a union type; only function calls carry .function.
        if call.type != "function":
            continue
        result = look_up_order(**json.loads(call.function.arguments))
        messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)})

    second = client.chat.completions.create(
        model=MODEL, messages=messages, tools=TOOLS, max_tokens=400
    )
    return {"response": second.choices[0].message.content or ""}


if __name__ == "__main__":
    experiment = configure_mlflow()
    print(answer("Where is my order A1002?", version="v2")["response"])
    print(f"\nTrace recorded in {experiment} — open it in the Databricks UI.")
