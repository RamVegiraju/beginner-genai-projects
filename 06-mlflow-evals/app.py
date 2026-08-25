"""The app we are going to measure: a LangGraph support agent with one tool.

Same graph as sample 3 -- model, tool, loop -- pointed at order lookups
instead of weather. Anything about a specific order has to come from the
`look_up_order` tool, because the model has no other way to know it.

Run this file directly to send one question through and record a trace.
"""

import functools
import os

import mlflow
from databricks.sdk import WorkspaceClient
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from mlflow.langchain import autolog as trace_langchain_calls

MODEL = os.environ.get("SERVING_ENDPOINT", "databricks-claude-haiku-4-5")
PROFILE = os.environ.get("DATABRICKS_PROFILE", "genai-series")

# Keep the whole series in one predictable place in the workspace. Beginners
# do not need to configure an experiment name before running this sample.
# Set MLFLOW_EXPERIMENT_NAME only when intentionally creating a separate run.
EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME", "beginner-genai-eval-final")

# Stands in for the database a real support agent would query.
ORDERS = {
    "A1001": {"status": "delivered", "delivered_on": "2026-08-19"},
    "A1002": {"status": "in transit", "expected_on": "2026-08-27"},
    "A1003": {"status": "preparing", "ships_on": "2026-09-01"},
    "A1004": {"status": "cancelled", "cancelled_on": "2026-08-11"},
    "A1005": {"status": "delayed", "expected_on": "2026-09-08"},
    "A1006": {"status": "delivered", "delivered_on": "2026-07-30"},
}

POLICY = """Bean Box policy:
- Refunds: request within 30 days of delivery for a full refund.
- Shipping: free on orders over $40, otherwise $5.
- Orders ship on the first Tuesday of every month."""

PROMPT = f"""You are a support agent for Bean Box.

Never guess the status of an order. Always call look_up_order first, and
report exactly what it returns. If an order is not found, say so plainly.
Only answer policy questions from the policy below. Keep replies short.

{POLICY}"""


@tool
def look_up_order(order_id: str) -> dict[str, str]:
    """Look up the current status of a customer order by its ID, e.g. A1001."""
    return ORDERS.get(order_id, {"status": "not found"})


@functools.cache
def _credentials() -> tuple[str, str]:
    """Mint one OAuth token and reuse it for everything. Runs once.

    Every client built after this reuses the result, so nothing shells out to
    the Databricks CLI again. That matters under the evaluation harness, which
    builds clients from several threads at once -- concurrent CLI refreshes
    contend over the OS keyring and can stall.

    Returns:
        (workspace host, bearer token)
    """
    workspace = WorkspaceClient(profile=PROFILE)
    token = workspace.config.authenticate()["Authorization"].removeprefix("Bearer ")
    return workspace.config.host, token


def configure_mlflow() -> str:
    """Point MLflow at Databricks-managed tracking and pick the experiment.

    Returns:
        The experiment name traces and evaluations are written to.
    """
    # Hand the one token to every client MLflow builds internally, so its
    # judges do not each shell out to the CLI. Short-lived, never on disk.
    #
    # This is the fix for local, browser-based (U2M) login. Running unattended?
    # Use a service principal instead -- see "Anything unattended is different"
    # in SETUP.md -- and the SDK mints tokens in-process with nothing to race.
    host, token = _credentials()
    os.environ["DATABRICKS_HOST"] = host
    os.environ["DATABRICKS_TOKEN"] = token

    mlflow.set_tracking_uri("databricks")

    # No profile= here: it picks up the token above, so this does not shell
    # out to the CLI either.
    user = WorkspaceClient().current_user.me().user_name
    experiment = f"/Users/{user}/{EXPERIMENT_NAME}"
    mlflow.set_experiment(experiment)

    # Records the graph, the model call, and every tool call as spans. The
    # TOOL spans are what MLflow's ToolCallCorrectness scorer looks for.
    trace_langchain_calls()
    return experiment


@functools.cache
def _agent():
    """Build the support agent graph. Compiled once and reused."""
    host, token = _credentials()
    llm = ChatOpenAI(
        model=MODEL,
        api_key=token,
        base_url=f"{host}/serving-endpoints",
        max_tokens=400,
    ).bind_tools([look_up_order])

    def agent(state: MessagesState) -> dict:
        system = SystemMessage(PROMPT)
        return {"messages": [llm.invoke([system] + state["messages"])]}

    # The same two-node loop as sample 3: the model decides, the tools run,
    # and control goes back to the model to write the answer.
    return (
        StateGraph(MessagesState)
        .add_node("agent", agent)
        .add_node("tools", ToolNode([look_up_order]))
        .add_edge(START, "agent")
        .add_conditional_edges("agent", tools_condition)
        .add_edge("tools", "agent")
        .compile()
    )


@mlflow.trace
def answer(question: str) -> dict[str, str]:
    """Answer a customer question, letting the agent call the tool if it wants.

    Args:
        question: What the customer asked.
    Returns:
        A dict with the agent's reply under "response".
    """
    result = _agent().invoke({"messages": [HumanMessage(question)]})
    return {"response": result["messages"][-1].content}


if __name__ == "__main__":
    experiment = configure_mlflow()
    print(answer("Where is my order A1002?")["response"])
    print(f"\nTrace recorded in {experiment} — open it in the Databricks UI.")
