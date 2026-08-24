"""The app we are going to measure: support for a fictional coffee subscription.

Two versions of the same app that differ only in the system prompt. Sample 6
asks which one is actually better, and answers with numbers instead of vibes.

Run this file directly to send one question through and record a trace.
"""

import functools
import os

import mlflow
from databricks.sdk import WorkspaceClient
from mlflow.openai import autolog as trace_openai_calls
from openai import OpenAI

MODEL = os.environ.get("SERVING_ENDPOINT", "databricks-claude-haiku-4-5")
PROFILE = os.environ.get("DATABRICKS_PROFILE", "genai-series")

# Everything the assistant is allowed to know. Anything outside this is a
# question it should decline rather than invent an answer to.
POLICY = """Bean Box policy:
- Refunds: request within 30 days of delivery for a full refund. Customers do
  not need to return the coffee.
- Shipping: free on orders over $40, otherwise $5.
- Subscriptions can be paused for up to 3 months from the account page.
- Orders ship on the first Tuesday of every month."""

# The only difference between the two versions of the app.
PROMPTS = {
    "v1": f"You are a support agent for Bean Box.\n\n{POLICY}",
    "v2": f"""You are a support agent for Bean Box.

Answer in at most three sentences.
Use only the policy below. If it does not answer the question, say you do not
know and offer to pass it to a human. Never guess.

{POLICY}""",
}


def configure_mlflow() -> str:
    """Point MLflow at Databricks-managed tracking and pick the experiment.

    Returns:
        The experiment name traces and evaluations are written to.
    """
    # One profile mechanism, used by every client MLflow builds internally.
    # Setting this AND a databricks://<profile> tracking URI conflicts.
    os.environ["DATABRICKS_CONFIG_PROFILE"] = PROFILE
    mlflow.set_tracking_uri("databricks")

    user = WorkspaceClient(profile=PROFILE).current_user.me().user_name
    experiment = f"/Users/{user}/beginner-genai-evals"
    mlflow.set_experiment(experiment)

    # Records the model call itself as a span inside each trace, so you can see
    # the exact prompt that was sent and the tokens it cost.
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
    """Answer a customer question using one version of the system prompt.

    The @mlflow.trace decorator is the whole cost of observability here: every
    call is recorded with its inputs, outputs, latency, and token usage.

    Args:
        question: What the customer asked.
        version: Which prompt in PROMPTS to use.

    Returns:
        A dict with the assistant's reply under "response".
    """
    response = _client().chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": PROMPTS[version]},
            {"role": "user", "content": question},
        ],
        max_tokens=250,
    )
    return {"response": response.choices[0].message.content or ""}


if __name__ == "__main__":
    experiment = configure_mlflow()
    result = answer("How long do I have to ask for a refund?", version="v2")
    print(result["response"])
    print(f"\nTrace recorded in {experiment} — open it in the Databricks UI.")
