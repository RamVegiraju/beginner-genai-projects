"""Score both versions of the app on the same questions, then compare.

"It seems better" is not a quality bar. This runs a fixed set of questions
through each prompt version and has LLM judges score the answers, so the
question "did my change help?" gets an actual number.

Run:  python evaluate.py
"""

import mlflow
import mlflow.genai
from app import answer, configure_mlflow
from mlflow.genai.scorers import Correctness, Guidelines, RelevanceToQuery

# The eval set. Hand-written, small, and the most valuable thing here: these
# are the questions you actually care about getting right.
#
# "inputs" is passed to the app as keyword arguments.
# "expectations" is ground truth -- what a correct answer has to contain.
EVAL_SET = [
    {
        "inputs": {"question": "How long do I have to ask for a refund?"},
        "expectations": {"expected_facts": ["within 30 days of delivery", "full refund"]},
    },
    {
        "inputs": {"question": "Do I have to mail the coffee back to get my money?"},
        "expectations": {"expected_facts": ["no need to return the coffee"]},
    },
    {
        "inputs": {"question": "When will my order actually ship?"},
        "expectations": {"expected_facts": ["first Tuesday of every month"]},
    },
    {
        "inputs": {"question": "Is shipping free?"},
        "expectations": {"expected_facts": ["free on orders over $40", "otherwise $5"]},
    },
    {
        # Deliberately not in the policy. A good answer admits that; a bad one
        # invents a grind policy that does not exist.
        "inputs": {"question": "Can I change the grind size on my subscription?"},
        "expectations": {
            "expected_facts": [
                "the policy does not cover grind size",
                "offers to hand the question to a human",
            ]
        },
    },
]

SCORERS = [
    # Did the answer contain the facts we said it must? Needs expectations.
    Correctness(),
    # Did it actually address the question? Needs no ground truth.
    RelevanceToQuery(),
    # Our own rule. Guidelines see the request and the response.
    Guidelines(name="concise", guidelines="The response must be at most three sentences."),
]


def score(version: str) -> dict[str, float]:
    """Run the whole eval set through one prompt version.

    Args:
        version: Which prompt in app.PROMPTS to evaluate.

    Returns:
        The aggregate metrics, keyed like "correctness/mean".
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

    print("\n" + "=" * 46)
    print(f"{'metric':<24}{'v1':>10}{'v2':>10}")
    print("-" * 46)
    for metric in sorted({m for scores in results.values() for m in scores}):
        v1, v2 = results["v1"].get(metric), results["v2"].get(metric)
        print(f"{metric:<24}{v1:>10.2f}{v2:>10.2f}")
    print("=" * 46)
    print("\nOpen the experiment in Databricks to see every answer and the")
    print("judge's reasoning for each score.")
