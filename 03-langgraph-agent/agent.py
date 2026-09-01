"""Use LangGraph when the application needs a custom review loop.

The `draft` node runs an ordinary LangChain agent. The `check` node applies an
application rule in Python. A conditional edge either accepts the answer or
sends it back for one revision:

    START -> draft -> check -> END
               ^         |
               +-- retry-+

Run `simple_agent.py` first to see the standard loop without custom wiring.
"""

from typing import Literal

from langchain.agents import create_agent
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict
from weather_tool import get_weather, make_model

MAX_DRAFTS = 2

weather_agent = create_agent(
    model=make_model(),
    tools=[get_weather],
    system_prompt="Use the weather tool and only report facts it returns.",
)


class State(TypedDict):
    """Values shared by every node in the review workflow."""

    question: str
    answer: str
    feedback: str
    drafts: int


def draft_answer(state: State) -> dict:
    """Ask the LangChain agent for an answer or a revised answer."""
    messages = [{"role": "user", "content": state["question"]}]
    if state.get("feedback"):
        messages += [
            {"role": "assistant", "content": state["answer"]},
            {"role": "user", "content": f"Revise that answer. {state['feedback']}"},
        ]

    result = weather_agent.invoke({"messages": messages})
    return {"answer": str(result["messages"][-1].content), "drafts": state["drafts"] + 1}


def check_answer(state: State) -> dict:
    """Check that weather is not presented as proof of a flight delay."""
    answer = state["answer"].lower()
    if "airline" in answer and "cannot determine" in answer:
        return {"feedback": ""}
    return {
        "feedback": (
            "Explain that weather readings alone cannot determine flight delays, "
            "and recommend checking the airline."
        )
    }


def route_after_check(state: State) -> Literal["draft", "fallback", "done"]:
    """Choose the next node from the check result and retry count."""
    if not state["feedback"]:
        return "done"
    if state["drafts"] < MAX_DRAFTS:
        return "draft"
    return "fallback"


def add_fallback(state: State) -> dict:
    """Guarantee the rule even if the model did not fix its second draft."""
    return {
        "answer": (
            f"{state['answer']}\n\nWeather readings alone cannot determine flight delays. "
            "Check your airline for the latest status."
        )
    }


graph = (
    StateGraph(State)
    .add_node("draft", draft_answer)
    .add_node("check", check_answer)
    .add_node("fallback", add_fallback)
    .add_edge(START, "draft")
    .add_edge("draft", "check")
    .add_conditional_edges(
        "check",
        route_after_check,
        {"draft": "draft", "fallback": "fallback", "done": END},
    )
    .add_edge("fallback", END)
    .compile()
)


if __name__ == "__main__":
    question = "I'm flying out of Chicago today. Should I expect weather delays?"
    result = graph.invoke({"question": question, "answer": "", "feedback": "", "drafts": 0})
    print(f"Review loop used {result['drafts']} draft(s).\n")
    print(result["answer"])
