"""Turn a finished conversation into a short, reusable user profile.

Short-term memory grows forever and is resent every turn, so it gets more
expensive the longer it lives. Distillation is the counterweight: when a
conversation ends, read it once with an LLM, keep a few durable facts about
the person, and leave conversation-specific details out of future prompts.

The model is given the profile it already has plus the conversation that just
ended, and returns the complete updated profile. Rewriting rather than
appending is what lets it merge duplicates and correct facts that changed.

Distillation does not delete the old thread or its checkpoints. ``app.py``
stores the distilled profile and then switches the UI to a new empty thread.
"""

from collections.abc import Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.store.base import BaseStore

PROMPT = """You keep a short profile on a user so you can help them better in
future conversations.

What you already know about them:
{known}

A conversation with them just ended:
{transcript}

Write the UPDATED profile: durable facts that would still be true in a
different conversation next month.

Keep: dietary needs, where they live, their job, how they like answers written.
Skip: this conversation's topic, questions they asked, one-off details.
Merge anything that duplicates what you already know.
If the conversation contradicts what you already know, prefer the new
information and drop the old.

Reply with the COMPLETE updated profile, not just what changed.
One fact per line, third person, no bullets or numbering.
If there is nothing worth keeping at all, reply NOTHING."""


def distill(llm: BaseChatModel, messages: Sequence[BaseMessage], known: list[str]) -> list[str]:
    """Use the model to create a complete updated user profile.

    The model receives both the existing profile and the conversation that
    just ended. It can keep new durable facts, merge duplicates, and replace
    facts that are no longer true.

    Args:
        llm: Chat model used for the distillation call.
        messages: Complete message history from the finished thread.
        known: Facts already stored in the user's long-term profile.

    Returns:
        The full replacement profile, or an empty list when there is nothing
        useful to store.
    """
    transcript = "\n".join(f"{m.type}: {m.content}" for m in messages)
    prompt = PROMPT.format(
        known="\n".join(f"- {fact}" for fact in known) or "(nothing yet)",
        transcript=transcript,
    )

    reply = llm.invoke([HumanMessage(prompt)]).content
    facts = [line.strip("-* ").strip() for line in reply.splitlines() if line.strip()]
    return [] if facts == ["NOTHING"] else facts


def remember(store: BaseStore, namespace: tuple[str, ...], facts: list[str]) -> None:
    """Replace every stored profile fact with a new complete profile.

    Args:
        store: Long-term memory store that contains the user's profile.
        namespace: Store location that identifies the user and memory type.
        facts: Complete set of facts that should replace the current profile.
    """
    for item in store.search(namespace, limit=100):
        store.delete(namespace, item.key)

    for i, fact in enumerate(facts):
        store.put(namespace, f"fact-{i}", {"fact": fact})
