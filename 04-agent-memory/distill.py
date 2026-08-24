"""
Distillation: turning a finished conversation into a short user profile.

Short-term memory grows forever and is resent every turn, so it gets more
expensive the longer it lives. Distillation is the counterweight: when a
conversation ends, read it once with an LLM, keep a few durable facts about
the person, and drop the rest.

The model is given the profile it already has plus the conversation that just
ended, and returns the complete updated profile. Rewriting rather than
appending is what lets it merge duplicates and correct facts that changed.

app.py imports these two functions.
"""

from langchain_core.messages import HumanMessage

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


def distill(llm, messages, known: list[str]) -> list[str]:
    """Fold a finished conversation into the user's profile.

    Returns the full replacement profile, or an empty list if there is
    nothing worth keeping.
    """
    transcript = "\n".join(f"{m.type}: {m.content}" for m in messages)
    prompt = PROMPT.format(
        known="\n".join(f"- {fact}" for fact in known) or "(nothing yet)",
        transcript=transcript,
    )

    reply = llm.invoke([HumanMessage(prompt)]).content
    facts = [line.strip("-* ").strip() for line in reply.splitlines() if line.strip()]
    return [] if facts == ["NOTHING"] else facts


def remember(store, namespace, facts: list[str]) -> None:
    """Replace the user's profile with `facts`."""
    for item in store.search(namespace, limit=100):
        store.delete(namespace, item.key)

    for i, fact in enumerate(facts):
        store.put(namespace, f"fact-{i}", {"fact": fact})
