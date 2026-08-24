"""
Distillation: turning a conversation into the part worth keeping.

The chatbot saves every message forever, and that is the problem. History
grows without bound, you resend all of it every turn, and you pay for all of
it every turn. Short-term memory does not scale.

Long-term memory is the opposite move: read the transcript ONCE, throw almost
all of it away, and keep the handful of durable facts about the person.

app.py imports these two functions. There is nothing to run here.
"""

from langchain_core.messages import HumanMessage

# The whole skill is in this prompt, and one word does the work: DURABLE.
# That is the line between a preference and a passing detail.
PROMPT = """Read this conversation and list durable facts about the user --
things that would still be true in a different conversation next month.

Keep: dietary needs, where they live, their job, how they like answers written.
Skip: today's topic, questions they asked, anything about this conversation.

One fact per line, third person, no bullets or numbering.
If there is nothing worth keeping, write NOTHING.

Conversation:
{transcript}"""


def distill(llm, messages) -> list[str]:
    """Read a transcript, return the few facts worth keeping about the user.

    Worth saying out loud: this is a model summarising a model's output, and
    the result gets pasted into every future conversation. It is a judgement
    call, not a database write. It will sometimes keep the wrong thing.
    """
    transcript = "\n".join(f"{m.type}: {m.content}" for m in messages)
    reply = llm.invoke([HumanMessage(PROMPT.format(transcript=transcript))]).content
    facts = [line.strip("-* ").strip() for line in reply.splitlines() if line.strip()]
    return [] if facts == ["NOTHING"] else facts


def remember(store, namespace, thread_id, facts) -> None:
    """Replace this thread's facts in long-term memory.

    Keys are prefixed with the thread they came from, so re-distilling the
    same conversation refreshes its facts instead of piling up duplicates --
    while facts learned from OTHER conversations are left alone.
    """
    for item in store.search(namespace, limit=100):
        if item.key.startswith(f"{thread_id}:"):
            store.delete(namespace, item.key)

    for i, fact in enumerate(facts):
        store.put(namespace, f"{thread_id}:{i}", {"fact": fact})
