"""
Distillation: turning a conversation into the part worth keeping.

chat.py saves every message forever, and that is the problem. History grows
without bound, you resend all of it every turn, and you pay for all of it
every turn. Short-term memory does not scale.

Long-term memory is the opposite move: read the transcript ONCE, throw almost
all of it away, and keep the handful of durable facts about the person. An
LLM does the summarising. Note that this is a model reading a model's output
-- the memory is a judgement call, not a database write, so it can be wrong.

Run:  python distill.py trip
"""

import os
import sys

from databricks.sdk import WorkspaceClient
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.store.sqlite import SqliteStore

MODEL = os.environ.get("SERVING_ENDPOINT", "databricks-claude-haiku-4-5")
PROFILE = os.environ.get("DATABRICKS_PROFILE", "DEFAULT")

DB = "memory.db"
USER_ID = "ram"
THREAD_ID = sys.argv[1] if len(sys.argv) > 1 else "default"

# The whole skill is in this prompt. "Durable" is doing the work: it is the
# line between a preference and a passing detail.
PROMPT = """Read this conversation and list durable facts about the user --
things that would still be true in a different conversation next month.

Keep: dietary needs, where they live, their job, how they like answers written.
Skip: today's topic, questions they asked, anything about this conversation.

One fact per line, third person, no bullets or numbering.
If there is nothing worth keeping, write NOTHING.

Conversation:
{transcript}"""


w = WorkspaceClient(profile=PROFILE)
token = w.config.authenticate()["Authorization"].removeprefix("Bearer ")

llm = ChatOpenAI(
    model=MODEL,
    api_key=token,
    base_url=f"{w.config.host}/serving-endpoints",
    max_tokens=300,
)

with SqliteSaver.from_conn_string(DB) as checkpointer, SqliteStore.from_conn_string(DB) as store:
    store.setup()

    # Read the thread back off disk. We only need the checkpointer, so the
    # graph is a stub -- get_state() does not care what the nodes do.
    graph = (
        StateGraph(MessagesState)
        .add_node("noop", lambda state: {})
        .add_edge(START, "noop")
        .compile(checkpointer=checkpointer)
    )
    config = {"configurable": {"thread_id": THREAD_ID}}
    messages = graph.get_state(config).values.get("messages", [])

    if not messages:
        sys.exit(f"thread {THREAD_ID!r} is empty. run `python chat.py {THREAD_ID}` first.")

    transcript = "\n".join(f"{m.type}: {m.content}" for m in messages)
    print(f"reading {len(messages)} messages from thread {THREAD_ID!r}...\n")

    reply = llm.invoke([HumanMessage(PROMPT.format(transcript=transcript))]).content
    facts = [line.strip("-* ").strip() for line in reply.splitlines() if line.strip()]

    if not facts or facts == ["NOTHING"]:
        sys.exit("nothing worth keeping.")

    # Re-distilling the same thread should replace its facts, not pile more on
    # top, so clear this thread's old entries first. Facts learned from OTHER
    # threads are keyed differently and survive.
    namespace = (USER_ID, "preferences")
    for item in store.search(namespace, limit=100):
        if item.key.startswith(f"{THREAD_ID}:"):
            store.delete(namespace, item.key)

    for i, fact in enumerate(facts):
        store.put(namespace, f"{THREAD_ID}:{i}", {"fact": fact})
        print(f"  remembered: {fact}")

    print(f"\n{len(messages)} messages -> {len(facts)} facts.")
    print("start a brand new thread and it will still know these:")
    print(f"  python chat.py something-new")
