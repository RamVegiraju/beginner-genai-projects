"""
Two kinds of memory.

Sample 2's chatbot kept the message list in RAM. Close the tab and you were a
stranger again. This one writes things down -- two different ways, because
"remember me" is really two different problems:

  SHORT-TERM   the conversation you are in right now.
               Keyed by thread_id. Saved by a CHECKPOINTER after every step.
               Answers "what did I just say?"

  LONG-TERM    what the assistant knows about YOU, in every conversation.
               Keyed by user_id. Saved in a STORE, written by distill.py.
               Answers "how do you like your answers?"

Both live in one file, memory.db. Delete it and the bot forgets you entirely.

Run:
    python chat.py trip          # talk to it, Ctrl-D when done
    python chat.py trip          # same thread -- it remembers
    python chat.py groceries     # new thread -- blank slate, same preferences
"""

import os
import sys

from databricks.sdk import WorkspaceClient
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.runtime import Runtime
from langgraph.store.sqlite import SqliteStore

MODEL = os.environ.get("SERVING_ENDPOINT", "databricks-claude-haiku-4-5")
PROFILE = os.environ.get("DATABRICKS_PROFILE", "DEFAULT")

DB = "memory.db"
USER_ID = "ram"  # long-term memory belongs to a PERSON...
THREAD_ID = sys.argv[1] if len(sys.argv) > 1 else "default"  # ...short-term to a CONVERSATION


# --- The model (same three lines as every sample so far) ---------------------

w = WorkspaceClient(profile=PROFILE)
token = w.config.authenticate()["Authorization"].removeprefix("Bearer ")

llm = ChatOpenAI(
    model=MODEL,
    api_key=token,
    base_url=f"{w.config.host}/serving-endpoints",
    max_tokens=500,
)


# --- The one node ------------------------------------------------------------


def agent(state: MessagesState, runtime: Runtime) -> dict:
    # Long-term memory: whatever distill.py has learned about this user, read
    # fresh every turn. Nothing here is specific to the current conversation.
    facts = [
        item.value["fact"]
        for item in runtime.store.search((USER_ID, "preferences"), limit=100)
    ]

    system = "You are a helpful assistant."
    if facts:
        system += " Things you know about this user:\n- " + "\n- ".join(facts)

    # Note what is NOT happening: the system message is built here and thrown
    # away. It never enters state, so it is never checkpointed. Change a
    # preference and the very next turn picks it up -- no stale copy on disk.
    #
    # state["messages"] is the short-term half, and LangGraph loaded it off
    # disk for us before this function ran.
    return {"messages": [llm.invoke([SystemMessage(system)] + state["messages"])]}


# --- Wiring ------------------------------------------------------------------
# The checkpointer and the store are two tables in the same SQLite file. The
# graph itself is sample 2's chatbot with no tools -- memory is the only new
# idea here.

with SqliteSaver.from_conn_string(DB) as checkpointer, SqliteStore.from_conn_string(DB) as store:
    store.setup()  # creates the store table on first run

    graph = (
        StateGraph(MessagesState)
        .add_node("agent", agent)
        .add_edge(START, "agent")
        .compile(checkpointer=checkpointer, store=store)
    )

    # thread_id is the whole trick. Same id, same conversation -- even across
    # a full process restart. Different id, clean slate.
    config = {"configurable": {"thread_id": THREAD_ID}}

    prior = graph.get_state(config).values.get("messages", [])
    known = store.search((USER_ID, "preferences"), limit=100)

    print(f"thread {THREAD_ID!r}: {len(prior)} messages loaded from {DB}")
    print(f"user {USER_ID!r}: {len(known)} things remembered about you")
    if known:
        for item in known:
            print(f"  - {item.value['fact']}")
    print("\nCtrl-D to exit.")

    while True:
        try:
            text = input("\nyou > ").strip()
        except EOFError:
            break
        if not text:
            continue

        # We send ONE message. Sample 2 had to resend the whole list by hand;
        # the checkpointer does that part now.
        result = graph.invoke({"messages": [HumanMessage(text)]}, config)
        print(f"\nbot > {result['messages'][-1].content}")

    total = len(graph.get_state(config).values.get("messages", []))
    print(f"\n\nsaved. thread {THREAD_ID!r} now holds {total} messages.")
    print(f"run `python distill.py {THREAD_ID}` to keep what matters long-term.")
