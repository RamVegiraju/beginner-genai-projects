"""
The chatbot from sample 2, with a memory.

Sample 2's sidebar showed you one list and made one point: the "memory" was a
list in RAM, and closing the tab destroyed it.

This sidebar shows you TWO lists, and the point is that they behave
differently:

  SHORT-TERM   the messages in this conversation. On disk, keyed by thread_id.
               Start a new thread and it empties.

  LONG-TERM    facts about you. On disk, keyed by user_id.
               Start a new thread and it stays.

Try it in this order, watching the sidebar the whole time:
  1. "I'm vegetarian and I live in Seattle. Keep answers short."
  2. Watch the message count climb.
  3. Hit "Distill" -- the messages collapse into two or three facts.
  4. Hit "New thread" -- messages go to 0, facts stay.
  5. Ask "what should I cook tonight?" -- it has never met you, and it knows.

Run:  streamlit run app.py
"""

import os
import sqlite3
from pathlib import Path

import streamlit as st
from databricks.sdk import WorkspaceClient
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.store.base import BaseStore
from langgraph.store.sqlite import SqliteStore

from distill import distill, remember

MODEL = os.environ.get("SERVING_ENDPOINT", "databricks-claude-haiku-4-5")
PROFILE = os.environ.get("DATABRICKS_PROFILE", "DEFAULT")

# Next to this file, NOT the working directory. Otherwise running the app from
# the repo root quietly creates a second, empty memory.db and the bot has
# amnesia with no error message.
DB = str(Path(__file__).parent / "memory.db")

USER_ID = "ram"  # long-term memory belongs to a PERSON...
NAMESPACE = (USER_ID, "preferences")


@st.cache_resource
def build():
    w = WorkspaceClient(profile=PROFILE)
    token = w.config.authenticate()["Authorization"].removeprefix("Bearer ")
    llm = ChatOpenAI(
        model=MODEL,
        api_key=token,
        base_url=f"{w.config.host}/serving-endpoints",
        max_tokens=500,
    )

    # Two connections to one file. The checkpointer and the store each want
    # their own -- sharing one raises "cannot start a transaction within a
    # transaction", because the store runs in autocommit and the saver does not.
    #
    # check_same_thread=False is required, not optional: Streamlit runs this
    # script on a worker thread and LangGraph writes checkpoints from another.
    saver = SqliteSaver(sqlite3.connect(DB, check_same_thread=False))
    store = SqliteStore(sqlite3.connect(DB, check_same_thread=False, isolation_level=None))
    store.setup()  # creates the store tables; the saver does its own, lazily

    # Naming a parameter `store` is what makes LangGraph hand you the store it
    # was compiled with. It matches on the NAME, and on the annotation being
    # BaseStore -- annotate it SqliteStore and you will silently get nothing.
    def agent(state: MessagesState, store: BaseStore) -> dict:
        facts = [item.value["fact"] for item in store.search(NAMESPACE, limit=100)]

        system = "You are a helpful assistant."
        if facts:
            system += " Things you know about this user:\n- " + "\n- ".join(facts)

        # The system message is built here and thrown away. It never enters
        # state, so it is never checkpointed. Change a preference and the very
        # next turn picks it up -- no stale copy frozen into an old thread.
        return {"messages": [llm.invoke([SystemMessage(system)] + state["messages"])]}

    graph = (
        StateGraph(MessagesState)
        .add_node("agent", agent)
        .add_edge(START, "agent")
        .compile(checkpointer=saver, store=store)
    )
    return graph, store, llm


graph, store, llm = build()


def thread_config(name):
    return {"configurable": {"thread_id": name}}


def next_empty_thread():
    """Find a thread name nobody has used yet.

    Just incrementing a counter is not enough: session state resets when the
    server restarts, so the counter would walk back over conversations that
    already exist and "New thread" would silently reopen an old one.
    """
    n = 1
    while graph.get_state(thread_config(f"chat-{n}")).values.get("messages"):
        n += 1
    return f"chat-{n}"


if "thread" not in st.session_state:
    st.session_state.thread = "chat-1"

# thread_id is the whole trick. Same id, same conversation -- even after you
# stop the server. Different id, clean slate.
config = thread_config(st.session_state.thread)

# The source of truth is the DATABASE, not a Python variable. Sample 2 read
# st.session_state here; we read the disk.
messages = graph.get_state(config).values.get("messages", [])
facts = store.search(NAMESPACE, limit=100)

st.title("Chatbot with a memory")

for message in messages:
    if message.type in ("human", "ai") and message.content:
        with st.chat_message("user" if message.type == "human" else "assistant"):
            st.markdown(message.content)

if prompt := st.chat_input("Tell it something about yourself"):
    with st.chat_message("user"):
        st.markdown(prompt)

    # We send ONE message. Sample 2 resent the whole list by hand; the
    # checkpointer loads it and saves it for us now.
    with st.chat_message("assistant"):
        st.write_stream(
            chunk.content
            for chunk, _metadata in graph.stream(
                {"messages": [HumanMessage(prompt)]}, config, stream_mode="messages"
            )
        )
    st.rerun()  # re-draw from disk, so the sidebar counts update


with st.sidebar:
    st.subheader("Short-term")
    st.caption(f"thread `{st.session_state.thread}` — {len(messages)} messages on disk")
    st.caption("Sent to the model every turn. Grows forever. You pay for all of it.")
    st.json([{"role": m.type, "content": m.content} for m in messages], expanded=False)

    if st.button("New thread", use_container_width=True):
        st.session_state.thread = next_empty_thread()
        st.rerun()

    st.divider()

    st.subheader("Long-term")
    st.caption(f"user `{USER_ID}` — {len(facts)} facts, sent in EVERY thread")
    if facts:
        for item in facts:
            st.markdown(f"- {item.value['fact']}")
    else:
        st.caption("_Nothing yet. Say something about yourself, then distill._")

    if st.button("Distill this conversation", disabled=not messages, use_container_width=True):
        remember(store, NAMESPACE, st.session_state.thread, distill(llm, messages))
        st.rerun()

    if st.button("Forget me", disabled=not facts, use_container_width=True):
        for item in facts:
            store.delete(NAMESPACE, item.key)
        st.rerun()
