"""
A chatbot with two kinds of memory.

  SHORT-TERM   the messages in this conversation, keyed by thread_id.
               Saved to SQLite by a checkpointer after every step.

  LONG-TERM    a short profile of the user, keyed by user_id.
               Written by distillation when a conversation ends.

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
USER_ID = os.environ.get("USER_ID", "demo-user")

NAMESPACE = (USER_ID, "profile")

# Resolved against this file, not the working directory, so the app finds the
# same database wherever you launch it from.
DB = str(Path(__file__).parent / "memory.db")


@st.cache_resource
def build():
    """Create the model client, the two memories, and the graph. Runs once."""
    w = WorkspaceClient(profile=PROFILE)
    token = w.config.authenticate()["Authorization"].removeprefix("Bearer ")
    llm = ChatOpenAI(
        model=MODEL,
        api_key=token,
        base_url=f"{w.config.host}/serving-endpoints",
        max_tokens=500,
    )

    # One file, two connections. The store needs autocommit and the
    # checkpointer does not, so they cannot share a connection.
    # check_same_thread=False is required: Streamlit and LangGraph use
    # different threads.
    checkpointer = SqliteSaver(sqlite3.connect(DB, check_same_thread=False))
    store = SqliteStore(sqlite3.connect(DB, check_same_thread=False, isolation_level=None))
    store.setup()

    # Name the parameter `store` and annotate it `BaseStore`: that is how
    # LangGraph knows to inject the store the graph was compiled with.
    def agent(state: MessagesState, store: BaseStore) -> dict:
        profile = [item.value["fact"] for item in store.search(NAMESPACE, limit=100)]

        system = "You are a helpful assistant."
        if profile:
            system += " Things you know about this user:\n- " + "\n- ".join(profile)

        # Built fresh each turn and never added to state, so the profile is
        # re-read every time instead of being frozen into a checkpoint.
        return {"messages": [llm.invoke([SystemMessage(system)] + state["messages"])]}

    graph = (
        StateGraph(MessagesState)
        .add_node("agent", agent)
        .add_edge(START, "agent")
        .compile(checkpointer=checkpointer, store=store)
    )
    return graph, store, llm


graph, store, llm = build()


def thread_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def next_empty_thread() -> str:
    """Find an unused thread name, so a new conversation is always blank."""
    n = 1
    while graph.get_state(thread_config(f"chat-{n}")).values.get("messages"):
        n += 1
    return f"chat-{n}"


if "thread" not in st.session_state:
    st.session_state.thread = "chat-1"

config = thread_config(st.session_state.thread)

# Both memories are read from the database, not from st.session_state.
messages = graph.get_state(config).values.get("messages", [])
profile = store.search(NAMESPACE, limit=100)

st.title("Chatbot with a memory")

for message in messages:
    if message.type in ("human", "ai") and message.content:
        with st.chat_message("user" if message.type == "human" else "assistant"):
            st.markdown(message.content)

if prompt := st.chat_input("Tell it something about yourself"):
    with st.chat_message("user"):
        st.markdown(prompt)

    # Only the new message is sent. The checkpointer loads the history before
    # the node runs and saves the reply after.
    with st.chat_message("assistant"):
        st.write_stream(
            chunk.content
            for chunk, _metadata in graph.stream(
                {"messages": [HumanMessage(prompt)]}, config, stream_mode="messages"
            )
        )
    st.rerun()


with st.sidebar:
    st.subheader("Short-term")
    st.caption(f"thread `{st.session_state.thread}` — {len(messages)} messages")
    st.caption("Resent to the model every turn, so every turn costs more.")
    st.json([{"role": m.type, "content": m.content} for m in messages], expanded=False)

    st.divider()

    st.subheader("Long-term")
    st.caption(f"user `{USER_ID}` — {len(profile)} facts, carried into every conversation")
    for item in profile:
        st.markdown(f"- {item.value['fact']}")
    if not profile:
        st.caption("_Empty. Say something about yourself, then end the conversation._")

    st.divider()

    if st.button(
        "End conversation", disabled=not messages, use_container_width=True, type="primary"
    ):
        updated = distill(llm, messages, [item.value["fact"] for item in profile])

        # An empty result means the model returned nothing usable. Skip the
        # write rather than erasing a profile built from real conversations.
        if updated:
            remember(store, NAMESPACE, updated)
            st.toast(f"{len(messages)} messages → {len(updated)} facts")

        st.session_state.thread = next_empty_thread()
        st.rerun()

    st.caption("Distills this conversation into the profile above, then starts a fresh one.")

    if st.button("Forget me", disabled=not profile, use_container_width=True):
        for item in profile:
            store.delete(NAMESPACE, item.key)
        st.rerun()
