"""A chatbot that demonstrates short-term and long-term memory.

Short-term memory is the message history for one conversation. A LangGraph
checkpointer saves graph state after each super-step and uses ``thread_id`` to
load the latest state when that conversation continues.

Long-term memory is a small user profile shared across conversations. A
LangGraph store keeps these facts under ``user_id``. When a conversation ends,
the app distills useful facts into this profile and starts a new thread. The
old thread's checkpoints remain in SQLite as history.

Run with ``streamlit run app.py``.
"""

import os
import sqlite3
from pathlib import Path

import streamlit as st
from databricks.sdk import WorkspaceClient
from distill import distill, remember
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from langgraph.store.sqlite import SqliteStore

MODEL = os.environ.get("SERVING_ENDPOINT", "databricks-claude-haiku-4-5")
PROFILE = os.environ.get("DATABRICKS_PROFILE")
USER_ID = os.environ.get("USER_ID", "demo-user")

NAMESPACE = (USER_ID, "profile")

# Resolved against this file, not the working directory, so the app finds the
# same database wherever you launch it from.
DB = str(Path(__file__).parent / "memory.db")


@st.cache_resource
def build() -> tuple[CompiledStateGraph, SqliteSaver, SqliteStore, ChatOpenAI]:
    """Build and cache the model, memory backends, and graph.

    Streamlit reruns this file after every interaction. ``st.cache_resource``
    keeps these expensive, shared resources alive instead of rebuilding them
    on every rerun.

    Returns:
        The compiled graph, short-term memory checkpointer, long-term memory
        store, and chat model client.
    """
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
    # SqliteSaver is the demo-grade checkpointer -- its own docs call it
    # "lightweight, synchronous... demos and small projects". Swap in Postgres
    # for anything real; thread_id and the node code stay exactly the same.
    checkpointer = SqliteSaver(sqlite3.connect(DB, check_same_thread=False))
    store = SqliteStore(sqlite3.connect(DB, check_same_thread=False, isolation_level=None))
    store.setup()

    # Name the parameter `store` and annotate it `BaseStore`: that is how
    # LangGraph knows to inject the store the graph was compiled with.
    def agent(state: MessagesState, store: BaseStore) -> dict[str, list[BaseMessage]]:
        """Generate one reply using short-term and long-term memory.

        Before this node runs, the checkpointer restores the latest messages
        for the configured ``thread_id``. The node also reads the user's
        cross-thread profile from the store and adds it to the system prompt.

        Args:
            state: Current graph state, including this thread's messages.
            store: Long-term memory store injected by LangGraph.

        Returns:
            A state update containing only the new assistant message. The
            ``MessagesState`` reducer appends it to the conversation.
        """
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
    return graph, checkpointer, store, llm


graph, checkpointer, store, llm = build()


def thread_config(thread_id: str) -> RunnableConfig:
    """Build the configuration that selects a conversation.

    Args:
        thread_id: Stable identifier for one conversation.

    Returns:
        LangGraph configuration that tells the checkpointer which thread to
        load and save. Reusing the ID resumes that conversation; a new ID
        starts with empty state.
    """
    return {"configurable": {"thread_id": thread_id}}


def saved_thread_ids(checkpointer: SqliteSaver) -> list[str]:
    """List every conversation ID currently known to the checkpointer.

    One thread has many checkpoints, so the IDs returned by ``list(None)`` are
    deduplicated. Only root-graph checkpoints are included; this sample has no
    subgraphs, but the namespace check keeps the intent explicit.

    Args:
        checkpointer: SQLite checkpointer that stores conversation state.

    Returns:
        Sorted thread IDs discovered from saved checkpoints.
    """
    return sorted(
        {
            str(checkpoint.config["configurable"]["thread_id"])
            for checkpoint in checkpointer.list(None)
            if checkpoint.config["configurable"].get("checkpoint_ns", "") == ""
        }
    )


def next_empty_thread(existing_thread_ids: list[str]) -> str:
    """Create the first numbered thread ID that is not already in use.

    Args:
        existing_thread_ids: Thread IDs already saved or currently selected.

    Returns:
        An unused thread ID such as ``chat-2`` so a new conversation starts
        with empty short-term memory.
    """
    existing = set(existing_thread_ids)
    n = 1
    while f"chat-{n}" in existing:
        n += 1
    return f"chat-{n}"


def open_selected_thread() -> None:
    """Make the thread chosen in the sidebar the active conversation.

    Streamlit runs this callback before rerunning the page, so every state read
    later in the script uses the newly selected ``thread_id``.
    """
    selected_thread = st.session_state.thread_picker
    st.session_state.thread = selected_thread
    st.session_state.memory_event = (
        f"Opened {selected_thread}. The checkpointer restored only this thread's latest state."
    )


if "thread" not in st.session_state:
    st.session_state.thread = "chat-1"

# Discover saved conversations from the checkpointer instead of maintaining a
# second, hard-coded thread list. Include the selected thread because a brand-
# new empty thread has no checkpoint yet and therefore is not returned by list.
thread_options = sorted(set(saved_thread_ids(checkpointer)) | {st.session_state.thread})

# Buttons change the active thread programmatically. Synchronize the selector
# before recreating its widget; Streamlit does not allow changing a widget's
# state after that widget has already rendered during the same run.
if st.session_state.pop("sync_thread_picker", False) or (
    st.session_state.get("thread_picker") not in thread_options
):
    st.session_state.thread_picker = st.session_state.thread

with st.sidebar:
    st.subheader("Conversations")
    st.selectbox(
        "Open a thread",
        thread_options,
        index=thread_options.index(st.session_state.thread),
        key="thread_picker",
        on_change=open_selected_thread,
        help="Selecting an ID reloads only that thread's latest checkpoint.",
    )

    if st.button("Start new empty thread", use_container_width=True):
        new_thread = next_empty_thread(thread_options)
        st.session_state.thread = new_thread
        st.session_state.sync_thread_picker = True
        st.session_state.memory_event = (
            f"Started {new_thread} with empty conversation state. Long-term "
            "profile facts remain available from the Store."
        )
        st.rerun()

config = thread_config(st.session_state.thread)

# Both memories are read from the database, not from st.session_state.
snapshot = graph.get_state(config)
messages = snapshot.values.get("messages", [])
checkpoint_history = list(graph.get_state_history(config))
profile = store.search(NAMESPACE, limit=100)

st.title("Chatbot with a memory")

# ``End conversation`` sets this message immediately before switching threads.
# Showing it after the rerun makes the handoff from distillation to a new
# conversation visible instead of hiding it inside the button click.
if memory_event := st.session_state.pop("memory_event", None):
    st.success(memory_event)

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
    st.subheader("What the next reply loads")
    st.markdown(
        f"""
1. **Checkpointer → state:** restored `{len(messages)}` messages from
   thread `{st.session_state.thread}`.
2. **Store → profile:** loaded `{len(profile)}` durable facts for user
   `{USER_ID}`.
3. **Agent:** combines both when the next message arrives.
"""
    )

    st.divider()

    st.subheader("Short-term")
    st.caption(f"thread `{st.session_state.thread}` — {len(messages)} messages")
    st.caption("Restored by the checkpointer and resent to the model every turn.")
    st.caption(f"{len(checkpoint_history)} state snapshots belong to this thread only.")
    st.json([{"role": m.type, "content": m.content} for m in messages], expanded=False)

    with st.expander("Checkpoint history", expanded=False):
        if checkpoint_history:
            st.caption("Newest snapshot first. Switching threads replaces this list.")
            for saved_snapshot in checkpoint_history[:10]:
                checkpoint_id = saved_snapshot.config["configurable"].get("checkpoint_id", "")
                step = (saved_snapshot.metadata or {}).get("step", "input")
                saved_messages = saved_snapshot.values.get("messages", [])
                st.code(
                    f"step {step} | {len(saved_messages)} messages | {checkpoint_id}",
                    language=None,
                )
            if len(checkpoint_history) > 10:
                st.caption(f"Showing 10 of {len(checkpoint_history)} snapshots.")
        else:
            st.caption("No checkpoints yet. Send a message to create them.")

    st.divider()

    st.subheader("Long-term")
    st.caption(f"user `{USER_ID}` — {len(profile)} facts, carried into every conversation")
    st.caption("Loaded from the Store and added to the system prompt for every reply.")
    for item in profile:
        st.markdown(f"- {item.value['fact']}")
    if not profile:
        st.caption("_Empty. Say something about yourself, then end the conversation._")

    st.divider()

    if st.button(
        "End conversation", disabled=not messages, use_container_width=True, type="primary"
    ):
        with st.status("Distilling this conversation...", expanded=True) as status:
            st.write(f"Reading {len(messages)} messages from this thread.")
            st.write(f"Combining them with {len(profile)} existing profile facts.")
            updated = distill(llm, messages, [item.value["fact"] for item in profile])

            # An empty result means the model returned nothing usable. Skip the
            # write rather than erasing a profile built from real conversations.
            if updated:
                remember(store, NAMESPACE, updated)
                st.write(f"Saved {len(updated)} replacement facts to the Store.")
                status.update(label="Long-term memory updated", state="complete")
                result = f"saved {len(updated)} profile facts"
            else:
                st.write("No durable facts found, so the existing profile was kept.")
                status.update(label="No profile changes needed", state="complete")
                result = "kept the existing profile"

        previous_thread = st.session_state.thread
        new_thread = next_empty_thread(thread_options)
        st.session_state.thread = new_thread
        st.session_state.sync_thread_picker = True
        st.session_state.memory_event = (
            f"Distilled {len(messages)} messages from {previous_thread}, {result}, and "
            f"started empty thread {new_thread}. The Store profile will be loaded for "
            "every reply in this new conversation."
        )
        st.rerun()

    st.caption("Distills this conversation into the profile above, then starts a fresh one.")

    if st.button("Forget me", disabled=not profile, use_container_width=True):
        for item in profile:
            store.delete(NAMESPACE, item.key)
        st.rerun()
