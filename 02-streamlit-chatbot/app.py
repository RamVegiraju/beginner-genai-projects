"""
A chatbot in ~40 lines.

The important idea: the API is STATELESS. The model remembers nothing between
calls. A chatbot "remembers" only because we keep a list of messages and
resend the whole thing every turn.

Run:  streamlit run app.py
"""

import os

import streamlit as st
from databricks.sdk import WorkspaceClient
from openai import OpenAI

MODEL = os.environ.get("SERVING_ENDPOINT", "databricks-claude-haiku-4-5")
PROFILE = os.environ.get("DATABRICKS_PROFILE")

st.title("Chatbot")


# Same three lines as sample 1. @st.cache_resource means we build the client
# once instead of on every keystroke -- Streamlit re-runs this whole file top
# to bottom every time you interact with the page.
@st.cache_resource
def get_client():
    w = WorkspaceClient(profile=PROFILE)
    token = w.config.authenticate()["Authorization"].removeprefix("Bearer ")
    return OpenAI(api_key=token, base_url=f"{w.config.host}/serving-endpoints")


client = get_client()

# THIS LIST IS THE MEMORY. Nothing else is going on.
# st.session_state survives those top-to-bottom re-runs; a plain variable
# would be wiped every time.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Turn this off to send only the newest message. The model then has nothing to
# read back and the "memory" vanishes -- which is the point: the list is the
# memory, not the model.
with st.sidebar:
    send_history = st.checkbox("Send full history", value=True)

# Re-draw the conversation so far. The screen is rebuilt from this list on
# every re-run, so what you see is always just the list rendered.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything"):
    # 1. append what the user said
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. send the ENTIRE list, every time -- this is the whole trick
    history = st.session_state.messages if send_history else st.session_state.messages[-1:]
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=MODEL,
            messages=history,
            max_tokens=500,
            stream=True,
        )
        # st.write_stream renders chunks as they arrive and returns the full
        # text at the end -- the sample 1 streaming loop, done for you.
        reply = st.write_stream(
            chunk.choices[0].delta.content or "" for chunk in stream if chunk.choices
        )

    # 3. append the reply, so it's there for the next turn
    st.session_state.messages.append({"role": "assistant", "content": reply})

# Proof that the "memory" is just a list being resent. Watch it grow: with the
# toggle on, everything here goes out on every turn, and you pay for all of it
# each time.
with st.sidebar:
    total = len(st.session_state.messages)
    st.caption(f"Sent on the next turn: {total if send_history else min(total, 1)} of {total}")
    st.json(st.session_state.messages, expanded=False)
    if st.button("Clear"):
        st.session_state.messages = []
        st.rerun()
