# 2 — Streamlit chatbot

Sample 1 asked one question and exited. This turns it into a conversation you
can actually hand to someone.

## What this sample showcases

**1. The API is stateless.** This is the whole lesson. The model remembers
*nothing* between calls. There is no session, no thread, no history stored on
the server.

**2. "Memory" is just a list you resend.** The chatbot adds no intelligence —
it keeps a list of messages and sends the entire thing on every turn. Open the
sidebar and watch the list grow.

**3. A chat UI is about 40 lines.** `st.chat_message`, `st.chat_input`, and
`st.write_stream` do the work. The streaming from sample 1 is one call here.

## Setup

**First time?** Do [SETUP.md](../SETUP.md) first — install the CLI, log in,
and create the virtual environment. Five minutes, once for the whole series.

## Run

Open a terminal at the repo root and paste the whole block:

```bash
export DATABRICKS_PROFILE=genai-series   # once per terminal
uv pip install --python .venv/bin/python -r 02-streamlit-chatbot/requirements.txt
cd 02-streamlit-chatbot
../.venv/bin/streamlit run app.py
```

Your browser opens at `http://localhost:8501`.

## Try this

Type these two messages in order:

```
My name is Ram.
What's my name?
```

It answers correctly — because we resent the whole conversation, not because
the model remembered anything.

**Now open the sidebar.** Every message is listed there, and *all of them* are
sent on every single turn. Two things follow, and they drive the rest of the
series:

- The list only grows, so each turn costs more than the last.
- Refresh the browser and it's gone. This memory lives in RAM.

## How it works

```python
# the list IS the memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# 1. append what the user said
st.session_state.messages.append({"role": "user", "content": prompt})

# 2. send the ENTIRE list, every time
stream = client.chat.completions.create(
    model=MODEL, messages=st.session_state.messages, stream=True
)

# 3. append the reply, so it's there next turn
st.session_state.messages.append({"role": "assistant", "content": reply})
```

Two Streamlit details worth knowing, because they surprise people:

- **Streamlit re-runs the entire file** top to bottom on every interaction.
  That's why history lives in `st.session_state` — a normal variable would be
  wiped each time.
- **`@st.cache_resource`** builds the API client once instead of rebuilding it
  on every keystroke.

## Next

Close the tab and the conversation is gone. Making it survive — across
restarts, across days — is [sample 4](../04-agent-memory/). First,
[sample 3](../03-langgraph-agent/) gives the model tools so it can do more than
talk.
