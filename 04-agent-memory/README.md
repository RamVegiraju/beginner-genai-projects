# 4 — Memory

Sample 2's chatbot kept the conversation in Streamlit session state, so ending
that session or restarting the server erased it. This is the same chatbot with
its memory moved to disk.

"Remember me" turns out to be two different problems:

| | Short-term | Long-term |
|---|---|---|
| **Answers** | "What did I just say?" | "How do you like your answers?" |
| **Scoped to** | one conversation (`thread_id`) | one person (`user_id`) |
| **Stored by** | a checkpointer | a store |
| **Written** | every graph super-step, automatically | once, when a conversation ends |
| **Grows** | forever | barely |

## What this sample showcases

**1. A checkpointer replaces the manually managed message list.** LangGraph
loads the latest state before your node runs and saves the updated state after,
so you send one new message per turn instead of resending everything yourself.

**2. `thread_id` is the conversation.** Same id, same conversation — even
after you restart the server. New id, blank slate.

**3. A store holds what outlives the conversation.** It is keyed by user, not
by thread, so it comes along into every new chat.

**4. Distillation connects the two.** When a conversation ends, an LLM reads
the transcript and copies the few facts still worth knowing next month into
the long-term store. The old thread remains available as checkpoint history.

## Setup

**First time?** Do [SETUP.md](../SETUP.md) first — install the CLI, log in,
and create the virtual environment. Five minutes, once for the whole series.

## Run

Open a terminal at the repo root and paste the whole block:

```bash
export DATABRICKS_PROFILE=genai-series   # once per terminal
uv pip install --python .venv/bin/python -r 04-agent-memory/requirements.txt
cd 04-agent-memory
../.venv/bin/streamlit run app.py
```

Both memories are shown in the sidebar. Watch it while you try the steps below.

## Try this

**1. Tell it something about yourself.**

```
I'm vegetarian and I live in Seattle. Keep answers short.
```

The message count climbs. Every message in that list is resent on the next
turn, so each turn costs a little more than the last.

**2. Hit "End conversation".**

Two or three durable facts are copied into the profile, and a fresh empty
thread opens. The previous thread is no longer sent to the model, but its
checkpoint history remains in SQLite.

**3. Ask a new question.**

```
What should I cook tonight?
```

This new conversation has no message history — and you get vegetarian ideas
for Seattle. Short-term memory stayed with the previous thread; long-term
memory came along.

**4. Restart the server.** Everything is still there. It is a SQLite file,
not a variable in RAM.

**5. Change your mind.**

```
Actually I moved to Denver last month.
```

End that conversation and the profile says Denver. The old fact is replaced,
not stacked on top, because distillation rewrites the whole profile each time
instead of appending to it.

## Distillation is an LLM call

Ending a conversation costs one extra model call over the transcript. That is
the trade: pay once at the end so you stop paying for those messages on every
future turn. It is cheap because it is rare — which is why it runs at the end
of a conversation rather than on every turn.

Real apps detect that moment with a session timeout or a nightly job. Here you
click a button, because a button is easier to see.

Retrieval is deliberately simple: the whole profile is loaded and added to the
system prompt every turn. No embeddings, no search. That works because the
profile stays small.

## Where it lives

One file, `memory.db`, holds both kinds of memory. Stop the app and delete that
file to make the demo forget everything. A production app would replace the
SQLite implementations with a production database, while keeping the same
`thread_id` and `user_id` concepts.

## Next

This serves one user, in one process, at a time. Making it handle many at once
is [sample 5](../05-fastapi-server/).
