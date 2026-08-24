# 4 — Memory

**The wall:** sample 2's chatbot kept the conversation in a Python list, so
closing the tab erased you. Sample 3 rebuilt state from scratch on every run.

This is the same chatbot as sample 2, with the memory moved to disk — and it
turns out "remember me" is really two different problems:

| | Short-term | Long-term |
|---|---|---|
| **Answers** | "What did I just say?" | "How do you like your answers?" |
| **Scoped to** | one conversation (`thread_id`) | one person (`user_id`) |
| **Mechanism** | checkpointer — saves state every step | store — a few key/value facts |
| **Written by** | LangGraph, automatically | you, by hitting **Distill** |
| **Grows** | forever, and you pay for it every turn | barely |

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

`app.py` is the only thing you run. `distill.py` is a two-function module it
imports — worth reading, nothing to execute.

**Watch the sidebar.** It shows both memories at once, which is the whole
point: one of them empties when you switch threads and the other doesn't.

1. Say *"I'm vegetarian and I live in Seattle. Keep answers short."*
2. The message count climbs. That list is resent on every single turn.
3. Hit **Distill**. The conversation collapses into two or three facts.
4. Hit **New thread**. Messages drop to 0. The facts stay.
5. Ask *"What should I cook tonight?"*

Step 5 is the payoff. The thread is empty — it has never met you — and it
answers with *"quick vegetarian ideas for Seattle"*. Short-term memory was
left behind; long-term memory came along.

Then stop the server and start it again. Your conversation is still there,
because `thread_id` is a key in a SQLite file rather than a variable in RAM.

## What distillation actually is

The whole skill sits in one prompt in `distill.py`, and one word does the
work: **durable**. "Facts that would still be true in a different conversation
next month" is the line between a preference and a passing detail. In testing
it kept *vegetarian* and dropped the dinner it had just recommended.

Worth saying out loud on camera: this is a model summarising a model's output,
and the result gets pasted into every future conversation. It is a judgement
call, not a database write. It will sometimes keep the wrong thing and then be
confidently wrong about you for months — which is why the sidebar shows you
exactly what it thinks it knows, and why **Forget me** is a button.

## Three things in the code worth pausing on

**The system prompt is built inside the node and thrown away.** It never
enters state, so it is never checkpointed. Do the obvious thing instead —
prepend it once at startup — and your preferences get frozen into the first
checkpoint, so old threads keep quoting a stale version of you. Memory you
cannot correct is worse than no memory.

**The parameter is named `store` on purpose.** LangGraph injects it by
matching the parameter *name*, and by checking the annotation is `BaseStore`.
Annotate it `SqliteStore` and you silently get nothing.

**Two connections to one file.** The checkpointer and the store can share
`memory.db` but not a connection — the store runs in autocommit and the saver
doesn't, so sharing raises *"cannot start a transaction within a
transaction"*. Both need `check_same_thread=False`, because Streamlit runs
the script on one thread and LangGraph writes checkpoints from another.

## Deliberately local

Plain SQLite on disk. No Lakebase, no external store, one file you can `rm`.
Swapping in `PostgresSaver` is a two-line change when you need it; the
concepts — `thread_id`, `user_id`, checkpointer, store — do not change at all.

**Ends on:** all of this serves one user, in one process, one at a time.
→ sample 5.
