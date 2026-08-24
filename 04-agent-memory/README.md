# 4 — Memory

**The wall:** sample 2 held the conversation in RAM and sample 3 rebuilt state
on every run. Both forget you the moment the process ends.

"Remember me" turns out to be two different problems, and this sample builds
both:

| | Short-term | Long-term |
|---|---|---|
| **Question it answers** | "What did I just say?" | "How do you like your answers?" |
| **Scoped to** | one conversation (`thread_id`) | one person (`user_id`) |
| **Mechanism** | checkpointer — saves state every step | store — key/value facts |
| **Written by** | LangGraph, automatically | `distill.py`, deliberately |
| **Grows** | forever, and you pay for it | barely |

Both tables live in one file, `memory.db`. Delete it and the bot forgets you
entirely.

## Run it

```bash
pip install -r requirements.txt

python chat.py trip     # tell it something about yourself, Ctrl-D to exit
python chat.py trip     # SAME thread -- the conversation is still there
```

The second run is the point. The process died completely in between, and it
still knows what you said. That is the checkpointer: `thread_id` is the key,
and LangGraph loads and saves state around every turn for you. Note that
`chat.py` sends **one** message per turn — sample 2 had to resend the whole
list by hand.

```bash
python distill.py trip  # read the transcript, keep what's durable
python chat.py brand-new-thread
```

Now the interesting part. The new thread has **zero** messages in it — ask
what you talked about and it has no idea — but it still knows you're
vegetarian. Short-term memory was left behind; long-term memory came along.

A real run:

```
$ python distill.py trip
reading 6 messages from thread 'trip'...

  remembered: They are vegetarian.
  remembered: They live in Seattle.
  remembered: They prefer short answers of one or two sentences maximum.

6 messages -> 3 facts.
```

Six messages became three lines. It kept the dietary need and dropped the
dinner it recommended — that was about the topic, not about the person.

## What distillation actually is

The whole skill sits in one prompt in `distill.py`, and one word in it does
the work: *durable*. "Facts that would still be true in a different
conversation next month" is the line between a preference and a passing
detail.

Worth saying out loud: this is a model summarising a model's output, then
writing the result into the next conversation's system prompt. It is a
judgement call, not a database write. It will sometimes keep the wrong thing,
and then be confidently wrong about you for months. Read what it stored — the
sample prints the facts on every startup for exactly that reason.

## Why the system prompt is built inside the node

`agent()` reads preferences and constructs the system message on every turn,
then throws it away. It never enters state, so it is never checkpointed.

If you did the obvious thing instead — prepend the system message once at
startup — it would be saved into the thread's history on the first turn and
frozen there. Update a preference and old threads would keep quoting the old
one. Memory you cannot correct is worse than no memory.

## Deliberately local

Plain SQLite on disk. No Lakebase, no external store, one file you can
`rm`. Swapping in `PostgresSaver` is a two-line change when you need it; the
concepts — `thread_id`, `user_id`, checkpointer, store — do not change at all.

**Ends on:** all of this works for one user, one process, one at a time.
→ sample 5.
