# 4 — Memory

> **Not built yet.** Design notes only.

**The wall:** sample 2 held history in RAM; sample 3 rebuilt state per run.
Both forget you the moment the process ends.

**Important framing:** sample 2 already taught in-conversation memory (resend
the list). This sample is about the harder thing — **persistent, cross-session
memory**. Close the app, come back tomorrow, and it still knows you.

Planned:
- LangGraph's `SqliteSaver` checkpointer writing to a local `.db` file
- `thread_id` as the conversation key — same id resumes, new id starts fresh
- Demo: run, exit the process entirely, re-run, and it still remembers
- Wire it back into the sample 2 chatbot
- Name the cost problem: history grows forever and you pay for all of it

**Deliberately local.** Plain SQLite on disk, no Lakebase, no external store.
One file you can delete. The concept transfers; the infrastructure would only
get in the way this early.

**Ends on:** it works for one user at a time. → sample 5.
