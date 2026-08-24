# 5 — Serving it: an agent is a systems problem

> **Not built yet.** Design notes only.

**The wall:** Streamlit runs your agent for one person. What happens when
fifty people hit it at once?

This is the mindset shift the series builds toward: an agent isn't a script,
it's a **service** — with latency, concurrency, and failure modes.

Planned:
- `server.py` — FastAPI wrapping the sample 4 agent
- `async def` endpoints + the async OpenAI client, so slow model calls don't
  block the event loop
- `/chat` (JSON) and `/chat/stream` (SSE)
- `thread_id` in the request body → per-user memory falls out of sample 4
- `load_test.py` — fire N concurrent requests, show sync vs async wall-clock

**The demo that sells it:** the same workload, blocking vs async. The numbers
make the point better than any explanation.
