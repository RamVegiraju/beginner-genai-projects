# 5 — Serving it with FastAPI

> **Not built yet.**

Streamlit runs your agent for one person at a time. This sample turns it into
a service that many people can call at once.

Will cover:

- `server.py` — FastAPI wrapping the sample 4 agent
- `async` endpoints, so a slow model call doesn't block everyone else
- `/chat` for JSON and `/chat/stream` for server-sent events
- `thread_id` in the request body, so each caller keeps their own memory
- `load_test.py` — the same workload run blocking and async, side by side
