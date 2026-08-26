# 5 — Serving the agent with FastAPI

Samples 2 and 4 ran your agent for one person, in one browser tab. This puts
the LangGraph agent behind an HTTP API, so anything can call it — a web app, a
job, another service.

Agent inference is almost all *waiting*: the model is thinking, a tool is
calling an API, the network is in flight. That shapes how you serve it.

## What this sample showcases

**1. Serving an agent is a few lines of FastAPI.** Take a message, run the
graph, return the reply.

**2. LangGraph has an async twin of every method.** There is nothing to
hand-roll:

| Method | Does |
|---|---|
| `await graph.ainvoke(...)` | one request |
| `await graph.abatch([...])` | many at once, concurrently, results in order |
| `async for ... graph.astream(...)` | tokens as they are produced |

**3. The node has to await too.** `async def` on the endpoint does nothing on
its own — the work inside it must yield, or the server still blocks.

## Setup

**First time?** Do [SETUP.md](../SETUP.md) first — install the CLI, log in,
and create the virtual environment. Five minutes, once for the whole series.

## Run

**Terminal 1 — the server.** Open it at the repo root and paste the whole
block:

```bash
export DATABRICKS_PROFILE=genai-series   # once per terminal
uv pip install --python .venv/bin/python -r 05-fastapi-server/requirements.txt
cd 05-fastapi-server
../.venv/bin/uvicorn server:app
```

**Terminal 2 — the load test.** Also from the repo root. No credentials here;
this one only talks to your local server:

```bash
cd 05-fastapi-server
../.venv/bin/python load_test.py
```

```
8 concurrent requests to /chat:

  all replies in     2.2s
  slowest single     2.2s

They overlap, so the set costs about what its slowest request costs.

/chat/batch (graph.abatch, one request)    2.5s
  returned 8 replies
```

Timings depend on your network and the model, so yours will differ. The point
is the relationship between the two numbers: eight requests cost about what
one costs, because the server spent each wait on somebody else's work.

## The endpoints

```python
@app.post("/chat")
async def chat(body: ChatRequest, request: Request) -> ChatResponse:
    state = await graph.ainvoke({"messages": [HumanMessage(body.message)]})
```

**`/chat/batch`** hands several messages to `graph.abatch(...)`, which runs
them concurrently and returns them in order — what `asyncio.gather` would do
by hand, minus the hand.

**`/chat/stream`** wraps `graph.astream(..., stream_mode="messages")` in a
`StreamingResponse` to send server-sent events:

```
data: {"token": "Hello,"}
data: {"token": " how"}
data: {"token": " are you today"}
```

The caller sees the first words immediately — sample 1's streaming, now over
HTTP.

## Awaiting is what makes it concurrent

`async def` on the endpoint is not what makes anything concurrent. The work
inside has to yield. This node awaits, so the server can switch away while the
model thinks:

```python
async def agent(state: MessagesState) -> dict:
    messages = [SystemMessage("Answer directly in no more than two sentences.")] + state["messages"]
    return {"messages": [await llm.ainvoke(messages)]}
```

The short instruction also prevents the small response budget from cutting a
long answer off mid-sentence.

Write `llm.invoke(...)` there instead and every caller queues up behind every
other one, even though the endpoint still says `async def`.

Worth knowing: a node written as `async def` **cannot** be run by the
synchronous `graph.invoke()` — LangGraph raises *"No synchronous function
provided"*. Pick async and stay async.

If you are stuck with a library that has no async version, do not fake it with
`async def`. Use a plain `def` endpoint and let FastAPI run it in a thread
pool, or wrap the call in `starlette.concurrency.run_in_threadpool`.

## Taking it to production

This sample stays deliberately small: one process, no limits, no retries. The
primitives you add first, roughly in the order they start to matter:

| When you need to | Reach for |
|---|---|
| Do I/O without blocking the loop | an async client and `await` — `graph.ainvoke`, `llm.ainvoke`, `httpx.AsyncClient` |
| Fan out work inside one request | `graph.abatch(...)`, or `asyncio.gather` / `asyncio.TaskGroup` for mixed work |
| Keep unbounded load off the model | an `asyncio.Semaphore` around the model call, and a 429 once it is full |
| Give up on a slow upstream | `asyncio.timeout(...)` plus a client-side timeout, so one hung call cannot pin a request forever |
| Call a library with no async version | `starlette.concurrency.run_in_threadpool`, or `asyncio.to_thread` |
| Reuse connections instead of reopening them | one long-lived `httpx.AsyncClient` on `app.state`, sized with `httpx.Limits(...)` |
| Use more than one CPU core | more uvicorn worker processes — after the loop is non-blocking, never instead of it |
| Let in-flight requests finish on deploy | the `lifespan` handler this sample already has |

Two of those are easy to get backwards. A semaphore is there to protect the
*model endpoint* from your server, so size it against that endpoint's rate
limit rather than against your traffic. And workers multiply processes, not
concurrency — if a blocking call is holding the loop, four workers just block
four times.

## Notes

- The graph is compiled in a `lifespan` handler rather than at import, so the
  module stays importable without credentials.
- Run it with the default single worker; the table above says when more help.
- FastAPI generates docs at `http://localhost:8000/docs` from the request and
  response models — you can call every endpoint from the browser.

## Next

The server works. But is the agent any *good*? Measuring answer quality
instead of guessing at it is [sample 6](../06-mlflow-evals/).
