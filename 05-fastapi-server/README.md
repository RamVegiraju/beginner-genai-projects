# 5 — Serving the agent with FastAPI

Samples 2 and 4 ran your agent for one person, in one browser tab. This turns
the LangGraph agent into an API that many people can call at once.

Agent inference is almost all *waiting* — the model is thinking, a tool is
calling an API, the network is in flight. Async is how one process serves
everyone through that waiting instead of one at a time.

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

**4. Waiting is the whole opportunity.** Eight questions take 8x as long one
at a time as they do together, on the same one process.

## Setup

**First time?** Do [SETUP.md](../SETUP.md) first — install the CLI, log in,
and create the virtual environment. Five minutes, once for the whole series.

Then install this sample's dependencies, **from the repo root**:

```bash
uv pip install --python .venv/bin/python -r 05-fastapi-server/requirements.txt
```

## Run

Start the server in one terminal:

```bash
cd 05-fastapi-server
../.venv/bin/uvicorn server:app
```

Then run the load test in another:

```bash
cd 05-fastapi-server
../.venv/bin/python load_test.py
```

```
8 questions, same endpoint:

  one at a time    13.3s
  all at once       2.2s

6.1x faster, on one process and one thread.

/chat/batch (graph.abatch, one request)    1.9s
  returned 8 replies
```

Same endpoint, same model, same eight questions. Sent together, they finish in
about the time of a single request, because the server used each wait to work
on somebody else's.

Your exact timings will differ — they depend on network and model latency. The
speedup lands somewhere around 5x on a typical run. What matters is the shape:
sending them together costs roughly one request's worth of time.

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

## The mistake to avoid

`async def` on the endpoint is not what makes anything concurrent. The work
inside has to yield. This node awaits, so the server can switch away while the
model thinks:

```python
async def agent(state: MessagesState) -> dict:
    return {"messages": [await llm.ainvoke(state["messages"])]}
```

Write `llm.invoke(...)` there instead and every caller queues up behind every
other one, even though the endpoint still says `async def`.

Worth knowing: a node written as `async def` **cannot** be run by the
synchronous `graph.invoke()` — LangGraph raises *"No synchronous function
provided"*. Pick async and stay async.

If you are stuck with a library that has no async version, do not fake it with
`async def`. Use a plain `def` endpoint and let FastAPI run it in a thread
pool, or wrap the call in `starlette.concurrency.run_in_threadpool`.

## Notes

- The graph is compiled in a `lifespan` handler rather than at import, so the
  module stays importable without credentials.
- Run it with the default single worker. Adding workers hides a blocking
  server by running several copies of it, which is not the same as fixing it.
- FastAPI generates docs at `http://localhost:8000/docs` from the request and
  response models — you can call every endpoint from the browser.

## Next

The server works. But is the agent any *good*? Measuring answer quality
instead of guessing at it is [sample 6](../06-mlflow-evals/).
