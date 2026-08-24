# 5 — Serving it with FastAPI

Samples 2 and 4 ran your agent for one person, in one browser tab. This turns
it into an API that many people can call at once.

The interesting part isn't the web framework — it's that the obvious way to
write the server is about **7x slower** under load, and it looks completely
fine.

## What this sample showcases

**1. A model endpoint is a few lines of FastAPI.** Take a message, call the
model, return the reply.

**2. A model call is almost all waiting.** Your server spends that time doing
nothing — unless you let it.

**3. `async def` alone does not make anything concurrent.** Putting a blocking
call inside an `async` function is the single most common way to build a slow
server.

**4. The fix is one word: `await`.** With an async client, one process on one
thread handles all the requests together.

## Setup

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

Output:

```
one request, on its own:     1.7s

8 requests, all sent at the same moment:
  /chat/blocking            12.6s   (7.5x one request)
  /chat                      2.1s   (1.3x one request)

5.9x faster. Same model, same machine, same work.
```

The blocking endpoint handled the requests one after another, so eight of them
took eight times as long. The async endpoint overlapped them, so eight
requests finished in about the time of one.

## The two endpoints

They do identical work. The only difference is which client they use.

```python
@app.post("/chat/blocking")
async def chat_blocking(request: ChatRequest) -> dict:
    response = sync_client.chat.completions.create(...)   # no await
    return {"reply": response.choices[0].message.content}


@app.post("/chat")
async def chat(request: ChatRequest) -> dict:
    response = await async_client.chat.completions.create(...)
    return {"reply": response.choices[0].message.content}
```

`await` means *"I am waiting — go do something else."* Without it, the server
sits on one request for the whole model call and nobody else gets served. The
`async def` on the first endpoint is a promise the function never keeps.

The load test itself is the same for both: it fires all eight requests at the
same instant and waits. The client cannot fix a server that refuses to work on
more than one thing.

## What if your library has no async version?

Then don't write `async def`. Use a plain `def`, and FastAPI runs it in a
thread pool instead of on the event loop:

```python
@app.post("/chat/threadpool")
def chat_threadpool(body: ChatRequest) -> ChatResponse:   # note: no async
    ...
```

That is concurrent too, and it's the right answer for a blocking library.
Threads cost more memory than awaiting does, so they run out sooner — but
running out sooner is still better than serialising, which is what
`/chat/blocking` does.

The rule: `async def` means everything inside must be awaited. If it isn't,
use `def`.

## Try this

**Change the number of requests.** Set `REQUESTS = 20` in `load_test.py`. The
async endpoint barely moves; the blocking one gets 20x worse.

**Check it from the docs page.** FastAPI generates one at
`http://localhost:8000/docs` from the `ChatRequest` and `ChatResponse` models
— you can call both endpoints from the browser.

## Notes

- Clients are built in a `lifespan` handler rather than at import, so the
  module imports without credentials and the async client gets closed when
  the server stops.
- The token is fetched once at startup. Fine for a demo; a service that runs
  for days should refresh it, since these tokens are short-lived on purpose.
- Run it with the default single worker. Adding workers hides the problem by
  running several copies of the server, which is not the same as fixing it.
- There is no memory here. Adding sample 4's checkpointer means giving each
  caller a `thread_id` and using LangGraph's async saver, so the database
  doesn't block the event loop either.

## Next

The server works. But is it any *good*? Measuring answer quality instead of
guessing at it is [sample 6](../06-mlflow-evals/).
