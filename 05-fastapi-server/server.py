"""
Serving the model as an API.

Two endpoints do exactly the same work, and one of them is roughly N times
slower under load:

  POST /chat/blocking   async def + the SYNC client. Looks fine. Isn't.
  POST /chat            async def + the ASYNC client. Handles many at once.

Run:  uvicorn server:app
Then, in another terminal:  python load_test.py
"""

import os

from databricks.sdk import WorkspaceClient
from fastapi import FastAPI
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel

MODEL = os.environ.get("SERVING_ENDPOINT", "databricks-claude-haiku-4-5")
PROFILE = os.environ.get("DATABRICKS_PROFILE", "DEFAULT")
MAX_TOKENS = 150

# Fetched once at startup. Fine for a demo; a long-running service should
# refresh it, since these tokens are short-lived by design.
w = WorkspaceClient(profile=PROFILE)
TOKEN = w.config.authenticate()["Authorization"].removeprefix("Bearer ")
BASE_URL = f"{w.config.host}/serving-endpoints"

# The same credentials, wrapped in two different clients.
sync_client = OpenAI(api_key=TOKEN, base_url=BASE_URL)
async_client = AsyncOpenAI(api_key=TOKEN, base_url=BASE_URL)

app = FastAPI(title="Chat service")


class ChatRequest(BaseModel):
    message: str


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "model": MODEL}


@app.post("/chat/blocking")
async def chat_blocking(request: ChatRequest) -> dict:
    """The mistake: `async def` with a blocking call inside it.

    A model call spends almost all its time waiting on the network. Nothing
    here is awaited, so the server cannot start anyone else's request during
    that wait -- callers queue up behind each other one at a time.
    """
    response = sync_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": request.message}],
        max_tokens=MAX_TOKENS,
    )
    return {"reply": response.choices[0].message.content}


@app.post("/chat")
async def chat(request: ChatRequest) -> dict:
    """The fix: an async client, and `await`.

    `await` means "I am waiting -- go do something else." The server starts
    the next request instead of sitting idle, so all of them are in flight
    together. Still one process, still one thread.
    """
    response = await async_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": request.message}],
        max_tokens=MAX_TOKENS,
    )
    return {"reply": response.choices[0].message.content}
