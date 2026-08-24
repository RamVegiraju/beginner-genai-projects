"""Chat API over a Databricks serving endpoint.

Two endpoints do identical work, and one of them is roughly N times slower
under load:

    POST /chat/blocking   async def + the SYNC client. Looks fine. Isn't.
    POST /chat            async def + the ASYNC client. Handles many at once.

Run:
    uvicorn server:app

Then, in another terminal:
    python load_test.py
"""

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import openai
from databricks.sdk import WorkspaceClient
from fastapi import FastAPI, HTTPException, Request
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel

MODEL = os.environ.get("SERVING_ENDPOINT", "databricks-claude-haiku-4-5")
PROFILE = os.environ.get("DATABRICKS_PROFILE", "genai-series")
MAX_TOKENS = 150


class ChatRequest(BaseModel):
    """A message from the caller."""

    message: str


class ChatResponse(BaseModel):
    """The model's reply."""

    reply: str


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build the clients on startup and close them on shutdown.

    Doing this here rather than at import time keeps the module importable
    without credentials, and gives the async client somewhere to release its
    connections when the server stops.
    """
    workspace = WorkspaceClient(profile=PROFILE)
    token = workspace.config.authenticate()["Authorization"].removeprefix("Bearer ")
    base_url = f"{workspace.config.host}/serving-endpoints"

    # The same credentials, wrapped in two different clients.
    app.state.sync_client = OpenAI(api_key=token, base_url=base_url)
    app.state.async_client = AsyncOpenAI(api_key=token, base_url=base_url)

    yield

    app.state.sync_client.close()
    await app.state.async_client.close()


app = FastAPI(title="Chat service", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    """Report that the server is up and which model it will call."""
    return {"status": "ok", "model": MODEL}


@app.post("/chat/blocking")
async def chat_blocking(body: ChatRequest, request: Request) -> ChatResponse:
    """Answer a message, one caller at a time.

    This is the mistake: `async def` with a blocking call inside it. A model
    call spends almost all its time waiting on the network, and nothing here
    is awaited, so the server cannot start anyone else's request during that
    wait. Callers queue up behind each other.

    Raises:
        HTTPException: 502 if the serving endpoint rejects the call.
    """
    client: OpenAI = request.app.state.sync_client
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": body.message}],
            max_tokens=MAX_TOKENS,
        )
    except openai.OpenAIError as exc:
        raise HTTPException(status_code=502, detail=f"Model call failed: {exc}") from exc

    # content is optional in the API schema, so fall back to an empty string.
    return ChatResponse(reply=response.choices[0].message.content or "")


@app.post("/chat")
async def chat(body: ChatRequest, request: Request) -> ChatResponse:
    """Answer a message, many callers at once.

    The fix is the async client and `await`. `await` means "I am waiting, go
    do something else", so the server starts the next request instead of
    sitting idle. Still one process, still one thread.

    Raises:
        HTTPException: 502 if the serving endpoint rejects the call.
    """
    client: AsyncOpenAI = request.app.state.async_client
    try:
        response = await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": body.message}],
            max_tokens=MAX_TOKENS,
        )
    except openai.OpenAIError as exc:
        raise HTTPException(status_code=502, detail=f"Model call failed: {exc}") from exc

    # content is optional in the API schema, so fall back to an empty string.
    return ChatResponse(reply=response.choices[0].message.content or "")
