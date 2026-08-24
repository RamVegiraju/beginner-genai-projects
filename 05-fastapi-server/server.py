"""Serving a LangGraph agent over FastAPI, asynchronously.

Agent inference is almost all waiting: the model is thinking, the tool is
calling an API, the network is in flight. Async is how one process serves many
callers through all that waiting instead of one at a time.

LangGraph gives every compiled graph an async twin of each method, so there is
nothing to hand-roll:

    await graph.ainvoke(...)    one request
    await graph.abatch([...])   many at once, concurrently
    async for ... graph.astream(...)   tokens as they are produced

Run:
    uvicorn server:app

Then, in another terminal:
    python load_test.py
"""

import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import openai
from databricks.sdk import WorkspaceClient
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

MODEL = os.environ.get("SERVING_ENDPOINT", "databricks-claude-haiku-4-5")
PROFILE = os.environ.get("DATABRICKS_PROFILE", "genai-series")
MAX_TOKENS = 150


class ChatRequest(BaseModel):
    """A message from the caller."""

    message: str


class BatchRequest(BaseModel):
    """Several messages to answer in one call."""

    messages: list[str]


class ChatResponse(BaseModel):
    """The agent's reply."""

    reply: str


class BatchResponse(BaseModel):
    """One reply per message, in the order they were sent."""

    replies: list[str]


def build_graph() -> CompiledStateGraph:
    """Compile the agent. One node, and it awaits.

    The node is `async def` and awaits the model. That is what lets the server
    start someone else's request while this one is waiting.
    """
    workspace = WorkspaceClient(profile=PROFILE)
    token = workspace.config.authenticate()["Authorization"].removeprefix("Bearer ")
    llm = ChatOpenAI(
        model=MODEL,
        api_key=token,
        base_url=f"{workspace.config.host}/serving-endpoints",
        max_tokens=MAX_TOKENS,
    )

    async def agent(state: MessagesState) -> dict:
        return {"messages": [await llm.ainvoke(state["messages"])]}

    return StateGraph(MessagesState).add_node("agent", agent).add_edge(START, "agent").compile()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Compile the graph on startup rather than at import time.

    Keeps the module importable without credentials, and gives the app one
    place to hold shared state.
    """
    app.state.graph = build_graph()
    yield


app = FastAPI(title="Agent service", lifespan=lifespan)


def _text(state: dict) -> str:
    """Pull the last message out of the graph's final state."""
    content = state["messages"][-1].content
    return content if isinstance(content, str) else str(content)


@app.get("/health")
async def health() -> dict[str, str]:
    """Report that the server is up and which model it will call."""
    return {"status": "ok", "model": MODEL}


@app.post("/chat")
async def chat(body: ChatRequest, request: Request) -> ChatResponse:
    """Answer one message.

    Raises:
        HTTPException: 502 if the serving endpoint rejects the call.
    """
    graph: CompiledStateGraph = request.app.state.graph
    try:
        state = await graph.ainvoke({"messages": [HumanMessage(body.message)]})
    except openai.OpenAIError as exc:
        raise HTTPException(status_code=502, detail=f"Model call failed: {exc}") from exc
    return ChatResponse(reply=_text(state))


@app.post("/chat/batch")
async def chat_batch(body: BatchRequest, request: Request) -> BatchResponse:
    """Answer several messages concurrently.

    `abatch` runs them all at once and returns them in order -- the same thing
    asyncio.gather would do by hand, minus the hand.

    Raises:
        HTTPException: 502 if the serving endpoint rejects the call.
    """
    graph: CompiledStateGraph = request.app.state.graph
    try:
        states = await graph.abatch(
            [{"messages": [HumanMessage(message)]} for message in body.messages]
        )
    except openai.OpenAIError as exc:
        raise HTTPException(status_code=502, detail=f"Model call failed: {exc}") from exc
    return BatchResponse(replies=[_text(state) for state in states])


@app.post("/chat/stream")
async def chat_stream(body: ChatRequest, request: Request) -> StreamingResponse:
    """Answer one message, sending tokens as they are produced.

    Server-sent events. The caller sees the first words immediately instead of
    waiting for the whole reply -- the same idea as sample 1's streaming, now
    over HTTP.
    """
    graph: CompiledStateGraph = request.app.state.graph

    async def events() -> AsyncIterator[str]:
        async for chunk, _metadata in graph.astream(
            {"messages": [HumanMessage(body.message)]}, stream_mode="messages"
        ):
            if chunk.content:
                yield f"data: {json.dumps({'token': chunk.content})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")
