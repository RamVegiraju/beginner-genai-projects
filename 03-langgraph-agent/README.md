# 3 — Giving the model tools

Samples 1 and 2 could only talk. This one can *do* something.

## The problem

Ask a model "what's the weather in Chicago?" and it cannot know. It has no
live data. It will hedge, or it will invent a plausible-sounding answer.

**No prompt fixes a missing fact.** You have to give it a way to go get one.

## What this sample showcases

**1. A tool is just a Python function.** No magic. The `@tool` decorator
exposes it, and the model decides when to call it.

**2. The docstring is a prompt.** The model reads the function name, type
hints, and docstring to decide whether this tool is relevant. That docstring
is written for the model, not for you.

**3. An agent is a loop.** model → tool → model → answer. LangGraph runs it.

**4. The model chooses.** Ask it something unrelated and it skips the tool
entirely. Ask about two cities and it calls the tool twice.

## Setup

**First time?** Do [SETUP.md](../SETUP.md) first — install the CLI, log in,
and create the virtual environment. Five minutes, once for the whole series.

Then install this sample's dependencies, **from the repo root**:

```bash
uv pip install --python .venv/bin/python -r 03-langgraph-agent/requirements.txt
```

No API key is needed for the weather data — [Open-Meteo](https://open-meteo.com/)
is free and open.

## Run

```bash
cd 03-langgraph-agent
../.venv/bin/python agent.py
```

Output (your weather will differ — the tool fetches it live):

```
user> I'm flying out of Chicago today. Should I expect weather delays?

[AIMessage] calling get_weather({'city': 'Chicago'})
[ToolMessage] returned: Chicago, US: 64.8F, precipitation 0.0mm, wind 5.1mph

bot > Good news! The weather in Chicago today looks favorable for flying...
```

Every step is printed on purpose. You can see the model *decide* to call the
tool, see what came back, and see it reason over the real number.

## The graph

```
   START → agent → (tool calls?) → tools ─┐
             ↑                            │
             └────────────────────────────┘
             │
             └── no tool calls → END
```

Two nodes and a loop:

- **`agent`** — asks the model what to do next
- **`tools`** — runs whatever the model asked for (`ToolNode`)
- **`tools_condition`** — looks at the last message: tool calls → `tools`,
  otherwise → `END`

The edge from `tools` back to `agent` is the important one. After a tool runs,
the model gets to see the result and decide what's next — answer, or call
another tool.

## Try this

**It skips the tool when it isn't needed.**

```
What is 2+2?
```
→ **no tool call at all.** It just answers. The model isn't required to use
what it's given.

**It calls the tool twice, in one turn.**

```
Compare the weather in Denver and Miami right now.
```
→ **two tool calls in a single turn** (`Denver`, then `Miami`), then one
combined answer. The loop runs until the model stops asking for tools.

**It handles bad input.**

```
What's the weather in Xyzzyville12345?
```
→ the tool returns "Could not find a place called...", and the model relays it
instead of crashing. Tools should return errors as *text* — that's information
the model can act on.

## Note on the model

Same serving endpoint as samples 1 and 2, reached through LangChain's
`ChatOpenAI` pointed at your workspace — because the endpoint is
OpenAI-compatible, exactly as in sample 1. `.bind_tools()` is what tells the
model which functions it may call.

## Next

Run this twice and the second run remembers nothing about the first. Making
that stick is [sample 4](../04-agent-memory/).
