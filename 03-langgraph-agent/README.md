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

**3. An agent is a loop.** model → tool → model → answer.

**4. The model chooses.** Ask it something unrelated and it skips the tool
entirely. Ask about two cities and it calls the tool twice.

## Two files, one agent

There are two versions here. They answer the same question the same way — the
only difference is how much of the loop you write yourself.

| File | What it is | Wiring |
|---|---|---|
| **`simple_agent.py`** | LangChain's `create_agent` | 1 line |
| **`agent.py`** | the same loop, built by hand in LangGraph | ~15 lines |

Both share `weather_tool.py`, so nothing differs except the wiring.

**Start with `simple_agent.py`.** This is how you should build an agent when a
standard loop is all you need:

```python
agent = create_agent(llm, [get_weather], system_prompt=SYSTEM)
```

That's it. No `.bind_tools()`, no nodes, no edges — `create_agent` does all of
it. Run it and you get a correct, tool-using agent.

## Setup

**First time?** Do [SETUP.md](../SETUP.md) first — install the CLI, log in,
and create the virtual environment. Five minutes, once for the whole series.

No API key is needed for the weather data — [Open-Meteo](https://open-meteo.com/)
is free and open.

## Run

Open a terminal at the repo root and paste the whole block:

```bash
export DATABRICKS_PROFILE=genai-series   # once per terminal
uv pip install --python .venv/bin/python -r 03-langgraph-agent/requirements.txt
cd 03-langgraph-agent
../.venv/bin/python simple_agent.py      # the easy way
../.venv/bin/python agent.py             # the same thing, unfolded
```

`simple_agent.py` prints the answer, then the steps it took to get there
(your weather will differ — the tool fetches it live):

```
bot > The current weather in Chicago shows:
- Temperature: 76.3F
- Precipitation: 0.0 mm (no rain)
- Wind: 4.5 mph

However, weather readings alone cannot determine whether airport or airline
delays will occur...

(steps: HumanMessage -> AIMessage -> ToolMessage -> AIMessage)
```

That step list is the loop. The tool call happened — it's just hidden inside
`create_agent`.

`agent.py` prints the same steps as they happen, because you wrote them:

```
user> I'm flying out of Chicago today. Should I expect weather delays?

[AIMessage] calling get_weather({'city': 'Chicago'})
[ToolMessage] returned: Chicago, US: 77.1F, precipitation 0.0mm, wind 3.4mph

bot > Here are the current weather conditions in Chicago:
- **Temperature:** 77.1F
- **Precipitation:** 0.0 mm (no rain)
- **Wind:** 3.4 mph

However, weather readings alone cannot determine whether airport or airline
delays will occur. To get accurate information about potential delays, I
recommend checking directly with your airline or with Chicago's airports...
```

Every step is printed on purpose. You can see the model *decide* to call the
tool, see what came back, and see it reason over the real number.

The tool does not know airline or airport operations. A short system
instruction keeps the answer grounded in its weather readings and sends the
user to the airline or airport for actual delay status.

## The graph (`agent.py`)

Here is what `create_agent` was doing for you, and what `agent.py` writes out:

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

## So why not always use `create_agent`?

Often you should. If a standard loop does the job, one line beats fifteen.

But `create_agent` is not a different *kind* of thing from the graph in
`agent.py` — it builds one. Both return a `CompiledStateGraph`, with the same
nodes:

```
simple_agent.py   ['__start__', 'model', 'tools', '__end__']
agent.py          ['__start__', 'agent', 'tools', '__end__']
```

Same shape; `create_agent` just names the first node `model`. So `agent.py`
isn't a lower-level alternative to the prebuilt — it's what the prebuilt
builds, with the lid off.

That matters the moment you want something the standard loop doesn't do:

- put a node **between** the model and the tools — to check a budget, ask a
  human to approve, or log the request
- **cap the loop** so a stuck agent can't call tools forever
- **route** to a cheaper model for easy turns and an expensive one for hard turns
- **edit state** on the way past — trim old messages, inject a retrieved document

You can't reach inside a single function call to do any of that. You can add a
node to a graph. That's the trade: `create_agent` for the common case, the
explicit graph when you need the control.

**Sample 4 needs exactly this**, which is why the series continues with
LangGraph from here.

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
model which functions it may call — `create_agent` calls it for you, which is
why `simple_agent.py` never mentions it.

## Next

Run this twice and the second run remembers nothing about the first. Making
that stick is [sample 4](../04-agent-memory/).
