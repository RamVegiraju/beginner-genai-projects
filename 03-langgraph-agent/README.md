# 3 — LangChain first, LangGraph when you need control

Samples 1 and 2 could only talk. This sample gives a model live weather data,
then shows two levels of abstraction for building an agent.

| Start with | Reach for it when | File |
|---|---|---|
| **LangChain** | A model needs a fixed set of tools and the usual tool-calling loop | [`simple_agent.py`](simple_agent.py) |
| **LangGraph** | Your application needs custom steps, branches, or loops | [`agent.py`](agent.py) |

These are layers, not competitors. LangChain's `create_agent()` is itself built
on LangGraph. Start with LangChain and drop down to LangGraph only when you need
to control the workflow. This follows LangChain's [framework comparison](https://www.langchain.com/blog/deep-agents-vs-langchain-vs-langgraph).

## Setup

Do [SETUP.md](../SETUP.md) once, then install this sample from the repo root:

```bash
uv pip install --python .venv/bin/python -r 03-langgraph-agent/requirements.txt
```

The weather comes from [Open-Meteo](https://open-meteo.com/), which needs no API key.

## Part 1: let LangChain run the standard loop

Run:

```bash
cd 03-langgraph-agent
../.venv/bin/python simple_agent.py
```

Measured output on September 1, 2026 (live weather will differ):

```text
bot > The current weather in Chicago is:
- Temperature: 92.1°F
- Precipitation: 0.0 mm (no rain)
- Wind: 7.4 mph

(steps: HumanMessage -> AIMessage -> ToolMessage -> AIMessage)
```

The important code is deliberately small:

```python
agent = create_agent(model, [get_weather], system_prompt=SYSTEM)
result = agent.invoke({"messages": [{"role": "user", "content": question}]})
```

`create_agent()` handles the familiar loop for us:

```text
model → call get_weather → model reads result → final answer
```

That is the right abstraction for a focused agent with a fixed set of tools.
There is no benefit in drawing the loop by hand.

The `@tool` decorator turns a Python function into something the model can
choose to call. Its name, type hints, and docstring become instructions for the
model, so the docstring in `weather_tool.py` is part of the agent's behavior.

## The new requirement

Now ask: “Should I expect weather delays?” Current weather cannot answer that
on its own. Our application requires every answer to say so and point the user
to their airline.

We could bury that rule in a prompt, but then the model is also responsible for
checking whether it followed the rule. Instead, the application owns the check:

```text
START → draft with LangChain → check in Python ── pass ─→ END
                  ↑                 │
                  └──── revise ─────┤
                                    └── two misses → safe fallback → END
```

## Part 2: use LangGraph for the custom loop

Run:

```bash
../.venv/bin/python agent.py
```

Measured output from the same run, shortened here to emphasize the loop:

```text
Review loop used 2 draft(s).

Based on the current weather in Chicago:
- Temperature: 92.1°F
- Precipitation: 0.0mm (no rain)
- Wind: 7.4 mph (light winds)

Weather readings alone cannot determine whether you'll experience flight
delays. I recommend checking directly with your airline.
```

The `draft` node runs the same kind of LangChain agent from part 1. The `check`
node is ordinary Python. A conditional edge decides whether to finish or request
one revision. The graph caps the work at two drafts so it cannot loop forever;
after two misses, normal Python adds the required safety note.

This is the reason to use LangGraph directly: not because an agent has tools,
but because the application needs to own the control flow.

## What to notice

- **LangChain removes boilerplate.** It owns the standard model/tool loop.
- **LangGraph makes custom flow visible.** Nodes, state, and routes are explicit.
- **They compose.** A LangGraph node can run a LangChain agent.
- **Use the highest useful abstraction.** The first example does not need a
  custom graph; the second one does.

`weather_tool.py` contains plumbing shared by both examples: the tool and the
Databricks-backed `ChatOpenAI` model. Read the two agent files first.

## Next

This state lasts for only one run. [Sample 4](../04-agent-memory/) adds memory
that survives across conversations.
