"""
The same agent as `simple_agent.py`, built by hand so you can see inside it.

`create_agent` gave us the loop in one line. Here it is written out: two nodes
and an edge that loops back. The result is the same kind of object -- a compiled
LangGraph graph -- with the same shape:

    START -> agent -> tools -> agent -> END

Why bother? Not to customize the loop -- `create_agent` takes middleware for
that (capping calls, human approval, model fallback), and using it beats
hand-rolling every time.

Bother for two other reasons. First, to see what an agent actually is: a loop,
two nodes and an edge that goes back. Second, for when you need a different
*shape* -- several agents with a supervisor, an agent as one step in a longer
pipeline, branches that fan out. Sample 4 is that case: one node, no tools, plus
a checkpointer for memory.

Two calls in this file are easy to confuse:

  llm.invoke()    calls the model. It happens in one place: the `agent` node.
  graph.invoke()  runs the graph. It calls no model itself -- it moves between
                  nodes until one of them stops asking for tools.

Run:  python agent.py
"""

import os

from databricks.sdk import WorkspaceClient
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from weather_tool import SYSTEM, get_weather

MODEL = os.environ.get("SERVING_ENDPOINT", "databricks-claude-haiku-4-5")
PROFILE = os.environ.get("DATABRICKS_PROFILE")

# The same tool and system prompt `simple_agent.py` uses.
tools = [get_weather]

# --- The model --------------------------------------------------------------
# Same endpoint as samples 1 and 2, reached through LangChain this time.
#
# .bind_tools() runs nothing. It turns each function into a JSON schema and
# attaches it to every request, so the model knows what it may ask for. When
# the model wants one it replies with a `tool_calls` field -- a *request* to
# run something, not a result. Executing it is ToolNode's job below.
#
# That is why the same list is passed in two places, for two different jobs:
#
#   bind_tools(tools)   the schemas the model reads
#   ToolNode(tools)     the callables the graph runs
w = WorkspaceClient(profile=PROFILE)
token = w.config.authenticate()["Authorization"].removeprefix("Bearer ")
llm = ChatOpenAI(
    model=MODEL,
    api_key=token,
    base_url=f"{w.config.host}/serving-endpoints",
    max_tokens=500,
).bind_tools(tools)


# --- The graph --------------------------------------------------------------
# Two nodes and a loop:
#
#        START -> agent -> (tool calls?) -> tools -> back to agent
#                   |                                     |
#                   +-- no tool calls --> END <-----------+
#
# The loop matters: after a tool runs, control returns to the model so it can
# read the result and either answer or call another tool.
#
# MessagesState is a TypedDict with one key, `messages`, whose reducer is
# `add_messages`. A node returns only what is NEW and LangGraph appends it, so
# the list grows into the full transcript without anyone rebuilding it.


def agent(state: MessagesState) -> dict:
    """Ask the model what to do next, given the conversation so far.

    This is the only model call in the file. The node runs once per pass
    through the loop: first to decide whether a tool is needed, then again on
    the way back, to turn the tool result into an answer.

    Returning one message instead of the whole list is deliberate --
    `add_messages` appends it to whatever is already in state.
    """
    return {"messages": [llm.invoke([SystemMessage(SYSTEM)] + state["messages"])]}


graph = (
    StateGraph(MessagesState)
    .add_node("agent", agent)
    # ToolNode reads `tool_calls` off the last message, runs those tools -- in
    # parallel if the model asked for several -- and appends one ToolMessage
    # per call.
    .add_node("tools", ToolNode(tools))
    .add_edge(START, "agent")
    # tools_condition inspects the last message and returns the literal string
    # "tools" if it carries tool calls, otherwise "__end__". Both are node
    # names, which is why no path map is needed here.
    .add_conditional_edges("agent", tools_condition)
    # After the tool runs, go back to the model. This is what makes it a loop.
    .add_edge("tools", "agent")
    .compile()
)


# How close is this to the prebuilt? Nearly identical. `create_agent` returns a
# CompiledStateGraph too, with the same nodes -- it just calls the first one
# "model" instead of "agent":
#
#   simple_agent.py  ['__start__', 'model', 'tools', '__end__']
#   agent.py         ['__start__', 'agent', 'tools', '__end__']
#
# So this is not a lower-level alternative to `create_agent`. It is what
# `create_agent` builds, with the lid off.
#
# (LangGraph's older `create_react_agent` is deprecated in favour of
# `langchain.agents.create_agent`. Use the latter.)


if __name__ == "__main__":
    question = "I'm flying out of Chicago today. Should I expect weather delays?"
    print(f"user> {question}\n")

    # Runs the graph, not the model: START -> agent -> tools -> agent -> END.
    # For this question the agent node runs twice, so two model calls in total.
    result = graph.invoke({"messages": [{"role": "user", "content": question}]})

    # Print every step so the loop is visible instead of magic.
    for message in result["messages"]:
        kind = message.__class__.__name__
        if getattr(message, "tool_calls", None):
            for call in message.tool_calls:
                print(f"[{kind}] calling {call['name']}({call['args']})")
        elif kind == "ToolMessage":
            print(f"[{kind}] returned: {message.content}")
        elif message.content and kind == "AIMessage":
            print(f"\nbot > {message.content}")
