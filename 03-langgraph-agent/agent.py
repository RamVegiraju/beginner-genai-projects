"""
Giving the model a tool: "will weather delay my flight?"

An LLM cannot know today's weather. Ask it directly and it will hedge or make
something up -- no prompt fixes a missing fact.

A tool fixes it. We hand the model a Python function it can choose to call.
LangGraph runs the loop: model -> tool -> model -> answer.

Two calls in this file are easy to confuse:

  llm.invoke()    calls the model. It happens in one place: the `agent` node.
  graph.invoke()  runs the graph. It calls no model itself -- it moves between
                  nodes until one of them stops asking for tools.

Run:  python agent.py
"""

import os

import requests
from databricks.sdk import WorkspaceClient
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

MODEL = os.environ.get("SERVING_ENDPOINT", "databricks-claude-haiku-4-5")
PROFILE = os.environ.get("DATABRICKS_PROFILE")

SYSTEM = """Use the weather tool for current conditions. Only report facts the tool
returns. Do not predict whether delays are likely or whether the readings will cause
delays, and do not characterize the conditions as favorable or unfavorable for flying.
Say that weather readings alone cannot determine airport or airline delays, then
recommend checking the airline or airport for delay status."""


# --- The tool ---------------------------------------------------------------
# A tool is just a Python function. The @tool decorator exposes it to the
# model, which reads the NAME, the TYPE HINTS, and the DOCSTRING to decide
# when to call it. That docstring is not a comment -- it's the model's
# instructions. Write it for the model, not for yourself.


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city. Use this whenever the user asks
    about weather, flights, or travel conditions."""
    # Open-Meteo is free and needs no API key. Two calls: name -> coordinates,
    # then coordinates -> conditions.
    geo = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1},
        timeout=10,
    ).json()

    if not geo.get("results"):
        return f"Could not find a place called {city!r}."

    place = geo["results"][0]

    weather = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            "current": "temperature_2m,precipitation,wind_speed_10m",
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
        },
        timeout=10,
    ).json()["current"]

    return (
        f"{place['name']}, {place.get('country_code', '')}: "
        f"{weather['temperature_2m']}F, "
        f"precipitation {weather['precipitation']}mm, "
        f"wind {weather['wind_speed_10m']}mph"
    )


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


# Written out longhand on purpose -- the point of the sample is to see the loop.
# LangChain ships an equivalent prebuilt: `from langchain.agents import create_agent`,
# then `create_agent(llm, tools)`. (LangGraph's older `create_react_agent` is
# deprecated in favour of it.) This sample does not install `langchain`.


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
