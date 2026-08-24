"""
Giving the model a tool: "will weather delay my flight?"

An LLM cannot know today's weather. Ask it directly and it will hedge or make
something up -- no prompt fixes a missing fact.

A tool fixes it. We hand the model a Python function it can choose to call.
LangGraph runs the loop: model -> tool -> model -> answer.

Run:  python agent.py
"""

import os

import requests
from databricks.sdk import WorkspaceClient
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

MODEL = os.environ.get("SERVING_ENDPOINT", "databricks-claude-haiku-4-5")
PROFILE = os.environ.get("DATABRICKS_PROFILE", "genai-series")


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
# .bind_tools() tells the model what it's allowed to call.
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


def agent(state: MessagesState) -> dict:
    """Ask the model what to do next, given the conversation so far."""
    return {"messages": [llm.invoke(state["messages"])]}


graph = (
    StateGraph(MessagesState)
    .add_node("agent", agent)
    # ToolNode runs whichever tool the model asked for and appends the result.
    .add_node("tools", ToolNode(tools))
    .add_edge(START, "agent")
    # tools_condition inspects the last message: if the model requested a tool
    # it routes to "tools", otherwise it ends.
    .add_conditional_edges("agent", tools_condition)
    # After the tool runs, go back to the model. This is what makes it a loop.
    .add_edge("tools", "agent")
    .compile()
)


if __name__ == "__main__":
    question = "I'm flying out of Chicago today. Should I expect weather delays?"
    print(f"user> {question}\n")

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
