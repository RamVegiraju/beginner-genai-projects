"""
The model setup and weather tool shared by both agents in this sample.

Keeping this plumbing here lets `simple_agent.py` focus on LangChain and
`agent.py` focus on LangGraph's custom control flow.

A tool is just a Python function. The @tool decorator exposes it to the model,
which reads the NAME, the TYPE HINTS, and the DOCSTRING to decide when to call
it. That docstring is not a comment -- it's the model's instructions. Write it
for the model, not for yourself.
"""

import os

import requests
from databricks.sdk import WorkspaceClient
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

MODEL = os.environ.get("SERVING_ENDPOINT", "databricks-claude-haiku-4-5")
PROFILE = os.environ.get("DATABRICKS_PROFILE")


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Use this when a question depends on current weather conditions.

    Args:
        city: The city whose weather should be retrieved.
    """
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


# The simple agent only needs one instruction: use live data for live weather.
SYSTEM = "Use the weather tool for questions about current weather."


def make_model() -> ChatOpenAI:
    """Create a LangChain model connected to a Databricks serving endpoint."""
    workspace = WorkspaceClient(profile=PROFILE)
    token = workspace.config.authenticate()["Authorization"].removeprefix("Bearer ")
    return ChatOpenAI(
        model=MODEL,
        api_key=token,
        base_url=f"{workspace.config.host}/serving-endpoints",
        max_tokens=500,
    )
