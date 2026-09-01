"""
The tool, shared by both agents in this sample.

`simple_agent.py` and `agent.py` build the SAME agent two different ways. Keeping
the tool here means the only difference between those two files is the wiring --
which is the whole point of the comparison.

A tool is just a Python function. The @tool decorator exposes it to the model,
which reads the NAME, the TYPE HINTS, and the DOCSTRING to decide when to call
it. That docstring is not a comment -- it's the model's instructions. Write it
for the model, not for yourself.
"""

import requests
from langchain_core.tools import tool


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


# The system prompt is shared too. The tool reports weather; it knows nothing
# about airline operations, so this keeps the answer honest about that.
SYSTEM = """Use the weather tool for current conditions. Only report facts the tool
returns. Do not predict whether delays are likely or whether the readings will cause
delays, and do not characterize the conditions as favorable or unfavorable for flying.
Say that weather readings alone cannot determine airport or airline delays, then
recommend checking the airline or airport for delay status."""
