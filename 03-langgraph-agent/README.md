# 3 — Adding tools with LangGraph

> **Not built yet.** Design notes only.

**The wall:** the model cannot know today's weather, and confidently makes it
up. No amount of prompting fixes a missing fact.

**The use case:** *"Should I expect my flight to be delayed?"* — a weather
lookup agent using [Open-Meteo](https://open-meteo.com/), which is free and
needs **no API key** (so nobody gets blocked on signup).

Why this use case: the failure is obvious and visceral. Ask without the tool
and the model hedges or invents. Ask with it and you get a real forecast. The
value of tool calling lands in one side-by-side demo.

Planned:
- `tools.py` — one `get_forecast(city)` function hitting Open-Meteo
- `agent.py` — a minimal LangGraph `StateGraph`: `agent → tools → agent`
- Print the tool-call round trip so the loop is visible, not magic

**Ends on:** the graph works but is reborn empty on every run. → sample 4.
