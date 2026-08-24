"""
Talking to an LLM on Databricks: non-streaming vs streaming.

Same model, same question, two ways of getting the answer back.

Run:  python invoke.py
"""

import os
import sys

from databricks.sdk import WorkspaceClient
from openai import OpenAI

MODEL = os.environ.get("SERVING_ENDPOINT", "databricks-claude-haiku-4-5")
PROFILE = os.environ.get("DATABRICKS_PROFILE", "genai-series")

# Databricks serving endpoints are OpenAI-compatible: use the real OpenAI SDK
# and just point base_url at your workspace. Auth comes from your CLI profile,
# so there's no API key in the code.
w = WorkspaceClient(profile=PROFILE)
token = w.config.authenticate()["Authorization"].removeprefix("Bearer ")
client = OpenAI(api_key=token, base_url=f"{w.config.host}/serving-endpoints")

messages = [{"role": "user", "content": "Explain what an API is, in about 60 words."}]


# --- Non-streaming: wait for the whole answer -------------------------------
# One request, one response. Simple, and what you want when something else
# consumes the output: a database write, another function, a batch job.

print("=== non-streaming ===\n")

response = client.chat.completions.create(
    model=MODEL,
    messages=messages,
    max_tokens=300,
)

print(response.choices[0].message.content)
if response.usage:
    print(f"\n[{response.usage.total_tokens} tokens]")


# --- Streaming: take the tokens as they're generated ------------------------
# Set stream=True and you get an iterator of small chunks instead of one
# response. The total time is the same -- but the user sees words appear
# immediately instead of staring at nothing. That's why chat UIs type at you.

print("\n=== streaming ===\n")

stream = client.chat.completions.create(
    model=MODEL,
    messages=messages,
    max_tokens=300,
    stream=True,
)

for chunk in stream:
    # Some chunks carry no text (they just mark start/stop), so guard for None.
    if not chunk.choices:
        continue
    piece = chunk.choices[0].delta.content
    if piece:
        sys.stdout.write(piece)
        sys.stdout.flush()  # without this your terminal buffers and it won't look live

print()
