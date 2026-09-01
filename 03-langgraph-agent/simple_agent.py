"""
The easy way: one function call builds the whole agent.

Read this file first, then `agent.py`. They do the same job. This one is short
because LangChain's `create_agent` assembles the agent loop for you:

    model -> tool -> model -> answer

Everything below is setup. The agent itself is a single line.

Run:  python simple_agent.py
"""

import os

from databricks.sdk import WorkspaceClient
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from weather_tool import SYSTEM, get_weather

MODEL = os.environ.get("SERVING_ENDPOINT", "databricks-claude-haiku-4-5")
PROFILE = os.environ.get("DATABRICKS_PROFILE")

# Same endpoint as samples 1 and 2, reached through LangChain.
w = WorkspaceClient(profile=PROFILE)
token = w.config.authenticate()["Authorization"].removeprefix("Bearer ")
llm = ChatOpenAI(
    model=MODEL,
    api_key=token,
    base_url=f"{w.config.host}/serving-endpoints",
    max_tokens=500,
)

# This is the whole agent.
#
# Note there is no .bind_tools() here -- create_agent does that for you, along
# with running the tools and looping back to the model afterwards.
agent = create_agent(llm, [get_weather], system_prompt=SYSTEM)


if __name__ == "__main__":
    question = "I'm flying out of Chicago today. Should I expect weather delays?"
    print(f"user> {question}\n")

    result = agent.invoke({"messages": [{"role": "user", "content": question}]})

    print(f"bot > {result['messages'][-1].content}")

    # The tool call really did happen -- it's just hidden inside create_agent.
    # Every step is still in the message list:
    steps = [m.__class__.__name__ for m in result["messages"]]
    print(f"\n(steps: {' -> '.join(steps)})")
