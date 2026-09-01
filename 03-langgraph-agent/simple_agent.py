"""
The easy way: one function call builds the whole agent.

Read this file first, then `agent.py`. This one is short because LangChain's
`create_agent` assembles the standard agent loop for you:

    model -> tool -> model -> answer

Everything below is setup. The agent itself is a single line.

Run:  python simple_agent.py
"""

from langchain.agents import create_agent
from weather_tool import SYSTEM, get_weather, make_model

# This is the whole agent.
#
# Note there is no .bind_tools() here -- create_agent does that for you, along
# with running the tools and looping back to the model afterwards.
agent = create_agent(make_model(), [get_weather], system_prompt=SYSTEM)


if __name__ == "__main__":
    question = "What is the weather in Chicago right now?"
    print(f"user> {question}\n")

    result = agent.invoke({"messages": [{"role": "user", "content": question}]})

    print(f"bot > {result['messages'][-1].content}")

    # The tool call really did happen -- it's just hidden inside create_agent.
    # Every step is still in the message list:
    steps = [m.__class__.__name__ for m in result["messages"]]
    print(f"\n(steps: {' -> '.join(steps)})")
