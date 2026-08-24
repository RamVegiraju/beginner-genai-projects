"""Show what async serving buys you.

The same eight questions, sent two ways to the same endpoint:

  one at a time   wait for each reply before sending the next
  all at once     send them together and let the server overlap the waiting

Agent inference is mostly waiting on the model, so the second one finishes in
about the time of a single request. That gap is the reason to serve agents
asynchronously.

Run the server first:  uvicorn server:app
Then:                  python load_test.py
"""

import asyncio
import sys
import time

import httpx

URL = "http://localhost:8000"
QUESTIONS = [
    "In two sentences, what is an API?",
    "In two sentences, what is a database index?",
    "In two sentences, what is a queue?",
    "In two sentences, what is caching?",
    "In two sentences, what is a load balancer?",
    "In two sentences, what is a webhook?",
    "In two sentences, what is idempotency?",
    "In two sentences, what is backpressure?",
]


async def ask(client: httpx.AsyncClient, question: str) -> str:
    """Send one question to /chat and return the reply."""
    response = await client.post(f"{URL}/chat", json={"message": question})
    response.raise_for_status()
    return response.json()["reply"]


async def one_at_a_time(client: httpx.AsyncClient) -> float:
    """Send each question only after the previous reply arrives."""
    start = time.perf_counter()
    for question in QUESTIONS:
        await ask(client, question)
    return time.perf_counter() - start


async def all_at_once(client: httpx.AsyncClient) -> float:
    """Send every question together and wait for the last one."""
    start = time.perf_counter()
    await asyncio.gather(*(ask(client, question) for question in QUESTIONS))
    return time.perf_counter() - start


async def main() -> None:
    async with httpx.AsyncClient(timeout=180) as client:
        try:
            (await client.get(f"{URL}/health")).raise_for_status()
        except httpx.HTTPError:
            sys.exit(f"No server at {URL}. Start it with:  uvicorn server:app")

        serial = await one_at_a_time(client)
        concurrent = await all_at_once(client)

        print(f"\n{len(QUESTIONS)} questions, same endpoint:\n")
        print(f"  one at a time   {serial:5.1f}s")
        print(f"  all at once     {concurrent:5.1f}s")
        print(f"\n{serial / concurrent:.1f}x faster, on one process and one thread.")
        print("The server spent the waiting on other people's requests.")

        # The graph can do the same fan-out server-side, in one call.
        start = time.perf_counter()
        batch = await client.post(f"{URL}/chat/batch", json={"messages": QUESTIONS})
        batch.raise_for_status()
        print(f"\n/chat/batch (graph.abatch, one request)  {time.perf_counter() - start:5.1f}s")
        print(f"  returned {len(batch.json()['replies'])} replies")


if __name__ == "__main__":
    asyncio.run(main())
