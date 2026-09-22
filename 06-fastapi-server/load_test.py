"""Send concurrent traffic at the running server.

Eight questions go out together. Agent inference is mostly waiting -- on the
model, on the network -- so one process can carry all eight by overlapping
the waiting.

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


async def ask(client: httpx.AsyncClient, question: str) -> float:
    """Send one question to /chat and return how long it took."""
    start = time.perf_counter()
    response = await client.post(f"{URL}/chat", json={"message": question})
    response.raise_for_status()
    return time.perf_counter() - start


async def main() -> None:
    async with httpx.AsyncClient(timeout=180) as client:
        try:
            (await client.get(f"{URL}/health")).raise_for_status()
        except httpx.HTTPError:
            sys.exit(f"No server at {URL}. Start it with:  uvicorn server:app")

        start = time.perf_counter()
        durations = await asyncio.gather(*(ask(client, q) for q in QUESTIONS))
        wall = time.perf_counter() - start

        print(f"\n{len(QUESTIONS)} concurrent requests to /chat:\n")
        print(f"  all replies in   {wall:5.1f}s")
        print(f"  slowest single   {max(durations):5.1f}s")
        print("\nThey overlap, so the set costs about what its slowest request costs.")

        # The graph can do the same fan-out server-side, in one call.
        start = time.perf_counter()
        batch = await client.post(f"{URL}/chat/batch", json={"messages": QUESTIONS})
        batch.raise_for_status()
        print(f"\n/chat/batch (graph.abatch, one request)  {time.perf_counter() - start:5.1f}s")
        print(f"  returned {len(batch.json()['replies'])} replies")


if __name__ == "__main__":
    asyncio.run(main())
