"""
The same work, sent to two endpoints.

This script is identical for both: it fires every request at the same instant
and waits for all of them. The only thing that changes is which endpoint is
answering -- so any difference in the numbers comes from the server.

Run the server first:  uvicorn server:app
Then:                  python load_test.py
"""

import asyncio
import sys
import time

import httpx

URL = "http://localhost:8000"
REQUESTS = 8
PROMPT = "In two sentences, explain what an API is."


async def one_call(client: httpx.AsyncClient, path: str) -> None:
    response = await client.post(f"{URL}{path}", json={"message": PROMPT})
    response.raise_for_status()


async def time_all_at_once(path: str, n: int) -> float:
    """Fire n requests simultaneously, return how long until the last finishes."""
    async with httpx.AsyncClient(timeout=180) as client:
        start = time.perf_counter()
        await asyncio.gather(*(one_call(client, path) for _ in range(n)))
        return time.perf_counter() - start


async def median_single_request() -> float:
    """Time one request a few times and take the middle value.

    A single sample is noisy enough to make the comparison below look odd,
    and this number is what everything else is measured against.
    """
    times = [await time_all_at_once("/chat", 1) for _ in range(3)]
    return sorted(times)[1]


async def main() -> None:
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            (await client.get(f"{URL}/health")).raise_for_status()
        except httpx.HTTPError:
            sys.exit(f"No server at {URL}. Start it with:  uvicorn server:app")

    # How long does one request take with nothing else going on? Everything
    # below is measured against this.
    baseline = await median_single_request()
    print(f"one request, on its own:   {baseline:5.1f}s\n")

    print(f"{REQUESTS} requests, all sent at the same moment:")
    blocking = await time_all_at_once("/chat/blocking", REQUESTS)
    print(f"  /chat/blocking           {blocking:5.1f}s   ({blocking / baseline:.1f}x one request)")

    concurrent = await time_all_at_once("/chat", REQUESTS)
    print(
        f"  /chat                    {concurrent:5.1f}s   ({concurrent / baseline:.1f}x one request)"
    )

    print(f"\n{blocking / concurrent:.1f}x faster. Same model, same machine, same work.")
    print("The blocking server did them one after another; the async one overlapped them.")


if __name__ == "__main__":
    asyncio.run(main())
