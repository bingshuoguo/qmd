"""Thin stdlib client for reward-server.ts (no third-party deps)."""

import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor


class RewardClient:
    def __init__(self, url: str = "http://127.0.0.1:8787", timeout: float = 600.0):
        self.url = url.rstrip("/")
        self.timeout = timeout

    def health(self) -> dict:
        with urllib.request.urlopen(f"{self.url}/health", timeout=self.timeout) as response:
            return json.load(response)

    def score(self, items: list[dict]) -> list[dict]:
        """items: [{"source_id", "qid", "completion", "truncated"?}] -> per-item metrics."""
        request = urllib.request.Request(
            f"{self.url}/score",
            data=json.dumps({"items": items}).encode("utf-8"),
            headers={"content-type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.load(response)["results"]
        except urllib.error.HTTPError as error:
            raise RuntimeError(f"reward server error: {error.read().decode('utf-8')}") from error


class PooledRewardClient:
    """Scores a batch across several reward-server processes in parallel.

    Each server is single-threaded (the vector scan is CPU-bound), so N servers give up to N-way
    parallelism. Items are grouped by (source_id, qid) and cut into N contiguous, near-equal chunks:
    a query lands on one server, or two at a chunk boundary, so the per-server raw cache stays useful.
    All servers must embed on the same device, or scores would depend on which server got the item.
    """

    def __init__(self, urls: list[str], timeout: float = 600.0):
        self.clients = [RewardClient(url, timeout) for url in urls]
        self.pool = ThreadPoolExecutor(len(self.clients))

    def health(self) -> list[dict]:
        return list(self.pool.map(lambda c: c.health(), self.clients))

    def score(self, items: list[dict]) -> list[dict]:
        first = {}
        for i, it in enumerate(items):
            first.setdefault((it["source_id"], it["qid"]), i)
        order = sorted(range(len(items)), key=lambda i: (first[(items[i]["source_id"], items[i]["qid"])], i))
        n = min(len(self.clients), len(items))
        bounds = [len(order) * k // n for k in range(n + 1)]
        chunks = [order[bounds[k]:bounds[k + 1]] for k in range(n)]
        futures = [self.pool.submit(c.score, [items[i] for i in chunk]) for c, chunk in zip(self.clients, chunks)]
        out = [None] * len(items)
        for chunk, future in zip(chunks, futures):
            for i, result in zip(chunk, future.result()):
                out[i] = result
        return out


def make_client(url: str):
    """One URL -> RewardClient; a comma-separated list -> PooledRewardClient."""
    urls = [u for u in url.split(",") if u]
    return RewardClient(urls[0]) if len(urls) == 1 else PooledRewardClient(urls)
