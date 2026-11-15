"""Thin stdlib client for reward-server.ts (no third-party deps)."""

import json
import urllib.request


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
