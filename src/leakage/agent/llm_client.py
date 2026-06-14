"""LLM client abstraction.

Two implementations:
  * ``OllamaClient`` — talks to a local ollama server (the real experiment engine).
  * ``MockClient``  — a deterministic, seeded heuristic "trader" with NO look-ahead. It lets
                      us validate the full backtest/metrics/stats pipeline offline (no GPU,
                      no model download). The EC2 run swaps in OllamaClient with identical I/O.

Both return a raw JSON *string*; parsing/validation lives in ``agent.py`` so the two clients
are interchangeable.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np


@dataclass
class LLMResponse:
    text: str
    model: str
    ok: bool = True
    error: str | None = None


class LLMClient:
    name = "base"

    def complete(self, system: str, user: str, *, temperature: float, seed: int) -> LLMResponse:
        raise NotImplementedError


class OllamaClient(LLMClient):
    """Local open-source model via ollama. Used for the actual experiment groups."""

    def __init__(self, model: str, host: str | None = None, num_predict: int = 512):
        self.model = model
        self.name = model
        self.num_predict = num_predict
        self._host = host

    def complete(self, system: str, user: str, *, temperature: float, seed: int) -> LLMResponse:
        import ollama  # imported lazily so the package isn't required for mock runs

        client = ollama.Client(host=self._host) if self._host else ollama
        try:
            resp = client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                format="json",
                options={
                    "temperature": temperature,
                    "seed": seed,
                    "num_predict": self.num_predict,
                },
            )
            return LLMResponse(text=resp["message"]["content"], model=self.model)
        except Exception as e:  # noqa: BLE001 - surfaced to caller for retry/logging
            return LLMResponse(text="", model=self.model, ok=False, error=f"{type(e).__name__}: {e}")


class MockClient(LLMClient):
    """Seeded momentum trader with NO future knowledge.

    Produces realistic target weights from trailing returns only, so the offline pipeline
    test exercises the same code paths as a real model. Because it has no look-ahead, its
    expected leakage/foresight metrics are ~0 — a useful null reference for the metric code.
    """

    def __init__(self, name: str = "mock-momentum", aggressiveness: float = 1.0):
        self.name = name
        self.aggressiveness = aggressiveness

    def complete(self, system: str, user: str, *, temperature: float, seed: int) -> LLMResponse:
        # Parse the rendered context out of the user prompt: the recent-returns block.
        rng = np.random.default_rng(seed + abs(hash(user)) % (2**31))
        tickers, mom = _parse_recent_momentum(user)
        if not tickers:
            return LLMResponse(text=json.dumps({"target_weights": {}, "confidence": 0.1,
                                                "rationale": "no data"}), model=self.name)
        # Momentum tilt + noise; long-only, capped, sum<=1.
        raw = np.maximum(0.0, np.array(mom) * 5.0 * self.aggressiveness
                         + rng.normal(0, 0.3, len(mom)) * temperature)
        if raw.sum() == 0:
            raw = np.ones(len(mom))
        w = raw / max(raw.sum(), 1.0)
        w = np.minimum(w, 0.4)  # position cap
        weights = {t: round(float(x), 3) for t, x in zip(tickers, w) if x > 0.01}
        payload = {
            "target_weights": weights,
            "confidence": round(float(min(0.9, 0.3 + abs(np.mean(mom)) * 5)), 2),
            "rationale": "Momentum tilt from trailing 5-day returns; no view on future events.",
        }
        return LLMResponse(text=json.dumps(payload), model=self.name)


def _parse_recent_momentum(user: str) -> tuple[list[str], list[float]]:
    """Extract tickers and a momentum proxy (mean of last-5 recent returns) from the prompt."""
    tickers: list[str] = []
    mom: list[float] = []
    in_block = False
    for line in user.splitlines():
        if line.strip().startswith("Recent daily returns"):
            in_block = True
            continue
        if in_block:
            s = line.strip()
            if not s or ":" not in s or "[" not in s:
                continue
            tk, arr = s.split(":", 1)
            try:
                vals = json.loads(arr.strip())
                tickers.append(tk.strip())
                mom.append(float(np.mean(vals[-5:])) if vals else 0.0)
            except Exception:  # noqa: BLE001
                continue
    return tickers, mom
