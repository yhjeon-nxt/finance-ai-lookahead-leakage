"""Trading agent: wraps an LLMClient, renders prompts, parses + validates decisions.

A ``Decision`` is the structured output for one trading day. Parsing is defensive: malformed
JSON or out-of-range weights are repaired/clamped, and a HOLD-cash fallback is used if the
model returns nothing usable (logged so failures are visible, not silently masked).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from leakage.agent.llm_client import LLMClient
from leakage.agent.prompts import SYSTEM_PROMPT, build_user_prompt


@dataclass
class Decision:
    date: str
    target_weights: dict[str, float]
    confidence: float
    analysis: str
    rationale: str
    raw: str = ""
    parse_ok: bool = True
    error: str | None = None

    @property
    def exposure(self) -> float:
        """Total invested fraction (1 - cash). The portfolio 'risk dial'."""
        return float(sum(self.target_weights.values()))


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _coerce_decision(date: str, raw: str, universe: list[str]) -> Decision:
    m = _JSON_RE.search(raw or "")
    if not m:
        return Decision(date, {}, 0.0, "", "", raw=raw, parse_ok=False,
                        error="no JSON object found")
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        return Decision(date, {}, 0.0, "", "", raw=raw, parse_ok=False, error=str(e))

    weights = {}
    for k, v in (obj.get("target_weights") or {}).items():
        if k in universe:
            try:
                w = float(v)
            except (TypeError, ValueError):
                continue
            if w > 0:
                weights[k] = min(max(w, 0.0), 1.0)
    # Enforce sum <= 1 (scale down if the model over-allocated).
    total = sum(weights.values())
    if total > 1.0:
        weights = {k: v / total for k, v in weights.items()}

    try:
        conf = min(max(float(obj.get("confidence", 0.0)), 0.0), 1.0)
    except (TypeError, ValueError):
        conf = 0.0

    return Decision(
        date=date,
        target_weights=weights,
        confidence=conf,
        analysis=str(obj.get("analysis", ""))[:2000],
        rationale=str(obj.get("rationale", ""))[:1000],
        raw=raw,
        parse_ok=True,
    )


class TradingAgent:
    def __init__(self, client: LLMClient, universe: list[str], temperature: float = 0.7):
        self.client = client
        self.universe = universe
        self.temperature = temperature

    def decide(self, date: str, context_text: str, seed: int,
               reflection: str | None = None) -> Decision:
        user = build_user_prompt(context_text, reflection)
        resp = self.client.complete(SYSTEM_PROMPT, user,
                                    temperature=self.temperature, seed=seed)
        if not resp.ok:
            return Decision(date, {}, 0.0, "", "", raw="", parse_ok=False, error=resp.error)
        return _coerce_decision(date, resp.text, self.universe)
