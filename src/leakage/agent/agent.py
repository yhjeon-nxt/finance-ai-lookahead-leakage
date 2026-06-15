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


def _extract_json(raw: str):
    """Balanced-brace extraction: return the first valid top-level JSON object in `raw`.

    Robust to prose-then-JSON, markdown fences, or a trailing second object — unlike a greedy
    `\\{.*\\}` which grabs the widest span and fails when two braces appear.
    """
    if not raw:
        return None
    depth, start = 0, None
    for i, ch in enumerate(raw):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(raw[start:i + 1])
                except json.JSONDecodeError:
                    start = None  # malformed candidate; keep scanning
    return None


def _coerce_decision(date: str, raw: str, universe: list[str]) -> Decision:
    obj = _extract_json(raw or "")
    if obj is None:
        truncated = bool(raw) and not raw.rstrip().endswith("}")
        return Decision(date, {}, 0.0, "", "", raw=raw, parse_ok=False,
                        error="truncated JSON" if truncated else "no JSON object found")

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
               reflection: str | None = None, max_attempts: int = 3) -> Decision:
        # Retry ONLY on a genuine parse failure. An empty target_weights with parse_ok=True is a
        # VALID all-cash (risk-off) decision — never retry that, or we'd bias exposure upward.
        dec = Decision(date, {}, 0.0, "", "", raw="", parse_ok=False, error="not attempted")
        for attempt in range(max_attempts):
            user = build_user_prompt(context_text, reflection)
            if attempt > 0:
                user += "\n\nReturn ONLY a single valid JSON object with all four fields."
            resp = self.client.complete(SYSTEM_PROMPT, user, temperature=self.temperature,
                                        seed=seed + 1000 * attempt)  # +1000*: avoid other days' seeds
            if not resp.ok:
                dec = Decision(date, {}, 0.0, "", "", raw="", parse_ok=False, error=resp.error)
                continue
            dec = _coerce_decision(date, resp.text, self.universe)
            if dec.parse_ok:
                return dec
        return dec
