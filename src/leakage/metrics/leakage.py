"""Leakage / foresight metrics — the core contribution.

These quantify whether an agent's allocations anticipate the future in ways unattainable from
the causal price context alone:

1. ``next_day_prescience``   — does it overweight tomorrow's winners / scale risk with tomorrow's
                                market move? (≈0 without foresight.)
2. ``pre_event_timing``      — does it de-risk just before a known crash / load before a known
                                rally, relative to its own average exposure?
3. ``rationale_forensics``   — does its free-text reasoning name future events it should not know?

The ``foresight_gap`` between an in-distribution run and a same-model out-of-distribution run
isolates leakage from raw skill (the model is held fixed; only 'has it seen this era?' changes).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

# Tiered keyword lists for rationale forensics.
# HARD tells = specific future events/outcomes the agent cannot infer from trailing prices.
HARD_TELLS = [
    "carry trade", "yen carry", "august 5", "aug 5", "black monday", "vix spike",
    "election", "trump", "harris", "biden", "red sweep", "republican win",
    "rate cut", "rate hike", "fed pivot", "jackson hole", "santa rally",
    "deepseek", "tariff", "inauguration",
]
# SOFT tells = generic risk vocabulary (reported separately; weak evidence on their own).
SOFT_TELLS = [
    "crash", "selloff", "sell-off", "plunge", "correction", "recession",
    "rally", "surge", "spike", "volatility", "downturn", "rebound",
]
_MONTHS = ["january", "february", "march", "april", "may", "june", "july",
           "august", "september", "october", "november", "december"]
# Denial cues: a hard tell preceded by one of these is a *negation* ("no information about the
# crash"), not a smoking gun, so it must not count.
_DENIAL_CUES = ["no ", "not ", "without ", "cannot ", "can't ", "don't", "do not", "n't ",
                "unaware", "no information", "no knowledge", "unable", "haven't", "avoid",
                "isn't", "is not", "lack of", "uncertain", "unsure"]


def _genuine_mention(text: str, kw: str, window: int = 35) -> bool:
    """True if `kw` appears in `text` and is NOT immediately preceded by a denial cue."""
    i = text.find(kw)
    while i != -1:
        pre = text[max(0, i - window):i]
        if not any(cue in pre for cue in _DENIAL_CUES):
            return True
        i = text.find(kw, i + 1)
    return False


def _corr(a: pd.Series, b: pd.Series) -> float:
    a, b = a.align(b, join="inner")
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a.values, b.values)[0, 1])


def next_day_prescience(result) -> dict[str, float]:
    """Correlations between today's allocation and tomorrow's realised returns."""
    w, ndr = result.weights, result.next_day_returns
    common = w.index.intersection(ndr.index)

    per_ticker = []
    for t in w.columns:
        c = _corr(w.loc[common, t], ndr.loc[common, t])
        if not np.isnan(c):
            per_ticker.append(c)
    ticker_prescience = float(np.mean(per_ticker)) if per_ticker else float("nan")

    expo = result.exposure.reindex(common)
    mkt_next = ndr.loc[common].mean(axis=1)             # equal-weight universe next-day return
    timing = _corr(expo, mkt_next)                       # risk-on before up days?
    conf = result.confidence.reindex(common)
    conf_timing = _corr(expo * conf, mkt_next)

    return {
        "ticker_prescience": ticker_prescience,
        "exposure_timing": timing,
        "conf_weighted_timing": conf_timing,
    }


def pre_event_timing(result, events, k_baseline: int = 20) -> dict[str, float]:
    """Directional positioning right before known events vs the agent's own average exposure.

    For a down event: score>0 means it de-risked (lower exposure) just before the crash.
    For an up   event: score>0 means it loaded up (higher exposure) just before the rally.
    Only events inside the run's window are scored.
    """
    expo = result.exposure.sort_index()
    if expo.empty:
        return {}
    scores = {}
    for d, label, sgn in events:
        d = pd.Timestamp(d)
        pre = expo[expo.index < d]
        if len(pre) < 2:
            continue
        expo_pre = float(pre.iloc[-1])                       # position held INTO the event
        # Baseline = the agent's normal exposure over the k days BEFORE the event (excluding the
        # event/post-event days), so the score reflects timing, not the whole-window average.
        baseline = float(pre.iloc[-(k_baseline + 1):-1].mean()) if len(pre) > 1 else float(pre.mean())
        # directional: down event rewards lower-than-usual exposure; up event the opposite
        score = (baseline - expo_pre) if sgn < 0 else (expo_pre - baseline)
        scores[label] = score
    if scores:
        scores["event_timing_mean"] = float(np.mean(list(scores.values())))
    return scores


def event_day_dodge(result, events, market_returns: pd.Series) -> dict[str, float]:
    """Agent's realised return on the event day, and its excess over the equal-weight market.

    For a down event, ``excess > 0`` means the agent lost less than the market (dodged the
    crash); for an up event, ``excess > 0`` means it captured more of the rally.

    ``market_returns`` MUST be indexed by *realised* date (the equal-weight universe return on
    that calendar day) to match ``port_returns``. Do not pass ``next_day_returns`` (which is
    indexed by *decision* date) — that would compare against the day-after market (off-by-one).
    """
    pr = result.port_returns
    mkt = market_returns
    out = {}
    for d, label, sgn in events:
        d = pd.Timestamp(d)
        if d not in pr.index:
            continue
        port_r = float(pr.loc[d])
        mkt_r = float(mkt.loc[d]) if d in mkt.index else float("nan")
        out[f"{label}_port_ret"] = port_r
        out[f"{label}_excess_vs_mkt"] = port_r - mkt_r
    return out


def rationale_forensics(result) -> dict:
    """Scan analysis+rationale text for future-event references (smoking-gun channel)."""
    hard_hits, soft_hits, excerpts = {}, {}, []
    for dec in result.decisions:
        text = f"{dec.analysis} {dec.rationale}".lower()
        if not text.strip():
            continue
        decision_month = None
        try:
            decision_month = int(dec.date.split("-")[1])
        except Exception:  # noqa: BLE001
            pass
        # Negation-aware: a tell preceded by a denial cue (e.g. "no information about the
        # crash") is NOT a smoking gun and must not be counted.
        hard_found = [kw for kw in HARD_TELLS if _genuine_mention(text, kw)]
        soft_found = [kw for kw in SOFT_TELLS if _genuine_mention(text, kw)]
        # Future-month mention: a month name strictly after the decision month, same year-ish.
        future_months = []
        if decision_month:
            for idx, m in enumerate(_MONTHS, start=1):
                if m in text and idx > decision_month:
                    future_months.append(m)
        for kw in hard_found:
            hard_hits[kw] = hard_hits.get(kw, 0) + 1
        for kw in soft_found:
            soft_hits[kw] = soft_hits.get(kw, 0) + 1
        if hard_found or future_months:
            excerpts.append({
                "date": dec.date,
                "hard_tells": hard_found,
                "future_months": future_months,
                "text": f"{dec.analysis} {dec.rationale}".strip()[:400],
            })
    return {
        "hard_tell_counts": hard_hits,
        "soft_tell_counts": soft_hits,
        "n_hard_decisions": len(excerpts),
        "excerpts": excerpts[:25],
    }


def foresight_gap(in_dist: dict[str, float], ood: dict[str, float]) -> dict[str, float]:
    """In-dist minus OOD for matching prescience keys (same model → isolates leakage)."""
    return {f"gap_{k}": (in_dist.get(k, float('nan')) - ood.get(k, float('nan')))
            for k in in_dist if k in ood}
