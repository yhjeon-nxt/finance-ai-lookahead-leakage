"""Date-conditioned PRICE-RECALL probe — directly measures the parametric memory the experiment
assumes (that the treatment's weights store 2024-H2 market specifics), instead of only inferring
it from trades.

Asks the model up/down direction questions about specific dated market moves, scored against the
ACTUAL realised returns from our price data. In-distribution (2024-H2) accuracy ≫ chance is direct
evidence of memorisation; out-of-distribution (2026) accuracy ≈ chance confirms the OOD control.
Run AFTER the multi-period job to avoid GPU contention:
    PYTHONPATH=src python -m leakage.run.price_recall_probe
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from leakage.config import IN_DIST, OOD, RESULTS_DIR, TREATMENT_MODEL  # noqa: E402
from leakage.data.ingest import load_prices  # noqa: E402

# (ticker, date, human label). Direction is scored from actual data, not hard-coded.
IN_DIST_Q = [
    ("SPY", "2024-08-05", "the day of the early-August 2024 global selloff"),
    ("SPY", "2024-11-06", "the day after the November 2024 US election"),
    ("IWM", "2024-11-06", "US small caps the day after the 2024 election"),
    ("NVDA", "2024-12-31", "NVIDIA over the second half of 2024 (Jul→Dec)"),
    ("TSLA", "2024-11-06", "Tesla the day after the 2024 election"),
]
OOD_Q = [
    ("SPY", "2026-03-16", "the S&P 500 in mid-March 2026"),
    ("SPY", "2026-02-02", "the S&P 500 in early February 2026"),
    ("NVDA", "2026-03-16", "NVIDIA in mid-March 2026"),
    ("QQQ", "2026-04-15", "the Nasdaq-100 in mid-April 2026"),
    ("IWM", "2026-02-17", "US small caps in mid-February 2026"),
]


def _actual_dir(ticker: str, datestr: str, horizon_days: int = 1) -> int | None:
    """Realised sign of the move at/after datestr (multi-day for the H2-2024 'over period' Qs)."""
    for win in (IN_DIST, OOD):
        try:
            df = load_prices(win)
        except Exception:  # noqa: BLE001
            continue
        close = df["Close"] if "Close" in df.columns.get_level_values(0) else df
        if ticker not in close.columns:
            continue
        s = close[ticker].dropna()
        d = pd.Timestamp(datestr)
        idx = s.index[s.index >= d]
        if len(idx) == 0:
            continue
        i = s.index.get_loc(idx[0])
        if i == 0:
            continue
        prev = s.iloc[i - 1]
        fut = s.iloc[min(i + horizon_days - 1, len(s) - 1)]
        return 1 if fut >= prev else -1
    return None


def _ask_dir(model: str, ticker: str, label: str) -> int:
    import ollama
    client = ollama.Client(timeout=600)
    q = (f"For {ticker} ({label}): did the price go UP or DOWN over that move? "
         f"Answer with exactly one word: UP or DOWN.")
    think = any(k in model.lower() for k in ("qwen3", "r1"))
    kw = dict(model=model, messages=[{"role": "user", "content": q}],
              options={"temperature": 0.0, "num_predict": 50})
    try:
        r = client.chat(**({**kw, "think": False} if think else kw))
        t = r["message"]["content"].upper()
    except Exception as e:  # noqa: BLE001
        return 0
    if "UP" in t and "DOWN" not in t:
        return 1
    if "DOWN" in t and "UP" not in t:
        return -1
    return 0


def run(model: str | None = None) -> dict:
    model = model or TREATMENT_MODEL.tag
    out = {"model": model, "cells": {}}
    for cell, qs in (("in_dist_2024H2", IN_DIST_Q), ("ood_2026", OOD_Q)):
        correct = total = 0
        rows = []
        for ticker, datestr, label in qs:
            actual = _actual_dir(ticker, datestr)
            pred = _ask_dir(model, ticker, label)
            ok = (pred != 0 and actual is not None and pred == actual)
            correct += int(ok); total += int(actual is not None)
            rows.append({"q": f"{ticker} {label}", "pred": pred, "actual": actual, "correct": ok})
            print(f"  [{cell}] {ticker} {datestr}: pred={pred} actual={actual} {'✓' if ok else '✗'}")
        out["cells"][cell] = {"accuracy": (correct / total if total else None),
                              "correct": correct, "total": total, "rows": rows}
    path = RESULTS_DIR / "price_recall_probe.json"
    path.write_text(json.dumps(out, indent=2))
    ind = out["cells"]["in_dist_2024H2"]["accuracy"]
    ood = out["cells"]["ood_2026"]["accuracy"]
    print(f"\nprice-recall accuracy — in-dist 2024H2: {ind}  vs  OOD 2026: {ood}")
    print(f"[price-recall] wrote {path}")
    return out


if __name__ == "__main__":
    run()
