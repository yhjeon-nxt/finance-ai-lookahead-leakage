"""Extra result figures from the EC2 decision logs (no GPU, no new runs).

  1. per-ticker prescience: corr(today's weight, tomorrow's return) per symbol × group.
  2. election-window allocation: mean weight per symbol around Nov-5 2024, T-in vs C-A.
  3. headline timing-prescience per group with bootstrap CIs (from eval_ec2.json).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from leakage.config import IN_DIST, OOD, RESULTS_DIR, UNIVERSE, Window  # noqa: E402
from leakage.data.ingest import load_prices, trading_days  # noqa: E402

DEC = RESULTS_DIR / "ec2" / "decisions"
FIG = RESULTS_DIR / "figures"
FIG.mkdir(parents=True, exist_ok=True)
NOV5 = pd.Timestamp("2024-11-05")
TRUMP_TRADE = {"IWM", "JPM", "TSLA", "COIN"}  # expected 2024 election beneficiaries


def _weights_by_date(group: str, model_tag: str, window: Window) -> pd.DataFrame:
    safe = model_tag.replace(":", "-")
    frames = []
    for f in sorted(DEC.glob(f"{group}_{safe}_seed*.jsonl")):
        rows = {}
        for line in f.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                if d.get("parse_ok", True):
                    w = d.get("target_weights") or {}
                    rows[pd.Timestamp(d["date"])] = {t: float(w.get(t, 0.0)) for t in UNIVERSE}
        if rows:
            frames.append(pd.DataFrame(rows).T[UNIVERSE])
    if not frames:
        return pd.DataFrame(columns=UNIVERSE)
    return sum(f.reindex(frames[0].index.union(frames[1].index if len(frames) > 1 else frames[0].index)).fillna(0)
               for f in frames) / len(frames) if len(frames) > 1 else frames[0]


def _next_day_rets(window: Window) -> pd.DataFrame:
    close = load_prices(window)
    close = (close["Close"] if "Close" in close.columns.get_level_values(0) else close)[UNIVERSE]
    rets = close.pct_change()
    days = trading_days(window)
    # map decision day -> next-day return row
    out = {}
    for i, d in enumerate(days[:-1]):
        out[d] = rets.loc[days[i + 1]]
    return pd.DataFrame(out).T[UNIVERSE]


def _ticker_prescience(group, tag, window) -> pd.Series:
    w = _weights_by_date(group, tag, window)
    nd = _next_day_rets(window)
    common = w.index.intersection(nd.index)
    out = {}
    for t in UNIVERSE:
        a, b = w.loc[common, t], nd.loc[common, t]
        out[t] = np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else np.nan
    return pd.Series(out)


def _plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = [("T-in", "qwen3:8b", IN_DIST), ("C-A", "llama3.1:8b", IN_DIST),
              ("C-B", "qwen3:8b", OOD)]
    colors = {"T-in": "tab:green", "C-A": "tab:blue", "C-B": "tab:orange"}

    # --- Figure 1: per-ticker prescience ---
    presc = {g: _ticker_prescience(g, tag, w) for g, tag, w in groups}
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(UNIVERSE)); bw = 0.26
    for i, g in enumerate(["T-in", "C-A", "C-B"]):
        ax.bar(x + (i - 1) * bw, [presc[g][t] for t in UNIVERSE], bw, label=g, color=colors[g])
    ax.axhline(0, color="k", lw=0.6); ax.set_xticks(x); ax.set_xticklabels(UNIVERSE)
    ax.set_ylabel("corr(today's weight, next-day return)")
    ax.set_title("Per-ticker next-day prescience — only the treatment (T-in) is consistently positive")
    ax.legend(); fig.tight_layout(); fig.savefig(FIG / "ticker_prescience.png", dpi=130); plt.close(fig)

    # --- Figure 2: election-window mean allocation (T-in vs C-A) ---
    lo, hi = NOV5 - pd.Timedelta(days=4), NOV5 + pd.Timedelta(days=4)
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (g, tag) in enumerate([("T-in", "qwen3:8b"), ("C-A", "llama3.1:8b")]):
        w = _weights_by_date(g, tag, IN_DIST)
        win = w[(w.index >= lo) & (w.index <= hi)]
        means = win.mean() if len(win) else pd.Series(0, index=UNIVERSE)
        ax.bar(x + (i - 0.5) * 0.4, [means[t] for t in UNIVERSE], 0.4, label=g,
               color=colors[g])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t}*" if t in TRUMP_TRADE else t for t in UNIVERSE])
    ax.set_ylabel("mean target weight")
    ax.set_title("Mean allocation around the Nov-5 2024 election (±4 days) — * = expected Trump-trade winners")
    ax.legend(); fig.tight_layout(); fig.savefig(FIG / "election_allocation.png", dpi=130); plt.close(fig)

    # --- Figure 3: headline timing prescience with bootstrap CIs ---
    rep = json.loads((RESULTS_DIR / "eval_ec2.json").read_text())
    cmp = rep["comparisons"]
    pts = {"T-in": cmp["T-in_vs_C-A"]["T-in_timing_prescience"],
           "C-A": cmp["T-in_vs_C-A"]["C-A_timing_prescience"],
           "C-B": cmp["T-in_vs_C-B"]["C-B_timing_prescience"]}
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, g in enumerate(["T-in", "C-A", "C-B"]):
        p = pts[g]
        ax.bar(i, p["point"], 0.6, color=colors[g])
        ax.errorbar(i, p["point"], yerr=[[p["point"] - p["lo"]], [p["hi"] - p["point"]]],
                    fmt="none", ecolor="k", capsize=5)
    ax.axhline(0, color="k", lw=0.6); ax.set_xticks(range(3)); ax.set_xticklabels(["T-in", "C-A", "C-B"])
    ax.set_ylabel("exposure-timing prescience (mean ± 95% bootstrap CI)")
    ax.set_title("Headline timing prescience: positive only for the treatment in-distribution")
    fig.tight_layout(); fig.savefig(FIG / "timing_prescience_ci.png", dpi=130); plt.close(fig)

    print("wrote ticker_prescience.png, election_allocation.png, timing_prescience_ci.png")
    print("\nper-ticker prescience:")
    print(pd.DataFrame(presc).round(3).to_string())


if __name__ == "__main__":
    _plot()
