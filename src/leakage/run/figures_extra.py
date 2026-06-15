"""Extra result figures from the EC2 decision logs (no GPU, no new runs).

  1. per-ticker prescience: corr(today's weight, tomorrow's return) per symbol × group.
  2. election-window allocation: mean weight per symbol around Nov-5 2024, T-in vs C-A.
  3. headline timing-prescience per group with bootstrap CIs (from eval_ec2.json).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from leakage.config import IN_DIST, OOD, RESULTS_DIR, UNIVERSE, Window  # noqa: E402
from leakage.data.ingest import load_prices, trading_days  # noqa: E402
from leakage.metrics import stats  # noqa: E402

# Run-config via env so the same script serves the qwen3 (ec2) and gemma runs. Defaults preserve
# the original ec2 behaviour and filenames (so existing report figures are untouched).
_RUN = os.environ.get("LEAKAGE_RUN_TAG", "ec2")
_TREAT = os.environ.get("LEAKAGE_TREATMENT_MODEL", "qwen3:8b")
_CTRL = os.environ.get("LEAKAGE_CONTROL_MODEL", "llama3.1:8b")
_SFX = "" if _RUN == "ec2" else f"_{_RUN}"  # suffix only for non-ec2 runs

DEC = RESULTS_DIR / _RUN / "decisions"
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


def _window_weights_raw(group: str, model_tag: str, window: Window,
                        lo: pd.Timestamp, hi: pd.Timestamp) -> pd.DataFrame:
    """All per-(seed, day) weight rows inside [lo, hi] (NOT seed-averaged), for permutation tests."""
    safe = model_tag.replace(":", "-")
    rows = []
    for f in sorted(DEC.glob(f"{group}_{safe}_seed*.jsonl")):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            ts = pd.Timestamp(d["date"])
            if d.get("parse_ok", True) and lo <= ts <= hi:
                w = d.get("target_weights") or {}
                rows.append({t: float(w.get(t, 0.0)) for t in UNIVERSE})
    return pd.DataFrame(rows, columns=UNIVERSE)


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
    from leakage.run import figstyle as fs
    plt = fs.plt
    fs.use()

    groups = [("T-in", _TREAT, IN_DIST), ("C-A", _CTRL, IN_DIST),
              ("C-B", _TREAT, OOD)]
    colors = fs.GROUP_COLORS
    glabel = {"T-in": f"T-in  ({_TREAT})", "C-A": f"C-A  ({_CTRL})", "C-B": f"C-B  ({_TREAT}, OOD)"}

    # --- Figure 1: per-ticker prescience ---
    presc = {g: _ticker_prescience(g, tag, w) for g, tag, w in groups}
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    x = np.arange(len(UNIVERSE)); bw = 0.26
    for i, g in enumerate(["T-in", "C-A", "C-B"]):
        ax.bar(x + (i - 1) * bw, [presc[g][t] for t in UNIVERSE], bw, label=glabel[g],
               color=colors[g], edgecolor="white", lw=0.6, zorder=3)
    fs.ygrid(ax); fs.zero_line(ax); fs.despine(ax)
    ax.set_xticks(x); ax.set_xticklabels(UNIVERSE)
    ax.set_ylabel("corr( today's weight,  next-day return )")
    fs.title(ax, "Per-ticker next-day prescience",
             "only the treatment (T-in) is consistently positive across tickers")
    fs.legend(ax, loc="upper left", ncol=3)
    fig.tight_layout(); fs.save(fig, FIG / f"ticker_prescience{_SFX}.png")

    # --- Figure 2: election-window mean allocation (T-in vs C-A) + per-ticker permutation p ---
    lo, hi = NOV5 - pd.Timedelta(days=7), NOV5 + pd.Timedelta(days=7)  # ~±7d for a few more obs
    raw_t = _window_weights_raw("T-in", _TREAT, IN_DIST, lo, hi)
    raw_c = _window_weights_raw("C-A", _CTRL, IN_DIST, lo, hi)
    pvals, diffs = {}, {}
    for t in UNIVERSE:
        r = stats.permutation_diff(raw_t[t], raw_c[t], n_perm=20000)
        pvals[t], diffs[t] = r["p_value"], r["diff"]

    fig, ax = plt.subplots(figsize=(12, 5.6))
    ax.bar(x - 0.2, [raw_t[t].mean() for t in UNIVERSE], 0.4, label=glabel["T-in"],
           color=colors["T-in"], edgecolor="white", lw=0.6, zorder=3)
    ax.bar(x + 0.2, [raw_c[t].mean() for t in UNIVERSE], 0.4, label=glabel["C-A"],
           color=colors["C-A"], edgecolor="white", lw=0.6, zorder=3)
    fs.ygrid(ax); fs.despine(ax)
    ymax = max([raw_t[t].mean() for t in UNIVERSE] + [raw_c[t].mean() for t in UNIVERSE])
    ax.set_ylim(0, ymax * 1.42)
    for j, t in enumerate(UNIVERSE):
        top = max(raw_t[t].mean(), raw_c[t].mean())
        p = pvals[t]
        ax.text(j, top + ymax * 0.04, f"Δ={diffs[t]:+.3f}\np={p:.3f} {fs.sig_star(p)}",
                ha="center", va="bottom", fontsize=8.5, color=fs.sig_color(p))
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t} *" if t in TRUMP_TRADE else t for t in UNIVERSE])
    ax.set_ylabel("mean target weight")
    fs.title(ax, "Allocation around the Nov-5 2024 election  (±7 trading days)",
             "* = expected Trump-trade winners · per-ticker permutation p (T-in − C-A) · "
             "***p<.01  **p<.05  *p<.1")
    fs.legend(ax, loc="upper right")
    fig.tight_layout()
    fs.save(fig, FIG / f"election_allocation{_SFX}.png")
    print("\nelection-window per-ticker T-in−C-A diff and permutation p:")
    for t in UNIVERSE:
        print(f"  {t:5} Δ={diffs[t]:+.3f}  p={pvals[t]:.3f}  (winner={t in TRUMP_TRADE})")

    # --- Figure 3: headline timing prescience with bootstrap CIs ---
    rep = json.loads((RESULTS_DIR / f"eval_{_RUN}.json").read_text())
    cmp = rep["comparisons"]
    pts = {"T-in": cmp["T-in_vs_C-A"]["T-in_timing_prescience"],
           "C-A": cmp["T-in_vs_C-A"]["C-A_timing_prescience"],
           "C-B": cmp["T-in_vs_C-B"]["C-B_timing_prescience"]}
    p_ac = cmp["T-in_vs_C-A"]["permutation_diff"]["p_value"]
    p_ab = cmp["T-in_vs_C-B"]["permutation_diff"]["p_value"]
    fig, ax = plt.subplots(figsize=(7.8, 5.6))
    glab3 = {"T-in": glabel["T-in"], "C-A": glabel["C-A"], "C-B": glabel["C-B"]}
    for i, g in enumerate(["T-in", "C-A", "C-B"]):
        p = pts[g]
        ax.bar(i, p["point"], 0.62, color=colors[g], edgecolor="white", lw=0.6, zorder=3, label=glab3[g])
        ax.errorbar(i, p["point"], yerr=[[p["point"] - p["lo"]], [p["hi"] - p["point"]]],
                    fmt="none", ecolor=fs.INK, capsize=5, lw=1.3, zorder=4)
    fs.ygrid(ax); fs.zero_line(ax); fs.despine(ax)
    ax.set_xticks(range(3)); ax.set_xticklabels(["T-in", "C-A", "C-B"])
    ax.set_ylabel("exposure-timing prescience  (mean ± 95% bootstrap CI)")

    top = max(pts[g]["hi"] for g in ("T-in", "C-A", "C-B"))
    fs.bracket(ax, 0, 1, top + 0.02, p_ac, h=0.013, label=f"perm p = {p_ac:.3f}  {fs.sig_star(p_ac)}")
    fs.bracket(ax, 0, 2, top + 0.075, p_ab, h=0.013, label=f"perm p = {p_ab:.3f}  {fs.sig_star(p_ab)}")
    ax.set_ylim(min(pts[g]["lo"] for g in pts) - 0.03, top + 0.15)
    fs.title(ax, "Headline timing prescience + pairwise permutation tests",
             "positive only for the treatment in-distribution")
    fig.tight_layout(); fs.save(fig, FIG / f"timing_prescience_ci{_SFX}.png")
    print(f"\npairwise permutation p — T-in vs C-A: {p_ac:.3f}   T-in vs C-B: {p_ab:.3f}")

    print("wrote ticker_prescience.png, election_allocation.png, timing_prescience_ci.png")
    print("\nper-ticker prescience:")
    print(pd.DataFrame(presc).round(3).to_string())


if __name__ == "__main__":
    _plot()
