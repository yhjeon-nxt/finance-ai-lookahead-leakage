"""Post-hoc analyses on the EC2 decision logs (no new model runs).

1. Pseudo-event null: gives the Aug-5 pre-event de-risk score a calibrated empirical p-value by
   comparing it to the distribution of de-risk scores at random pseudo-event dates in the window.
   (Addresses the verification finding that pre-event timing lacked a null distribution.)
2. Per-event exposure timeline figure: seed-averaged exposure for T-in vs C-A across 2024-H2 with
   the Aug-5 crash and Nov-5 election marked, plus the inter-seed min/max band (kills the
   "lucky seed" objection).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from leakage.config import RESULTS_DIR  # noqa: E402

_RUN = os.environ.get("LEAKAGE_RUN_TAG", "ec2")
_SFX = "" if _RUN == "ec2" else f"_{_RUN}"
DEC_DIR = RESULTS_DIR / _RUN / "decisions"
FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
AUG5 = pd.Timestamp("2024-08-05")
NOV5 = pd.Timestamp("2024-11-05")


def _exposure_panel(group: str) -> pd.DataFrame:
    """date x seed exposure panel for a group (parse_ok days only)."""
    cols = {}
    for f in sorted(DEC_DIR.glob(f"{group}_*_seed*.jsonl")):
        if "mock" in f.stem:   # exclude the no-memory baseline files (real runs only)
            continue
        seed = f.stem.split("seed")[-1]
        rows = {}
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            if not d.get("parse_ok", True):
                continue
            rows[pd.Timestamp(d["date"])] = float(sum((d.get("target_weights") or {}).values()))
        if rows:
            cols[f"seed{seed}"] = pd.Series(rows)
    return pd.DataFrame(cols).sort_index()


def _derisk_score(expo: pd.Series, d: pd.Timestamp, k: int = 20) -> float:
    """baseline(prior k days) − exposure held into day d. >0 ⇒ de-risked before d."""
    pre = expo[expo.index < d]
    if len(pre) < 2:
        return np.nan
    baseline = pre.iloc[-(k + 1):-1].mean()
    return float(baseline - pre.iloc[-1])


def pseudo_event_null(group: str, event=AUG5, n: int = 5000, k: int = 20, seed: int = 0) -> dict:
    panel = _exposure_panel(group)
    if panel.empty:
        return {"group": group, "p": float("nan")}
    expo = panel.mean(axis=1)  # seed-averaged
    obs = _derisk_score(expo, event, k)
    # candidate pseudo-event days: any day with >=k prior days and not adjacent to the real event
    cand = [d for d in expo.index if len(expo[expo.index < d]) > k and abs((d - event).days) > 5]
    rng = np.random.default_rng(seed)
    null = np.array([_derisk_score(expo, d, k) for d in cand])
    null = null[~np.isnan(null)]
    if len(null) < 10 or np.isnan(obs):
        return {"group": group, "observed_derisk": obs, "p": float("nan"), "n_null": len(null)}
    p = float((null >= obs).mean())  # one-sided: is the observed de-risk unusually large?
    return {"group": group, "observed_derisk": round(obs, 4),
            "null_mean": round(float(null.mean()), 4), "null_std": round(float(null.std()), 4),
            "p_value": round(p, 4), "n_null": int(len(null))}


def exposure_timeline_fig():
    import matplotlib.dates as mdates

    from leakage.run import figstyle as fs
    plt = fs.plt
    fs.use()

    fig, ax = plt.subplots(figsize=(12, 5.2))
    colors = {"T-in": fs.TREAT, "C-A": fs.CTRL}
    label = {"T-in": "T-in exposure (seed-avg)", "C-A": "C-A exposure (seed-avg)"}
    xmin, xmax = None, None
    for g, c in colors.items():
        panel = _exposure_panel(g)
        if panel.empty:
            continue
        mean = panel.mean(axis=1)
        ax.plot(mean.index, mean.values, color=c, lw=1.9, zorder=4, label=label[g])
        ax.fill_between(panel.index, panel.min(axis=1), panel.max(axis=1), color=c,
                        alpha=0.16, lw=0, zorder=2, label=f"{g} seed min–max")
        xmin = mean.index.min() if xmin is None else min(xmin, mean.index.min())
        xmax = mean.index.max() if xmax is None else max(xmax, mean.index.max())
    fs.ygrid(ax); fs.despine(ax)
    ax.set_ylim(0, 1.08)
    if xmin is not None:
        ax.set_xlim(xmin, xmax)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    fs.title(ax, "Portfolio exposure (risk dial) through 2024-H2",
             "treatment vs control · shaded band = inter-seed min–max (rules out a lucky seed)")
    ax.set_ylabel("total invested fraction")
    for d, lab in [(AUG5, "Aug-5 crash"), (NOV5, "Nov-5 election")]:
        fs.event_marker(ax, d, lab, y_frac=0.12, va="bottom", ha="left")
    fs.legend(ax, loc="lower right", ncol=2)
    fig.tight_layout()
    out = FIG_DIR / f"exposure_timeline_2024H2{_SFX}.png"
    fs.save(fig, out)
    return out


if __name__ == "__main__":
    print("=== Pseudo-event null for the Aug-5 de-risk ===")
    res = {g: pseudo_event_null(g) for g in ("T-in", "C-A")}
    for g, r in res.items():
        print(f"  {g}: {r}")
    fig = exposure_timeline_fig()
    print(f"=== wrote {fig} ===")
    (RESULTS_DIR / f"pseudo_event_null{_SFX}.json").write_text(json.dumps(res, indent=2))
