"""Honest equity figure: reconstruct equity-by-CALENDAR-DATE from the EC2 decision logs.

The original overlay used an ordinal "trading day in window" x-axis, which silently put T-in
(2024-H2) and C-B (2026) on the same axis despite being different calendar dates and lengths.
This redraw fixes that:
  * Left panel  — T-in vs C-A on a SHARED 2024-H2 date axis (the only truly like-for-like
    comparison: same dates, same window), with the Aug-5 crash and Nov-5 election marked.
  * Right panel — C-B (same model, 2026) on its own date axis (the within-model time control).
Equity is recomputed deterministically from the logged target weights and realised prices.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from leakage.config import IN_DIST, OOD, RESULTS_DIR, UNIVERSE, Window  # noqa: E402
from leakage.data.ingest import load_prices, trading_days  # noqa: E402

DEC = RESULTS_DIR / "ec2" / "decisions"
FIG = RESULTS_DIR / "figures"
AUG5, NOV5 = pd.Timestamp("2024-08-05"), pd.Timestamp("2024-11-05")


def _equity_by_date(group: str, model_tag: str, window: Window) -> pd.Series:
    close = load_prices(window)
    close = (close["Close"] if "Close" in close.columns.get_level_values(0) else close)[UNIVERSE]
    rets = close.pct_change()
    days = trading_days(window)
    safe = model_tag.replace(":", "-")
    curves = []
    for f in sorted(DEC.glob(f"{group}_{safe}_seed*.jsonl")):
        w = {}
        for line in f.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                w[pd.Timestamp(d["date"])] = d.get("target_weights") or {}
        eq, val, idx = [], 1.0, []
        for i, day in enumerate(days[:-1]):
            nd = days[i + 1]
            wd = w.get(day, {})
            r = float(sum(wd.get(t, 0.0) * float(rets.loc[nd, t]) for t in UNIVERSE))
            val *= (1 + r); eq.append(val); idx.append(nd)
        if eq:
            curves.append(pd.Series(eq, index=idx))
    if not curves:
        return pd.Series(dtype=float)
    return pd.concat(curves, axis=1).mean(axis=1)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t_in = _equity_by_date("T-in", "qwen3:8b", IN_DIST)
    c_a = _equity_by_date("C-A", "llama3.1:8b", IN_DIST)
    c_b = _equity_by_date("C-B", "qwen3:8b", OOD)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.3, 1]})
    axL.plot(t_in.index, t_in.values, color="tab:green", label="T-in qwen3:8b (KNOWS 2024-H2)")
    axL.plot(c_a.index, c_a.values, color="tab:blue", label="C-A llama3.1:8b (control, same dates)")
    for d, lab in [(AUG5, "Aug-5 crash"), (NOV5, "Nov-5 election")]:
        axL.axvline(d, color="red", ls="--", lw=1); axL.text(d, axL.get_ylim()[1], lab,
                                                             rotation=90, va="top", fontsize=8, color="red")
    axL.axhline(1.0, color="k", lw=0.5, ls=":")
    axL.set_title("In-distribution (2024-H2) — like-for-like, same calendar dates")
    axL.set_xlabel("date (2024-H2)"); axL.set_ylabel("equity (start=1.0)"); axL.legend(loc="upper left")
    axL.tick_params(axis="x", rotation=30)

    axR.plot(c_b.index, c_b.values, color="tab:orange", label="C-B qwen3:8b (same model, OOD)")
    axR.axhline(1.0, color="k", lw=0.5, ls=":")
    axR.set_title("Out-of-distribution time control (2026)")
    axR.set_xlabel("date (2026)"); axR.legend(loc="upper left"); axR.tick_params(axis="x", rotation=30)

    fig.suptitle("Equity by calendar date — treatment separates only where it has memory", y=1.02)
    fig.tight_layout()
    out = FIG / "equity_ec2_bydate.png"
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}")
    print(f"final equity — T-in={t_in.iloc[-1]:.3f}  C-A={c_a.iloc[-1]:.3f}  C-B={c_b.iloc[-1]:.3f}")


if __name__ == "__main__":
    main()
