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
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from leakage.config import IN_DIST, OOD, RESULTS_DIR, UNIVERSE, Window  # noqa: E402
from leakage.data.ingest import load_prices, trading_days  # noqa: E402

_RUN = os.environ.get("LEAKAGE_RUN_TAG", "ec2")
_TREAT = os.environ.get("LEAKAGE_TREATMENT_MODEL", "qwen3:8b")
_CTRL = os.environ.get("LEAKAGE_CONTROL_MODEL", "llama3.1:8b")
DEC = RESULTS_DIR / _RUN / "decisions"
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
    import matplotlib.dates as mdates

    from leakage.run import figstyle as fs
    plt = fs.plt
    fs.use()

    t_in = _equity_by_date("T-in", _TREAT, IN_DIST)
    c_a = _equity_by_date("C-A", _CTRL, IN_DIST)
    c_b = _equity_by_date("C-B", _TREAT, OOD)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.4), gridspec_kw={"width_ratios": [1.35, 1]})
    axL.plot(t_in.index, t_in.values, color=fs.TREAT, lw=2.0, zorder=4,
             label=f"T-in  {_TREAT}  (knows 2024-H2)")
    axL.plot(c_a.index, c_a.values, color=fs.CTRL, lw=1.8, zorder=3,
             label=f"C-A  {_CTRL}  (control, same dates)")
    axL.axhline(1.0, color=fs.FAINT, lw=1.0, ls=":", zorder=1)
    fs.ygrid(axL); fs.despine(axL)
    axL.set_xlim(t_in.index.min(), t_in.index.max())
    fs.title(axL, "In-distribution (2024-H2)", "same calendar dates — the only like-for-like overlay")
    axL.set_xlabel("date"); axL.set_ylabel("equity  (start = 1.0)")
    axL.legend(loc="lower right")
    for d, lab in [(AUG5, "Aug-5 crash"), (NOV5, "Nov-5 election")]:
        fs.event_marker(axL, d, lab, y_frac=0.96, va="top", ha="left")

    axR.plot(c_b.index, c_b.values, color=fs.OOD, lw=2.0, zorder=4,
             label=f"C-B  {_TREAT}  (same model, OOD)")
    axR.axhline(1.0, color=fs.FAINT, lw=1.0, ls=":", zorder=1)
    fs.ygrid(axR); fs.despine(axR)
    axR.set_xlim(c_b.index.min(), c_b.index.max())
    fs.title(axR, "Out-of-distribution (2026)", "within-model time control — no memory of these dates")
    axR.set_xlabel("date"); axR.legend(loc="upper left")

    for ax in (axL, axR):
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.tick_params(axis="x", rotation=0)

    # The headline claim ("separates only where it has memory") is the qwen3 finding; for the
    # Gemma run the treatment does NOT separate (it confabulates), so keep the suptitle neutral.
    suptitle = {"ec2": "Equity by calendar date — the treatment separates only where it has memory"}.get(
        _RUN, f"Equity by calendar date — {_TREAT} in-distribution vs out-of-distribution time control")
    fig.suptitle(suptitle, y=1.00, fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = FIG / f"equity_{_RUN}_bydate.png"
    fs.save(fig, out)
    print(f"wrote {out}")
    print(f"final equity — T-in={t_in.iloc[-1]:.3f}  C-A={c_a.iloc[-1]:.3f}  C-B={c_b.iloc[-1]:.3f}")


if __name__ == "__main__":
    main()
