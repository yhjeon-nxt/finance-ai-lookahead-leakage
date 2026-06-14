"""Per-day *causal* context construction.

For each trading day T we build the information set the agent is allowed to see: trailing
price history and derived features for each ticker, using ONLY rows dated <= T. A hard
assertion enforces the causality guard so the experiment cannot accidentally leak future
prices through the feed (the only intended leakage channel is the model's parametric memory).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from leakage.config import TRAILING_DAYS, UNIVERSE, Window  # noqa: E402
from leakage.data.ingest import load_prices  # noqa: E402


@dataclass
class DayContext:
    date: pd.Timestamp
    tickers: list[str]
    last_close: dict[str, float]
    features: pd.DataFrame          # index=ticker, cols=ret_1d/ret_5d/ret_20d/vol_20d
    recent_returns: dict[str, list[float]]  # last ~15 daily returns per ticker
    max_date_in_context: pd.Timestamp       # for the causality assertion


def _close_frame(window: Window) -> pd.DataFrame:
    df = load_prices(window)
    close = df["Close"] if "Close" in df.columns.get_level_values(0) else df
    return close[UNIVERSE].sort_index()


def build_context(window: Window, day: pd.Timestamp,
                  trailing_days: int = TRAILING_DAYS,
                  recent: int = 15) -> DayContext:
    close = _close_frame(window)
    hist = close.loc[:day].tail(trailing_days)        # rows dated <= day only
    if hist.empty:
        raise ValueError(f"No price history up to {day}")

    # Causality guard. `loc[:day]` already excludes future rows, so `max <= day` is necessary
    # but tautological. The substantive guard is that the context must NOT reach the day whose
    # return realises the decision (day+1): if it did, the agent would see its own outcome.
    full = close.index
    pos = full.get_loc(full[full <= day][-1])
    realize_day = full[pos + 1] if pos + 1 < len(full) else None
    assert hist.index.max() <= day, (
        f"LEAKAGE: context for {day.date()} contains data dated {hist.index.max().date()}")
    assert realize_day is None or hist.index.max() < realize_day, (
        f"LEAKAGE: context for {day.date()} reaches the realization day {realize_day}")

    rets = hist.pct_change()
    feats = pd.DataFrame(index=UNIVERSE)
    feats["ret_1d"] = rets.iloc[-1]
    feats["ret_5d"] = hist.iloc[-1] / hist.iloc[-6] - 1 if len(hist) > 5 else np.nan
    feats["ret_20d"] = hist.iloc[-1] / hist.iloc[-21] - 1 if len(hist) > 20 else np.nan
    feats["vol_20d"] = rets.tail(20).std() * np.sqrt(252)

    recent_returns = {
        t: [round(float(x), 4) for x in rets[t].tail(recent).fillna(0).tolist()]
        for t in UNIVERSE
    }
    return DayContext(
        date=day,
        tickers=list(UNIVERSE),
        last_close={t: round(float(hist[t].iloc[-1]), 2) for t in UNIVERSE},
        features=feats.round(4),
        recent_returns=recent_returns,
        max_date_in_context=hist.index.max(),
    )


def render_context(ctx: DayContext, portfolio: dict[str, float], cash: float) -> str:
    """Compact text rendering for the LLM prompt (token-frugal for 8B models)."""
    lines = [
        f"DATE: {ctx.date.date()} ({ctx.date.day_name()}).",
        f"CASH: ${cash:,.0f}. CURRENT POSITIONS (market value): "
        + (", ".join(f"{k}=${v:,.0f}" for k, v in portfolio.items() if v) or "none") + ".",
        "",
        "TRAILING MARKET DATA (as of today's close; you may not see any later data):",
        "ticker  last_close   ret_1d   ret_5d  ret_20d  ann_vol_20d",
    ]
    for t in ctx.tickers:
        f = ctx.features.loc[t]
        lines.append(
            f"{t:<6}  {ctx.last_close[t]:>9.2f}  "
            f"{f.ret_1d:>+7.3f}  {f.ret_5d:>+7.3f}  {f.ret_20d:>+7.3f}  {f.vol_20d:>9.3f}"
        )
    lines.append("")
    lines.append("Recent daily returns (oldest→newest, last 15):")
    for t in ctx.tickers:
        lines.append(f"  {t}: {ctx.recent_returns[t]}")
    return "\n".join(lines)


if __name__ == "__main__":
    from leakage.config import IN_DIST
    from leakage.data.ingest import trading_days

    days = trading_days(IN_DIST)
    # Sample a day just before the Aug 5 2024 crash.
    target = next(d for d in days if d >= pd.Timestamp("2024-08-02"))
    ctx = build_context(IN_DIST, target)
    print(render_context(ctx, portfolio={t: 0.0 for t in UNIVERSE}, cash=1_000_000))
    print(f"\n[causality ok] max date in context = {ctx.max_date_in_context.date()} <= {target.date()}")
