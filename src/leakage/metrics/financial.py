"""Standard financial performance metrics from a daily-return series."""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def total_return(rets: pd.Series) -> float:
    return float((1.0 + rets).prod() - 1.0)


def annualized_return(rets: pd.Series) -> float:
    n = len(rets)
    if n == 0:
        return 0.0
    return float((1.0 + total_return(rets)) ** (TRADING_DAYS_PER_YEAR / n) - 1.0)


def annualized_vol(rets: pd.Series) -> float:
    return float(rets.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))


def sharpe(rets: pd.Series, rf_annual: float = 0.0) -> float:
    """Annualized Sharpe ratio (excess over a constant rf)."""
    if len(rets) < 2 or rets.std(ddof=1) == 0:
        return 0.0
    rf_daily = rf_annual / TRADING_DAYS_PER_YEAR
    excess = rets - rf_daily
    return float(excess.mean() / excess.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))


def max_drawdown(rets: pd.Series) -> float:
    """Most negative peak-to-trough drawdown (a negative number)."""
    if len(rets) == 0:
        return 0.0
    equity = (1.0 + rets).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def turnover(weights: pd.DataFrame) -> float:
    """Average daily one-way turnover (sum of |Δweight|), a cost/activity proxy."""
    if len(weights) < 2:
        return 0.0
    return float(weights.diff().abs().sum(axis=1).iloc[1:].mean())


def summary(rets: pd.Series, weights: pd.DataFrame | None = None) -> dict[str, float]:
    out = {
        "total_return": total_return(rets),
        "ann_return": annualized_return(rets),
        "ann_vol": annualized_vol(rets),
        "sharpe": sharpe(rets),
        "max_drawdown": max_drawdown(rets),
        "n_days": int(len(rets)),
    }
    if weights is not None:
        out["turnover"] = turnover(weights)
    return out
