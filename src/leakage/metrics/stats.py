"""Statistical tests: block-bootstrap CIs and permutation tests.

Prescience is expressed as a per-day **contribution** ``z(signal_t) * z(return_{t+1})`` whose
mean approximates the Pearson correlation. Because it is a sum over days, it supports honest
block bootstrap (preserving autocorrelation) and label-permutation tests for group differences.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else np.zeros_like(x)


def prescience_contrib(signal: pd.Series, target: pd.Series) -> pd.Series:
    """Per-day standardized product; its mean ≈ corr(signal, target)."""
    s, t = signal.align(target, join="inner")
    if len(s) < 3:
        return pd.Series(dtype=float)
    return pd.Series(zscore(s.values) * zscore(t.values), index=s.index)


def _block_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Stationary/circular block bootstrap row indices covering ~n samples."""
    idx = []
    while len(idx) < n:
        start = int(rng.integers(0, n))
        idx.extend([(start + j) % n for j in range(block)])
    return np.array(idx[:n])


def block_bootstrap_mean_ci(series: pd.Series, n_boot: int = 3000, block: int = 5,
                            ci: float = 0.95, seed: int = 0) -> dict[str, float]:
    """Point estimate, bootstrap CI, and one-sided p-value for mean(series) > 0."""
    x = series.dropna().values
    n = len(x)
    if n < 3:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "p_gt_0": float("nan"), "n": n}
    rng = np.random.default_rng(seed)
    boots = np.array([x[_block_indices(n, block, rng)].mean() for _ in range(n_boot)])
    alpha = (1 - ci) / 2
    return {
        "point": float(x.mean()),
        "lo": float(np.quantile(boots, alpha)),
        "hi": float(np.quantile(boots, 1 - alpha)),
        "p_gt_0": float((boots <= 0).mean()),       # bootstrap evidence the mean is > 0
        "n": n,
    }


def permutation_diff(a: pd.Series, b: pd.Series, n_perm: int = 5000,
                     seed: int = 0) -> dict[str, float]:
    """Two-sided permutation test for mean(a) - mean(b) via label shuffling."""
    a = a.dropna().values
    b = b.dropna().values
    if len(a) < 3 or len(b) < 3:
        return {"diff": float("nan"), "p_value": float("nan")}
    obs = a.mean() - b.mean()
    pooled = np.concatenate([a, b])
    na = len(a)
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        d = pooled[:na].mean() - pooled[na:].mean()
        if abs(d) >= abs(obs):
            count += 1
    return {"diff": float(obs), "p_value": float((count + 1) / (n_perm + 1))}


def stationary_bootstrap_sharpe(rets: pd.Series, n_boot: int = 3000, block: int = 5,
                                ci: float = 0.95, seed: int = 0) -> dict[str, float]:
    from leakage.metrics.financial import sharpe

    x = rets.dropna()
    n = len(x)
    if n < 3:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": n}
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = _block_indices(n, block, rng)
        boots.append(sharpe(pd.Series(x.values[idx])))
    alpha = (1 - ci) / 2
    return {
        "point": float(sharpe(x)),
        "lo": float(np.quantile(boots, alpha)),
        "hi": float(np.quantile(boots, 1 - alpha)),
        "n": n,
    }
