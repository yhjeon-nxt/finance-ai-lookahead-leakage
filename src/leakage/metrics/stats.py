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
    """Circular fixed-length block bootstrap row indices covering ~n samples.

    (A true *stationary* bootstrap would draw geometric-length blocks; we use fixed-length
    circular blocks, which still preserve short-range autocorrelation. Inference is corroborated
    by the permutation test, which makes no block-length assumption.)
    """
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


def block_bootstrap_did(in_real: pd.Series, out_real: pd.Series,
                        in_mock: pd.Series, out_mock: pd.Series,
                        n_boot: int = 5000, block: int = 5, ci: float = 0.95,
                        seed: int = 0) -> dict:
    """Block-bootstrap test of the difference-in-differences leakage statistic:

        DiD = [mean(in_real) - mean(out_real)] - [mean(in_mock) - mean(out_mock)]

    Resamples each per-day contribution series in CIRCULAR BLOCKS (preserving short-range
    autocorrelation, unlike individual-day permutation) and reports the DiD point estimate, a
    bootstrap CI, and a one-sided p-value (bootstrap evidence that DiD > 0). This tests the
    actual regime-adjusted leakage quantity, not the regime-confounded raw in-vs-out gap.
    """
    arrs = [s.dropna().values for s in (in_real, out_real, in_mock, out_mock)]
    if any(len(a) < 3 for a in arrs):
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "p_gt_0": float("nan")}

    def _did(samples):
        return ((samples[0].mean() - samples[1].mean())
                - (samples[2].mean() - samples[3].mean()))

    point = _did(arrs)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        rs = [a[_block_indices(len(a), block, rng)] for a in arrs]
        boots[b] = _did(rs)
    alpha = (1 - ci) / 2
    return {
        "point": float(point),
        "lo": float(np.quantile(boots, alpha)),
        "hi": float(np.quantile(boots, 1 - alpha)),
        "p_gt_0": float((boots <= 0).mean()),   # one-sided evidence DiD>0 (autocorr-robust)
    }


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
