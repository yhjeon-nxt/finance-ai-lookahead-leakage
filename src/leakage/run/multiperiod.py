"""Multi-period within-backbone leakage test (strengthens the T-in vs C-B comparison).

Holds ONE backbone fixed (qwen3:8b) and varies only the traded window between periods the model
demonstrably KNOWS (2024 H1+H2) and periods strictly AFTER its cutoff (2026 Q1+Q2). Pooling
several windows on each side averages over market regimes, so the in-dist−OOD foresight gap is
not driven by a single regime. A no-memory momentum baseline gives the difference-in-differences.

Run locally: PYTHONPATH=src python -m leakage.run.multiperiod
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from leakage.agent.llm_client import MockClient, OllamaClient  # noqa: E402
from leakage.backtest.engine import run_backtest  # noqa: E402
from leakage.config import RESULTS_DIR, SEEDS, UNIVERSE, Group, ModelSpec, Window  # noqa: E402
from leakage.data.ingest import download_prices  # noqa: E402
from leakage.metrics import financial, stats  # noqa: E402
from leakage.run.orchestrate import _market_next  # noqa: E402

BACKBONE = ModelSpec("qwen3:8b", "2024-aware (verified)", "treatment")

IN_DIST = [
    Window("2024H1", date(2024, 1, 1), date(2024, 6, 30), date(2023, 10, 1)),
    Window("2024H2", date(2024, 7, 1), date(2024, 12, 31), date(2024, 3, 1)),
]
OOD = [
    Window("2026Q1", date(2026, 1, 1), date(2026, 3, 31), date(2025, 10, 1)),
    Window("2026Q2", date(2026, 4, 1), date(2026, 6, 13), date(2026, 1, 1)),
]


def _client(mock: bool):
    return MockClient(name="mock-momentum") if mock else OllamaClient(BACKBONE.tag)


def _pooled_contrib(windows, mock: bool, seeds) -> tuple[pd.Series, list[dict]]:
    """Seed-averaged per-day timing contributions, concatenated across `windows`."""
    series, fin = [], []
    client = _client(mock)
    for w in windows:
        grp = Group(w.name, BACKBONE, w, f"{BACKBONE.tag} on {w.name}")
        per_seed = []
        rets = []
        for s in seeds:
            r = run_backtest(grp, client, s, temperature=0.7)
            c = stats.prescience_contrib(r.exposure, _market_next(r))
            if not c.empty:
                per_seed.append(c.reset_index(drop=True))
            rets.append(r.port_returns)
        if per_seed:
            series.append(pd.concat(per_seed, axis=1).mean(axis=1))  # seed-avg per day
        if not mock:
            allr = pd.concat(rets)
            fin.append({"window": w.name, **financial.summary(allr)})
    pooled = pd.concat(series, ignore_index=True) if series else pd.Series(dtype=float)
    return pooled, fin


def main():
    seeds = SEEDS
    for w in IN_DIST + OOD:
        download_prices(w)  # ensure cached

    print("[multiperiod] running qwen3:8b across in-dist (2024H1,H2) and OOD (2026Q1,Q2)...",
          flush=True)
    real_in, fin_in = _pooled_contrib(IN_DIST, mock=False, seeds=seeds)
    real_ood, fin_ood = _pooled_contrib(OOD, mock=False, seeds=seeds)
    print("[multiperiod] running no-memory momentum baseline for DiD...", flush=True)
    mock_in, _ = _pooled_contrib(IN_DIST, mock=True, seeds=seeds)
    mock_ood, _ = _pooled_contrib(OOD, mock=True, seeds=seeds)

    real_gap = float(real_in.mean() - real_ood.mean())
    mock_gap = float(mock_in.mean() - mock_ood.mean())
    did = real_gap - mock_gap
    ci_in = stats.block_bootstrap_mean_ci(real_in)
    ci_ood = stats.block_bootstrap_mean_ci(real_ood)
    perm = stats.permutation_diff(real_in, real_ood)

    report = {
        "design": "within-backbone multi-period (qwen3:8b)",
        "in_dist_windows": [w.name for w in IN_DIST],
        "ood_windows": [w.name for w in OOD],
        "n_in_dist_days": int(len(real_in)),
        "n_ood_days": int(len(real_ood)),
        "in_dist_timing_prescience": ci_in,
        "ood_timing_prescience": ci_ood,
        "real_gap": real_gap,
        "regime_baseline_gap": mock_gap,
        "DiD": did,
        "permutation_in_vs_ood": perm,
        "financial_in_dist": fin_in,
        "financial_ood": fin_ood,
    }
    out = RESULTS_DIR / "eval_multiperiod.json"
    out.write_text(json.dumps(report, indent=2, default=str))

    print("\n================ MULTI-PERIOD RESULT (qwen3:8b backbone) ================")
    print(f"in-dist days={report['n_in_dist_days']}  OOD days={report['n_ood_days']}")
    print(f"in-dist timing prescience: {ci_in['point']:+.4f}  CI[{ci_in['lo']:+.3f},{ci_in['hi']:+.3f}]")
    print(f"OOD     timing prescience: {ci_ood['point']:+.4f}  CI[{ci_ood['lo']:+.3f},{ci_ood['hi']:+.3f}]")
    print(f"raw gap (in-dist − OOD)  : {real_gap:+.4f}")
    print(f"regime baseline gap      : {mock_gap:+.4f}")
    print(f"DiD (leakage estimate)   : {did:+.4f}")
    print(f"permutation in vs OOD    : diff={perm['diff']:+.4f}  p={perm['p_value']:.3f}")
    print(f"[multiperiod] wrote {out}")


if __name__ == "__main__":
    main()
