"""Run all experiment groups and evaluate them.

Identical analysis path for the offline mock smoke test and the real ollama run — only the
client differs (``mock=True`` uses the no-foresight MockClient as a null reference).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from leakage.agent.llm_client import MockClient, OllamaClient  # noqa: E402
from leakage.backtest.engine import BacktestResult, run_backtest  # noqa: E402
from leakage.config import EVENT_ANCHORS, GROUPS, RESULTS_DIR, SEEDS, Group  # noqa: E402
from leakage.metrics import financial, leakage, stats  # noqa: E402

FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def make_client(group: Group, mock: bool, ollama_host: str | None = None):
    if mock:
        return MockClient(name=f"mock-{group.model.label}")
    return OllamaClient(group.model.tag, host=ollama_host)


def run_experiment(seeds=SEEDS, mock=False, max_days=None,
                   ollama_host: str | None = None) -> dict[tuple[str, int], BacktestResult]:
    results = {}
    for group in GROUPS:
        client = make_client(group, mock=mock, ollama_host=ollama_host)
        for seed in seeds:
            print(f"[run] group={group.name} model={group.model.tag} seed={seed} "
                  f"(mock={mock})", flush=True)
            res = run_backtest(group, client, seed, temperature=0.7, max_days=max_days)
            results[(group.name, seed)] = res
    return results


def _market_next(res: BacktestResult) -> pd.Series:
    return res.next_day_returns.mean(axis=1)


def _pooled_timing_contrib(results_for_group: list[BacktestResult]) -> pd.Series:
    """Concatenate per-day exposure-timing contributions across seeds for one group."""
    parts = []
    for r in results_for_group:
        c = stats.prescience_contrib(r.exposure, _market_next(r))
        if not c.empty:
            parts.append(c.reset_index(drop=True))
    return pd.concat(parts, ignore_index=True) if parts else pd.Series(dtype=float)


def evaluate(results: dict[tuple[str, int], BacktestResult], tag: str = "run") -> dict:
    groups = sorted({g for g, _ in results})
    report: dict = {"tag": tag, "groups": {}}

    # Per-group aggregates across seeds.
    fin_by_group, presc_by_group, contrib_by_group = {}, {}, {}
    for g in groups:
        rs = [results[(g, s)] for s in sorted({s for gg, s in results if gg == g})]
        fin = [financial.summary(r.port_returns, r.weights) for r in rs]
        fin_mean = {k: float(np.mean([f[k] for f in fin])) for k in fin[0]}
        presc = [leakage.next_day_prescience(r) for r in rs]
        presc_mean = {k: float(np.nanmean([p[k] for p in presc])) for k in presc[0]}
        pre_evt = [leakage.pre_event_timing(r, EVENT_ANCHORS) for r in rs]
        pre_evt_mean = ({k: float(np.nanmean([p.get(k, np.nan) for p in pre_evt]))
                         for k in pre_evt[0]} if pre_evt and pre_evt[0] else {})
        dodge = [leakage.event_day_dodge(r, EVENT_ANCHORS, _market_next(r)) for r in rs]
        dodge_mean = ({k: float(np.nanmean([d.get(k, np.nan) for d in dodge]))
                       for k in dodge[0]} if dodge and dodge[0] else {})
        forensic = leakage.rationale_forensics(rs[0])  # representative seed
        n_parse_fail = int(np.sum([r.n_parse_fail for r in rs]))

        fin_by_group[g] = fin_mean
        presc_by_group[g] = presc_mean
        contrib_by_group[g] = _pooled_timing_contrib(rs)
        report["groups"][g] = {
            "model": rs[0].model,
            "financial": fin_mean,
            "prescience": presc_mean,
            "pre_event_timing": pre_evt_mean,
            "event_day_dodge": dodge_mean,
            "rationale_forensics": {k: v for k, v in forensic.items() if k != "excerpts"},
            "forensic_excerpts": forensic["excerpts"],
            "n_parse_fail": n_parse_fail,
        }

    # Headline comparisons (timing prescience): T-in vs C-A, T-in vs C-B.
    report["comparisons"] = {}
    for a, b in [("T-in", "C-A"), ("T-in", "C-B")]:
        if a in contrib_by_group and b in contrib_by_group:
            ci_a = stats.block_bootstrap_mean_ci(contrib_by_group[a])
            ci_b = stats.block_bootstrap_mean_ci(contrib_by_group[b])
            perm = stats.permutation_diff(contrib_by_group[a], contrib_by_group[b])
            report["comparisons"][f"{a}_vs_{b}"] = {
                f"{a}_timing_prescience": ci_a,
                f"{b}_timing_prescience": ci_b,
                "permutation_diff": perm,
            }

    # Foresight gap (same model, in-dist vs OOD).
    if "T-in" in presc_by_group and "C-B" in presc_by_group:
        report["foresight_gap_Tin_minus_CB"] = leakage.foresight_gap(
            presc_by_group["T-in"], presc_by_group["C-B"])

    _plot_equity(results, groups, tag)
    out = RESULTS_DIR / f"eval_{tag}.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"[eval] wrote {out}")
    return report


def _plot_equity(results, groups, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    for g in groups:
        rs = [results[(g, s)] for s in sorted({s for gg, s in results if gg == g})]
        # mean equity curve across seeds (align on index)
        eq = pd.concat([r.equity.reset_index(drop=True) for r in rs], axis=1).mean(axis=1)
        ax.plot(eq.values, label=f"{g} ({rs[0].model})")
    ax.axhline(1.0, color="k", lw=0.5, ls="--")
    ax.set_title(f"Equity curves — {tag}")
    ax.set_xlabel("trading day in window")
    ax.set_ylabel("equity (start=1.0)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"equity_{tag}.png", dpi=130)
    plt.close(fig)
