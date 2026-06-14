"""Render the Empirical Results (§4) markdown from an eval_<tag>.json file.

Usage: python -m leakage.run.report_tables ec2  > report/section4_results.md
Keeps results reporting deterministic and copy-pasteable into the main report.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from leakage.config import RESULTS_DIR  # noqa: E402

ORDER = ["T-in", "C-A", "C-B"]


def _f(x, nd=3):
    try:
        return f"{float(x):+.{nd}f}"
    except (TypeError, ValueError):
        return "n/a"


def render(tag: str) -> str:
    rep = json.loads((RESULTS_DIR / f"eval_{tag}.json").read_text())
    g = rep["groups"]
    groups = [x for x in ORDER if x in g] + [x for x in g if x not in ORDER]
    L = []

    L.append("### 4.1 Financial performance\n")
    L.append("| Group | Model | Total return | Sharpe | Max DD | Turnover | Parse-fail |")
    L.append("|---|---|---|---|---|---|---|")
    for k in groups:
        f = g[k]["financial"]
        L.append(f"| {k} | `{g[k]['model']}` | {_f(f['total_return'])} | "
                 f"{_f(f['sharpe'],2)} | {_f(f['max_drawdown'])} | "
                 f"{_f(f.get('turnover',0))} | {g[k]['n_parse_fail']} |")

    L.append("\n### 4.2 Leakage / foresight metrics\n")
    L.append("| Group | Ticker prescience | Exposure timing | Conf-wtd timing |")
    L.append("|---|---|---|---|")
    for k in groups:
        p = g[k]["prescience"]
        L.append(f"| {k} | {_f(p['ticker_prescience'])} | {_f(p['exposure_timing'])} | "
                 f"{_f(p['conf_weighted_timing'])} |")

    L.append("\n### 4.3 Pre-event timing (in-distribution groups)\n")
    L.append("| Group | Aug-5 crash (de-risk>0) | Nov-5 election (load>0) | mean |")
    L.append("|---|---|---|---|")
    for k in groups:
        pe = g[k].get("pre_event_timing") or {}
        if pe:
            L.append(f"| {k} | {_f(pe.get('yen_carry_unwind_crash'))} | "
                     f"{_f(pe.get('us_election_trump_trade'))} | {_f(pe.get('event_timing_mean'))} |")

    if "comparisons" in rep:
        L.append("\n### 4.4 Headline statistical tests\n")
        L.append("| Comparison | Δ (timing prescience) | permutation p |")
        L.append("|---|---|---|")
        for name, c in rep["comparisons"].items():
            perm = c["permutation_diff"]
            L.append(f"| {name.replace('_',' ')} | {_f(perm['diff'],4)} | {perm['p_value']:.3f} |")

    if "foresight_gap_Tin_minus_CB" in rep:
        L.append("\n### 4.5 Within-model foresight gap + regime-adjusted DiD\n")
        L.append("Leakage is supported only if the LLM in-dist−OOD gap EXCEEDS the no-memory "
                 "momentum baseline's gap (DiD > 0); a raw gap alone can arise from the "
                 "2024-H2-vs-2026 regime difference.\n")
        real = rep["foresight_gap_Tin_minus_CB"]
        base = rep.get("regime_baseline_gap_Tin_minus_CB", {})
        did = rep.get("foresight_gap_DiD", {})
        L.append("| Metric | LLM gap | Regime baseline gap | **DiD (leakage)** |")
        L.append("|---|---|---|---|")
        for k in real:
            L.append(f"| {k} | {_f(real[k])} | {_f(base.get(k))} | **{_f(did.get(k))}** |")

    L.append("\n### 4.6 Rationale forensics (smoking-gun scan)\n")
    for k in groups:
        rf = g[k]["rationale_forensics"]
        L.append(f"**{k}** — hard-tell decisions: {rf.get('n_hard_decisions',0)}; "
                 f"hard tells: {rf.get('hard_tell_counts',{})}")
        for ex in (g[k].get("forensic_excerpts") or [])[:3]:
            L.append(f"  - `{ex['date']}` tells={ex['hard_tells']} months={ex['future_months']}: "
                     f"\"{ex['text'][:200]}\"")
    return "\n".join(L)


if __name__ == "__main__":
    print(render(sys.argv[1] if len(sys.argv) > 1 else "ec2"))
