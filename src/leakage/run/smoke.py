"""Offline end-to-end smoke test with the no-foresight MockClient.

Validates the full pipeline (data → agent → backtest → metrics → stats → figure) without any
LLM. Because the mock has no look-ahead, all prescience/foresight metrics should be ~0 and the
T-in vs C-A/C-B permutation tests should be non-significant — confirming the metrics do not
manufacture false leakage signals.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from leakage.run.orchestrate import evaluate, run_experiment  # noqa: E402


def main():
    results = run_experiment(seeds=[0, 1], mock=True)
    report = evaluate(results, tag="smoke")

    print("\n================ SMOKE SUMMARY (MockClient, no foresight) ================")
    for g, blk in report["groups"].items():
        fin, presc = blk["financial"], blk["prescience"]
        print(f"\n[{g}] model={blk['model']}  parse_fail={blk['n_parse_fail']}")
        print(f"   return={fin['total_return']:+.3f}  sharpe={fin['sharpe']:+.2f}  "
              f"maxDD={fin['max_drawdown']:+.3f}  turnover={fin.get('turnover', 0):.3f}")
        print(f"   ticker_prescience={presc['ticker_prescience']:+.3f}  "
              f"exposure_timing={presc['exposure_timing']:+.3f}")
        if blk["pre_event_timing"]:
            print(f"   pre_event_timing={blk['pre_event_timing']}")
    print("\n---- headline comparisons ----")
    for name, c in report["comparisons"].items():
        perm = c["permutation_diff"]
        print(f"   {name}: diff={perm['diff']:+.4f}  p={perm['p_value']:.3f}")
    if "foresight_gap_Tin_minus_CB" in report:
        print(f"\n   foresight_gap (T-in − C-B): {report['foresight_gap_Tin_minus_CB']}")
    print("\nExpect: prescience ≈ 0 and p-values non-significant (mock has no look-ahead).")


if __name__ == "__main__":
    main()
