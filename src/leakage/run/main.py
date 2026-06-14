"""Real-run entrypoint (local Mac or EC2).

Order: empirical cutoff probe → backtests for all groups/seeds (real ollama models) →
evaluation + figures → optional S3 upload. The decision cache makes the whole thing resumable,
so re-invoking after an interruption continues where it stopped.

Examples:
    python -m leakage.run.main --max-days 5 --no-s3       # quick local validation
    python -m leakage.run.main                            # full run (3 groups x 3 seeds)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from leakage.config import SEEDS  # noqa: E402
from leakage.run.orchestrate import evaluate, run_experiment  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--max-days", type=int, default=None,
                    help="limit trading days per window (smoke/validation)")
    ap.add_argument("--tag", default="real")
    ap.add_argument("--ollama-host", default=None)
    ap.add_argument("--skip-probe", action="store_true")
    ap.add_argument("--no-s3", action="store_true")
    args = ap.parse_args()

    if not args.skip_probe:
        from leakage.run.cutoff_probe import run as run_probe
        run_probe()

    results = run_experiment(seeds=args.seeds, mock=False, max_days=args.max_days,
                             ollama_host=args.ollama_host)
    report = evaluate(results, tag=args.tag)

    print("\n================ RESULTS ================")
    for g, blk in report["groups"].items():
        fin, presc = blk["financial"], blk["prescience"]
        print(f"[{g}] model={blk['model']} parse_fail={blk['n_parse_fail']} "
              f"ret={fin['total_return']:+.3f} sharpe={fin['sharpe']:+.2f} "
              f"ticker_presc={presc['ticker_prescience']:+.3f} "
              f"expo_timing={presc['exposure_timing']:+.3f}")
    for name, c in report.get("comparisons", {}).items():
        print(f"  {name}: permutation p={c['permutation_diff']['p_value']:.3f} "
              f"diff={c['permutation_diff']['diff']:+.4f}")

    if not args.no_s3:
        from leakage.config import RESULTS_DIR
        from leakage.run.s3_sync import upload_dir
        upload_dir(RESULTS_DIR, subprefix=f"results/{args.tag}")


if __name__ == "__main__":
    main()
