"""Cross-family comparison: qwen3:8b (ec2) vs gemma3:12b (gemma) treatment runs.

Reads the two eval JSONs and emits one comparison table + grouped bar chart answering the
headline question: does the leakage signature reproduce on an independent model family?
Run after the Gemma results land:  PYTHONPATH=src python -m leakage.run.compare_runs
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from leakage.config import RESULTS_DIR  # noqa: E402

RUNS = [("qwen3:8b", "ec2"), ("gemma3:12b", "gemma")]


def _load(tag: str):
    for p in (RESULTS_DIR / f"eval_{tag}.json", RESULTS_DIR / tag / f"eval_{tag}.json"):
        if p.exists():
            return json.loads(p.read_text())
    return None


def _row(rep) -> dict:
    g = rep["groups"]
    cmp = rep.get("comparisons", {})
    did = rep.get("foresight_gap_DiD", {})
    return {
        "T-in Sharpe": g["T-in"]["financial"]["sharpe"],
        "T-in total return": g["T-in"]["financial"]["total_return"],
        "C-B Sharpe (OOD)": g["C-B"]["financial"]["sharpe"],
        "T-in ticker prescience": g["T-in"]["prescience"]["ticker_prescience"],
        "T-in exposure timing": g["T-in"]["prescience"]["exposure_timing"],
        "T-in vs C-A perm p": cmp.get("T-in_vs_C-A", {}).get("permutation_diff", {}).get("p_value"),
        "DiD exposure timing": did.get("gap_exposure_timing"),
    }


def main():
    data = {f"{m} ({tag})": _load(tag) for m, tag in RUNS}
    missing = [k for k, v in data.items() if v is None]
    if missing:
        print(f"[compare] not ready — missing eval for: {missing}")
        return
    rows = {k: _row(v) for k, v in data.items()}
    metrics = list(next(iter(rows.values())).keys())

    lines = ["# Cross-family comparison — does the leakage signature replicate?\n",
             "| Metric | " + " | ".join(rows.keys()) + " |",
             "|---|" + "|".join(["---"] * len(rows)) + "|"]
    for m in metrics:
        cells = []
        for k in rows:
            v = rows[k][m]
            cells.append("n/a" if v is None else (f"{v:.3f}" if abs(v) < 100 else f"{v:.1f}"))
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    out_md = RESULTS_DIR.parent / "report" / "cross_family_comparison.md"
    out_md.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\n[compare] wrote {out_md}")

    # grouped bar of the leakage-relevant metrics
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    show = ["T-in Sharpe", "T-in exposure timing", "DiD exposure timing"]
    labels = list(rows.keys())
    x = np.arange(len(show)); bw = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, k in enumerate(labels):
        ax.bar(x + (i - 0.5) * bw, [rows[k][m] for m in show], bw, label=k)
    ax.axhline(0, color="k", lw=0.6); ax.set_xticks(x); ax.set_xticklabels(show)
    ax.set_title("Cross-family leakage signature: qwen3:8b vs gemma3:12b (treatment, in-distribution)")
    ax.legend(); fig.tight_layout()
    fig.savefig(RESULTS_DIR / "figures" / "cross_family_comparison.png", dpi=130); plt.close(fig)
    print("[compare] wrote results/figures/cross_family_comparison.png")


if __name__ == "__main__":
    main()
