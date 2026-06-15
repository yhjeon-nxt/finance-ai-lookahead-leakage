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

    # grouped bar of the leakage-relevant metrics. Sharpe (~1.8) and the timing metrics (~0.1)
    # live on wildly different scales, so they get SEPARATE panels — on one shared axis the timing
    # bars were invisible.
    import numpy as np

    from leakage.run import figstyle as fs
    plt = fs.plt
    fs.use()
    labels = list(rows.keys())                       # ["qwen3:8b (ec2)", "gemma3:12b (gemma)"]
    fam_color = [fs.TREAT, fs.CONFAB]                # qwen3 genuinely recalls; gemma confabulates
    panels = [("Sharpe ratio (in-distribution)", ["T-in Sharpe"], ["T-in\nSharpe"]),
              ("Leakage timing metrics (corr units)",
               ["T-in exposure timing", "DiD exposure timing"],
               ["T-in exposure\ntiming", "DiD exposure\ntiming"])]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), gridspec_kw={"width_ratios": [1, 1.7]})
    for ax, (ptitle, metrics, mlabels) in zip(axes, panels):
        x = np.arange(len(metrics)); bw = 0.36
        for i, k in enumerate(labels):
            vals = [rows[k][m] for m in metrics]
            b = ax.bar(x + (i - 0.5) * bw, vals, bw, label=k, color=fam_color[i % 2],
                       edgecolor="white", lw=0.7, zorder=3)
            fs.bar_labels(ax, b, vals, fmt="{:.3f}", fontsize=9, dy_frac=0.025)
        fs.ygrid(ax); fs.zero_line(ax); fs.despine(ax)
        ax.set_xticks(x); ax.set_xticklabels(mlabels)
        ax.set_title(ptitle, loc="left", fontsize=12)
        ax.margins(y=0.18)
    axes[0].legend(loc="upper right")
    fig.suptitle("Cross-family leakage signature — qwen3:8b (recalls) vs gemma3:12b (confabulates)",
                 y=1.00, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fs.save(fig, RESULTS_DIR / "figures" / "cross_family_comparison.png")
    print("[compare] wrote results/figures/cross_family_comparison.png")


if __name__ == "__main__":
    main()
