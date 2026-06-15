"""Per-model in-train-window vs out-of-train-window leakage test (the core within-model design).

For EACH model we trade ~1 calendar year INSIDE its training cutoff (it "knows" the year) and
~1 year AFTER its cutoff (it cannot), holding model + prompt + scaffold fixed. Any in-vs-out
foresight gap is parametric leakage. A no-memory momentum baseline on the same windows gives the
difference-in-differences (nets out market regime). Aggregating across 4 models with different
cutoffs tests whether models that GENUINELY RECALL their in-year leak more than those that
confabulate it.

Designed to run on EC2 (LEAKAGE_RUN_MODULE=leakage.run.per_model_windows). Daily cadence,
seeds from LEAKAGE_SEEDS.
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from leakage.agent.llm_client import MockClient, OllamaClient  # noqa: E402
from leakage.backtest.engine import run_backtest  # noqa: E402
from leakage.config import RESULTS_DIR, SEEDS, Group, ModelSpec, Window  # noqa: E402
from leakage.data.ingest import download_prices  # noqa: E402
from leakage.metrics import financial, stats  # noqa: E402
from leakage.run.orchestrate import _market_next  # noqa: E402

Y2023 = Window("Y2023", date(2023, 1, 1), date(2023, 12, 31), date(2022, 10, 1))
Y2024 = Window("Y2024", date(2024, 1, 1), date(2024, 12, 31), date(2023, 10, 1))
Y2025 = Window("Y2025", date(2025, 1, 1), date(2025, 12, 31), date(2024, 10, 1))
Y2026 = Window("Y2026", date(2026, 1, 1), date(2026, 6, 13), date(2025, 10, 1))

# Each model: ~1 year INSIDE its training cutoff vs a window clearly AFTER it. The OUT window is
# chosen PER MODEL so it is genuinely post-cutoff: for qwen3 (a 2025-release model that partly
# knows 2025) the only clean OOD window is 2026; for the others 2025 is clean.
# Both size tiers, identical (new, richer) causal context — so size is the only variable in the
# §4.10 size×recall comparison. Window per model = ~1yr inside cutoff vs a clean post-cutoff year.
MODELS = [
    # --- small tier (~8-12B) ---
    {"tag": "llama3.1:8b",  "tier": "8B", "cutoff": "2023-12",  "recall": "denies 2024",          "in": Y2023, "out": Y2025},
    {"tag": "qwen2.5:7b",   "tier": "8B", "cutoff": "~2023-10", "recall": "knows 2023",           "in": Y2023, "out": Y2025},
    {"tag": "qwen3:8b",     "tier": "8B", "cutoff": "~2024-Q4", "recall": "recalls 2024-H2",      "in": Y2024, "out": Y2026},
    {"tag": "gemma3:12b",   "tier": "12B", "cutoff": "2024-08", "recall": "confabulates 2024",    "in": Y2024, "out": Y2025},
    # --- large tier (~24-32B) ---
    {"tag": "qwen3:32b",        "tier": "32B", "cutoff": "~2024-Q4", "recall": "2024-aware",          "in": Y2024, "out": Y2026},
    {"tag": "qwen2.5:32b",      "tier": "32B", "cutoff": "~2023",    "recall": "~2023 (denies 2024)", "in": Y2023, "out": Y2025},
    {"tag": "gemma3:27b",       "tier": "27B", "cutoff": "2024-08",  "recall": "Aug-2024 (confab?)",  "in": Y2024, "out": Y2025},
    {"tag": "mistral-small3.2", "tier": "24B", "cutoff": "~2023-10", "recall": "~2023 (denies 2024)", "in": Y2023, "out": Y2025},
]


def _contrib(results) -> pd.Series:
    """Seed-averaged per-day exposure-timing contribution (mean ≈ corr(exposure, next-day mkt))."""
    cols = [stats.prescience_contrib(r.exposure, _market_next(r)) for r in results]
    cols = [c for c in cols if not c.empty]
    if not cols:
        return pd.Series(dtype=float)
    return pd.concat(cols, axis=1).mean(axis=1)


def _run_cell(tag: str, window: Window, seeds, mock: bool):
    client = MockClient(name=f"mock-{tag}") if mock else OllamaClient(tag)
    spec = ModelSpec(tag, "n/a", "treatment")
    role = "mock" if mock else "real"
    grp = Group(f"{tag.replace(':', '-')}-{window.name}-{role}", spec, window, "")
    res, rets = [], []
    for s in seeds:
        r = run_backtest(grp, client, s, temperature=0.7)
        res.append(r); rets.append(r.port_returns)
    return res, pd.concat(rets) if rets else pd.Series(dtype=float)


def main():
    import os
    seeds = SEEDS
    # LEAKAGE_PERMODEL_ONLY=<tag[,tag]> runs a subset (one model per EC2 instance, in parallel);
    # each instance's partial eval_permodel.json is merged locally afterwards.
    only = [t.strip() for t in os.environ.get("LEAKAGE_PERMODEL_ONLY", "").split(",") if t.strip()]
    models = [m for m in MODELS if m["tag"] in only] if only else MODELS
    for w in (Y2023, Y2024, Y2025, Y2026):
        download_prices(w)
    out = {"design": "per-model in-train vs out-of-train (1yr each)", "seeds": seeds, "models": {}}

    for m in models:
        tag = m["tag"]
        print(f"\n=== {tag}  IN={m['in'].name}  OUT={m['out'].name}  ({m['recall']}) ===", flush=True)
        in_res, in_ret = _run_cell(tag, m["in"], seeds, mock=False)
        out_res, out_ret = _run_cell(tag, m["out"], seeds, mock=False)
        min_res, _ = _run_cell(tag, m["in"], seeds, mock=True)
        mout_res, _ = _run_cell(tag, m["out"], seeds, mock=True)

        in_c, out_c = _contrib(in_res), _contrib(out_res)
        min_c, mout_c = _contrib(min_res), _contrib(mout_res)
        real_gap = float(in_c.mean() - out_c.mean())
        mock_gap = float(min_c.mean() - mout_c.mean())
        perm = stats.permutation_diff(in_c, out_c)
        parse_fail = int(sum(r.n_parse_fail for r in in_res + out_res))

        m_out = {
            "cutoff": m["cutoff"], "recall": m["recall"],
            "in_window": m["in"].name, "out_window": m["out"].name,
            "sharpe_in": financial.sharpe(in_ret), "sharpe_out": financial.sharpe(out_ret),
            "return_in": financial.total_return(in_ret), "return_out": financial.total_return(out_ret),
            "timing_in": float(in_c.mean()) if len(in_c) else float("nan"),
            "timing_out": float(out_c.mean()) if len(out_c) else float("nan"),
            "real_in_minus_out": real_gap, "regime_baseline_gap": mock_gap,
            "DiD": real_gap - mock_gap,
            "perm_in_vs_out_p": perm["p_value"], "perm_diff": perm["diff"],
            "n_in_days": int(len(in_c)), "n_out_days": int(len(out_c)), "parse_fail": parse_fail,
        }
        out["models"][tag] = m_out
        print(f"  Sharpe in={m_out['sharpe_in']:+.2f} out={m_out['sharpe_out']:+.2f} | "
              f"DiD={m_out['DiD']:+.4f} perm_p={m_out['perm_in_vs_out_p']:.3f} parsefail={parse_fail}")

    path = RESULTS_DIR / "eval_permodel.json"
    path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n[per-model] wrote {path}")
    _plot(out)


def _plot(out: dict):
    from matplotlib.patches import Patch

    from leakage.run import figstyle as fs
    plt = fs.plt
    fs.use()

    # Display order: group by family, small→large, so the within-family size effect (§4.10)
    # is read off adjacent bars; then the lone-model controls. Tier shown under each tick.
    pref = ["qwen3:8b", "qwen3:32b", "qwen2.5:7b", "qwen2.5:32b",
            "gemma3:12b", "gemma3:27b", "llama3.1:8b", "mistral-small3.2"]
    tags = [t for t in pref if t in out["models"]] + [t for t in out["models"] if t not in pref]
    M = out["models"]
    did = [M[t]["DiD"] for t in tags]
    # Prefer the block-bootstrap DiD p (autocorrelation-robust, tests the DiD itself);
    # fall back to the raw-gap permutation p if the block stats haven't been computed yet.
    pval = [M[t].get("did_block", {}).get("p_gt_0", M[t].get("perm_in_vs_out_p", float("nan")))
            for t in tags]
    sin = [M[t]["sharpe_in"] for t in tags]
    sout = [M[t]["sharpe_out"] for t in tags]

    def _kind(r):  # robust to label casing (the old "RECALL" check silently matched nothing)
        rl = r.lower()
        return "recall" if "recall" in rl else ("confab" if "confab" in rl else "deny")
    kind = [_kind(M[t]["recall"]) for t in tags]
    cmap = {"recall": fs.TREAT, "confab": fs.CONFAB, "deny": fs.NEUTRAL}
    colr = [cmap[k] for k in kind]
    tier_of = {m["tag"]: m["tier"] for m in MODELS}
    ticklabels = [f"{t}\n({tier_of[t]})" if t in tier_of else t for t in tags]

    x = np.arange(len(tags))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.6))

    # --- panel 1: leakage DiD + one-sided bootstrap p ---
    bars = a1.bar(x, did, color=colr, width=0.66, edgecolor="white", lw=0.7, zorder=3)
    fs.ygrid(a1); fs.zero_line(a1); fs.despine(a1)
    lo, hi = min(did + [0]), max(did + [0.01])
    a1.set_ylim(lo - (hi - lo) * 0.18, hi * 1.30)
    for j, (d, p) in enumerate(zip(did, pval)):
        yy = d + (hi - lo) * (0.03 if d >= 0 else -0.03)
        a1.text(j, yy, f"p={p:.2f}\n{fs.sig_star(p)}", ha="center",
                va="bottom" if d >= 0 else "top", fontsize=8.5, color=fs.sig_color(p))
    a1.set_xticks(x); a1.set_xticklabels(ticklabels, fontsize=9)
    a1.set_ylabel("leakage DiD  (in − out, regime-adjusted)")
    fs.title(a1, "Per-model leakage signal",
             "DiD with one-sided bootstrap p  (H₀: no leakage — smaller p = stronger leakage)")
    a1.legend(handles=[Patch(fc=fs.TREAT, label="genuinely recalls window"),
                       Patch(fc=fs.CONFAB, label="confabulates window"),
                       Patch(fc=fs.NEUTRAL, label="denies / pre-cutoff")],
              loc="upper left", fontsize=9)

    # --- panel 2: in-train vs out-of-train Sharpe ---
    bw = 0.4
    b_in = a2.bar(x - bw / 2, sin, bw, label="in-train year", color=fs.TREAT,
                  edgecolor="white", lw=0.7, zorder=3)
    b_out = a2.bar(x + bw / 2, sout, bw, label="out-of-train year", color=fs.OOD,
                   edgecolor="white", lw=0.7, zorder=3)
    fs.ygrid(a2); fs.zero_line(a2); fs.despine(a2)
    a2.set_xticks(x); a2.set_xticklabels(ticklabels, fontsize=9)
    a2.set_ylabel("Sharpe ratio")
    fs.title(a2, "In-train vs out-of-train Sharpe",
             "every model is sharper inside its training window")
    a2.legend(loc="upper right")

    fig.tight_layout()
    fs.save(fig, RESULTS_DIR / "figures" / "per_model_in_vs_out.png")
    print("[per-model] wrote results/figures/per_model_in_vs_out.png")


if __name__ == "__main__":
    main()
