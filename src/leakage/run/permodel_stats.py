"""Recompute the per-model leakage significance with the CORRECT statistic + method.

Fixes two issues with the original `perm_in_vs_out_p`:
  (1) it tested the regime-confounded RAW in-vs-out gap, not the regime-adjusted DiD the figure
      reports; (2) it permuted individual days, ignoring autocorrelation (anti-conservative).

Here we reconstruct the four per-day exposure-timing contribution series per model (in/out ×
real/mock) from the decision logs and run a CIRCULAR-BLOCK bootstrap of the DiD itself. Offline —
no model re-run needed. Updates eval_permodel.json and re-renders the figure.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from leakage.config import RESULTS_DIR, UNIVERSE  # noqa: E402
from leakage.data.ingest import load_prices, trading_days  # noqa: E402
from leakage.metrics import stats  # noqa: E402
from leakage.run.per_model_windows import MODELS, _plot  # noqa: E402

DEC = RESULTS_DIR / "permodel" / "decisions"


def _market_next(window) -> pd.Series:
    close = load_prices(window)
    close = (close["Close"] if "Close" in close.columns.get_level_values(0) else close)[UNIVERSE]
    rets = close.pct_change()
    days = trading_days(window)
    return pd.Series({days[i]: float(rets.loc[days[i + 1]].mean()) for i in range(len(days) - 1)})


def _contrib(tagdash: str, window, role: str) -> pd.Series:
    """Seed-averaged per-day exposure-timing contribution for one (model, window, role)."""
    mn = _market_next(window)
    series = []
    for f in sorted(DEC.glob(f"{tagdash}-{window.name}-{role}_*_seed*.jsonl")):
        expo = {}
        for line in f.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                if d.get("parse_ok", True):
                    expo[pd.Timestamp(d["date"])] = float(sum((d.get("target_weights") or {}).values()))
        c = stats.prescience_contrib(pd.Series(expo), mn)
        if not c.empty:
            series.append(c.reset_index(drop=True))
    if not series:
        return pd.Series(dtype=float)
    return pd.concat(series, axis=1).mean(axis=1)


def main():
    rep = json.loads((RESULTS_DIR / "eval_permodel.json").read_text())
    print("model         DiD     block-CI            p(DiD>0)   [old raw-gap p]")
    for m in MODELS:
        tag, td = m["tag"], m["tag"].replace(":", "-")
        ir = _contrib(td, m["in"], "real")
        orr = _contrib(td, m["out"], "real")
        im = _contrib(td, m["in"], "mock")
        om = _contrib(td, m["out"], "mock")
        did = stats.block_bootstrap_did(ir, orr, im, om, n_boot=5000, block=5)
        rep["models"][tag]["did_block"] = did
        old = rep["models"][tag].get("perm_in_vs_out_p", float("nan"))
        print(f"{tag:13} {did['point']:+.4f}  [{did['lo']:+.3f},{did['hi']:+.3f}]   "
              f"p={did['p_gt_0']:.3f}      (raw {old:.3f})")
    rep["did_significance_method"] = ("circular block bootstrap (block=5) of the DiD = "
                                      "[in-out]_real - [in-out]_mock; one-sided p = P(DiD<=0)")
    (RESULTS_DIR / "eval_permodel.json").write_text(json.dumps(rep, indent=2, default=str))
    _plot(rep)
    print("[permodel-stats] updated eval_permodel.json + figure")


if __name__ == "__main__":
    main()
