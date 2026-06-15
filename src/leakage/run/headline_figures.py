"""Marquee report figures that the section tables under-serve.

Built from local artifacts only — eval_ec2.json, cutoff_probe.json (+ the documented §3.4/§4.9
verdicts), and the decision logs. No EC2, no model re-runs.

  graphical_abstract -> figures/graphical_abstract.png  one-glance leakage signature (3 panels)
  probe_scorecard    -> figures/probe_scorecard.png     recall vs confabulate vs deny grid (§3.4)
  stat_evidence      -> figures/stat_evidence.png        3-group prescience CIs + DiD slope (§4.4–4.5)
  aug5_forensics     -> figures/aug5_forensics.png       exposure timeline + pseudo-event null (§4.7)

The latter two REPLACE the standalone timing_prescience_ci / exposure_timeline figures (which they
subsume), per the figure restructure.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from leakage.config import IN_DIST, OOD, RESULTS_DIR, UNIVERSE  # noqa: E402
from leakage.data.ingest import load_prices, trading_days  # noqa: E402
from leakage.metrics import stats  # noqa: E402
from leakage.run import figstyle as fs  # noqa: E402
from leakage.run.extra_analyses import AUG5, NOV5, _derisk_score, _exposure_panel  # noqa: E402

FIG = RESULTS_DIR / "figures"
FIG.mkdir(parents=True, exist_ok=True)
DEC = RESULTS_DIR / "ec2" / "decisions"


def _eval() -> dict:
    return json.loads((RESULTS_DIR / "eval_ec2.json").read_text())


# --------------------------------------------------------------------------------------------
# shared: seed-averaged per-day exposure-timing contribution for any (decision-file prefix, window)
# --------------------------------------------------------------------------------------------
def _market_next(window) -> pd.Series:
    close = load_prices(window)
    close = (close["Close"] if "Close" in close.columns.get_level_values(0) else close)[UNIVERSE]
    rets = close.pct_change()
    days = trading_days(window)
    return pd.Series({days[i]: float(rets.loc[days[i + 1]].mean()) for i in range(len(days) - 1)})


def _timing_mean(prefix: str, window) -> float:
    mn = _market_next(window)
    series = []
    for f in sorted(DEC.glob(f"{prefix}_seed*.jsonl")):
        ex = {}
        for ln in f.read_text().splitlines():
            if ln.strip():
                o = json.loads(ln)
                if o.get("parse_ok", True):
                    ex[pd.Timestamp(o["date"])] = float(sum((o.get("target_weights") or {}).values()))
        c = stats.prescience_contrib(pd.Series(ex), mn)
        if not c.empty:
            series.append(c.reset_index(drop=True))
    return float(pd.concat(series, axis=1).mean(axis=1).mean()) if series else float("nan")


# =============================================================================================
# 1. GRAPHICAL ABSTRACT — one figure that carries the whole story
# =============================================================================================
def graphical_abstract():
    fs.use()
    plt = fs.plt
    rep = _eval()
    g = rep["groups"]

    sharpe = [g["T-in"]["financial"]["sharpe"], g["C-A"]["financial"]["sharpe"],
              g["C-B"]["financial"]["sharpe"]]
    # Aug-5 pre-event de-risk (from §4.3/§4.7): positive = cut risk into the known crash.
    derisk = {"T-in": 0.115, "C-A": -0.125}
    derisk_p = {"T-in": 0.051, "C-A": 0.92}
    did = rep["foresight_gap_DiD"]
    did_vals = [did["gap_ticker_prescience"], did["gap_exposure_timing"], did["gap_conf_weighted_timing"]]

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(14.5, 5.0))

    # -- A: in-distribution outperformance --
    cols = [fs.TREAT, fs.CTRL, fs.OOD]
    labs = ["T-in\n(knows window)", "C-A\n(other model)", "C-B\n(same model, OOD)"]
    b = a1.bar(range(3), sharpe, color=cols, edgecolor="white", lw=0.8, zorder=3, width=0.66)
    fs.ygrid(a1); fs.despine(a1); a1.set_xticks(range(3)); a1.set_xticklabels(labs, fontsize=9)
    a1.set_ylabel("Sharpe ratio (2024-H2)")
    a1.set_ylim(0, max(sharpe) * 1.22)
    fs.bar_labels(a1, b, sharpe, fmt="{:.2f}", fontsize=11, dy_frac=0.02)
    fs.title(a1, "1)  Outperforms only in-distribution", "Sharpe 1.76 vs 0.30 / 0.49")

    # -- B: de-risks before its KNOWN crash --
    gb = ["T-in", "C-A"]
    vals = [derisk[k] for k in gb]
    bb = a2.bar(range(2), vals, color=[fs.TREAT, fs.CTRL], edgecolor="white", lw=0.8, zorder=3, width=0.6)
    fs.ygrid(a2); fs.zero_line(a2); fs.despine(a2)
    a2.set_xticks(range(2)); a2.set_xticklabels(["T-in\n(knows crash)", "C-A\n(control)"], fontsize=9)
    a2.set_ylabel("Aug-5 pre-event de-risk  (>0 = cut risk)")
    a2.set_ylim(min(vals) * 1.5, max(vals) * 1.6)
    for j, k in enumerate(gb):
        v = derisk[k]
        a2.text(j, v + (0.012 if v >= 0 else -0.012), f"{v:+.3f}\np={derisk_p[k]:.2f}",
                ha="center", va="bottom" if v >= 0 else "top", fontsize=9.5,
                color=fs.SIG if derisk_p[k] < 0.1 else fs.FAINT)
    fs.title(a2, "2)  De-risks into the Aug-5 crash", "in the top ~5% of random timing (p=0.051)")

    # -- C: foresight beyond regime (DiD) --
    mets = ["ticker\nprescience", "exposure\ntiming", "conf-wtd\ntiming"]
    bc = a3.bar(range(3), did_vals, color=fs.TREAT, edgecolor="white", lw=0.8, zorder=3, width=0.6)
    fs.ygrid(a3); fs.zero_line(a3); fs.despine(a3)
    a3.set_xticks(range(3)); a3.set_xticklabels(mets, fontsize=9)
    a3.set_ylabel("regime-adjusted DiD  (>0 = leakage)")
    a3.set_ylim(0, max(did_vals) * 1.25)
    fs.bar_labels(a3, bc, did_vals, fmt="{:+.3f}", fontsize=10, dy_frac=0.02)
    fs.title(a3, "3)  Foresight exceeds market regime", "DiD > 0 on every metric")

    fig.suptitle("The leakage signature — a model that genuinely recalls the backtest window "
                 "outperforms, de-risks before its known crash, and shows foresight beyond regime "
                 "(only in-distribution)", y=1.005, fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    fs.save(fig, FIG / "graphical_abstract.png")
    print("wrote graphical_abstract.png")


# =============================================================================================
# 2. KNOWLEDGE-PROBE SCORECARD — recall vs confabulate vs deny (the keystone)
# =============================================================================================
def probe_scorecard():
    fs.use()
    plt = fs.plt
    from matplotlib.patches import Patch, Rectangle

    facts = ["2024 election\nwinner", "Harris's\nVP pick", "NVIDIA\n10:1 split", "Aug-5\ncrash cause"]
    # Verdicts curated from §3.3–3.4 probe transcripts + §4.9 (gemma). One row per model, ordered
    # treatment → confabulator → controls. Values: recall / refuse / confab / border / deny / unver.
    rows = [
        ("qwen3:8b",    "treatment", ["refuse", "refuse", "recall", "recall"], "RECALLS  (2/4 market facts)"),
        ("gemma3:12b",  "treatment", ["confab", "unver", "unver", "confab"],   'CONFABULATES  ("Biden won")'),
        ("qwen2.5:7b",  "control",   ["deny", "deny", "border", "deny"],       "DENIES  (~1/4, borderline)"),
        ("llama3.1:8b", "control",   ["deny", "deny", "deny", "deny"],         "DENIES  (0/4, clean control)"),
        ("phi4",        "control",   ["deny", "deny", "deny", "deny"],         "DENIES  (0/4)"),
    ]
    cmap = {"recall": fs.TREAT, "refuse": fs.CTRL, "confab": fs.SIG,
            "border": fs.CONFAB, "deny": "#cfd4da", "unver": "#eef0f2"}
    txt = {"recall": "recall", "refuse": "refuse\n(RLHF)", "confab": "confab",
           "border": "border", "deny": "deny", "unver": "—"}
    tcol = {"recall": "white", "refuse": "white", "confab": "white",
            "border": fs.INK, "deny": fs.FAINT, "unver": "#aab0b6"}

    nrow, ncol = len(rows), len(facts)
    fig, ax = plt.subplots(figsize=(12.0, 5.4))
    for i, (model, role, verdicts, _) in enumerate(rows):
        y = nrow - 1 - i
        for j, v in enumerate(verdicts):
            ax.add_patch(Rectangle((j, y), 0.96, 0.92, facecolor=cmap[v],
                                   edgecolor="white", lw=2, zorder=2))
            ax.text(j + 0.48, y + 0.46, txt[v], ha="center", va="center",
                    fontsize=9, color=tcol[v], fontweight="bold", zorder=3)
    # verdict summary column
    for i, (model, role, verdicts, verdict) in enumerate(rows):
        y = nrow - 1 - i
        ax.text(ncol + 0.15, y + 0.46, verdict, ha="left", va="center", fontsize=9.5,
                color=fs.INK, zorder=3)

    ax.set_xlim(-0.05, ncol + 3.0)
    ax.set_ylim(-0.05, nrow + 0.18)
    # fact column headers drawn manually just above the grid (kept clear of the title)
    for j, f in enumerate(facts):
        ax.text(j + 0.48, nrow + 0.05, f, ha="center", va="bottom", fontsize=9.5, color=fs.INK)
    ax.set_xticks([])
    ax.set_yticks([nrow - 1 - i + 0.46 for i in range(nrow)])
    ax.set_yticklabels([f"{m}\n({r})" for m, r, _, _ in rows], fontsize=9.5)
    ax.tick_params(length=0)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    fig.suptitle("Empirical knowledge-probe scorecard — genuine recall ≠ documented cutoff",
                 y=1.04, fontsize=14, fontweight="bold")
    fig.text(0.5, 0.985, "each model probed on 4 hard-to-confabulate 2024-H2 facts · "
             "the basis for treatment selection (§3.4)", ha="center", va="top",
             fontsize=9.5, color=fs.FAINT)
    legend_items = [Patch(fc=fs.TREAT, label="genuine recall (can leak)"),
                    Patch(fc=fs.CTRL, label="refuses (RLHF guardrail)"),
                    Patch(fc=fs.SIG, label="confabulates (false 'memory')"),
                    Patch(fc=fs.CONFAB, label="borderline / keyword match"),
                    Patch(fc="#cfd4da", label="denies (cannot leak)")]
    ax.legend(handles=legend_items, loc="lower center", bbox_to_anchor=(0.5, -0.17),
              ncol=5, fontsize=8.5, frameon=False, handlelength=1.2, columnspacing=1.3)
    fig.tight_layout()
    fs.save(fig, FIG / "probe_scorecard.png")
    print("wrote probe_scorecard.png")


# =============================================================================================
# 3. STATISTICAL EVIDENCE — 3-group prescience CIs (left) + regime-adjusted DiD slope (right)
#    (subsumes the old timing_prescience_ci.png)
# =============================================================================================
def stat_evidence():
    fs.use()
    plt = fs.plt
    rep = _eval()
    cmp = rep["comparisons"]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.4), gridspec_kw={"width_ratios": [1, 1.12]})

    # -- LEFT: raw 3-group exposure-timing prescience with bootstrap CIs --
    pts = {"T-in": cmp["T-in_vs_C-A"]["T-in_timing_prescience"],
           "C-A": cmp["T-in_vs_C-A"]["C-A_timing_prescience"],
           "C-B": cmp["T-in_vs_C-B"]["C-B_timing_prescience"]}
    p_ac = cmp["T-in_vs_C-A"]["permutation_diff"]["p_value"]
    p_ab = cmp["T-in_vs_C-B"]["permutation_diff"]["p_value"]
    cols = [fs.TREAT, fs.CTRL, fs.OOD]
    for i, gname in enumerate(["T-in", "C-A", "C-B"]):
        p = pts[gname]
        axL.bar(i, p["point"], 0.62, color=cols[i], edgecolor="white", lw=0.7, zorder=3)
        axL.errorbar(i, p["point"], yerr=[[p["point"] - p["lo"]], [p["hi"] - p["point"]]],
                     fmt="none", ecolor=fs.INK, capsize=5, lw=1.3, zorder=4)
    fs.ygrid(axL); fs.zero_line(axL); fs.despine(axL)
    axL.set_xticks(range(3)); axL.set_xticklabels(["T-in", "C-A", "C-B"])
    axL.set_ylabel("exposure-timing prescience  (mean ± 95% bootstrap CI)")
    top = max(pts[g]["hi"] for g in pts)
    fs.bracket(axL, 0, 1, top + 0.02, p_ac, h=0.013, label=f"perm p = {p_ac:.3f}  {fs.sig_star(p_ac)}")
    fs.bracket(axL, 0, 2, top + 0.075, p_ab, h=0.013, label=f"perm p = {p_ab:.3f}  {fs.sig_star(p_ab)}")
    axL.set_ylim(min(pts[g]["lo"] for g in pts) - 0.03, top + 0.15)
    fs.title(axL, "Raw prescience by group", "positive only for the treatment in-distribution")

    # -- RIGHT: difference-in-differences slope (the regime adjustment) --
    li, lo = _timing_mean("T-in_qwen3-8b", IN_DIST), _timing_mean("C-B_qwen3-8b", OOD)
    bi, bo = _timing_mean("T-in_mock-treatment", IN_DIST), _timing_mean("C-B_mock-treatment", OOD)
    did = (li - lo) - (bi - bo)
    x = [0, 1]  # 0 = in-distribution, 1 = out-of-distribution
    # actual LLM line
    axR.plot(x, [li, lo], "-o", color=fs.TREAT, lw=2.4, ms=8, zorder=5,
             label="LLM agent (qwen3:8b)")
    # no-memory momentum baseline
    axR.plot(x, [bi, bo], "--s", color=fs.NEUTRAL, lw=2.0, ms=7, zorder=4,
             label="no-memory momentum baseline")
    # counterfactual: LLM under regime-only (parallel to baseline, anchored at LLM's OOD point)
    cf_in = lo + (bi - bo)
    axR.plot(x, [cf_in, lo], ":", color=fs.SIG, lw=1.8, zorder=4,
             label="LLM counterfactual (regime only)")
    axR.plot([0], [cf_in], "o", color=fs.SIG, ms=7, mfc="white", zorder=5)
    # DiD brace at in-distribution between actual and counterfactual
    axR.annotate("", xy=(0.04, li), xytext=(0.04, cf_in),
                 arrowprops=dict(arrowstyle="<->", color=fs.SIG, lw=1.6))
    axR.text(0.075, (li + cf_in) / 2, f"DiD = {did:+.3f}\n(leakage)", color=fs.SIG,
             fontsize=10.5, va="center", ha="left", fontweight="bold")
    fs.ygrid(axR); fs.zero_line(axR); fs.despine(axR)
    axR.set_xlim(-0.12, 1.12); axR.set_xticks([0, 1])
    axR.set_xticklabels(["in-distribution\n(2024-H2)", "out-of-distribution\n(2026)"])
    axR.set_ylabel("exposure-timing prescience")
    axR.legend(loc="upper right", fontsize=8.8)
    other = rep["foresight_gap_DiD"]
    fs.title(axR, "Regime-adjusted difference-in-differences",
             f"DiD>0 on all metrics: ticker {other['gap_ticker_prescience']:+.3f} · "
             f"conf-wtd {other['gap_conf_weighted_timing']:+.3f}")
    fig.tight_layout()
    fs.save(fig, FIG / "stat_evidence.png")
    print(f"wrote stat_evidence.png  (DiD={did:+.4f})")


# =============================================================================================
# 4. AUG-5 FORENSICS — exposure timeline (left) + pseudo-event null distribution (right)
#    (subsumes the old exposure_timeline_2024H2.png)
# =============================================================================================
def _null_distribution(group: str, k: int = 20):
    panel = _exposure_panel(group)
    if panel.empty:
        return None
    expo = panel.mean(axis=1)
    obs = _derisk_score(expo, AUG5, k)
    cand = [d for d in expo.index if len(expo[expo.index < d]) > k and abs((d - AUG5).days) > 5]
    null = np.array([_derisk_score(expo, d, k) for d in cand])
    null = null[~np.isnan(null)]
    p = float((null >= obs).mean()) if len(null) else float("nan")
    return {"expo": expo, "panel": panel, "obs": obs, "null": null, "p": p}


def aug5_forensics():
    fs.use()
    plt = fs.plt
    import matplotlib.dates as mdates

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14.0, 5.2), gridspec_kw={"width_ratios": [1.55, 1]})

    # -- LEFT: exposure timeline (risk dial) with inter-seed band --
    dat = {}
    for g, c in (("T-in", fs.TREAT), ("C-A", fs.CTRL)):
        panel = _exposure_panel(g)
        if panel.empty:
            continue
        mean = panel.mean(axis=1)
        dat[g] = mean
        axL.plot(mean.index, mean.values, color=c, lw=1.9, zorder=4, label=f"{g} exposure (seed-avg)")
        axL.fill_between(panel.index, panel.min(axis=1), panel.max(axis=1), color=c, alpha=0.15,
                         lw=0, zorder=2)
    fs.ygrid(axL); fs.despine(axL); axL.set_ylim(0, 1.08)
    if dat:
        idx = next(iter(dat.values())).index
        axL.set_xlim(idx.min(), idx.max())
    axL.xaxis.set_major_locator(mdates.MonthLocator())
    axL.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    axL.set_ylabel("total invested fraction")
    for d, lab in [(AUG5, "Aug-5 crash"), (NOV5, "Nov-5 election")]:
        fs.event_marker(axL, d, lab, y_frac=0.12, va="bottom", ha="left")
    axL.legend(loc="lower right")
    fs.title(axL, "Portfolio exposure through 2024-H2",
             "treatment cuts risk into the Aug-5 crash · band = inter-seed min–max")

    # -- RIGHT: pseudo-event null distribution for the Aug-5 de-risk --
    t = _null_distribution("T-in")
    c = _null_distribution("C-A")
    axR.hist(t["null"], bins=18, color="#c7ccd2", edgecolor="white", zorder=2)
    fs.ygrid(axR); fs.despine(axR)
    ymax = axR.get_ylim()[1]
    axR.set_ylim(0, ymax * 1.18)
    ymax = axR.get_ylim()[1]
    # null label sits on the histogram body (center), out of the way of the obs markers
    axR.text(t["null"].mean(), ymax * 0.62, "null:\n98 random\npseudo-events", ha="center",
             va="center", fontsize=8.6, color=fs.FAINT)
    # T-in observed (far right, significant); label to the LEFT of the line to stay on-axis
    axR.axvline(t["obs"], color=fs.TREAT, lw=2.4, zorder=5)
    axR.text(t["obs"] - 0.006, ymax * 0.96, f"T-in obs {t['obs']:+.3f}\np = {t['p']:.3f}  (top ~5%)",
             color=fs.TREAT, fontsize=9.5, va="top", ha="right", fontweight="bold")
    # C-A observed (far left, opposite side); label to the RIGHT of the line
    axR.axvline(c["obs"], color=fs.CTRL, lw=2.4, zorder=5)
    axR.text(c["obs"] + 0.006, ymax * 0.42, f"C-A obs {c['obs']:+.3f}\np = {c['p']:.2f}",
             color=fs.CTRL, fontsize=9.5, va="top", ha="left", fontweight="bold")
    axR.set_xlabel("pre-event de-risk score  (>0 = cut risk before the date)")
    axR.set_ylabel("count")
    fs.title(axR, "Is the de-risk significant?",
             "observed vs random-timing null · treatment in the top ~5%")
    fig.tight_layout()
    fs.save(fig, FIG / "aug5_forensics.png")
    print(f"wrote aug5_forensics.png  (T-in p={t['p']:.3f}, C-A p={c['p']:.2f})")


def main():
    graphical_abstract()
    probe_scorecard()
    stat_evidence()
    aug5_forensics()


if __name__ == "__main__":
    main()
