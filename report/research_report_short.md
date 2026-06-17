# Look-ahead Leakage in Open-Source LLM Trading Agents — Short Report

**Younghoon Jeon (전영훈)** · Korea University · BREIN Lab · Student ID 2025010657 · Financial AI (final project) 2026
**Code:** <https://github.com/yhjeon-nxt/finance-ai-lookahead-leakage> · Full report: `report/research_report.pdf`

*Backtesting an LLM trading agent on a window that lies **inside its pre-training corpus** is a
silent look-ahead bias: the outcome is baked into the weights, and no chronological train/test
split of the inputs can remove it. We call this **parametric look-ahead leakage**, and measure it
with controlled, **price-only** trading agents over an in-distribution window (2024-H2) and a
genuinely out-of-distribution one (2026).*

### 1. Problem and gap

Classical look-ahead bias is a *data* problem fixed by chronological splitting. An LLM adds a
second, invisible channel — its **pre-training corpus**: if it covers the backtest window, the
model can "recall" what happened and trade on it, and splitting the *input features* does nothing.
Of the seminar papers, only **CausalStock** and **TradeMaster** mention look-ahead, both treating
it purely as a splitting issue; **none** address a model whose pre-training already contains the
test period. This study targets exactly that gap: *does an agent whose training data covers a
window gain a non-causal edge there, versus (a) a model that cannot have seen it and (b) itself on
a window it cannot have seen?*

### 2. Design

Three arms run an **identical Reflection + ReAct agent** on a **price-only causal context** (no
news; a hard causality assertion) so any foresight must be *parametric*. Knowledge cutoffs are
**measured by probe, not trusted from model cards**: the treatment `qwen3:8b` genuinely recalls
the 2024-08-05 yen-carry crash and the NVDA split; the model-control `llama3.1:8b` (cutoff 2023-12)
denies every 2024-H2 fact; the time-control is the treatment on 2026. To strip out the
2024-vs-2026 **market-regime confound** we report a **difference-in-differences (DiD)** against an
identical **no-memory momentum baseline**. Inference uses 3 seeds, a circular block bootstrap,
permutation tests, and a **pseudo-event null** (98 random dates) for the crash-timing score.

### 3. Results

| metric | **T-in** `qwen3:8b` (recalls) | **C-A** `llama3.1:8b` (control) | **C-B** `qwen3:8b` (OOD) |
|---|---|---|---|
| Sharpe | **1.76** | 0.30 | 0.49 |
| next-day prescience | +0.021 | −0.032 | +0.016 |
| Aug-5 de-risk (pseudo-event *p*) | **+0.115 (0.051)** | −0.125 (0.92) | — |

**The knowledge–behaviour match is the core finding.** The treatment **cut risk into the Aug-5
crash it demonstrably remembers** (+0.115; top ~5% of random-timing outcomes, *p*=0.051), while
the control did the **opposite** (−0.125, *p*=0.92). It shows **no election-timing edge** — exactly
the event the probe shows it does *not* surface — yet its book still **significantly overweights
the election winner JPM** (Δ+0.088, *p*=0.001): leakage that is behavioural even where verbal
recall is suppressed. Behaviour mirrors *measured* memory item-by-item, which a generic "smarter
model" or a regime artifact would not produce.

**Regime is netted out, not the explanation.** The regime-adjusted **DiD is positive on every
metric** (exposure-timing **+0.136**); the no-memory baseline's own in–out gap is *negative*, so
regime works *against* the result. **Sharpe is descriptive** — formal inference rests on the
timing/DiD metrics, not raw Sharpe.

**Family dependence (pre-registered).** An independent-family co-treatment `gemma3:12b` (documented
Aug-2024 cutoff) does **not** replicate the signature — it **confabulates** the period (projects
"Biden" as the 2024 winner), with no crash de-risk (*p*=0.54). **Lesson: a documented in-window
cutoff is necessary but not sufficient — leakage needs *genuine recall*, so each model must be
probed.** An 8-model, two-size-tier sweep adds that, **within every family, the larger model shows
the *larger* regime-adjusted gap** (qwen3 8B→32B +0.075→+0.133), though all per-model effects are
individually non-significant.

**Honesty about strength.** Support is **moderate and internally consistent** but statistically
modest at this sample size (cross-model *p*≈0.075); the load-bearing evidence is the *pattern* —
knowledge-matched timing, sign-consistent DiD, cross-model ordering — not any single *p*-value.

![The leakage signature in one view: (1) higher Sharpe only in-distribution, (2) de-risking into the remembered Aug-5 crash (p=0.051), (3) foresight beyond what market regime explains (DiD > 0 on every metric).](figures/graphical_abstract.png)

### 4. Recommended robust-backtesting standards

(1) **Probe before you backtest** — empirically verify each model's knowledge of the test window;
never trust documented cutoffs. (2) **Prefer out-of-distribution windows** strictly after the
measured cutoff; treat in-distribution backtests as upper bounds. (3) **Use a model-control** with
a verifiably earlier cutoff. (4) **Audit rationales**, not just returns, for confident
confabulation. (5) **Report inference config** (thinking-mode, decoding, seeds) — it materially
changes behaviour. (6) **Isolate the channel** (e.g. price-only) so the leakage source is
unambiguous.
