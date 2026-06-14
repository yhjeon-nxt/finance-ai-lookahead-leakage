# Look-ahead Leakage in Open-Source LLM Trading Agents: A Forensic Study of Parametric Memory as a Backtest Confound

**Author:** Younghoon Jeon (전영훈) · Korea University AutoAI Lab
**Course:** Financial AI (final project) · 2026
**Code:** https://github.com/yhjeon-nxt/finance-ai-lookahead-leakage

---

## Abstract

Backtesting an LLM-based trading agent on a historical window that lies *inside the model's
pre-training corpus* is a silent form of look-ahead bias: the "future" is baked into the model
weights, and no chronological train/test split of the input data can remove it. We design and
execute an end-to-end experiment to detect and measure this *parametric* leakage in
open-source LLM trading agents. Using models with **empirically verified** knowledge cutoffs —
a control (`llama3.1:8b`, cutoff 2023-12, which cannot know the test window) and a treatment
(a 2024-aware model selected by probe) — we run identical reflection/ReAct trading agents over
the **2024-H2** window (bracketed by the 2024-08-05 yen-carry crash and the 2024-11-05 US
election) and over a genuinely out-of-distribution **2026** window. Crucially, the agent's
context is **price-only**, so any anticipatory behaviour can originate only in parametric
memory. We quantify leakage with four metrics (next-day prescience, pre-event timing, the
within-model in-distribution−OOD foresight gap, and rationale forensics) under block-bootstrap
and permutation inference. Beyond the headline result, our cutoff probes surface a second,
arguably more dangerous phenomenon for practitioners: small open models **confabulate** the
period — misreporting their own cutoff and inventing plausible future events with confidence —
rather than cleanly memorising it. We close with concrete robust-backtesting standards for the
LLM-agent era.

> *Empirical Results (§4) are populated from the EC2 experiment run; methodology, the empirical
> cutoff verification (§3.3–3.4), and forensic framing are complete.*

---

## 1. Introduction

Financial-AI research has moved through four eras — rule-based, reinforcement learning,
LLM-agent, and multimodal-agent systems. As LLM agents become the dominant trading paradigm,
an old hazard returns in a new and subtle form. Classical look-ahead bias is a *data* problem:
test-period information leaks into training through careless splitting. The standard fix is a
chronological train/test split. But an LLM carries a second, invisible information channel —
its **pre-training corpus**. If that corpus covers the backtest window, the model may "recall"
what happened and trade on it, and a chronological split of the *input features* does nothing
to prevent this. We call this **parametric look-ahead leakage**.

This study asks: *does an LLM trading agent whose training data covers a backtest window gain a
non-causal advantage on that window, relative to (a) a model that cannot have seen it and (b)
itself on a window it cannot have seen?* We answer it with a controlled, reproducible,
cost-disciplined experiment built around models with **measured** (not merely documented)
knowledge cutoffs.

## 2. Lecture Alignment

An audit of the seminar's presentations (FINCON, TradeMaster, QuantAgents, FLAG-TRADER,
CausalStock, CLER, Hierarchical Financial-QA, *Two Sides of the Same Coin*) shows a clear
architectural consensus and a clear blind spot.

**Architectural consensus.** The dominant paradigms were tool-use / retrieval (nearly all
papers), reflection / memory (≈6/8), ReAct-style reasoning (≈5/8), with multi-agent debate and
gradient-based RL less common. Our agent (§3.5) is deliberately the *intersection* of the three
most common patterns — reflection + ReAct + structured decisions — kept minimal so the leakage
signal is not masked by orchestration.

**The blind spot.** Only **CausalStock** and **TradeMaster** explicitly discussed look-ahead
bias, and both treated it purely as a data-pipeline issue solved by chronological splitting
(CausalStock: *"the data is split chronologically, not randomly… prevents unrealistic future
leakage"*; TradeMaster: *"last year for test, penultimate for validation, remaining for
training"*). **None** addressed the case where the *model's pre-training* already contains the
test period. This project targets precisely that gap.

## 3. Methodology

### 3.1 Hypotheses

- **H1 (leakage):** the treatment, on its in-distribution window, shows a positive foresight
  signature absent in both controls.
- **H0 (null):** no foresight gap beyond capability/noise.
- **H2 (degenerate, pre-registered):** the "knowing" model does not cleanly cheat but
  *confabulates* the period, possibly degrading performance. Pre-registering H2 ensures the
  experiment can report a non-leakage outcome honestly rather than only "confirming" leakage.

### 3.2 Experimental groups

| Group | Model | Window | Can know the window? | Role |
|---|---|---|---|---|
| **T-in** | treatment (2024-aware) | 2024-07-01…12-31 | **yes** | leakage candidate |
| **C-A** | `llama3.1:8b` (cutoff 2023-12) | 2024-07-01…12-31 | no | model control |
| **C-B** | treatment | 2026-01-01…05-31 | no (post-cutoff) | time control |

The **T-in vs C-B** contrast holds the model fixed and varies only whether the traded window is
in-distribution; it is therefore the clean causal estimate of leakage, isolating it from raw
capability. **C-A** corroborates with an independent, verifiably-ignorant model. The
treatment/control model-family difference is a known confound for the C-A comparison; the
within-model C-B comparison neutralises it.

### 3.3 Empirical cutoff verification (not just documented)

Self-reported cutoffs are unreliable, so we *measured* each model's knowledge with specific,
hard-to-confabulate 2024-H2 facts. The control behaves exactly as a valid control must:

> **llama3.1:8b** — *"I'm unable to provide information about … the November 2024 [election]…"*;
> *"I don't have information about a … selloff … around August 5, 2024. My training data only
> goes up to [2023/2022]."*

It denies **all** four 2024-H2 facts — it genuinely cannot leak what it never saw.

### 3.4 Treatment-model selection

We probed four candidate open models on four 2024-H2 discriminators (Nov-2024 election winner,
Harris's VP pick, NVIDIA's June-2024 10-for-1 split, the Aug-5 yen-carry selloff):

| Model | 2024-H2 recall | Notes |
|---|---|---|
| **qwen3:8b** | **2/4** | correctly recalls the **NVIDIA split** and the **Aug-5 selloff** (the market-relevant facts); *refuses* the political ones via RLHF guardrails; misreports its own cutoff as "October 2023" |
| qwen2.5:7b | 1/4 | mostly denies (keyword match on NVIDIA is a borderline false positive) |
| llama3.1:8b | 0/4 | clean control — denies all |
| phi4 | 0/4 | denies all; self-reports "October 2023" despite a Dec-2024 release |

The treatment is selected as the highest-scoring model that still denies 2026 knowledge. On the
EC2 run we extend this probe to 32B candidates (`qwen3:32b`, `qwen2.5:32b`) and auto-select; a
larger model is expected to also recall the political events that the 8B variant guard-rails.

**Finding already visible here:** parametric knowledge of the test window is *real but uneven* —
market facts surface, politically-sensitive facts are suppressed by alignment, and models'
self-reported cutoffs are simply wrong. Leakage is not all-or-nothing.

### 3.5 Agent architecture

A single-agent **Reflection + ReAct** trader. At each day *T* it receives the causal context,
a one-line reflection on the prior day's P&L, and emits a single JSON object:
`{analysis, target_weights{ticker→[0,1], Σ≤1}, confidence, rationale}`. Long-only, cash =
1−Σweights. The free-text `analysis`/`rationale` are preserved verbatim as the qualitative
smoking-gun channel. The identical prompt/scaffold is used for every group; only the model and
the date window change. *Inference note:* reasoning models (qwen3) must run with thinking
disabled, else `format=json` yields empty output — itself a reproducibility hazard (§6).

### 3.6 Price-only causal context (leakage isolation)

The agent sees **only** trailing prices/returns (per-ticker OHLCV-derived features and the last
15 daily returns) dated ≤ *T*, enforced by a hard causality assertion. We deliberately exclude
news. This is the methodological keystone: with a price-only feed, the agent has **no
legitimate channel** to anticipate an unsignalled event such as the Aug-5 crash, so any
pre-emptive de-risking must come from parametric memory.

### 3.7 Metrics

Financial: total return, annualised Sharpe, max drawdown, turnover. Leakage/foresight:
(1) **next-day prescience** = corr(today's allocation, tomorrow's return), expected ≈0 absent
foresight; (2) **pre-event timing** = de-risking before a known crash / loading before a known
rally, relative to the agent's own average exposure; (3) **in-dist−OOD foresight gap** (same
model); (4) **rationale forensics** = automated scan for future-event references. A no-foresight
mock client was used to confirm the metrics return ≈0 under the null (they do).

### 3.8 Statistics

Prescience is expressed as a per-day contribution `z(signal)·z(return)` (mean ≈ Pearson r), so
it supports a **stationary block bootstrap** for CIs and a **label-permutation test** for group
differences. We report effect sizes and 95% CIs, not just p-values, and pool ≥3 seeds.

### 3.9 Infrastructure & cost

Developed and smoke-tested locally (Apple-Silicon ollama), then executed on **one
`g6e.xlarge` spot** instance (L40S 48 GB, ap-northeast-2, ≈$0.54/hr). Data prepared locally and
staged to S3 so the instance does no yfinance I/O; a resumable decision cache (keyed by
group·model·window·seed·date) makes spot interruption cost ≤ one decision; logs/equity/raw
outputs stream to `s3://neuroxt-personal/yhjeon/finance-ai-leakage/` every 30–60 s; the
instance self-terminates on completion. Estimated total compute cost **< $2**.

### 3.10 Pivots from the baseline prompt

The assignment prompt supplied example parameters and explicit adaptation rights. Deviations:
(i) **open local models instead of GPT-4o** — yields a *real, controllable* cutoff gap at zero
API cost; (ii) **empirical treatment-model selection** rather than assuming a model knows the
window (the assumption failed for qwen3:8b on political facts); (iii) **price-only context** to
make parametric memory the sole leakage channel; (iv) **2026** as the OOD window (true
post-cutoff data now exists). Each is documented here and in `findings.md`.

## 4. Empirical Results

*Populated from the EC2 run (`results/eval_ec2.json`, `results/figures/equity_ec2.png`).*

Planned tables/figures: (a) per-group financial summary (return, Sharpe, MaxDD, turnover) with
bootstrap CIs; (b) prescience metrics per group with CIs; (c) T-in vs C-A and T-in vs C-B
permutation tests; (d) the in-dist−OOD foresight gap; (e) equity curves with the Aug-5 and
Nov-5 event lines; (f) curated rationale excerpts flagged by the forensic scan.

## 5. Discussion & Forensic Analysis

### 5.1 The cutoff probe is itself forensic evidence
Before a single trade, the probes establish the experiment's validity and reveal the *texture*
of leakage: the control is verifiably blind to 2024-H2, while the treatment demonstrably recalls
the market events the backtest hinges on. This direct, quotable evidence is stronger than
relying on vendor-stated cutoffs.

### 5.2 Backtest forensics
*To be completed from the run:* whether T-in de-risked into the Aug-5 crash and loaded into the
Nov-5 rally while the controls did not; the size and significance of the in-dist−OOD gap; and
any "smoking-gun" rationales referencing future events.

### 5.3 Confabulation can be worse than memorisation
A striking qualitative finding: probed about Q1-2026 (beyond any model's cutoff), qwen3:8b did
not say "I don't know" — it **invented** a specific, confident, false event ("a sharp decline
following the Federal Reserve's unexpected 50bps hike in early March 2026"). For a trading
agent, confident confabulation of the future is at least as dangerous as accurate memorisation:
the former produces unfalsifiable, plausible-sounding rationales that can drive real positions.

## 6. Conclusion & Robust-Backtesting Standards

Parametric look-ahead leakage is a first-class threat to LLM-agent backtests, invisible to the
chronological-split discipline the field currently relies on. We recommend:

1. **Probe before you backtest.** Empirically verify each model's knowledge of the test window
   (as in §3.3–3.4); never trust documented cutoffs.
2. **Prefer out-of-distribution windows.** Evaluate on periods that strictly postdate the
   model's (measured) cutoff; treat in-distribution backtests as upper bounds, not estimates.
3. **Use a model-control.** Include a same-task agent with a verifiably earlier cutoff.
4. **Audit rationales, not just returns.** Scan free-text for future-event references and for
   confident confabulation.
5. **Report inference configuration.** Thinking-mode, decoding constraints, and seeds materially
   change behaviour and must be fixed and reported for reproducibility.
6. **Isolate the channel.** Where possible, restrict the agent's feed (e.g. price-only) so the
   leakage source is unambiguous.

---

### Appendix A — Reproducibility
All code, prompts, the causality guard, the cutoff/selection probes, metrics, statistics, and
the EC2 infra (spot launch, bootstrap, self-terminate) are in the repository. The full run is
reproducible via `infra/stage.sh` + `infra/launch_spot.sh`, or locally via
`python -m leakage.run.main`.

### Appendix B — The inference-config hazard
qwen3 is a hybrid reasoning model; under `format=json` with thinking enabled it emits empty
content (the token budget is consumed by suppressed `<think>` tokens). The fix (`think=False`)
changed the parse-success rate from 0% to 100%. We flag this as a concrete, easily-missed
reproducibility hazard for LLM-agent backtests.
