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

**Headline result.** On a real run (treatment `qwen3:8b`, control `llama3.1:8b`, executed on an
EC2 GPU spot instance), the treatment trading its in-distribution window earned **Sharpe 1.76 /
+25%** versus **0.30** (model control) and **0.49** (same model, out-of-distribution), showed
positive next-day prescience where the controls showed none, and — most tellingly — **de-risked
ahead of the 2024-08-05 crash it demonstrably remembers** (its clearest, portfolio-level
pre-event signal). Around the election it shows no *exposure*-timing edge but a narrower
*composition* tilt — a significant overweight of the bank winner JPM (§4.8) — even though it
verbally refuses the election question, i.e. leakage that is behavioural where verbal recall is
suppressed. The regime-adjusted difference-in-differences is positive on every metric. Support
for parametric leakage is **moderate and internally consistent**, though statistically modest at
this sample size (cross-model p≈0.075). **An independent-family co-treatment (`gemma3:12b`,
official Aug-2024 cutoff) did *not* reproduce the clean signature** — it **confabulates** the
period (projecting "Joe Biden" as the 2024 winner) rather than recalling it (the pre-registered
H2). The two-family contrast yields the study's sharpest lesson: **a documented in-window cutoff
is necessary but not sufficient for leakage — what matters is genuine recall, so each model must
be probed, not assumed contaminated from its cutoff date.** An **eight-model, two-size-tier**
in-train-year vs out-of-train-year sweep (§4.10) adds two patterns: every model trades its
training year at a higher Sharpe than an unseen year, and — within all three families that have
both tiers — the **larger model shows the *larger* regime-adjusted leakage gap** (qwen3 8B→32B
+0.075→+0.133; qwen2.5, gemma likewise), i.e. a more competent agent trades on its memory more
effectively. All per-model effects remain individually non-significant (underpowered), so the
evidence is the *direction* and the *within-family monotonicity*, not per-model p-values.

![Graphical abstract — the leakage signature](figures/graphical_abstract.png)

*The leakage signature in one view. A model that **genuinely recalls** the backtest window
(**1**) earns a far higher Sharpe than a different-cutoff control or itself on an unseen window
(1.76 vs 0.30 / 0.49); (**2**) **cuts risk into the Aug-5 crash it demonstrably remembers** — in
the top ~5% of random-timing outcomes (p=0.051) — while the control did the opposite; and (**3**)
shows exposure-timing foresight that **exceeds what market regime alone explains** (regime-adjusted
DiD > 0 on every metric). All three appear **only in-distribution**.*

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

## 2b. Related Work

Our study sits within the literature on **training-data contamination**, where exposure to
evaluation content during pre-training inflates measured capability and detection methods
struggle to certify cleanliness (survey: Cheng et al., 2025, arXiv:2502.14425; *Does Data
Contamination Detection Work (Well) for LLMs?*, arXiv:2410.18966). Temporal look-ahead in
financial agents is a distinctive special case: the "leaked" content is the *future outcome of
the very series being traded*, so contamination surfaces not as a higher benchmark score but as
spurious trading profit. The closest prior work establishes the effect. **Sarkar and Vafa
(2024)** show pre-trained models exhibit look-ahead bias in return prediction and *see through*
anonymisation in long documents, so masking firm names is an incomplete remedy
([SSRN 4754678](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4754678)). **Gao, Jiang and
Yan (2025)** formalise a test of this bias, finding LLMs predict *past* moves more accurately
than future ones — a signature of memorisation, not forecasting. At the agent level, **Li et al.
(2025, *Profit Mirage*, arXiv:2510.07920)** find agent profitability degrades systematically
*past* the knowledge cutoff across Claude-3.5/GPT-4o/Grok/Llama-3.1/Qwen-2.5. On effective
cutoffs, **Cheng et al. (2024, *Dated Data*, arXiv:2403.12958)** show that a model's *effective*
cutoff diverges from its reported one because web dumps mix old content and deduplication is
imperfect — which directly explains our counterintuitive finding that the larger 32B candidates
recall *less* of the target window than the 8B treatment (effective cutoff depends on data
mixture, not parameter count).

We extend this line in three ways. **First**, rather than anonymising a text context that prior
work shows models penetrate, we remove the textual channel entirely — a **price-only, strictly
causal context** — so any foresight must originate in parametric memory. **Second**, we *select*
the treatment model **empirically via a cutoff probe** rather than trusting the model card.
**Third**, we isolate the parametric channel from regime confounds with a
**difference-in-differences** design (same model in-distribution vs out-of-distribution, relative
to a memory-free momentum baseline) — converting "performance collapses post-cutoff" into an
estimable foresight gap with an explicit no-memory counterfactual.

## 3. Methodology

### 3.1 Hypotheses

- **H1 (leakage):** the treatment, on its in-distribution window, shows a positive foresight
  signature absent in both controls.
- **H0 (null):** no foresight gap beyond capability/noise.
- **H2 (degenerate, pre-registered):** the "knowing" model does not cleanly cheat but
  *confabulates* the period, possibly degrading performance. Pre-registering H2 ensures the
  experiment can report a non-leakage outcome honestly rather than only "confirming" leakage.

### 3.2 Experimental groups

| Group                | Model                                    | Window            | Can know the window?               | Role                                    |
| -------------------- | ---------------------------------------- | ----------------- | ---------------------------------- | --------------------------------------- |
| **T-in**       | `qwen3:8b` (Qwen, 2024-aware)          | 2024-07-01…12-31 | **yes (recalls)**            | leakage candidate (family 1)            |
| **T2-in**      | `gemma3:12b` (Google, Aug-2024 cutoff) | 2024-07-01…12-31 | **nominally** (confabulates) | independent-family co-treatment (§4.9) |
| **C-A**        | `llama3.1:8b` (cutoff 2023-12)         | 2024-07-01…12-31 | no                                 | model control                           |
| **C-B / C-B2** | treatment (qwen3 / gemma3)               | 2026-01-01…05-31 | no (post-cutoff)                   | time control                            |

The **T-in vs C-B** contrast holds the model fixed and varies only whether the traded window is
in-distribution; it isolates leakage from raw capability. **C-A** corroborates with an
independent, verifiably-ignorant model. The treatment/control model-family difference is a known
confound for the C-A comparison; the within-model C-B comparison neutralises it.

**Regime confound and the difference-in-differences correction.** An adversarial review (§3.9)
identified a first-order threat: the 2024-H2 and 2026 windows differ not only in whether the
model has "seen" them but in *market regime* — notably next-day return autocorrelation
(momentum-persistent vs mean-reverting). Because the agent is partly a trailing-return trader,
its exposure-timing metric can be pushed positive or negative by regime alone, so a *raw*
in-dist−OOD gap could arise even from a memoryless agent and be misread as leakage. We therefore
do **not** interpret the raw gap as leakage. Instead we run an identical **no-memory momentum
baseline** (the MockClient) over the same windows/seeds and report the **difference-in-differences**:
`DiD = (LLM in-dist−OOD gap) − (no-memory in-dist−OOD gap)`. Leakage is supported only if
`DiD > 0` (and significant); the baseline absorbs the pure regime effect.

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

| Model              | 2024-H2 recall | Notes                                                                                                                                                                                                    |
| ------------------ | -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **qwen3:8b** | **2/4**  | correctly recalls the**NVIDIA split** and the **Aug-5 selloff** (the market-relevant facts); *refuses* the political ones via RLHF guardrails; misreports its own cutoff as "October 2023" |
| qwen2.5:7b         | 1/4            | mostly denies (keyword match on NVIDIA is a borderline false positive)                                                                                                                                   |
| llama3.1:8b        | 0/4            | clean control — denies all                                                                                                                                                                              |
| phi4               | 0/4            | denies all; self-reports "October 2023" despite a Dec-2024 release                                                                                                                                       |

The treatment is selected (by a gated rule: highest 2024-H2 score among candidates that also
deny 2026 and pass a sanity check; fails loudly otherwise). On the EC2 run we extended the probe
to 32B candidates — and the "bigger ⇒ knows more" expectation was **empirically refuted**:
`qwen2.5:32b` self-reports an *October 2022/2023* cutoff and denies 2024 entirely (older
knowledge than the 8B), and `qwen3:32b` scored only **1/4** (recalls the NVIDIA split but, by its
own account, has an ~April/July-2024 cutoff that misses the Aug-5 crash and the election).
`qwen3:8b` (**2/4** — crash *and* split) had the most verifiable target-window knowledge, so the
gate correctly selected it. *Finding:* among the available open models, parametric coverage of a
recent window is not monotonic in size — a reproducibility caution for anyone assuming a larger
model is "more contaminated."

We additionally ran **`gemma3:12b`** (Google; **officially documented Aug-2024 cutoff**) as an
independent-family co-treatment (§4.9). Critically, its *documented* cutoff covers 2024-H2 but its
*effective* recall does not: probed, it projects **"Joe Biden"** as the 2024 election winner and
misattributes the Aug-5 crash — i.e. it **confabulates** the window. This dissociation between
documented and effective cutoff (cf. *Dated Data*, §2b) is exactly why the leakage test must use
*measured recall*, not model cards.

**Finding already visible here:** parametric knowledge of the test window is *real but uneven* —
market facts surface, politically-sensitive facts are suppressed by alignment, and models'
self-reported cutoffs are simply wrong. Leakage is not all-or-nothing.

![Knowledge-probe scorecard](figures/probe_scorecard.png)

*Empirical knowledge-probe scorecard — the basis for treatment selection and the keystone of the
whole design. **`qwen3:8b`** (treatment) genuinely recalls the market-relevant facts (NVIDIA split,
Aug-5 crash) while RLHF suppresses the political ones; **`gemma3:12b`** has a documented in-window
cutoff yet **confabulates** ("Biden won the 2024 election"); the control bank (`llama3.1:8b`,
`phi4`, `qwen2.5:7b`) **denies** the window outright. This dissociation — **genuine recall ≠
documented cutoff** — is exactly why leakage must be tested against *measured* recall, not
model-card dates, and it predicts which models do (§4.1–4.8) and do not (§4.9) leak.*

### 3.5 Agent architecture

A single-agent **Reflection + ReAct** trader. At each day *T* it receives the causal context,
a one-line reflection on the prior day's P&L, and emits a single JSON object:
`{analysis, target_weights{ticker→[0,1], Σ≤1}, confidence, rationale}`. Long-only, cash =
1−Σweights. The free-text `analysis`/`rationale` are preserved verbatim as the qualitative
smoking-gun channel. The identical prompt/scaffold is used for every group; only the model and
the date window change. *Inference note:* reasoning models (qwen3) must run with thinking
disabled, else `format=json` yields empty output — itself a reproducibility hazard (Appendix B).

### 3.6 Price-only causal context (leakage isolation)

The agent sees **only** trailing prices/returns (per-ticker OHLCV-derived features and the last
15 daily returns) dated ≤ *T*, enforced by a hard causality assertion. We deliberately exclude
news. This is the methodological keystone: with a price-only feed, the agent has **no
legitimate channel** to anticipate an unsignalled event such as the Aug-5 crash, so any
pre-emptive de-risking must come from parametric memory.

**Universe & window choice.** The 7-name universe (SPY, QQQ, NVDA, TSLA, JPM, IWM, COIN) is chosen
so the two 2024-H2 event anchors have *identifiable, namable* beneficiaries — the Nov-5
"Trump trade" (banks JPM, small-caps IWM, TSLA, crypto-proxy COIN) and the AI complex (NVDA, QQQ)
— which is what makes the per-ticker forensics in §4.8 interpretable. 2024-H2 is the window
bracketed by two memorable, dated shocks (Aug-5 yen-carry crash, Nov-5 election); 2026 is the
nearest fully-out-of-cutoff window with data available at run time.

### 3.7 Metrics

Financial: total return, annualised Sharpe, max drawdown, turnover. Leakage/foresight:
(1) **next-day prescience** = corr(today's allocation, tomorrow's return), expected ≈0 absent
foresight; (2) **pre-event timing** = de-risking before a known crash / loading before a known
rally, relative to the agent's own average exposure; (3) **in-dist−OOD foresight gap** (same
model); (4) **rationale forensics** = automated scan for future-event references. A no-foresight
mock client was used to confirm the metrics return ≈0 under the null (they do).

### 3.8 Statistics

Prescience is expressed as a per-day contribution `z(signal)·z(return)` (mean ≈ Pearson r). To
avoid pseudo-replication we **average the contribution across seeds per trading day** (one
observation per day, not day×seed) before inference. We then use a **circular fixed-length block
bootstrap** for CIs and a **label-permutation test** for group differences. We report effect
sizes and 95% CIs, and treat the windows' modest sample sizes (~100–128 days) as a power
limitation rather than over-claiming significance.

**Significance of the DiD (§4.10).** The headline leakage statistic is the difference-in-
differences, so it is tested directly with a **circular block bootstrap** (block=5) of the DiD —
resampling per-day contributions in blocks to respect autocorrelation, and bootstrapping the DiD
itself rather than the regime-confounded raw in-vs-out gap. *Caveat:* the simpler group-difference
**permutation tests** (§4.4) shuffle individual days and are therefore not autocorrelation-robust
(mildly anti-conservative); they are treated as corroborative, with the block-bootstrap DiD and
the cross-model ordering carrying the primary inference.

### 3.9 Pre-run adversarial verification

Before spending any compute on results, the full design and codebase were audited by a
**multi-agent adversarial review** (7 independent reviewers — backtest mechanics, pipeline
causality, metric validity, statistical rigor, design/confounds, agent/prompt neutrality, infra —
each finding then re-checked by a skeptic instructed to *refute* it). Of 23 raw findings, 18 were
confirmed and 5 refuted (e.g. the claim that treatment-model *selection* is circular was rejected:
selecting a model that demonstrably knows the period and then testing whether it *trades on* that
knowledge is sound). Confirmed issues were fixed before the run, including: the regime-confound
DiD correction (§3.2), an off-by-one in the event-day market benchmark, seed pseudo-replication in
the bootstrap/permutation tests, parse-failure days contaminating foresight metrics, denial
phrases ("no information about the crash") being mis-counted as smoking guns, and an unsafe model
auto-selection that ignored the OOD-denial gate. The full ledger is in `report/verification_findings.md`.
This verification step is itself part of the contribution: LLM-agent experiments are easy to get
subtly wrong, and adversarial pre-registration of the analysis materially hardened the conclusions.

### 3.10 Infrastructure & cost

Developed and smoke-tested locally (Apple-Silicon ollama), then executed on **one
`g6e.xlarge` spot** instance (L40S 48 GB, ap-northeast-2, ≈$0.54/hr). Data prepared locally and
staged to S3 so the instance does no yfinance I/O; a resumable decision cache (keyed by
group·model·window·seed·date) makes spot interruption cost ≤ one decision; logs/equity/raw
outputs stream to `s3://neuroxt-personal/yhjeon/finance-ai-leakage/` every 30–60 s; the
instance self-terminates on completion. The main qwen3 run cost **< $1**; **total across all
runs** (incl. the final 7-instance 32B+8B parallel fleet) **≈ $17** (Appendix A).

### 3.11 Pivots from the baseline prompt

The assignment prompt supplied example parameters and explicit adaptation rights. Deviations:
(i) **open local models instead of GPT-4o** — yields a *real, controllable* cutoff gap at zero
API cost; (ii) **empirical treatment-model selection** rather than assuming a model knows the
window (the assumption failed for qwen3:8b on political facts); (iii) **price-only context** to
make parametric memory the sole leakage channel; (iv) **2026** as the OOD window (true
post-cutoff data now exists). Each is documented here and in `findings.md`.

## 4. Empirical Results

Run on a single `g6e.xlarge` spot instance (Seoul), 3 groups × 3 seeds, treatment auto-selected
as `qwen3:8b` (the highest verified 2024-H2 recall; the 32B candidates scored *lower* — §3.4).

![Equity by calendar date](figures/equity_ec2_bydate.png)

*Equity by calendar date (start=1.0). **Left (like-for-like, identical 2024-H2 dates):** the
treatment (green, knows the period) tracks above the control (blue) through the Aug-5 crash and
then **separates decisively at the Nov-5 election** (→1.25 vs 1.04). **Right:** the same model on
the 2026 out-of-distribution window (orange) drifts and recovers to ~1.03. The treatment pulls
away **only where it has memory** — a different model on the same dates (blue) and the same model
on unseen dates (orange) both stay roughly flat. (An ordinal-axis overlay of all three is in
`figures/equity_ec2.png`; the calendar-date split here is the honest comparison, since
the only like-for-like pair is T-in vs C-A.)*

### 4.1 Financial performance

| Group                               | Model           | Total return     | Sharpe          | Max DD  | Turnover | Parse-fail |
| ----------------------------------- | --------------- | ---------------- | --------------- | ------- | -------- | ---------- |
| **T-in** (knows 2024-H2)      | `qwen3:8b`    | **+0.251** | **+1.76** | −0.136 | 0.728    | 0          |
| C-A (model control)                 | `llama3.1:8b` | +0.043           | +0.30           | −0.152 | 0.880    | 0          |
| C-B (time control, same model, OOD) | `qwen3:8b`    | +0.032           | +0.49           | −0.131 | 0.694    | 0          |

The treatment earns a Sharpe of **1.76** on the window it was trained on, versus **0.30**
(different model, same window) and **0.49** (same model, unseen window). The advantage appears
*only* in-distribution.

*Caveats.* These Sharpe figures are **descriptive** — at ~100–128 days the standard errors are
large and we run no between-group Sharpe test; formal inference rests on the foresight/timing
metrics and the DiD (§4.4–4.5, §4.10), not on raw Sharpe. Returns are **gross of transaction
costs**; since all arms share the same daily-rebalance cost structure and comparable turnover
(0.69–0.88), the *leakage contrast* (T-in vs C-A/C-B, and the DiD) is approximately cost-invariant
even though absolute returns are not net.

### 4.2 Leakage / foresight metrics

| Group          | Ticker prescience | Exposure timing  | Conf-wtd timing  |
| -------------- | ----------------- | ---------------- | ---------------- |
| **T-in** | **+0.021**  | **+0.054** | **+0.055** |
| C-A            | −0.032           | −0.071          | −0.037          |
| C-B            | +0.016            | −0.039          | −0.045          |

Only T-in shows *positive* prescience/timing; both controls are ≈0 or negative.

### 4.3 Pre-event timing (the behavioural smoking gun)

| Group          | Aug-5 crash (de-risk > 0)                   | Nov-5 election (load > 0) |
| -------------- | ------------------------------------------- | ------------------------- |
| **T-in** | **+0.115** (de-risked into the crash) | +0.011 (≈none)           |
| C-A            | −0.125 (did*not* de-risk)                | +0.060                    |

The treatment **cut risk before the 2024-08-05 crash** — an event the probe shows it *knows* —
while the control did not. (The agent is **long-only**, so anticipation can only show up as
*reducing exposure to cash*, never shorting; the de-risk metric is thus conservative — it cannot
capture short-side foresight.) It shows **no** election-timing edge, again *consistent with the
probe*, where qwen3:8b recalled the crash but refused/did not surface the election outcome. Leakage
tracks exactly what the model demonstrably remembers.

### 4.4 Headline statistical tests (seed-averaged per-day series)

| Comparison  | Δ timing prescience | permutation p |
| ----------- | -------------------- | ------------- |
| T-in vs C-A | +0.125               | 0.075         |
| T-in vs C-B | +0.093               | 0.400         |

The cross-model contrast is marginal (p≈0.075); the within-model contrast is directionally
consistent but not significant at this sample size (~100–128 days × 3 seeds) — an
**underpowering** limitation, not evidence of absence.

### 4.5 Within-model foresight gap + regime-adjusted difference-in-differences

| Metric            | LLM gap (T-in−C-B) | No-memory baseline gap (MockClient momentum, same windows/seeds) | **DiD (leakage)** |
| ----------------- | ------------------- | ---------------------------------------------------------------- | ----------------------- |
| ticker prescience | +0.005              | −0.021                                                          | **+0.026**        |
| exposure timing   | +0.093              | −0.042                                                          | **+0.136**        |
| conf-wtd timing   | +0.100              | −0.045                                                          | **+0.145**        |

Critically, the **no-memory momentum baseline's** in-dist−OOD gap is *negative*, so the regime
difference between 2024-H2 and 2026 works *against* the finding; the **DiD is positive on every
metric**, i.e. the treatment's in-distribution foresight exceeds what regime alone explains.

![Statistical evidence: prescience by group + regime-adjusted DiD](figures/stat_evidence.png)

*Left — raw exposure-timing prescience per group (mean ± 95% bootstrap CI): positive only for the
treatment in-distribution; both controls are ≤0 (T-in vs C-A permutation p=0.075). The wide
intervals make the underpowering explicit. **Right — the difference-in-differences.** The LLM
agent's in-distribution prescience (+0.054) sits far above the **regime-only counterfactual**
(dotted; where it would land if it merely tracked the market regime like the no-memory momentum
baseline). The gap between them is the leakage **DiD = +0.136** — positive on all three metrics
(ticker +0.026, conf-wtd +0.145). This is the figure that makes the claim causal rather than
correlational: the foresight is what's left after netting out the 2024-H2-vs-2026 regime shift.*

### 4.6 Rationale forensics

The automated scan found **0** future-event "confessions" in the trading rationales (all groups).
The leakage here is **behavioural, not verbalised**: the model acts on memorised structure
(cutting risk before the crash) without naming the event in its reasoning. (Contrast the *probe*,
§3.3/§5.3, where direct questioning does elicit both recall and confabulation.)

### 4.7 Pseudo-event null for the Aug-5 de-risk (calibrated significance)

A pre-event timing score is only meaningful against a null. We compare the treatment's observed
Aug-5 de-risk to the distribution of de-risk scores at **random pseudo-event dates** within the
window (same metric, 98 valid pseudo-events):

| Group                            | Observed Aug-5 de-risk       | Null mean ± σ | Empirical p (one-sided) |
| -------------------------------- | ---------------------------- | --------------- | ----------------------- |
| **T-in** (knows the crash) | **+0.115**             | +0.001 ± 0.065 | **0.051**         |
| C-A (control)                    | −0.125 (*increased* risk) | +0.002 ± 0.109 | 0.92                    |

The treatment's de-risking into the crash is in the **top ~5%** of random-timing outcomes; the
control did the **opposite** — it carried *more* risk than usual into the crash (p=0.92, i.e. far
from any de-risk). This directly answers the "pre-event timing has no null distribution" critique.

![Aug-5 forensics: exposure timeline + pseudo-event null](figures/aug5_forensics.png)

*Aug-5 crash forensics. **Left:** seed-averaged portfolio exposure through 2024-H2 (band =
inter-seed min/max) — the treatment (green) cuts its risk dial into the Aug-5 crash. **Right:** the
significance of that move. The treatment's observed de-risk (+0.115) lands in the **top ~5%** of
the null distribution of de-risk scores at 98 random pseudo-event dates (p=0.051), whereas the
control's observed value (−0.125) sits on the **opposite** tail (p=0.92) — it carried more risk
than usual into the crash. The behaviour is both present and calibrated-significant.*

### 4.8 Per-ticker and allocation views

Beyond portfolio-level metrics, the *composition* of the treatment's book is revealing. Around
the Nov-5 2024 election (±7 days), we test the treatment-minus-control weight difference per
ticker with a permutation test (20k perms over seed×day units):

![Election-window allocation](figures/election_allocation.png)

*Mean target weight around the Nov-5 election with per-ticker permutation p-values; * = expected
Trump-trade winners. Crimson = p<0.1.*

The treatment **significantly overweights JPM** (banks; Δ=+0.088, **p=0.001** — the only ticker
to survive a 7-ticker Bonferroni correction, α/7≈0.0071). **IWM** (small caps; Δ=+0.049, p=0.077)
tilts the same way but is **only suggestive** — it does not survive Bonferroni and is not
significant even uncorrected. The other two winners (TSLA, COIN) tilt positive but not
significantly, and the non-election names show **no positive tilt** (SPY/QQQ/NVDA, all p>0.19; the
control if anything holds *more* SPY). So the election tilt is **real but narrow** — robust only
on banks (JPM), not a blanket basket bet.

This is striking because the cutoff probe shows qwen3:8b **verbally refuses** the election
question (RLHF guardrail) — yet its *allocation* still tilts significantly toward a key winner.
Leakage can be **behavioural even where verbal recall is suppressed**. (A noisier per-ticker
next-day prescience view, n≈127/ticker, is in the repo — `leakage.run.figures_extra` — and tells
the same story: the treatment's positive values concentrate on the election-sensitive names while
the control is mostly ≤0.)

### 4.9 Independent-family test (Gemma 3 12B): a non-replication that sharpens the thesis

To break the treatment/control family confound, we re-ran the full pipeline with **`gemma3:12b`**
(Google; **official Aug-2024 cutoff**, an independent architecture and training corpus) as a
second treatment, same `llama3.1:8b` control and a `gemma3:12b` 2026 OOD arm. **The clean leakage
signature did *not* replicate** — the pre-registered **H2** outcome — and the reason is
diagnostic.

**Gemma confabulates the period despite an in-window cutoff.** Its cutoff probe:
*"As of today, November 7, 2024, **Joe Biden** has been projected to win"* (wrong — and Biden was
not a candidate) and it misattributes the Aug-5 crash to *"a hotter-than-expected jobs report"*
(backwards). So although its documented cutoff covers 2024-H2, it does **not genuinely recall**
the period — unlike `qwen3:8b`, which correctly recalled the Aug-5 yen-carry crash and the NVIDIA
split.

**Cross-family comparison (treatment, in-distribution):**

| Metric                                | `qwen3:8b` (recalls 2024-H2) | `gemma3:12b` (confabulates) |
| ------------------------------------- | ------------------------------ | ----------------------------- |
| In-dist Sharpe                        | **1.76**                 | 0.75                          |
| In-dist total return                  | +0.251                         | +0.041                        |
| OOD (C-B) Sharpe                      | +0.49                          | −0.63                        |
| Exposure-timing prescience            | +0.054                         | +0.025                        |
| T-in vs C-A permutation p             | 0.075                          | **0.221 (n.s.)**        |
| Regime-adjusted DiD (exposure timing) | +0.136                         | +0.092                        |
| Aug-5 crash de-risk (pseudo-event p)  | +0.115 (p=0.051)               | +0.004 (**p=0.54**)     |

![Cross-family comparison](figures/cross_family_comparison.png)

*Both treatments, in-distribution. Gemma is same-signed but much weaker on every leakage metric.*

![Gemma equity by date](figures/equity_gemma_bydate.png)

**Reading.** Gemma's signal is *directionally* consistent (in-dist Sharpe > OOD; positive DiD)
but **weak and non-significant** (p=0.22), with **no crash de-risk** (p=0.54) and a generally
cash-heavy, low-turnover book. It does not reproduce the qwen3 signature. This is the strongest
result in the study: **a documented, in-window knowledge cutoff is *necessary but not sufficient*
for parametric leakage — what matters is whether the model *genuinely recalls* the period.**
`qwen3:8b` recalls and leaks; `gemma3:12b` confabulates and does not. The leakage we measure is
therefore **model-specific, not a mechanical consequence of the cutoff date** — which makes
per-model probing (not model-card cutoffs) the operative safeguard.

### 4.10 Per-model in-vs-out across two size tiers (8 models, EC2 fleet)

The cleanest within-model test holds the backbone fixed and varies only the *traded year* between
one **inside** the model's training cutoff and one **after** it. We ran it for **eight models —
four small (≈8–12B) and four large (≈24–32B)** — on the **identical enriched price-only context**
(§3.6, augmented with multi-horizon momentum, drawdown-from-high, distance-from-MA and a
vol-regime flag), 3 seeds, on a 7-instance EC2 spot fleet, with a no-memory momentum baseline for
the regime-adjusted DiD (circular block bootstrap, footnote ¹).

| Family  | Model                | tier | IN/OUT    | Sharpe IN | Sharpe OUT | **DiD**    | p¹            |
| ------- | -------------------- | ---- | --------- | --------- | ---------- | ---------------- | -------------- |
| Qwen3   | `qwen3:8b`         | 8B   | 2024/2026 | +1.69     | −0.23     | +0.075           | 0.26           |
| Qwen3   | `qwen3:32b`        | 32B  | 2024/2026 | +1.75     | +0.04      | **+0.133** | **0.13** |
| Qwen2.5 | `qwen2.5:7b`       | 8B   | 2023/2025 | +1.71     | +0.72      | +0.004           | 0.48           |
| Qwen2.5 | `qwen2.5:32b`      | 32B  | 2023/2025 | +2.30     | +0.68      | +0.065           | 0.23           |
| Gemma3  | `gemma3:12b`       | 12B  | 2024/2025 | +1.08     | +0.48      | +0.066           | 0.16           |
| Gemma3  | `gemma3:27b`       | 27B  | 2024/2025 | +1.56     | +0.10      | +0.090           | 0.17           |
| Mistral | `mistral-small3.2` | 24B  | 2023/2025 | +2.47     | +0.68      | −0.018          | 0.58           |
| Llama   | `llama3.1:8b`      | 8B   | 2023/2025 | +1.96     | +0.45      | −0.154          | 0.97           |

¹ One-sided block-bootstrap p for H₀: DiD ≤ 0 (no leakage); **smaller = stronger leakage
evidence** (§3.8). **All are non-significant; every DiD CI includes 0.**

![Per-model in vs out](figures/per_model_in_vs_out.png)

**Three findings.** (1) **Every model trades its in-train year at a higher Sharpe than its
out-of-train year** — but much of that is regime, which is why we regime-adjust. (2) **A
consistent within-family size effect:** in all three families with both tiers, the *larger* model
has the *larger* in-vs-out DiD — qwen3 8B→32B (+0.075→**+0.133**), qwen2.5 7B→32B
(+0.004→+0.065), gemma3 12B→27B (+0.066→+0.090). So a more *competent* agent trades on its
parametric memory **more** effectively, not less — addressing the concern that an 8B trader is
"too weak" to reveal leakage. (3) `qwen3:32b` shows the **largest DiD overall** (+0.133, p=0.13).
Counterexamples remain (`llama3.1:8b` and `mistral` are ≈0/negative), and — crucially — **nothing
is individually significant**; the evidence is the *direction and the within-family monotonicity*,
not per-model p-values.

**Window-sensitivity caveat (the honest wrinkle).** On the 2024-**H2**-specific headline window
(§4.1–4.5), with a *same-family* control (`qwen2.5:32b`), `qwen3:32b`'s DiD was **negative**
(−0.05) — the opposite of its +0.133 on *full-year* 2024. The reconciliation: `qwen3:32b`'s
*effective* cutoff (~mid-2024) covers 2024-**H1** well but not the Aug-5/Nov-5 **H2** events, so
its leakage surfaces over full-year 2024 (which includes the known H1) but not the H2 sub-window —
whereas `qwen3:8b`, which *does* recall the Aug-5 crash, leaks specifically on H2. **Leakage is
sensitive to whether the model's effective cutoff covers the exact traded sub-window** — a further
argument for measuring recall per window, not trusting a single cutoff date.

## 5. Discussion & Forensic Analysis

### 5.1 The cutoff probe is itself forensic evidence

Before a single trade, the probes establish the experiment's validity and reveal the *texture*
of leakage: the control is verifiably blind to 2024-H2, while the treatment demonstrably recalls
the market events the backtest hinges on. This direct, quotable evidence is stronger than
relying on vendor-stated cutoffs.

### 5.2 Backtest forensics — verdict: moderate, consistent support for H1

The evidence triangulates:

1. **Financial.** The treatment's edge (Sharpe 1.76) materialises *only* in-distribution — not
   for a different model on the same window, nor for the same model on an unseen window.
2. **The knowledge↔behaviour match is the strongest signal.** The treatment cut risk before the
   Aug-5 crash (pre-event timing +0.115 vs the control's −0.125) — and the probe shows it *knows*
   that crash. It shows *no* election-timing edge — and the probe shows it does *not* know the
   election outcome. The behaviour mirrors the model's measured memory item-by-item; a generic
   "smarter model" or a regime artifact would not produce this selective pattern.
3. **Regime is accounted for (not the explanation).** The DiD against a no-memory momentum agent
   is positive on every metric, and the baseline's own in-dist−OOD gap is negative — so the
   2024-vs-2026 regime works *against* the result, not for it. (The DiD is positive but its CI
   still includes 0, so regime is netted out, not a *significant* effect "ruled out".)
4. **Cross-family dependence (pre-registered H2 on family 2).** The signature did **not** cleanly
   replicate on `gemma3:12b` (§4.9): in-dist Sharpe 0.75, no crash de-risk (p=0.54), weak/ns DiD.
   The reason is diagnostic — Gemma *confabulates* 2024-H2 (projects "Biden" as winner) despite an
   official Aug-2024 cutoff. So the leakage is **model-specific**: present where recall is genuine
   (Qwen3), absent where the model confabulates (Gemma). An in-window cutoff alone predicts neither.
5. **Honesty about strength.** Significance is marginal (cross-model p≈0.075) to non-significant
   (within-model p≈0.40) at this sample size. We therefore report *moderate* support, not proof,
   and identify a multi-period within-backbone design (more in-dist/OOD windows, averaging over
   regimes) as the natural power-increasing follow-up.

### 5.3 Confabulation can be worse than memorisation

A striking qualitative finding: probed about Q1-2026 (beyond any model's cutoff), qwen3:8b did
not say "I don't know" — it **invented** a specific, confident, false event ("a sharp decline
following the Federal Reserve's unexpected 50bps hike in early March 2026"). For a trading
agent, confident confabulation of the future is at least as dangerous as accurate memorisation:
the former produces unfalsifiable, plausible-sounding rationales that can drive real positions.

### 5.4 Threats to validity (and how they were addressed)

| Threat                                                            | Mitigation in this study                                                                                                                         |
| ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Pipeline leakage** (future data in the agent's feed)      | Price-only causal context + hard causality assertion; the only foresight channel is parametric memory.                                           |
| **Regime confound** (2024 vs 2026 differ in dynamics)       | Difference-in-differences vs a no-memory momentum baseline; the baseline's own gap is*negative*.                                               |
| **Model-family/capability confound** (treatment vs control) | Within-model in-dist vs OOD contrast (same backbone); independent-family co-treatment (§4.9); 8-model in-vs-out sweep (§4.10).                 |
| **Treatment-selection circularity**                         | Selecting a model that*knows* the period then testing whether it *trades on* it is sound (independently affirmed by the adversarial review). |
| **Lucky seed / cherry-picking**                             | ≥3 seeds; inter-seed band on the exposure figure; pseudo-event null.                                                                            |
| **Spurious pre-event timing**                               | Calibrated empirical p-value via 98 random pseudo-events (§4.7).                                                                                |
| **Self-reported cutoffs unreliable**                        | Empirical cutoff + price-recall probes, not model cards.                                                                                         |
| **Implementation bugs**                                     | 7-dimension adversarial multi-agent review; 18 confirmed issues fixed pre-run (§3.9,`verification_findings.md`).                              |
| **Parse-failure contamination**                             | Parse-fail days carried forward for equity but excluded from foresight metrics;`n_parse_fail = 0` on the final run.                            |

### 5.5 Limitations

- **Statistical power.** ~100–250 trading days per cell; the cross-model contrast is marginal
  (p≈0.075) and the single-pair within-model contrast is non-significant (p≈0.40). We report
  *moderate* support; the 8-model in-vs-out sweep (§4.10) is the power/robustness remedy.
- **Family dependence (tested).** We ran an independent-family co-treatment (`gemma3:12b`,
  official Aug-2024 cutoff, §4.9). It did **not** reproduce the clean signature — it *confabulates*
  the period rather than recalling it (pre-registered H2). So the leakage is **demonstrated on
  Qwen3 but is not a universal property of any in-window cutoff**; characterising *which*
  families/training mixtures yield genuine recall (and hence leakage) vs confabulation is the
  natural follow-up. We do **not** use a proprietary API model (GPT-4o/Claude) as it would forfeit
  the controllable-cutoff, reproducible, zero-cost design.
- **Small open models confabulate**, so absence of explicit rationale tells is expected; leakage
  here is behavioural. Behaviour-independent membership-inference/perplexity probes are future work.
- **Single asset universe / daily cadence / long-only.** Generalisation to other universes,
  intraday horizons, and short-selling is untested.

## 6. Conclusion & Robust-Backtesting Standards

Parametric look-ahead leakage is a first-class threat to LLM-agent backtests, invisible to the
chronological-split discipline the field currently relies on. It is, however, **model-specific**:
of two open models with in-window cutoffs, one (Qwen3 8B) genuinely recalled the period and traded
on it, while the other (Gemma 3 12B) *confabulated* it and showed no clean leakage — so a
documented cutoff predicts neither. This makes per-model probing non-negotiable. We recommend:

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

All code, prompts, the causality guard, the cutoff/selection/price-recall probes, metrics,
statistics, and the EC2 infra (spot launch, bootstrap, self-terminate) are in the repository.
Run the full pipeline via `infra/stage.sh` + `infra/launch_spot.sh`, or locally via
`python -m leakage.run.main`.

**Load-bearing settings (fix these to reproduce):**

- **Models:** treatments `qwen3:8b` **and `gemma3:12b`** (independent-family co-treatment, §4.9),
  control `llama3.1:8b` (ollama 0.30.8); Qwen treatment auto-selected by the gated probe among
  `{qwen3:32b, qwen3:8b}`; Gemma run via `LEAKAGE_FORCE_TREATMENT=gemma3:12b` (artifacts in
  `results/gemma/`).
- **Inference:** `format=json`, **`think=False`** (mandatory for qwen3 — see Appendix B),
  `temperature=0.7`, `num_predict=700`, `seed ∈ {0,1,2}`; 600 s client timeout for 32B cold-loads.
- **Universe:** SPY, QQQ, NVDA, TSLA, JPM, IWM, COIN; daily rebalance; long-only; 60-day trailing
  context; start equity 1.0.
- **Windows:** main run — in-dist 2024-07-01…12-31, OOD 2026-01-01…05-31. Per-model sweep (§4.10)
  — full calendar years 2023 / 2024 / 2025 and 2026-01…06, paired per model (IN inside cutoff,
  OUT after). Prices via `yfinance` (auto-adjusted), cached to parquet (needs `pyarrow`).
- **Infra:** 1× `g6e.xlarge` spot (ap-northeast-2), DL Base GPU AMI, IAM instance profile with S3
  + SSM, `instance-initiated-shutdown-behavior=terminate` + EXIT-trap self-terminate; artifacts to
    `s3://neuroxt-personal/yhjeon/finance-ai-leakage/`. Total compute cost ≈ $17 across all EC2 spot runs (qwen3 main ≈ $0.8 incl. 3 aborted boots
    that each self-terminated cleanly; Gemma co-treatment ≈ $0.6; the 8-model two-tier size sweep on a 7-instance spot fleet (§4.10) ≈ $15.5).
- **Stats:** seed-averaged per-day prescience contributions; circular block bootstrap + permutation
  tests; pseudo-event null with 98 random pseudo-events.

### Appendix B — The inference-config hazard

qwen3 is a hybrid reasoning model; under `format=json` with thinking enabled it emits empty
content (the token budget is consumed by suppressed `<think>` tokens). The fix (`think=False`)
changed the parse-success rate from 0% to 100%. We flag this as a concrete, easily-missed
reproducibility hazard for LLM-agent backtests.

### Appendix C — Agent prompt & reflection (verbatim)

The **identical** system prompt is used for every group; only the model and the date window vary.
This is the load-bearing "prompt neutrality" claim — it never hints the date is historical, names
no future event, and does not instruct the model to use outside knowledge:

```
You are a disciplined portfolio manager making a single daily allocation decision for a
long-only equity portfolio. You may allocate across the listed tickers and hold the rest in cash.

You are given ONLY the market data shown in the user message (trailing prices and returns as of
today's close). Base your decision on that information and sound risk management. Do not assume
access to any information beyond what is shown.

Respond with a SINGLE JSON object and nothing else, with exactly these fields:
{ "analysis": "<2-3 sentences ...>", "target_weights": {"TICKER": <float in [0,1]>, ...},
  "confidence": <float in [0,1]>, "rationale": "<one short sentence>" }

Constraints: weights are fractions of total portfolio value, each in [0,1], sum <= 1.0 (remainder
is cash). Use only tickers from the provided universe.
```

The per-day **user** message is the rendered causal context (§3.6) — trailing prices/returns ≤ T
— optionally prefixed by a one-line **reflection** on the prior day's realised P&L
(`"Portfolio gained/lost X% in the most recent session (you were Y% invested)."`). The reflection
carries no look-ahead (it reports an already-realised outcome). Reasoning models (qwen3) are run
with `think=False` (Appendix B).
