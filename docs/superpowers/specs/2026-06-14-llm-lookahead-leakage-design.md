# Design Spec — Look-ahead Leakage in Open-Source LLM Trading Agents

**Date:** 2026-06-14
**Author:** Younghoon Jeon (전영훈) · Korea University AutoAI Lab · Financial AI seminar
**Status:** Approved (design-gate passed); autonomous build in progress.
**Source assignment:** "Autonomous Financial AI Data Leakage Research & Experimentation Agent"
master prompt (treated as spec; its concrete numbers are examples and may be re-set under the
prompt's explicit *adaptation rights*).

---

## 1. Research question & hypotheses

**RQ.** Does an LLM trading agent whose pre-training data *covers* a backtest window gain
an unfair, non-causal advantage ("look-ahead bias / cheating") relative to (a) a model that
cannot have seen the window, and (b) itself on a window it cannot have seen?

- **H1 (leakage).** Treatment-in-dist shows a positive *foresight signature* (anomalous
  pre-event timing, positive next-day prescience) absent in Control A and Control B.
- **H0 (null).** No foresight gap beyond what model capability/noise explains.
- **H2 (degenerate, pre-registered alternative).** The "knowing" model does *not* cheat but
  instead hallucinates memorized-but-stale narratives, degrading performance. If observed,
  this is reported as the finding (a negative/forensic result), not hidden.

Pre-committing to H2 prevents the experiment from only being able to "confirm" leakage.

## 2. Experimental groups

Real knowledge-cutoff gap (not prompt-simulated) is the central methodological choice.

| Group | Model | Period | Can know period? | Role |
|---|---|---|---|---|
| **T-in** Treatment in-distribution | post-2024 model (`qwen3:8b` cand.) | 2024-07-01 … 2024-12-31 | **Yes** | leakage candidate |
| **C-A** Model control | `llama3.1:8b`, cutoff 2023-12 | 2024-07-01 … 2024-12-31 | No | "no future knowledge" baseline |
| **C-B** Time control | same as T-in | 2026-01-01 … 2026-05-31 | No (post-cutoff) | isolates leakage from skill |
| *(opt.)* **Ladder** | `qwen2.5:7b` (2024 straddle) | 2024-H2 | partial | robustness appendix |

**Confound handling.** T-in vs C-A differs in both *cutoff* and *model family* → capability
confound. The **T-in vs C-B** contrast uses the *same model* and differs only in whether the
traded window is in-distribution → any foresight gap there is attributable to memorization,
not capability. C-A remains as an independent corroboration. Model cutoffs are **verified
empirically in Phase 1** (probe each model with dated factual questions) and the pair is
locked or swapped accordingly.

## 3. Agent architecture

Grounded in a Phase-1 audit of `other_students/` class papers (FinAgent, FINCON, TradeMaster,
QuantAgents, FLAG-TRADER, CLER, …). Dominant paradigms expected: **reflection**, **ReAct-style
reasoning + tool use**, **structured decisions**, **multi-agent debate**.

Implemented agent = **minimal single-agent Reflection+ReAct trader** (intentionally simple so
the leakage signal is not masked by orchestration):

- **Input (day T):** trailing price/return window per ticker, optional dated news headlines,
  current portfolio state. Context is strictly *causal* — only data dated ≤ T is shown.
- **Reason:** short ReAct-style scratchpad + a one-line reflection on the prior day's P&L.
- **Output (structured):** JSON `{action: BUY|SELL|HOLD, ticker, target_weight ∈ [0,1], confidence ∈ [0,1], rationale: str}`.
- **Rationale is preserved verbatim** — it is the qualitative smoking-gun channel.

The same prompt/scaffold is used across all groups; only the model and the date window change.

## 4. Universe, cadence, controls

- **Universe (~6–8):** `SPY, QQQ, NVDA, TSLA` + a few names sensitive to H2-2024 events
  (e.g. `JPM`, `IWM`, `COIN`/`DJT` for the election trade). Final list fixed in Phase 2.
- **Events bracketing the window:** **2024-08-05** yen-carry-unwind crash (VIX spike) and
  **2024-11-05** US election "Trump trade" rally — memorable, exploitable-if-known shocks.
- **Cadence:** daily (~128 trading days/window) so single-day shock-dodging is observable.
- **Seeds:** ≥3 on headline comparisons for error bars (temperature > 0).
- **Causality guard:** an automated check asserts no datum dated > T enters any prompt.

## 5. Metrics

**Financial:** Total Return, annualized Sharpe, Max Drawdown, turnover (per group/seed).

**Leakage / foresight (the contribution):**
1. **Pre-event timing score** — net de-risking in the k days before a known drawdown / net
   loading before a known rally, vs. baseline and vs. controls.
2. **Next-day prescience** — corr(signed position·confidence, realized next-day return).
   Expected ≈0 for C-A/C-B, suspiciously >0 for T-in under H1.
3. **In-dist − OOD foresight gap** — (2) and (1) for T-in minus the same for C-B (same model).
4. **Rationale forensics** — automated scan for future-dated references / event names the
   model should not "know" at day T; counts + curated excerpts.

**Statistics:** stationary bootstrap and permutation tests on group differences in Sharpe and
in foresight scores; report effect sizes + 95% CIs, not just p-values. Multiple-comparison
note included.

## 6. Infrastructure & cost plan

- **Local first:** full pipeline developed and smoke-tested on the Mac (ollama on Apple
  Silicon, small model) so EC2 runs exactly once, clean.
- **Data staging:** prepare datasets locally → upload to `s3://neuroxt-personal/yhjeon/finance-ai-leakage/` → EC2 pulls. Minimizes EC2 wall-clock.
- **Compute:** single **`g5.xlarge` spot** (A10G 24 GB; ~$0.40–0.60/hr spot). Fallback
  `g4dn.xlarge` spot if cheaper/available. Models run 4-bit via ollama.
- **Resilience:** idempotent decision cache keyed by `(group, model, period, date, seed)` →
  resumable across spot interruptions; continuous S3 streaming of logs/equity/raw outputs;
  **self-terminate** on completion + CloudWatch idle backstop.
- **Estimated runtime ≤ ~1 hr → total cost < ~$2.**
- **Gates (honored even in autonomous mode):** (1) confirm instance type + live cost estimate
  before launch; (2) any scaling beyond the single instance needs fresh approval.

## 7. Deliverables

1. GitHub repo `finance-ai-lookahead-leakage` (private, `yhjeon-nxt`) — code, infra, results.
2. Reproducible pipeline (local smoke + EC2 full run).
3. **Publication-grade research report** (`report/`): Abstract · Introduction & Lecture
   Alignment · Methodology (with documented pivots) · Empirical Results · Discussion &
   Forensic Analysis · Conclusion & Robust-Backtesting Standards.

## 8. Pivots log

Material deviations from the master prompt's example parameters and from this spec are
recorded in `PROGRESS.md` / `findings.md` and summarized in the report's Methodology section.
Initial pivots from the prompt's examples:

- **Models:** open-source local (ollama) instead of GPT-4o API — per chosen "all-local" mode;
  yields a *real* cutoff gap and zero API cost.
- **Window:** kept 2024-H2 (well-bracketed by two marquee events) but explicitly re-derived
  from the model cutoffs rather than assumed.
- **OOD control:** 2026 Jan–May (true post-cutoff data available as of the run date).
