# Web-grounded review — synthesis (workflow wf_da972855-71e)

Both anchor citations verify. I have enough to consolidate. The key grounding facts confirmed: multiperiod run is barely started (no eval JSON, only 2024H1 seed0 log), decision logs exist for all groups/seeds plus mock-momentum, n_parse_fail=0 everywhere, DiD positive (+0.136 exposure timing), within-model permutation p=0.40 (weak), cross-model p=0.075. Here is the consolidated plan.

---

# Consolidated Action Plan — Finishing the Leakage Report

Grounding check (verified against artifacts): T-in Sharpe 1.76 vs C-A 0.30 / C-B 0.49; DiD +0.136 (exposure timing), +0.145 (conf-weighted); within-model permutation p=0.40 (weakest link), cross-model p=0.075; `n_parse_fail=0` for all groups; full per-day decision logs + mock-momentum logs on disk for all 3 seeds; **multiperiod run is barely started** (only `2024H1_qwen3-8b_seed0.jsonl` exists, no `eval_multiperiod.json`). The "bigger≠more-contaminated" finding (qwen3:32b 1/4, qwen2.5:32b 0/4 vs qwen3:8b 2/4) is in `model_selection.json`.

## Ranked actions (value/effort), each tagged DO-NOW / OPTIONAL / SKIP

**Tier 1 — DO-NOW, zero new compute, high value (pure post-processing of data on disk)**

1. **DO-NOW — Date-conditioned PRICE-RECALL probe** (contamination-methods #1). The single most decisive missing piece: it measures the thing the whole paper *assumes* (weights store 2024-H2 specifics) instead of inferring it from trades. Reuses the `cutoff_probe._ask()` harness; ~15 Q × 3 cells × seeds. Directly mechanistically explains the pre-event de-risk. Highest value/effort in the entire set.
2. **DO-NOW — Per-event exposure timeline figure** (report-enhancements #1). The behavioural smoking gun, currently two numbers. Seed-averaged exposure with min/max band, Aug-5 + Nov-6 lines. Pre-empts "lucky seed." Built from logs already present.
3. **DO-NOW — Prescience/timing bar chart with bootstrap CIs** (report-enh #2). CIs already in `eval_ec2.json`; visualizing them *is* the "moderate, underpowered" honesty argument.
4. **DO-NOW — Cutoff-probe scoreboard table** (report-enh #3). Turns the central validity claim + the "bigger knew less" finding into one auditable table. Data in `model_selection.json`/`cutoff_probe.json`.
5. **DO-NOW — Pre-event timing pseudo-event null** (report-enh #6). Directly kills verification finding #4 (HIGH): the single damning critique that Aug-5 de-risk "has no null distribution." 2000 random pseudo-events → empirical p. Pure post-processing.
6. **DO-NOW — Consolidated Threats to Validity §5.4** (report-enh #5). Converts the 7-reviewer adversarial audit into visible credit; publication-grade expectation; raw material already in `verification_findings.md`.
7. **DO-NOW — Reproducibility appendix** (report-enh #13). The `think=False` flag, seeds, universe, env vars are load-bearing and currently absent. Difference between "code available" and "reproducible." All in `config.py`/infra.
8. **DO-NOW — Explicit Limitations §5.5** (report-enh #11) + **parse-integrity note** (report-enh #14, fold in: `n_parse_fail=0` closes finding #5 in one sentence). Low effort, signals maturity.

**Tier 2 — DO-NOW, light new compute (now feasible — ollama v0.30.8 exposes logprobs, verified)**

9. **DO-NOW — Min-K% Prob membership inference** (contamination-methods #2) **+ calibrated perplexity gap** (#3, same code path, near-free marginal). A model-internal, behaviour-independent contamination signal with an AUROC — far more defensible than QA alone, and the project currently has *zero* direct parametric-memory measurement. Do these two together; perplexity gap is essentially free once the Min-K% harness exists. Frame both as corroborating (cite Duan et al. on MIA power limits).

**Tier 3 — OPTIONAL (do if time remains; clear value but lower than Tier 1–2)**

10. **OPTIONAL — Multi-period DiD figure/table slot §4.7** (report-enh #4). Pre-build the placeholder now, populate when the run lands. *Reality check:* run is at 2024H1 seed0 only — do NOT block the report on it. Add the slot + "pending" note; ship without it if it doesn't finish.
11. **OPTIONAL — Add Gemma 3 12B as independent-family co-treatment** (model-strategy #1). This is the single most inference-strengthening *new model* (breaks the family confound that the p=0.40 within-model contrast can't), official Aug-2024 cutoff, fits the box, near-zero infra cost. Tagged OPTIONAL not DO-NOW only because it needs a fresh EC2 run (gated on user approval per global instructions) and a probe-first gate. **If you green-light any new compute, this is the one to do.**
12. **OPTIONAL — DiD slope figure** (report-enh #9), **Sharpe CIs** (report-enh #7), **per-seed dispersion** (report-enh #8), **guided-vs-general completion probe** (contamination #4), **scored contamination quiz** (contamination #5). All low-effort hardening on existing data/harness; bundle whichever fit. Sharpe-CI and per-seed dispersion both reinforce the underpowered framing cheaply.
13. **OPTIONAL — Ethical Considerations §7** (report-enh #12). Content-rich here (dual-use + confabulation retail harm). Add if targeting a venue that expects it; otherwise skippable for a homework report.

**Tier 4 — SKIP (explicitly reject in text where noted)**

14. **SKIP — Mistral Small 3.2 24B third-family control** (model-strategy #2). A control is less informative than a co-treatment; only worth it *if* Gemma is added, and even then it's symmetry-hardening, not a result. Adds compute for marginal defensive value.
15. **SKIP — cutoff ladder on documented dates** (model-strategy #3). Would contradict the paper's own thesis (documented cutoffs are unreliable). Keep the honest probe-score-graded contamination axis instead — and *say so* in the methods. (This is guidance, not an action.)
16. **SKIP — Llama 4 Scout / DeepSeek V3** (model-strategy #4). Infeasible on 48GB L40S; mention as a scale-up future-work line only.
17. **SKIP — proprietary API treatment (GPT-4o/Claude)** (model-strategy #5). Breaks the controllable-cutoff, reproducible, zero-API-cost contribution. *Explicitly reject* it in Limitations/Future Work rather than implement.
18. **SKIP — CDD output-distribution detectors** (contamination #6). Near-chance on 8B models (cite Omer et al. 2026). Note as deliberately-excluded in methods.
19. **SKIP — membership-inference as primary evidence framing.** Use behavioural foresight gaps as primary; MIA/perplexity corroborate (pre-empts the "why not MIA?" reviewer question per literature cluster E).

**Conflict resolved:** model-strategy treats Gemma as "highest value addition"; report-enhancements + contamination treat on-disk analysis as highest. I rank the on-disk analyses (Tier 1) and the now-feasible local logprob probes (Tier 2) *above* Gemma, because they are zero-gate, zero-cost, and the report is the deliverable — Gemma requires a fresh gated EC2 run. Gemma remains the top *new-model* choice if compute is approved.

---

## (1) Add models? Final answer: **NO for the report as-is; YES (one model) only if you approve a fresh EC2 run.**

The report can ship strongly on existing data + on-disk probes. If you authorize new compute, add exactly one:

| Add? | Model | Official cutoff | Role | Source |
|---|---|---|---|---|
| **YES (if compute approved)** | **Gemma 3 12B** (27B if box allows) | **August 2024** (Google model card, all sizes) | 3rd-family in-dist **co-treatment** — breaks the family confound the within-model p=0.40 contrast can't. Probe-gate first; promote only if it recalls Aug-5 crash. | [model card](https://ai.google.dev/gemma/docs/core/model_card_3), [ollama](https://ollama.com/library/gemma3) |
| Optional only-if-Gemma-added | Mistral Small 3.2 24B | October 2023 | 3rd-family co-control (symmetry) | [HF discussion](https://huggingface.co/mistralai/Mistral-Small-3.2-24B-Instruct-2506/discussions/11) |
| No | Llama 4 Scout / DeepSeek V3 | Aug 2024 / Jul 2024 (latter third-party) | Too large for 48GB L40S; scale-up future work only | [Llama4 card](https://www.llama.com/docs/model-cards-and-prompt-formats/llama4/) |
| No | GPT-4o / Claude (API) | vendor-stated, shifts per snapshot | Breaks reproducibility/control/zero-cost pillars — reject in text | — |
| No | cutoff ladder by documented date | n/a | No open model documented cleanly between 2024-H2 and 2026; contradicts own thesis | [HaoooWang cutoff list](https://github.com/HaoooWang/llm-knowledge-cutoff-dates) |

**Decision rule:** Gemma 3 12B is the one materially inference-improving addition at near-zero marginal cost — but it needs the gated EC2 launch and a probe-first promotion gate. If you do not want another run, the report stands without it; do NOT add any others.

## (2) Ready-to-paste Related Work paragraph (real citations, verified)

> **Related Work.** Our study sits within the literature on training-data contamination, where exposure to evaluation content during pretraining inflates measured capability and detection methods struggle to certify cleanliness (survey: Cheng et al., 2025, arXiv:2502.14425; *Does Data Contamination Detection Work (Well) for LLMs?*, arXiv:2410.18966). Temporal look-ahead in financial agents is a distinctive special case: the "leaked" content is the future outcome of the very series being traded, so contamination surfaces not as a higher benchmark score but as spurious trading profit. The closest prior work establishes this effect. Sarkar and Vafa (2024) show pretrained models exhibit lookahead bias in return prediction and, critically, *see through* anonymization in long documents, so masking firm names is an incomplete remedy ([SSRN 4754678](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4754678)). Gao, Jiang and Yan (2025) formalize a test of this bias, finding LLMs predict *past* moves more accurately than future ones — a signature of memorization, not forecasting ([A Test of Lookahead Bias in LLM Forecasts](https://www.researchgate.net/publication/399275870_A_Test_of_Lookahead_Bias_in_LLM_Forecasts)). At the agent level, Li et al. (2025, *Profit Mirage*, arXiv:2510.07920) use counterfactual temporal evaluation across Claude 3.5, GPT-4o, Grok, Llama 3.1 and Qwen 2.5 and find profitability degrades systematically past the knowledge cutoff, while TradeTrap (2025, arXiv:2512.02261) shows such agents report rationales that do not survive perturbation. On mitigation, DatedGPT (2026, arXiv:2603.11838) trains cutoff-respecting models from scratch. Our knowledge-cutoff probe is motivated by Cheng et al. (2024, *Dated Data*, arXiv:2403.12958), who show *effective* cutoffs diverge from reported ones because CommonCrawl dumps mix old content and deduplication is imperfect — which also explains our counterintuitive finding that larger 32B candidates recall *less* of the target window than the 8B treatment, since effective cutoff depends on data mixture rather than parameter count. We extend this line in three ways. First, rather than anonymizing a text context prior work shows models penetrate, we remove the textual channel entirely — a price-only, strictly causal context — so any foresight must originate in parametric memory. Second, we *select* the treatment model empirically via a cutoff probe instead of trusting the model card. Third, we isolate the parametric channel from regime confounds with a difference-in-differences design benchmarking the same model in-distribution (2024-H2) against out-of-distribution (2026) relative to a memory-free momentum baseline — converting "performance collapses post-cutoff" into an estimable foresight gap with an explicit no-memory counterfactual.

(All URLs above verified live via search this session; the verbatim per-cluster suggested_text in the literature findings can be pasted as a longer multi-paragraph version if more space is available.)

## (3) Top report additions worth doing now (the 3–5 to ship first)

1. **Date-conditioned price-recall probe** — directly measures parametric memory of 2024-H2 (the assumption the whole paper rests on); control + OOD as built-in negatives. *Highest value/effort.*
2. **Per-event exposure timeline figure (§4.3)** — the smoking gun, visualized with an inter-seed band that kills the lucky-seed objection.
3. **Pre-event-timing pseudo-event null (§4.x)** — gives the Aug-5 de-risk (+0.115) a calibrated p-value, neutralizing the single most damaging open critique (verification finding #4).
4. **Cutoff-probe scoreboard table (§3.4)** — makes the validity argument *and* the "bigger≠more-contaminated" result auditable at a glance.
5. **Threats to Validity §5.4 + Limitations §5.5 + Reproducibility appendix** — the publication scaffolding the draft omits; together they convert the existing adversarial audit and config into visible rigor at low effort. (Fold the one-line `n_parse_fail=0` note into §5.5.)

Min-K%/perplexity probes (Tier 2) are the strong sixth if you want a quantitative internal contamination metric beyond QA.

Key files: report `/Users/jeon-younghoon/Desktop/lectures/finance_ai/homework/report/research_report.md`; eval `/Users/jeon-younghoon/.../results/eval_ec2.json`; probe data `/Users/jeon-younghoon/.../results/model_selection.json` and `/results/cutoff_probe.json`; decision logs `/Users/jeon-younghoon/.../results/ec2/decisions/*.jsonl` (all groups/seeds + mock-momentum present); stats lib `/Users/jeon-younghoon/.../src/leakage/metrics/stats.py`; multiperiod (still running, no eval JSON) `/Users/jeon-younghoon/.../results/multiperiod_run.log`.