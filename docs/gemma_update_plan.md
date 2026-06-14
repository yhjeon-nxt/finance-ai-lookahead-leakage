# Gemma 3 12B result — pre-staged update plan

Prepared by 3 parallel scout agents while the Gemma run executes. Goal: when `results/gemma/`
lands, application is (near) one command. Two outcomes are pre-written:
- **A = REPLICATES** (in-dist Sharpe edge + Aug-5 de-risk p≲0.05 + DiD>0)
- **B = DOES NOT** (confabulates / no signal) → the pre-registered **H2**.
Decide A vs B from `eval_gemma.json` (T-in Sharpe > C-B Sharpe? prescience>0?) + Aug-5 pseudo p + DiD sign.

## STEP 1 — pull results + run analyses (scripts already env-parameterized; ec2 output unchanged)
```bash
cd /Users/jeon-younghoon/Desktop/lectures/finance_ai/homework
aws s3 sync s3://neuroxt-personal/yhjeon/finance-ai-leakage/gemma/results/ results/gemma/
cp results/gemma/eval_gemma.json results/eval_gemma.json        # top-level, for report_tables/figures
export PYTHONPATH=src LEAKAGE_RUN_TAG=gemma LEAKAGE_TREATMENT_MODEL=gemma3:12b LEAKAGE_CONTROL_MODEL=llama3.1:8b
python -m leakage.run.report_tables gemma > report/section4_results_gemma.md
python -m leakage.run.redraw_equity        # -> results/figures/equity_gemma_bydate.png
python -m leakage.run.figures_extra        # -> *_gemma.png (ticker/election/timing)
python -m leakage.run.extra_analyses       # -> exposure_timeline_2024H2_gemma.png, pseudo_event_null_gemma.json
unset LEAKAGE_RUN_TAG LEAKAGE_TREATMENT_MODEL LEAKAGE_CONTROL_MODEL
python -m leakage.run.compare_runs         # -> report/cross_family_comparison.md + figures/cross_family_comparison.png
cp results/figures/*_gemma*.png results/figures/cross_family_comparison.png report/figures/
```
(Price-recall probe for Gemma needs a live `gemma3:12b` ollama — optional, run locally:
`LEAKAGE_TREATMENT_MODEL=gemma3:12b python -m leakage.run.price_recall_probe`.)

## STEP 2 — report patches (`report/research_report.md`)  [anchors verbatim]
- **Abstract design sentence** — anchor `(a 2024-aware model selected by probe)` → "two independent-family treatments (`qwen3:8b` and `gemma3:12b`)".
- **Abstract "Headline result."** — anchor `Support for parametric leakage is **moderate and internally consistent**` → append A: "and **replicates on an independent family (gemma3:12b)**"; B: "but **did not replicate on gemma3:12b** (confabulation; pre-registered H2)".
- **§3.2 groups table** — anchor `| **T-in** | treatment (2024-aware) |` → add T2-in (gemma3:12b, 2024-H2) + C-B2 (gemma3:12b, 2026) rows.
- **§3.2 confound sentence** — anchor `the within-model C-B comparison neutralises it.` → add: independent-family co-treatment (gemma3:12b) breaks the family confound directly.
- **§3.4** — anchor `**Finding already visible here:**` → add gemma3:12b probe row + 1 sentence (promoted to co-treatment; results §4.9).
- **NEW §4.9** — insert immediately before anchor `## 5. Discussion & Forensic Analysis`. Use the financial/foresight/DiD/election tables + `figures/equity_gemma_bydate.png`. Pick outcome A or B template (see scout-A output).
- **§4.4** — anchor `an **underpowering** limitation, not evidence of absence.` → if A, note cross-family corroboration.
- **§5.2 verdict** — anchor `### 5.2 Backtest forensics — verdict: moderate, consistent support for H1` → add cross-family item (A: replicated; B: family-dependent, H2 on family 2).
- **§5.4 threats table** — anchor `| **Model-family/capability confound** (treatment vs control) |` → cell: A "+ gemma3:12b co-treatment reproduces it (§4.9)"; B "co-treatment run, did NOT reproduce — leakage family-specific".
- **§5.5 Limitations "One treatment family"** — anchor `- **One treatment family.** Results rest on the Qwen3 backbone;` → A: rewrite to *resolved* (two families); B: reframe to *family-dependent non-replication*.
- **§6 Conclusion opening** — anchor `Parametric look-ahead leakage is a first-class threat` → A: "replicates across two families"; B: "is model-specific (Qwen3 leaked, Gemma confabulated)".
- **Appendix A models line** — anchor ``treatment `qwen3:8b`, control `llama3.1:8b` `` → add gemma3:12b co-treatment + `results/gemma/` artifacts.
- Embed `figures/cross_family_comparison.png` near §4.9.

## STEP 3 — meta/tracking patches
- **PROGRESS.md** — flip Phase row to ✅ Gemma done (rc, self-terminate, +cost); add dated log entry w/ replication verdict; bump cost total.
- **findings.md** — add gemma3:12b cutoff fact (Aug-2024, independent family) + a "Gemma co-treatment findings" entry w/ verdict; add pivot bullet.
- **README.md** — anchor table `| **Treatment (in-dist)** |` add Gemma co-treatment row; anchor `neutralizing the model-family confound` fix wording; cost line.
- **docs/…design.md** — groups table + confound-handling + pivots log: add Gemma co-treatment.
- **report/verification_findings.md** — note the family/size-confound concern is now mitigated by the Gemma run.
- **report/review_recommendations.md** — mark item 11 (Gemma) DONE; resolve the conditional "(1) Add models?" framing.
- **report/section4_results.md** — keep in sync (this is the qwen tables; gemma tables live in section4_results_gemma.md).
- **Cost ceilings** to re-check if total >$2: README `< ~$2`, design.md, infra/README.md.

## Outcome decision rule
`results/eval_gemma.json`: if `groups."T-in".financial.sharpe` > `groups."C-B".financial.sharpe`
AND `groups."T-in".prescience.exposure_timing` > 0 AND `foresight_gap_DiD.gap_exposure_timing` > 0
AND Aug-5 `pseudo_event_null_gemma.json` T-in p ≲ 0.05 → **OUTCOME A**, else **OUTCOME B**.
Under B, cite H2 by name everywhere (it is a confirmed prediction, not a failure).
