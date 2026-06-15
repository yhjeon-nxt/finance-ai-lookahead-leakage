# PROGRESS

Live status log for the look-ahead-leakage experiment. Newest entries at top.
Durable across context compaction — read this + `findings.md` to resume.

## Phase status

| # | Phase | Status |
|---|---|---|
| 0 | Repo scaffold + GitHub + spec | ✅ done |
| 1 | Audit class presentations + verify model cutoffs | ✅ audit done (cutoff probe at run-time) |
| 2 | Data engineering pipeline | ✅ done (both windows real; causality guard passes) |
| 3 | Agent + backtest engine + metrics (local smoke) | ✅ done (validated on real models) |
| 4 | Cost-optimized EC2 spot run (GATED) | ✅ DONE — run #4 i-025f474167e533f5c rc=0, self-terminated (~$0.8 total) |
| 5 | Evaluation + statistical proof | ✅ done — DiD positive, moderate H1 support |
| 6 | Research report | ✅ results+discussion+related-work+figures+Gemma §4.9 written |
| 7 | Gemma 3 12B independent-family test (RUN_TAG=gemma) | ✅ done — did NOT replicate (confabulates; H2) |
| 8 | Per-model in-vs-out (4 models, 3 seeds, EC2) | ✅ done — DiD orders by recall |
| 9 | 32B+8B parallel fleet (7 spot, size×recall) | ✅ done — within-family bigger→larger DiD (all ns); 32B-H2 headline null (cutoff/window) |

## Open gates (require user)

- **EC2 launch:** confirm instance type + live cost estimate before `aws ec2 run-instances`.
- **Scale beyond 1 instance:** fresh approval required.

## Log

### 2026-06-14
- Brainstormed design; user set mode = all-local open models / real EC2+S3 (cost-minimized) /
  design-gate→autonomous. User delegated target-window choice to me.
- Locked design: T-in `qwen3:8b` (2024-H2) vs C-A `llama3.1:8b` (2024-H2) vs C-B `qwen3:8b`
  (2026 Jan–May). Cutoffs to be verified in Phase 1.
- Wrote spec, README, requirements, scaffold dirs. Created+pushed GitHub repo.
- Phase 1 audit done (report/phase1_audit.md): class consensus = ReAct+Reflection+structured;
  only CausalStock/TradeMaster mention look-ahead, none address parametric leakage → report hook.
- Phase 2 data done: yfinance ingest + causal context builder. 2024H2=128 trading days,
  2026JanMay=102 trading days (real, through 2026-05-29). Causality guard verified.
- Decided context = price-only. Model-cutoff empirical probe deferred to run-time first step.

### 2026-06-15
- Run #4 (qwen3:8b treatment) succeeded: T-in Sharpe 1.76 vs C-A 0.30 / C-B 0.49; Aug-5 de-risk
  p=0.051; DiD positive. Added related-work (cited), per-ticker figures (election allocation
  JPM p=0.001), pseudo-event null, threats/limitations, reproducibility appendix.
- Gemma 3 12B independent-family co-treatment (i-08f3c35cd2e075530, rc=0, self-terminated): did
  NOT replicate the clean signature. Gemma CONFABULATES 2024-H2 (projects "Biden" as 2024 winner;
  wrong crash cause) despite official Aug-2024 cutoff → no crash de-risk (p=0.54), weak/ns DiD
  (+0.092), Sharpe 0.75. Pre-registered H2 confirmed on family 2. KEY LESSON: documented in-window
  cutoff is necessary-but-not-sufficient; leakage needs genuine recall → probe each model.
- Fixed a latent bug: extra_analyses._exposure_panel globbed mock files (seed-key collision);
  harmless for ec2 (qwen sorts after mock) but corrupted gemma. Excluded mock; ec2 unchanged.
- Total EC2 cost ≈ $1.4 (qwen run ~$0.8 incl 3 aborted boots + Gemma run ~$0.6).
- Phase 3 fully validated on REAL models (qwen3:8b/llama3.1:8b via local ollama): JSON parsing,
  ~7s/decision, 0 parse failures end-to-end.
- Treatment-selection probe (results/model_selection.json): qwen3:8b 2/4 (knows Aug-5 crash +
  NVDA split, refuses politics), qwen2.5:7b 1/4, llama3.1:8b 0/4 (clean control), phi4 0/4.
- User chose EC2 spot + BIGGER treatment. AWS recon: region ap-northeast-2, G/VT spot quota=256,
  g6e.xlarge spot $0.539/hr. Plan: g6e.xlarge spot, treatment auto-selected from {qwen3:32b,
  qwen2.5:32b} on-instance, control llama3.1:8b, est ~$1-2, self-terminate.
- Code+data staged to s3://neuroxt-personal/yhjeon/finance-ai-leakage/. Config now env-overridable.
- BLOCKED: user to name an instance profile with S3(neuroxt-personal)+SSM access. Then launch.
