# PROGRESS

Live status log for the look-ahead-leakage experiment. Newest entries at top.
Durable across context compaction — read this + `findings.md` to resume.

## Phase status

| # | Phase | Status |
|---|---|---|
| 0 | Repo scaffold + GitHub + spec | ✅ done |
| 1 | Audit class presentations + verify model cutoffs | ✅ audit done (cutoff probe at run-time) |
| 2 | Data engineering pipeline | ✅ done (both windows real; causality guard passes) |
| 3 | Agent + backtest engine + metrics (local smoke) | 🟡 in progress |
| 4 | Cost-optimized EC2 spot run (GATED) | ⬜ pending |
| 5 | Evaluation + statistical proof | ⬜ pending |
| 6 | Research report | ⬜ pending |

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
