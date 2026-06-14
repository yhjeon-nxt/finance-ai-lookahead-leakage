# PROGRESS

Live status log for the look-ahead-leakage experiment. Newest entries at top.
Durable across context compaction — read this + `findings.md` to resume.

## Phase status

| # | Phase | Status |
|---|---|---|
| 0 | Repo scaffold + GitHub + spec | 🟡 in progress |
| 1 | Audit class presentations + verify model cutoffs | ⬜ pending |
| 2 | Data engineering pipeline | ⬜ pending |
| 3 | Agent + backtest engine + metrics (local smoke) | ⬜ pending |
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
- Wrote spec, README, requirements, scaffold dirs. Creating GitHub repo.
