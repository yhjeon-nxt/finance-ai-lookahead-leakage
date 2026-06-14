# Look-ahead Leakage in Open-Source LLM Trading Agents

Final-project experiment for the **Financial AI** seminar (Korea University AutoAI Lab).
An end-to-end scientific study of **data leakage / look-ahead bias ("cheating")** in
LLM-based agentic trading systems, executed by an autonomous research agent.

> **One-line thesis.** An open LLM whose pre-training corpus *includes* a historical
> market window can exhibit statistically anomalous "foresight" when backtested on that
> window — foresight that a same-size model with an earlier knowledge cutoff does not show,
> and that the *same* model does not show on a genuinely out-of-distribution future window.

## Experimental design (summary)

| Group | Model (local, open-source) | Period traded | Knows the period? |
|---|---|---|---|
| **Treatment (in-dist)** | `qwen3:8b`; + `gemma3:12b` co-treatment (family 2) | **2024-H2** (Jul–Dec) | **Yes → leakage suspected** |
| **Control A (model control)** | `llama3.1:8b` (cutoff 2023-12) | 2024-H2 | No |
| **Control B (time control)** | same treatment model | **2026 Jan–May** (post-cutoff) | No |

The model cutoff gap is **real**, not prompt-simulated — that is the central methodological
choice. The within-treatment in-dist vs. OOD comparison isolates leakage from raw model
capability; an independent-family co-treatment (`gemma3:12b`) breaks the treatment/control
**family** confound directly.

### Key result
`qwen3:8b` (which *genuinely recalls* 2024-H2: the Aug-5 crash + NVIDIA split) shows the leakage
signature — **Sharpe 1.76 in-dist vs 0.30/0.49 controls**, de-risks before the Aug-5 crash
(p=0.051), positive regime-adjusted DiD. `gemma3:12b` (official Aug-2024 cutoff but **confabulates**
the period — projects "Biden" as 2024 winner) does **not** replicate it (Sharpe 0.75, no crash
de-risk, ns). **Lesson: a documented in-window cutoff is necessary but not sufficient — leakage
needs genuine recall, so probe each model.** Total EC2 cost ≈ $1.4.

See [`docs/superpowers/specs/2026-06-14-llm-lookahead-leakage-design.md`](docs/superpowers/specs/2026-06-14-llm-lookahead-leakage-design.md)
for the full design, and [`PROGRESS.md`](PROGRESS.md) for live status.

## Repo layout

```
src/leakage/
  data/       market data ingestion + per-day context bundles (yfinance)
  agent/      ollama-backed Reflection+ReAct trading agent (structured action + rationale)
  backtest/   modular simulation/portfolio engine, idempotent decision cache
  metrics/    financial metrics + leakage/foresight metrics + stats tests
  run/        orchestration, S3 sync, EC2 entrypoint
infra/        cost-optimized EC2 spot launch + bootstrap + self-terminate
results/      logs, equity curves, decision records, figures
report/       publication-grade research report (the deliverable)
docs/         design spec
```

## Cost discipline

Develop + smoke-test locally; burst to a **single `g5.xlarge` spot** instance for the real
run with an idempotent resumable cache and a self-terminate backstop. Artifacts stream to
`s3://neuroxt-personal/yhjeon/`. Target total compute cost **< ~$2**.

## Reproduce

```bash
pip install -r requirements.txt
# (Phase scripts and a Makefile/CLI land as the build proceeds — see PROGRESS.md)
```
