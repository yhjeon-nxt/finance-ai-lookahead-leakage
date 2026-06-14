# FINDINGS

Research findings, decisions, errors, and pivots. Separate from PROGRESS (status) so the
narrative reasoning survives context compaction. Newest at top.

## Model-cutoff facts (to verify empirically in Phase 1)

- `llama3.1:8b` — Meta states knowledge cutoff **2023-12**. Clean *pre-2024* control.
- `qwen3:8b` — Alibaba 2025 release; pre-training covers 2024 (and into 2025). Candidate
  *treatment* (knows 2024-H2). Verify it cannot reliably answer 2026 facts → validates C-B.
- `qwen2.5:7b` — 2024-09 release; partially covers 2024. Optional robustness "straddle".

## Event anchors (2024-H2)

- **2024-08-05** — global risk-off / yen carry-trade unwind; VIX spiked ~65 intraday;
  Nikkei −12%. Sharp down-then-recover. Ideal "did the model de-risk *before* it?" probe.
- **2024-11-05/06** — US presidential election → "Trump trade": small caps (IWM), banks (JPM),
  TSLA, crypto-adjacent (COIN/DJT) rallied. Ideal "did it load up *before* the result?" probe.

## Phase 1 audit — the report hook

Across all 10 class papers, **only CausalStock and TradeMaster discuss look-ahead bias, and
both only as a *data-pipeline* issue fixed by a chronological train/test split.** NONE address
**parametric leakage** — the LLM's pre-training corpus already containing the test period. A
chronological split of the *inputs* does not fix this; the future is in the weights. This is
exactly the gap this project isolates. Class consensus architecture = ReAct + Reflection +
structured output (tool use 8/8, reflection 6/8, ReAct 5/8) → confirms my minimal-agent design.

## Engineering findings (real models, local validation)

- **qwen3:8b is a *thinking* model.** With `format=json` + default thinking it returns EMPTY
  output (the token budget is spent on suppressed `<think>` tokens). Fix: pass `think=False`
  (ollama). Then it emits valid JSON in ~7s. llama3.1:8b has no thinking mode. → `OllamaClient`
  auto-detects qwen3/r1 and disables thinking; `cutoff_probe` does the same. **Report footnote:
  model inference config is itself a reproducibility hazard for LLM-agent backtests.**
- **Speed (Apple Silicon, this Mac):** ~6–8s / decision for 8B. Full run ≈ 3 groups × ~118 avg
  decision-days × 3 seeds ≈ 1065 calls ≈ ~2.2 h locally, $0. EC2 g5 spot would be faster but
  costs ~$1–2 + setup. → present as the gate choice.
- **Cache bug fixed:** decision cache now keyed by client name (mock vs real never collide).

## Decisions / pivots

- 2026-06-14: API → local open models (ollama) for a *real* cutoff gap, zero API cost.
- 2026-06-14: kept 2024-H2 window, re-derived from cutoffs rather than assumed.
- 2026-06-14: **context = price-only (no news)**. We test *parametric memory* leakage; if the
  agent only sees prices ≤ day T, any foresight must come from training memory, not the feed.
  News would open a second leakage channel and muddy the causal claim. Methodological upgrade.
