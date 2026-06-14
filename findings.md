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

## Decisions / pivots

- 2026-06-14: API → local open models (ollama) for a *real* cutoff gap, zero API cost.
- 2026-06-14: kept 2024-H2 window, re-derived from cutoffs rather than assumed.
