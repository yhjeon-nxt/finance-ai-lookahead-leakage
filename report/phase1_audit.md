# Phase 1 — Lecture Alignment: Class Paradigms & Leakage Concerns

Audit of the Financial AI seminar presentation PDFs in `other_students/` (+ the presenter's
own FinAgent talk). Purpose: ground the trading-agent architecture in the class consensus, and
locate where the class did / did not address look-ahead bias.

## Per-paper summary

| Paper | Paradigm tags | Inputs | Decision format | Backtest | Temporal / leakage notes |
|---|---|---|---|---|---|
| **FINCON** | ReAct, Reflection, Multi-agent, Tool use | News, prices, reports, fundamentals | BUY/SELL/HOLD + allocation | rolling split; Sharpe, TR, CR, MDD | rolling split implied; no explicit leakage statement |
| **TradeMaster** | RL (gradient), Tool use, Multi-agent | Prices, technicals, news | BUY/SELL/HOLD, continuous alloc | 13 datasets; SR, TR, MDD | **explicit**: "last year test, penultimate validation, rest train" → prevents look-ahead |
| **QuantAgents** | Multi-agent debate, Reflection, Tool use | Prices, news, fundamentals | BUY/SELL/HOLD + weights | simulated+real dual reward | dual reward ≠ temporal split; no explicit statement |
| **FLAG-TRADER** | ReAct, RL (gradient), Reflection | Prices, sentiment, macro | BUY/SELL/HOLD | stock/crypto/ETF envs; SR, TR | state→text prompts; no explicit temporal split |
| **CausalStock** | Supervised, CoT, Tool use | OHLCV, news | UP/DOWN (multi-stock) | 6 markets | **explicit & highest rigor**: "split chronologically, not randomly… prevents unrealistic future leakage" |
| **Hierarchical Retrieval (Fin-QA)** | RAG, CoT | SEC filings | extractive QA | LOFin benchmark | fixed corpus; no forward-looking risk |
| **Two Sides of the Same Coin** | Reflection, Tool use | 10-K narratives | dual-narrative detection | no backtest | n/a |
| **CLER** | Reflection, Multimodal, CoT | tables, charts, text | multimodal QA | FinMME benchmark | test-time only; n/a |

## Ranked paradigm prevalence (class consensus)

1. **Tool use / RAG** — ~all papers
2. **Reflection / memory** — 6/8
3. **ReAct / chain-of-thought** — 5/8
4. **Multi-agent roles/debate** — 3/8
5. **RL (gradient-based)** — 2/8 (TradeMaster, FLAG-TRADER)
6. **Supervised prediction** — 2/8
7. **Multimodal** — 1/8 (CLER)

## Architecture decision (grounded in the consensus)

Adopt a **minimal single-agent ReAct + Reflection trader with structured JSON output** — the
intersection of the class's three most common paradigms (tool/feature grounding, reflection,
ReAct reasoning) without RL or multi-agent orchestration that would mask the leakage signal:

- **Input:** market state (trailing prices/returns) rendered to text — FLAG-TRADER / CausalStock pattern.
- **Reasoning:** short chain-of-thought over the price context.
- **Reflection:** one-line review of the prior day's P&L before deciding (FINCON CVRF / QuantAgents memory, minimized).
- **Action:** JSON `{action, ticker, target_weight, confidence, rationale}`; **no gradient updates** (in-context only).

## The gap this project targets (report hook)

> Across the seminar, **only CausalStock and TradeMaster explicitly address look-ahead bias, and
> both treat it purely as a *data-pipeline* problem solved by a chronological train/test split.**
> **None address *parametric* leakage** — the case where the LLM's *pre-training corpus itself*
> already contains the "test" period. A chronological split of the *input data* does **not** fix
> this: the future is baked into the model weights. This is the leakage vector the present study
> isolates and measures.

All temporal-split remarks found:
- CausalStock: "split chronologically, not randomly … prevents unrealistic future leakage."
- TradeMaster: "last year for test, penultimate for validation, remaining for training."
- All others: no explicit look-ahead / temporal-split / knowledge-cutoff statement.
