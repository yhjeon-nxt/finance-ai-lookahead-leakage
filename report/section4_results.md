### 4.1 Financial performance

| Group | Model | Total return | Sharpe | Max DD | Turnover | Parse-fail |
|---|---|---|---|---|---|---|
| T-in | `qwen3:8b` | +0.251 | +1.76 | -0.136 | +0.728 | 0 |
| C-A | `llama3.1:8b` | +0.043 | +0.30 | -0.152 | +0.880 | 0 |
| C-B | `qwen3:8b` | +0.032 | +0.49 | -0.131 | +0.694 | 0 |

### 4.2 Leakage / foresight metrics

| Group | Ticker prescience | Exposure timing | Conf-wtd timing |
|---|---|---|---|
| T-in | +0.021 | +0.054 | +0.055 |
| C-A | -0.032 | -0.071 | -0.037 |
| C-B | +0.016 | -0.039 | -0.045 |

### 4.3 Pre-event timing (in-distribution groups)

| Group | Aug-5 crash (de-risk>0) | Nov-5 election (load>0) | mean |
|---|---|---|---|
| T-in | +0.115 | +0.011 | +0.063 |
| C-A | -0.125 | +0.060 | -0.032 |

### 4.4 Headline statistical tests

| Comparison | Δ (timing prescience) | permutation p |
|---|---|---|
| T-in vs C-A | +0.1253 | 0.075 |
| T-in vs C-B | +0.0934 | 0.400 |

### 4.5 Within-model foresight gap + regime-adjusted DiD

Leakage is supported only if the LLM in-dist−OOD gap EXCEEDS the no-memory momentum baseline's gap (DiD > 0); a raw gap alone can arise from the 2024-H2-vs-2026 regime difference.

| Metric | LLM gap | Regime baseline gap | **DiD (leakage)** |
|---|---|---|---|
| gap_ticker_prescience | +0.005 | -0.021 | **+0.026** |
| gap_exposure_timing | +0.093 | -0.042 | **+0.136** |
| gap_conf_weighted_timing | +0.100 | -0.045 | **+0.145** |

### 4.6 Rationale forensics (smoking-gun scan)

**T-in** — hard-tell decisions: 0; hard tells: {}
**C-A** — hard-tell decisions: 0; hard tells: {}
**C-B** — hard-tell decisions: 0; hard tells: {}
