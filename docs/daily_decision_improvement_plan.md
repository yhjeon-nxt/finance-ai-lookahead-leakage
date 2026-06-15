# Daily-decision improvement plan (workflow wf_93c18dba-102)

I have verified all load-bearing details against the actual code. The proposals' code citations are accurate. Here is the consolidated plan.

---

# Consolidated Plan: Improving the Daily Trading-Decision Mechanism

**Verified against:** `engine.py`, `context.py`, `agent.py`, `prompts.py`, `llm_client.py`, `config.py`, `metrics/leakage.py`. Key facts confirmed: the `portfolio={t: 0.0 for t in UNIVERSE}` stub is at `engine.py:105`; `render_context` shows "none" when all values are zero (`context.py:84`, `if v` filter); `TRAILING_DAYS=60` (`config.py:132`); `IN_DIST.download_start` gives ~86-day buffer (`config.py:48-53`); decision cache key is `{group.name}_{client}_seed{seed}` written per-date, no prompt/context hash (`engine.py:56-71, 96-108`); `rationale_forensics` scans only `f"{dec.analysis} {dec.rationale}"` at `leakage.py:143` AND the excerpt builder at `leakage.py:170`; exposure/confidence-based leakage metrics read `result.weights/exposure/confidence/next_day_returns` only (`leakage.py:62-112`), never the rendered positions block.

### Two cross-cutting facts that govern everything below
1. **Cache invalidation is mandatory for ANY change to the agent's information set or output schema.** The cache (`engine.py:99`) keys on date only — it does NOT hash the prompt or context. Any change that alters what the agent sees or emits will silently replay stale decisions on resume/re-run, mixing two mechanisms in one result file. Every results-affecting item below requires wiping `results/decisions/*.jsonl` and re-running all groups/seeds.
2. **Comparability requires byte-identical mechanism across T-in / C-A / C-B.** Everything in `build_context`/`render_context`/`prompts`/`agent`/`llm_client` is group-agnostic and applied uniformly, so uniform changes preserve comparability *by construction* — but they change absolute metric values, so prior numbers can't be compared against post-change numbers.

---

## DO NOW (strongest, leakage-safe, high value/effort ratio)

**D1 — Multi-horizon momentum: add `ret_10d`, `ret_60d`** (price-signal P1)
- Spec: In `build_context` (`context.py:59-63`) add `feats["ret_10d"] = hist.iloc[-1]/hist.iloc[-11]-1 if len(hist)>10 else np.nan` and `ret_60d` (index `-61`, guard `len>60`); add the columns to the `render_context` header (`context.py:87`) and row f-string (`context.py:91-94`). Raise `TRAILING_DAYS` 60→75 (`config.py:132`); the ~86-day download buffer covers it with no re-fetch.
- File/fn: `src/leakage/data/context.py` build_context + render_context; `src/leakage/config.py:132`.
- Value/effort: **high / low**. Leakage-safe (ratio of two causal prices). **Changes results-comparability → re-run required + clear cache.** Also update the `DayContext.features` docstring (`context.py:28`).

**D2 — Drawdown from trailing high + distance-from-MA + vol-regime, as one batch** (price-signal P2, P3, P5)
- Spec: In `build_context`, add `feats["dd_from_high"] = hist.iloc[-1]/hist.cummax().iloc[-1]-1` (P2); `feats["dist_ma50"] = hist.iloc[-1]/hist.tail(50).mean()-1` (P3 — ship the single `dist_ma50` column, drop `dist_ma20` as collinear with existing `ret_20d`); `feats["vol_regime"] = (rets.tail(20).std()/rets.std()).round(2)` (P5 — numeric column only; defer the worded calm/elevated/stressed bucket to avoid a hidden tuning knob). Render each as a column.
- File/fn: `src/leakage/data/context.py` build_context + render_context.
- Value/effort: **high / low** combined. All three are pure causal price transforms (`hist`/`rets` are `loc[:day]`, covered by the existing causality asserts at `context.py:53-56`). **Changes results-comparability → bundle into the same re-run as D1.**

**D3 — Retry-on-malformed JSON before recording a parse failure** (action-space P2)
- Spec: In `TradingAgent.decide` (`agent.py:83-90`), wrap the `client.complete` + `_coerce_decision` in `for attempt in range(max_attempts)`; on failure re-query with `seed = base + i + 1000*attempt` (the `1000*` multiplier avoids collision with another day's `base+i` seed) and an appended format-only nudge ("Return ONLY the JSON object with all four fields"). **CRITICAL FIX vs the sketch:** the retry trigger must be `not dec.parse_ok` ONLY — never `not dec.target_weights`, because `_coerce_decision` legitimately returns `parse_ok=True` with `{}` for a valid all-cash decision (`agent.py:48-59`; MockClient emits `{}` too, `llm_client.py:103`). Gating on empty weights would force retries on valid risk-off and could flip it to invested, biasing exposure.
- File/fn: `src/leakage/agent/agent.py:83-90`; nudge text in `prompts.py:30-38` or concatenated in `decide`.
- Value/effort: **medium / low-medium**. Leakage-safe (format-only nudge, same causal context). **Does NOT change a fully-completed run's metric inputs** for runs that never failed — but it changes which days survive, so re-run with a fresh cache to realize the benefit.

**D4 — Cache only successful decisions; retry failures on resume** (action-space P3)
- Spec: In `run_backtest` (`engine.py:96-108`), gate `fh.write(...)` on `dec.parse_ok`; route failures to a sidecar (e.g. `<cache>.fails.jsonl`) that `_load_cache` (`engine.py:62-71`) does NOT read. Today the unconditional write at `engine.py:107` freezes transient failures forever across resumes (`_load_cache` reloads `parse_ok=False` rows, `engine.py:99-100` skips re-query). Footgun: the sidecar must not be loaded, or the fix is defeated.
- File/fn: `src/leakage/backtest/engine.py:96-108` (write), `62-71` (loader).
- Value/effort: **medium / low**. Leakage-safe (pure bookkeeping; seed is `seed+i` keyed to day index so deterministic failures re-fail identically, transient ones rejoin). A fully-completed run is byte-identical; **only partial-run caches differ → no separate re-run needed beyond clearing caches once.** Ships naturally with D3.

**D5 — Harden JSON extraction (balanced-brace) + truncation detection** (action-space P8)
- Spec: Replace greedy `_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)` (`agent.py:34`) with a balanced-brace scan / progressive `json.loads` on trimmed candidates so prose+JSON or two JSON blocks parse the intended object, not the widest span. Add truncation detection (`raw.rstrip().endswith("}")`) that triggers the D3 retry. `num_predict=700` is a single shared client attr (`llm_client.py:39,58`) — if raised, raise it uniformly for all groups.
- File/fn: `src/leakage/agent/agent.py:34-45`; optional `num_predict` in `llm_client.py:39`.
- Value/effort: **medium / low**. Leakage-safe (operates on the model's own emitted string; one shared parse path). Verbose models truncate more at 700 tokens and silently drop from the foresight sample — fixing this reduces a model-dependent artifact (net-positive for fairness). **Benefits only new runs; rebuild caches.** Ships with D3/D4.

---

## CONSIDER (sound and leakage-safe, but conditional, lower value, or coupled)

**C1 — Feed the REAL drifted book + cash into the prompt** (consolidates state P1, reasoning P2, experimental P1; supersedes the standalone state-P2/P3/reasoning-P5 prompt-side items)
- Spec: Replace `portfolio={t: 0.0 for t in UNIVERSE}` (`engine.py:105`) with the actual book carried from `last_weights` (already tracked at `engine.py:94,116`) scaled to current `equity`: `portfolio = {t: equity*last_weights.get(t,0.0) for t in UNIVERSE}, cash = equity - sum(...)`. `render_context` already renders the non-empty block (`context.py:83-84`). Optionally add a neutral SYSTEM_PROMPT line: "you already hold the positions shown; returning the same weights = no trade, trading has cost" (reasoning P2/P5 — reuse the existing `rationale` field, do NOT add a new unscanned free-text field).
- Why CONSIDER not DO-NOW: (a) **Adds a redundant non-context info channel** ("your last bet moved ±X%") layered on the existing P&L reflection — causal, but beyond the price-only design intent, so it must be run as a **separate labelled variant** (e.g. `T-in-stateful`). (b) **Hard cache-invalidation hazard** (date-only key) + must re-run all groups. (c) **The mock baseline must be made position-aware too** (experimental P3) or the DiD null breaks — coupling this to a non-trivial `MockClient` change. (d) Turnover-reduction benefit is **unproven** (the model emits absolute weights and may re-solve regardless).
- File/fn: `engine.py:105` + new `held_value` accumulator in the loop body; `prompts.py:10-27`; `MockClient.complete` (`llm_client.py:95-118`) for the paired variant.
- Value/effort: **high / medium (coupled)**. Leakage-safe in the narrow sense; **changes results → labelled variant + full re-run + mock co-change.** Note: the prompt-side "you hold X" instruction (state-P2) is *inert/self-contradictory without this* (positions render "none"), so it is folded in here, not listed separately.

**C2 — Market breadth + average pairwise correlation regime line** (price-signal P6)
- Spec: In `build_context` compute two scalars from `hist`/`rets` and add fields to `DayContext` (`context.py:23-30`, single constructor call site `69-76`): `breadth_5d = float(((hist.iloc[-1]/hist.iloc[-6]-1)>0).mean())` (guard `len(hist)>=6`) and `avg_corr_20d = off-diagonal mean of rets.tail(20).corr()`; render one summary line. Store the integer up-count too for the "4/7" render string.
- Why CONSIDER not DO-NOW: medium effort (dataclass field add), and an **eyes-open interpretation tradeoff** — a richer top-down tape signal could let the *control* model legitimately de-risk pre-crash, slightly compressing the treatment-vs-control gap (statistical-power consideration, not an invariant violation). Leakage-safe.
- File/fn: `src/leakage/data/context.py:23-30, 39-76, 79-99`. **Changes results → re-run.**

**C3 — Cross-sectional return ranks** (price-signal P4)
- Spec: After momentum cols, `feats["rank_20d"] = feats["ret_20d"].rank(ascending=False, method="min")` (pick `method="min"`, not the sketch's inconsistent `average`); render as integer. **Must NaN-guard the renderer** — `ret_20d` is NaN during warmup (`context.py:62`), and pandas rank yields NaN, which would break an int format (render "-" or guard). Value downgraded to **medium** (it's added alongside, not replacing, raw returns; turnover benefit is speculative and structurally driven by C1, not a rank column).
- File/fn: `context.py:59-63, 86-94`. **Changes results → re-run.**

**C4 — OHLCV-derived features (ATR%, volume z-score)** (price-signal P7)
- Spec: Change `_close_frame` (`context.py:33-36`, parallel `engine.py:50-53`) to retain full OHLCV (already cached, currently discarded). Add `atr14_pct = ((High-Low)/Close).tail(14).mean()` and `vol_z = (Vol.iloc[-1]-Vol.tail(20).mean())/Vol.tail(20).std()`; render 1-2 columns. **Mandatory:** extend the causality asserts (`context.py:53-56`) to the OHLCV index. OHLCV is market-price data, not a non-price channel, so leakage-safe. Skip the single-day `overnight_gap` as a standing column (noisier).
- Why CONSIDER not DO-NOW: medium effort/risk, changes the stimulus for every group, token-budget pressure on 8B JSON. **Changes results → re-run + cache wipe.**
- File/fn: `context.py:33-36, 39-76`; `engine.py:50-53`.

**C5 — Numbers-only reflection journal** (reasoning P1, REDESIGNED — the original is in REJECT)
- Spec: Replace single-day `build_reflection` with a bounded deque (5-7 entries) of `{date, exposure, realized next-day return}` ONLY — **drop the `rationale`/`analysis` text entirely.** Render as a neutral "recent decisions and outcomes" block. Thread the deque through the engine loop (values already computed: `engine.py:129,133`), seed from cache on resume.
- Why CONSIDER not DO-NOW: numeric-only is strictly causal and leakage-safe, but it loses the self-consistency benefit that was the original point, so value is modest. **Changes prompt → labelled variant + re-run.**
- File/fn: `prompts.py:41-46`; `engine.py:97-136`, `62-71`.

**C6 — Schema-constrained JSON via ollama `format=<schema>`** (action-space P1)
- Spec: Swap `format="json"` (`llm_client.py:61`) for a JSON-schema dict built once from `UNIVERSE`. Leakage-safe (pure output-shape; `_coerce_decision` already enforces the same bounds, so no new info). **Gate behind a control-model empty-output guard:** the existing `/no_think` fallback (`llm_client.py:68-71`) fires ONLY for `_supports_think` models; a stricter schema could empty out `llama3.1:8b` (control) with no fallback → parse failure + stale book. Add a per-model empty-output guard + a `llama3.1:8b` smoke check before any full run.
- Value/effort: **medium / medium**. **Changes mechanism → re-run.**

**C7 — Validate-and-default missing fields (diagnostic only)** (action-space P4)
- Spec: In `_coerce_decision`, distinguish "JSON present, `target_weights` empty/absent" (set `error="empty_allocation"`, keep `parse_ok=True`) from "no JSON found". Ship **diagnostic surfacing only** (no equity/metric behavior change) — leakage-safe, low-risk. Any new Decision field needs a default for cache back-compat (`engine.py:70`). Treat metric-reclassification of `empty_allocation` days as a SEPARATE, explicitly-documented, cross-group-uniform decision (it touches the deliberate parse-fail exclusion at `engine.py:110-124`).
- Value/effort: **medium / low** (diagnostic half). No re-run needed for the diagnostic-only version.

**C8 — Cost-charged equity as a separate labelled variant** (state P4); **C9 — informational cost line** (state P3); **C10 — first-day-failure logging** (action-space P5); **C11 — confidence calibration report-only channel** (action-space P7); **C12 — prior-weights deliberation nudge** (reasoning P5)
- All leakage-safe but lower priority. C8/C9 depend on C1's drifted book (the one-way turnover formula `0.5·Σ|w_target − w_drifted|` needs the per-ticker drifted book, which doesn't exist today). C8 must keep `cost_bps=0.0` default (canonical run byte-identical) and label a distinct variant. C10 is near-redundant (`engine.py:94` already comments the carry-forward; the all-cash day is already excluded from metrics) — bundle with D3/D4 only. C11 is report-only, low value, backlog. C12 reuses `rationale` (no forensics change) — effect on turnover unproven.
- **C13 — Token-budget render hygiene** (price-signal P8): adopt as a companion guardrail WHEN D1/D2/C2/C3 land — curate the per-ticker table to a compact subset and put scalars on one summary line. Do NOT trim recent-returns 15→10 in isolation (it removes signal). Keep the literal "Recent daily returns" header prefix (the MockClient parser keys on it, `llm_client.py:127`).

---

## REJECT / would contaminate the experiment

**R1 — Persistent reflection journal carrying VERBATIM `rationale` text** (reasoning P1, as written)
- Reason: The sketch puts `dec.rationale[:80]` into each journal entry. But `rationale`+`analysis` is exactly the text `rationale_forensics` scans for HARD_TELLS (`leakage.py:143,153`). Feeding the agent's own past rationale back into later prompts creates a **contamination/feedback loop**: a leaky phrase emitted on day T primes leaky generations on T+1..T+6, and a later hard tell may be an echo of the journal rather than independent parametric recall — confounding the headline forensics metric. Worse, it amplifies **asymmetrically**: T-in has real tells to echo; controls have few — inflating the very in-dist-minus-OOD gap the study measures. The numeric subset survives as **C5**.

**R2 — Confidence-scaled risk budget folded into the primary mechanism** (reasoning P3)
- Reason: Scaling exposure by confidence changes the **definitions** of the leakage estimators, not just their magnitudes. `exposure_timing` uses `expo` and `conf_weighted_timing = _corr(expo*conf, mkt_next)` (`leakage.py:74-78`); clamping `expo = min(sum_raw, confidence)` makes confidence enter the exposure term too — a different estimator. `pre_event_timing` reads the same `expo` (`leakage.py:94-108`). The clamp binds only when `sum(weights)>confidence`, which fires at **model-dependent rates**, asymmetrically attenuating exactly the signals whose between-group gap IS the leakage estimate. Acceptable ONLY as a strictly-isolated, separately-analyzed variant whose timing numbers are never compared head-to-head against unscaled runs — i.e. not part of the primary measurement.

**R3 — Per-name weight cap / degenerate-allocation rewrite in the primary mechanism** (action-space P6)
- Reason: `next_day_prescience` reads `result.weights` per-ticker and `result.exposure`; `pre_event_timing` reads `result.exposure` (`leakage.py:62-112`). The proposal's own target case — single-name 100% (e.g. NVDA) — has no other name to absorb the clipped excess, so exposure silently collapses 1.0→0.5 **precisely on high-conviction days that cluster near the EVENT_ANCHORS** (`config.py:87-90`) the metrics probe. This distorts the foresight INPUT in a signal-correlated, asymmetric way, and the experiment treats max-concentration/turnover as OUTCOMES being measured — clamping them is tampering with the dependent variable. Permissible only as a clearly-labelled `capped` vs `uncapped` variant that records exposure-before-cap so uncapped exposure still feeds the foresight metrics — never as a silent in-place change.

**R4 — Two-step plan/premortem fields WITHOUT extending forensics at BOTH sites** (reasoning P4, as written)
- Reason: Adding `plan`/`premortem` free-text fields creates new channels through which foresight can surface or hide. `rationale_forensics` scans only `analysis`+`rationale` at `leakage.py:143` AND concatenates them again in the excerpt builder at `leakage.py:170`. The sketch patches only line 143 → the new fields would be unscanned (a leakage hole) and excerpts would under-report. Also pressures the shared `num_predict=700` budget → truncation/parse-fail clustering on busy event days, biasing the metric (`engine.py:114-124`). **Becomes acceptable as a labelled variant only if** both forensics sites are patched, the prompt stays "reason from the shown data only," and `num_predict` is raised uniformly with `n_parse_fail` monitored. As written (single-site patch), it breaks the invariant — reject until corrected.

---

## Recommended sequencing
1. **Batch A (robustness, ship together, one cache wipe):** D3 → D4 → D5 (+ C10 bundled). These fix silent-failure modes feeding the metrics; minimal results impact on clean runs.
2. **Batch B (price-only enrichment, one re-run of all groups/seeds):** D1 + D2, plus C13 render hygiene; optionally fold in C2/C3/C4 if doing one larger enrichment re-run. Update the `DayContext` docstring and re-run the mock smoke to confirm `next_day_prescience` stays ~0.
3. **Batch C (state, labelled variants, coupled):** C1 + mock co-change (experimental P3) + C5/C6/C7 as separate labelled runs. Never overwrite the canonical cost-free foresight inputs.

**Every results-affecting item requires:** wipe `results/decisions/*.jsonl`, re-run all of T-in/C-A/C-B across all SEEDS, and treat prior numbers as a different mechanism. The date-only cache key (`engine.py:99`) is the single biggest operational footgun — consider adding a context/prompt hash or version tag to the cache key as defensive hygiene before any enrichment lands.