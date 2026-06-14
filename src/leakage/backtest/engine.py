"""Daily-rebalance backtest with a resumable per-day decision cache.

Mechanics: at the close of day T the agent sees causal context (data <= T) and outputs target
weights. The portfolio is rebalanced to those weights and held into T+1, so the realized
portfolio return on T+1 is ``sum_i w_i(T) * r_i(T->T+1)``; the cash remainder earns 0.

Resumability: every decision is appended to ``results/decisions/<group>_seed<seed>.jsonl``.
Re-running reloads decided days and only queries the model for missing ones — so a spot-instance
interruption costs at most one day of work.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from leakage.agent.agent import Decision, TradingAgent  # noqa: E402
from leakage.agent.llm_client import LLMClient  # noqa: E402
from leakage.agent.prompts import build_reflection  # noqa: E402
from leakage.config import RESULTS_DIR, UNIVERSE, Group  # noqa: E402
from leakage.data.context import build_context, render_context  # noqa: E402
from leakage.data.ingest import load_prices, trading_days  # noqa: E402

DECISIONS_DIR = RESULTS_DIR / "decisions"
DECISIONS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class BacktestResult:
    group: str
    model: str
    seed: int
    dates: list[pd.Timestamp]
    equity: pd.Series           # indexed by realized date
    port_returns: pd.Series     # daily portfolio returns
    weights: pd.DataFrame       # decision_date x ticker target weights
    exposure: pd.Series         # decision_date -> total invested fraction
    confidence: pd.Series       # decision_date -> confidence
    next_day_returns: pd.DataFrame  # decision_date x ticker realized next-day return
    decisions: list[Decision]
    n_parse_fail: int


def _close_frame(group: Group) -> pd.DataFrame:
    df = load_prices(group.window)
    close = df["Close"] if "Close" in df.columns.get_level_values(0) else df
    return close[UNIVERSE].sort_index()


def _cache_file(group: Group, seed: int, client_name: str) -> Path:
    # Cache key includes the client name so mock and real (and different models) never collide.
    safe = client_name.replace(":", "-").replace("/", "-")
    return DECISIONS_DIR / f"{group.name}_{safe}_seed{seed}.jsonl"


def _load_cache(path: Path) -> dict[str, Decision]:
    out: dict[str, Decision] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        out[d["date"]] = Decision(**d)
    return out


def run_backtest(group: Group, client: LLMClient, seed: int,
                 temperature: float = 0.7, verbose: bool = False,
                 max_days: int | None = None) -> BacktestResult:
    close = _close_frame(group)
    rets = close.pct_change()
    days = trading_days(group.window)
    if max_days:
        days = days[: max_days + 1]
    decision_days = days[:-1]          # need T+1 to realize the return
    agent = TradingAgent(client, UNIVERSE, temperature=temperature)

    cache_path = _cache_file(group, seed, client.name)
    cache = _load_cache(cache_path)

    weights_rows, exposure, confidence, decisions = {}, {}, {}, []
    nd_returns = {}
    equity_vals, port_rets = [], []
    equity = 1.0
    n_parse_fail = 0
    prev_ret, prev_expo = None, None
    last_weights: dict[str, float] = {}  # carried forward on a parse failure

    with cache_path.open("a") as fh:
        for i, day in enumerate(decision_days):
            key = day.date().isoformat()
            if key in cache:
                dec = cache[key]
            else:
                ctx = build_context(group.window, day)
                reflection = (build_reflection(prev_ret, prev_expo)
                              if prev_ret is not None else None)
                text = render_context(ctx, portfolio={t: 0.0 for t in UNIVERSE}, cash=equity)
                dec = agent.decide(key, text, seed=seed + i, reflection=reflection)
                fh.write(json.dumps(asdict(dec)) + "\n")
                fh.flush()
            decisions.append(dec)

            # On a parse failure carry forward the prior book (realistic) for the EQUITY curve,
            # but DO NOT record the day in the foresight-metric inputs — coercing it to all-cash
            # would differentially attenuate leakage metrics if failures cluster on event days.
            if dec.parse_ok:
                w = dec.target_weights
                last_weights = w
                weights_rows[day] = {t: w.get(t, 0.0) for t in UNIVERSE}
                exposure[day] = dec.exposure
                confidence[day] = dec.confidence
                nxt = days[i + 1]
                nd_returns[day] = rets.loc[nxt].to_dict()
            else:
                n_parse_fail += 1
                w = last_weights  # hold yesterday's positions

            # Realize on next day (equity curve covers every day for a continuous series).
            nxt = days[i + 1]
            r_next = rets.loc[nxt]
            port_r = float(sum(w.get(t, 0.0) * float(r_next[t]) for t in UNIVERSE))
            equity *= (1.0 + port_r)
            equity_vals.append(equity)
            port_rets.append(port_r)
            prev_ret, prev_expo = port_r, float(sum(w.values()))
            if verbose:
                print(f"  {key} ok={dec.parse_ok} expo={sum(w.values()):.2f} "
                      f"r_next={port_r:+.4f} eq={equity:.4f}")

    realized_dates = days[1: len(decision_days) + 1]

    def _panel(d: dict) -> pd.DataFrame:
        return (pd.DataFrame(d).T[UNIVERSE] if d
                else pd.DataFrame(columns=UNIVERSE, dtype=float))

    if n_parse_fail:
        print(f"  [warn] {group.name} seed{seed}: {n_parse_fail}/{len(decision_days)} "
              f"parse failures (excluded from foresight metrics)")
    return BacktestResult(
        group=group.name,
        model=group.model.tag,
        seed=seed,
        dates=list(decision_days),
        equity=pd.Series(equity_vals, index=realized_dates, name="equity"),
        port_returns=pd.Series(port_rets, index=realized_dates, name="ret"),
        weights=_panel(weights_rows),
        exposure=pd.Series(exposure, name="exposure"),
        confidence=pd.Series(confidence, name="confidence"),
        next_day_returns=_panel(nd_returns),
        decisions=decisions,
        n_parse_fail=n_parse_fail,
    )
