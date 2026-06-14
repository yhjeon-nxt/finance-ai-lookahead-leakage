"""Price-data ingestion via yfinance, cached to local parquet.

We deliberately use **prices only** (OHLCV). The experiment tests *parametric* look-ahead
leakage: if the agent's context contains only data dated <= day T, any apparent foresight must
originate in the model's pre-training memory rather than in the input feed. Adding a news feed
would open a second leakage channel and confound the causal claim.

Caching is per-window and idempotent: re-running re-uses the parquet unless `force=True`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Allow `python -m leakage.data.ingest` and direct execution.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from leakage.config import RAW_DIR, UNIVERSE, WINDOWS, Window  # noqa: E402


def _cache_path(window: Window) -> Path:
    return RAW_DIR / f"prices_{window.name}.parquet"


def download_prices(window: Window, tickers: list[str] | None = None,
                    force: bool = False) -> pd.DataFrame:
    """Download adjusted OHLCV for `tickers` over the window's download range.

    Returns a tidy DataFrame indexed by date with a column MultiIndex (field, ticker).
    Cached to parquet; pass force=True to refresh.
    """
    import yfinance as yf

    tickers = tickers or UNIVERSE
    path = _cache_path(window)
    if path.exists() and not force:
        return pd.read_parquet(path)

    raw = yf.download(
        tickers,
        start=window.download_start.isoformat(),
        end=(pd.Timestamp(window.end) + pd.Timedelta(days=1)).date().isoformat(),
        auto_adjust=True,
        progress=False,
        group_by="column",
    )
    if raw.empty:
        raise RuntimeError(f"yfinance returned no data for {window} / {tickers}")

    # Normalise to a (field, ticker) column MultiIndex even for a single ticker.
    if not isinstance(raw.columns, pd.MultiIndex):
        raw.columns = pd.MultiIndex.from_product([raw.columns, [tickers[0]]])

    raw = raw.sort_index()
    raw.to_parquet(path)
    return raw


def load_prices(window: Window) -> pd.DataFrame:
    path = _cache_path(window)
    if not path.exists():
        return download_prices(window)
    return pd.read_parquet(path)


def trading_days(window: Window) -> list[pd.Timestamp]:
    """Actual trading days inside [window.start, window.end] (from the index)."""
    df = load_prices(window)
    mask = (df.index >= pd.Timestamp(window.start)) & (df.index <= pd.Timestamp(window.end))
    return list(df.index[mask])


if __name__ == "__main__":  # quick smoke / manual ingest
    for w in WINDOWS.values():
        try:
            df = download_prices(w, force=True)
            close = df["Close"] if "Close" in df.columns.get_level_values(0) else df
            td = trading_days(w)
            print(f"[ok] {w}: rows={len(df)} tickers={close.shape[1]} "
                  f"trading_days_in_window={len(td)} "
                  f"first={df.index.min().date()} last={df.index.max().date()}")
        except Exception as e:  # noqa: BLE001
            print(f"[FAIL] {w}: {type(e).__name__}: {e}")
