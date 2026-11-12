"""Data acquisition and cleaning.

Real data comes from Yahoo Finance (cached to data/raw so the raw download is never edited).
A synthetic generator is provided ONLY for offline smoke-tests of the pipeline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import RAW_DIR

OHLCV = ["Open", "High", "Low", "Close", "Volume"]


def make_synthetic(start: str, end: str, seed: int = 0) -> pd.DataFrame:
    """GARCH(1,1)-style daily returns with a very weak AR(1) term. NOT real market data."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, end)
    n = len(idx)
    omega, alpha, beta, phi = 2e-6, 0.08, 0.90, 0.03
    r = np.zeros(n)
    var = omega / (1 - alpha - beta)
    for t in range(1, n):
        var = omega + alpha * r[t - 1] ** 2 + beta * var
        r[t] = 0.0004 + phi * r[t - 1] + np.sqrt(var) * rng.standard_normal()
    close = 1000 * np.exp(np.cumsum(r))
    open_ = np.r_[close[0], close[:-1]] * np.exp(rng.normal(0, 0.003, n))
    spread = np.abs(rng.normal(0, 0.006, n)) + np.abs(r) * 0.5
    high = np.maximum(open_, close) * (1 + spread)
    low = np.minimum(open_, close) * (1 - spread)
    vol = (2e6 * np.exp(rng.normal(0, 0.25, n) + 20 * np.abs(r))).astype(float)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol}, index=idx)


def _download_yfinance(ticker: str, start: str, end: str) -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):  # newer yfinance returns (field, ticker)
        df.columns = df.columns.get_level_values(0)
    return df[OHLCV]


def load_prices(ticker: str, start: str, end: str, source: str = "yfinance",
                csv_path: str | None = None, seed: int = 0) -> pd.DataFrame:
    """Return a raw OHLCV DataFrame indexed by date."""
    if source == "synthetic":
        return make_synthetic(start, end, seed)
    if source == "csv":
        df = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")
        return df[OHLCV]
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache = RAW_DIR / f"{ticker.replace('^', '')}_{start}_{end}.csv"
    if cache.exists():
        return pd.read_csv(cache, parse_dates=["Date"], index_col="Date")
    df = _download_yfinance(ticker, start, end)
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker}. Check the ticker and your internet connection.")
    df.index.name = "Date"
    df.to_csv(cache)  # raw data is saved untouched
    return df


def clean_prices(df: pd.DataFrame, outlier_z: float = 6.0) -> tuple[pd.DataFrame, dict]:
    """Basic, auditable cleaning. Returns (clean_df, report).

    - sort by date, drop duplicate dates
    - drop rows with missing / non-positive prices (holidays are simply absent, never forward-filled)
    - FLAG (not remove) extreme returns: crashes such as March 2020 are real information
    """
    report = {"rows_raw": len(df)}
    df = df.sort_index()
    report["duplicate_dates"] = int(df.index.duplicated().sum())
    df = df[~df.index.duplicated(keep="first")]
    report["missing_values"] = int(df[OHLCV].isna().sum().sum())
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    bad = (df[["Open", "High", "Low", "Close"]] <= 0).any(axis=1)
    report["non_positive_prices"] = int(bad.sum())
    df = df[~bad]
    df["Volume"] = df["Volume"].fillna(0)
    r = np.log(df["Close"]).diff()
    z = (r - r.mean()) / r.std()
    report["outlier_days_flagged"] = int((z.abs() > outlier_z).sum())
    report["outlier_dates"] = [d.strftime("%Y-%m-%d") for d in df.index[z.abs() > outlier_z]]
    report["has_volume"] = bool((df["Volume"] > 0).mean() > 0.5)  # indices often report 0 volume
    report["rows_clean"] = len(df)
    report["first_date"] = df.index[0].strftime("%Y-%m-%d")
    report["last_date"] = df.index[-1].strftime("%Y-%m-%d")
    return df, report

