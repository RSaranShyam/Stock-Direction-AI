"""Feature engineering and target construction.

Every feature at day t uses data from day t or earlier only (verified in tests/test_leakage.py).
Features are *stationary* (returns / ratios), because raw prices drift and mislead neural networks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / down.replace(0, np.nan)
    return (100 - 100 / (1 + rs)) / 100  # scaled to 0-1


def engineered_features(df: pd.DataFrame, use_volume: bool = True) -> pd.DataFrame:
    c, o, h, l, v = df["Close"], df["Open"], df["High"], df["Low"], df["Volume"]
    logret = np.log(c).diff()
    f = pd.DataFrame(index=df.index)
    f["ret_1"] = logret
    f["ret_5"] = np.log(c).diff(5)
    f["ret_10"] = np.log(c).diff(10)
    for n in (5, 10, 20):
        f[f"ma{n}_ratio"] = c / c.rolling(n).mean() - 1
    f["rsi_14"] = _rsi(c)
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    f["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / c
    f["bb_width"] = 4 * c.rolling(20).std() / c.rolling(20).mean()
    f["vol_10"] = logret.rolling(10).std()
    f["vol_20"] = logret.rolling(20).std()
    f["hl_range"] = (h - l) / c
    f["oc_ret"] = (c - o) / o
    if use_volume:
        f["volume_chg"] = np.log1p(v).diff().clip(-2, 2)
    return f


def raw_features(df: pd.DataFrame, use_volume: bool = True) -> pd.DataFrame:
    """The 'textbook' set from the brief: OHLCV + moving averages (non-stationary on purpose)."""
    f = df[["Open", "High", "Low", "Close"]].copy()
    if use_volume:
        f["Volume"] = df["Volume"]
    f["MA5"] = df["Close"].rolling(5).mean()
    f["MA20"] = df["Close"].rolling(20).mean()
    return f


def build_features(df: pd.DataFrame, feature_set: str = "engineered", use_volume: bool = True) -> pd.DataFrame:
    fn = {"engineered": engineered_features, "raw": raw_features}[feature_set]
    return fn(df, use_volume).replace([np.inf, -np.inf], np.nan)


def make_target(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """y[t] = 1 if Close[t+1] > Close[t] else 0.  The last row has no tomorrow -> NaN.

    Also returns next-day simple return (used only for the backtest, never as a feature).
    """
    nxt = df["Close"].shift(-1)
    y = (nxt > df["Close"]).astype(float).where(nxt.notna())
    next_ret = nxt / df["Close"] - 1
    return y.rename("y"), next_ret.rename("next_ret")
