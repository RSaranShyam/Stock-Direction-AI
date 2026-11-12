"""Chronological split, train-only scaling and sliding-window sequence construction."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .features import build_features, make_target


@dataclass
class Split:
    X: np.ndarray            # (n, window, n_features)
    y: np.ndarray            # (n,)
    dates: pd.DatetimeIndex  # date t of the LAST row of each window (label = direction of t+1)
    next_ret: np.ndarray     # next-day simple return (backtest only)
    ret_t: np.ndarray        # today's log return (for the previous-day baseline only)


@dataclass
class Dataset:
    train: Split
    val: Split
    test: Split
    scaler: object
    feature_names: list
    window: int
    meta: dict = field(default_factory=dict)


def get_scaler(kind: str):
    return {"standard": StandardScaler, "minmax": MinMaxScaler}[kind]()


def make_windows(arr: np.ndarray, window: int) -> np.ndarray:
    """(N, F) -> (N-window+1, window, F); window i covers rows i..i+window-1."""
    w = sliding_window_view(arr, (window, arr.shape[1]))[:, 0]  # (N-w+1, w, F)
    return np.ascontiguousarray(w)


def build_dataset(prices: pd.DataFrame, *, window: int, feature_set: str, scaler_kind: str,
                  train_end: str, val_end: str, test_end: str | None = None,
                  use_volume: bool = True) -> Dataset:
    """Features -> target -> chronological split -> scaler fit on TRAIN rows only -> windows.

    A sample is identified by its end date t. It is assigned to train / val / test by t alone,
    so no label from a later period is ever seen during fitting. Windows of early val/test samples
    may contain *feature* rows from the previous period (that is just past information).
    """
    feats = build_features(prices, feature_set, use_volume)
    y, nret = make_target(prices)
    ret_t = np.log(prices["Close"]).diff().rename("ret_t")
    frame = pd.concat([feats, y, nret, ret_t], axis=1).dropna(subset=list(feats.columns) + ["ret_t"])
    # last row has no label; keep it out of supervised data
    frame = frame.dropna(subset=["y", "next_ret"])
    names = list(feats.columns)

    t_train, t_val = pd.Timestamp(train_end), pd.Timestamp(val_end)
    t_test = pd.Timestamp(test_end) if test_end else frame.index[-1]
    idx = frame.index

    scaler = get_scaler(scaler_kind).fit(frame.loc[idx <= t_train, names].values)  # TRAIN ONLY
    Xs = scaler.transform(frame[names].values)
    W = make_windows(Xs, window)                       # sample j ends at row j+window-1
    end_pos = np.arange(window - 1, len(frame))
    end_dates = idx[end_pos]

    def pack(mask):
        sel = end_pos[mask]
        return Split(W[mask], frame["y"].values[sel].astype(np.float32), end_dates[mask],
                     frame["next_ret"].values[sel], frame["ret_t"].values[sel])

    tr = end_dates <= t_train
    va = (end_dates > t_train) & (end_dates <= t_val)
    te = (end_dates > t_val) & (end_dates <= t_test)
    ds = Dataset(pack(tr), pack(va), pack(te), scaler, names, window,
                 meta={"train_end": train_end, "val_end": val_end, "scaler": scaler_kind,
                       "feature_set": feature_set})
    return ds


def walk_forward_datasets(prices: pd.DataFrame, years: list[int], *, window: int, feature_set: str,
                          scaler_kind: str, use_volume: bool = True):
    """Expanding-window folds. Fold for year Y: train < Y-6 months, val = last 6 months before Y, test = Y."""
    for yr in years:
        val_start = pd.Timestamp(f"{yr - 1}-07-01")
        yield yr, build_dataset(
            prices, window=window, feature_set=feature_set, scaler_kind=scaler_kind,
            train_end=(val_start - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            val_end=f"{yr - 1}-12-31", test_end=f"{yr}-12-31", use_volume=use_volume)
