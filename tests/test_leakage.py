"""Leakage and alignment tests. If any of these fail, every result in the project is invalid."""
import numpy as np
import pandas as pd
import pytest

from src.data import clean_prices, make_synthetic
from src.features import build_features, make_target
from src.windows import build_dataset

KW = dict(window=20, feature_set="engineered", scaler_kind="standard",
          train_end="2018-12-31", val_end="2020-06-30")


@pytest.fixture(scope="module")
def prices():
    df, _ = clean_prices(make_synthetic("2014-01-01", "2022-12-31", seed=1))
    return df


def test_target_is_next_day_direction(prices):
    y, _ = make_target(prices)
    c = prices["Close"]
    manual = (c.shift(-1) > c).astype(float)
    assert (y.iloc[:-1] == manual.iloc[:-1]).all()
    assert np.isnan(y.iloc[-1])            # no tomorrow for the last row


def test_features_are_causal(prices):
    """Features at day t must not change if we delete all data after t."""
    full = build_features(prices)
    for cut in (500, 900, 1500):
        part = build_features(prices.iloc[: cut + 1])
        a, b = full.iloc[cut].values, part.iloc[cut].values
        assert np.allclose(a, b, equal_nan=True), f"feature at row {cut} depends on the future"


def test_split_is_chronological_and_disjoint(prices):
    ds = build_dataset(prices, **KW)
    assert ds.train.dates.max() <= pd.Timestamp(KW["train_end"])
    assert ds.train.dates.max() < ds.val.dates.min()
    assert ds.val.dates.max() <= pd.Timestamp(KW["val_end"])
    assert ds.val.dates.max() < ds.test.dates.min()
    assert len(ds.train.X) and len(ds.val.X) and len(ds.test.X)


def test_scaler_fit_on_train_only(prices):
    ds = build_dataset(prices, **KW)
    feats = build_features(prices).dropna()
    train_rows = feats.loc[: KW["train_end"]]
    assert np.allclose(ds.scaler.mean_, train_rows.values.mean(axis=0))
    assert not np.allclose(ds.scaler.mean_, feats.values.mean(axis=0))  # differs from whole-data mean


def test_window_ends_on_label_day_and_label_is_tomorrow(prices):
    ds = build_dataset(prices, **KW)
    feats = build_features(prices).dropna()
    y, _ = make_target(prices)
    i = 100
    t = ds.test.dates[i]
    # last row of the window equals the scaled features of day t
    expected = ds.scaler.transform(feats.loc[[t]].values)[0]
    assert np.allclose(ds.test.X[i, -1], expected, atol=1e-6)
    # and the label is the direction from t to the NEXT trading day
    assert ds.test.y[i] == y.loc[t]


def test_ret_t_is_today_not_tomorrow(prices):
    ds = build_dataset(prices, **KW)
    t = ds.test.dates[10]
    today = np.log(prices["Close"]).diff().loc[t]
    assert np.isclose(ds.test.ret_t[10], today)
