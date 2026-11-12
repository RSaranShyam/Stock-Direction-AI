"""Final models + next-session prediction, for ANY ticker.

Everything for one stock lives in stocks/<TICKER>/ :
    model/                  final LSTM ensemble + scaler + meta.json
    test_predictions.csv    out-of-sample history: what the model would have said each day after the validation period
    latest_prediction.json  the newest next-session call

train_final(cfg):   (1) honest history  (2) live ensemble trained on all labelled data
predict_latest(t):  download newest prices, build the last window, return P(next session closes UP)
"""
from __future__ import annotations

import ast
import json
import re
from dataclasses import replace

import joblib
import numpy as np
import pandas as pd

from .config import RAW_DIR, ROOT, MODEL_DIR, Config
from .data import OHLCV, _download_yfinance, clean_prices, load_prices, make_synthetic
from .features import build_features
from .train import fit_model, predict
from .windows import build_dataset

STOCKS_DIR = ROOT / "stocks"
LEGACY_MODEL_DIR = MODEL_DIR / "final"      # location used by the first version of this project
REPORTS = ROOT / "reports"
VAL_DAYS = 120


def safe(ticker: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", ticker.strip())


def tdir(ticker: str):
    return STOCKS_DIR / safe(ticker)


def find_model_dir(ticker: str):
    d = tdir(ticker) / "model"
    if (d / "meta.json").exists():
        return d
    if (LEGACY_MODEL_DIR / "meta.json").exists():
        if json.loads((LEGACY_MODEL_DIR / "meta.json").read_text())["config"]["ticker"] == ticker:
            return LEGACY_MODEL_DIR
    return None


def config_from_results() -> Config:
    """Reuse source / dates of the last full run, so every stock is treated the same way."""
    p = REPORTS / "results.json"
    if p.exists():
        c = json.loads(p.read_text())["config"]
        c["units"] = tuple(c["units"])
        return Config(**c)
    return Config()


def best_hparams(cfg: Config):
    p = REPORTS / "results.json"
    if p.exists():
        t = json.loads(p.read_text()).get("tuning")
        if t:
            b = t[0]  # results.json stores the grid sorted by validation loss
            return tuple(ast.literal_eval(b["units"])), float(b["dropout"]), float(b["lr"])
    return cfg.units, cfg.dropout, cfg.lr


def fetch_prices(cfg: Config, drop_today: bool) -> tuple[pd.DataFrame, dict]:
    """Newest available daily bars. drop_today=True removes an unfinished bar (used for training)."""
    if cfg.source == "synthetic":   # offline smoke tests only; different tickers get different fake series
        seed = 0 if cfg.ticker == "TCS.NS" else sum(map(ord, cfg.ticker)) % 997
        df = make_synthetic(cfg.start, cfg.end, seed)
    elif cfg.source == "csv":
        df = load_prices(cfg.ticker, cfg.start, cfg.end, "csv", cfg.csv_path)
    else:
        end = (pd.Timestamp.today() + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
        cache = RAW_DIR / f"{safe(cfg.ticker)}_latest.csv"
        try:
            df = _download_yfinance(cfg.ticker, cfg.start, end)
            if df.empty:
                raise RuntimeError(f"no data returned for '{cfg.ticker}' (wrong ticker, or Yahoo is unreachable)")
            df.index.name = "Date"
            RAW_DIR.mkdir(parents=True, exist_ok=True)
            df.to_csv(cache)
        except Exception as e:  # offline / rate-limited: fall back to the last good download
            if not cache.exists():
                raise RuntimeError(f"Could not download {cfg.ticker}: {e}") from e
            df = pd.read_csv(cache, parse_dates=["Date"], index_col="Date")[OHLCV]
    df, report = clean_prices(df)
    if len(df) < 800:
        raise RuntimeError(f"{cfg.ticker} has only {len(df)} trading days since {cfg.start}; need at least 800 to train.")
    if drop_today and df.index[-1].date() >= pd.Timestamp.today().date():
        df = df.iloc[:-1]
    return df, report


def latest_window(prices, scaler, window, feature_set, use_volume):
    feats = build_features(prices, feature_set, use_volume).dropna()
    X = scaler.transform(feats.values)[-window:]
    return X[None].astype("float32"), feats.index[-1]


def _strength(p: float) -> str:
    d = abs(p - 0.5)
    return "Weak" if d < 0.03 else "Moderate" if d < 0.08 else "Strong"


def train_final(cfg: Config, seeds=(42, 43, 44), log=print) -> dict:
    units, dropout, lr = best_hparams(cfg)
    td = tdir(cfg.ticker)
    mdir = td / "model"
    td.mkdir(parents=True, exist_ok=True)
    log(f"[{cfg.ticker}] hyper-parameters: units={units} dropout={dropout} lr={lr}")
    prices, report = fetch_prices(cfg, drop_today=True)
    use_vol = report["has_volume"]
    kw = dict(window=cfg.window, feature_set=cfg.feature_set, scaler_kind=cfg.scaler, use_volume=use_vol)
    log(f"[{cfg.ticker}] data: {len(prices)} days, {report['first_date']} to {prices.index[-1].date()}")

    # ---- (1) honest out-of-sample history -------------------------------------------------
    ds = build_dataset(prices, train_end=cfg.train_end, val_end=cfg.val_end, **kw)
    ps = []
    for s in seeds:
        m, _ = fit_model("lstm", ds, replace(cfg, seed=s), units=units, dropout=dropout, lr=lr, tag=f"{safe(cfg.ticker)}_hist_s{s}")
        ps.append(predict(m, ds.test))
        log(f"[{cfg.ticker}] history model {seeds.index(s) + 1}/{len(seeds)} done")
    p = np.mean(ps, axis=0)
    hist = pd.DataFrame({"date": ds.test.dates, "close": prices["Close"].reindex(ds.test.dates).values,
                         "p_up": p, "actual_up": ds.test.y.astype(int), "next_ret": ds.test.next_ret})
    hist["pred_up"] = (hist.p_up >= 0.5).astype(int)
    hist["correct"] = (hist.pred_up == hist.actual_up).astype(int)
    hist["strength"] = hist.p_up.map(_strength)
    hist.to_csv(td / "test_predictions.csv", index=False)
    log(f"[{cfg.ticker}] history saved: {len(hist)} out-of-sample predictions, hit rate {hist.correct.mean():.1%}")

    # ---- (2) live model on all labelled data ----------------------------------------------
    labelled = prices.index[:-1]                       # last bar has no 'tomorrow' label yet
    dsb = build_dataset(prices, train_end=str(labelled[-1 - VAL_DAYS].date()), val_end=str(labelled[-1].date()), **kw)
    mdir.mkdir(parents=True, exist_ok=True)
    for old in mdir.glob("lstm_s*.keras"):
        old.unlink()
    for s in seeds:
        m, _ = fit_model("lstm", dsb, replace(cfg, seed=s), units=units, dropout=dropout, lr=lr, tag=f"{safe(cfg.ticker)}_live_s{s}")
        m.save(mdir / f"lstm_s{s}.keras")
        log(f"[{cfg.ticker}] live model {seeds.index(s) + 1}/{len(seeds)} saved")
    joblib.dump(dsb.scaler, mdir / "scaler.joblib")
    meta = {"config": cfg.to_dict(), "feature_names": dsb.feature_names, "use_volume": use_vol,
            "trained_through": str(labelled[-1].date()), "units": list(units), "dropout": dropout, "lr": lr,
            "seeds": list(seeds)}
    (mdir / "meta.json").write_text(json.dumps(meta, indent=2))
    return predict_latest(cfg.ticker, log=log)


def predict_latest(ticker: str | None = None, log=print) -> dict:
    import tensorflow as tf  # noqa: F401  (imported lazily so the dashboard opens fast)
    from tensorflow import keras

    ticker = ticker or config_from_results().ticker
    mdir = find_model_dir(ticker)
    if mdir is None:
        raise FileNotFoundError(f"No trained model for {ticker} yet. Run:  python predict_tomorrow.py --ticker {ticker} --train")
    meta = json.loads((mdir / "meta.json").read_text())
    c = meta["config"]
    c["units"] = tuple(c["units"])
    cfg = Config(**c)
    scaler = joblib.load(mdir / "scaler.joblib")
    models = [keras.models.load_model(p) for p in sorted(mdir.glob("lstm_s*.keras"))]

    prices, _ = fetch_prices(cfg, drop_today=False)
    X, asof = latest_window(prices, scaler, cfg.window, cfg.feature_set, meta["use_volume"])
    votes = [float(m.predict(X, verbose=0)[0, 0]) for m in models]
    p = float(np.mean(votes))
    tail = prices.tail(120)
    ma20 = prices["Close"].rolling(20).mean().reindex(tail.index)
    out = {
        "ticker": cfg.ticker, "source": cfg.source,
        "as_of": str(asof.date()), "last_close": float(prices["Close"].iloc[-1]),
        "next_session": str((asof + pd.offsets.BDay(1)).date()),
        "p_up": p, "votes": votes, "direction": "UP" if p >= 0.5 else "DOWN",
        "strength": _strength(p), "trained_through": meta["trained_through"],
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "tail": {"date": [str(d.date()) for d in tail.index], "close": tail["Close"].round(4).tolist(),
                 "ma20": [None if np.isnan(v) else round(float(v), 4) for v in ma20]},
    }
    td = tdir(cfg.ticker)
    td.mkdir(parents=True, exist_ok=True)
    (td / "latest_prediction.json").write_text(json.dumps(out, indent=2))
    log(f"[{cfg.ticker}] P(UP) = {p:.1%} for session after {out['as_of']} -> {out['direction']} ({out['strength']})")
    return out
