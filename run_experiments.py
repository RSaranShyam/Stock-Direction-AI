"""End-to-end experiment runner.

    python run_experiments.py                      # real data from Yahoo Finance (needs internet)
    python run_experiments.py --ticker ^NSEI       # NIFTY 50 index
    python run_experiments.py --source synthetic   # offline smoke test on FAKE data
    python run_experiments.py --quick              # fewer seeds / epochs, for a fast check

Outputs: reports/figures/*.png, reports/results.md, reports/results.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from dataclasses import replace

import numpy as np
import pandas as pd

from src import evaluate as ev
from src.config import ROOT, Config
from src.data import clean_prices, load_prices
from src.features import build_features, make_target
from src.models import LogisticBaseline, MajorityBaseline, PreviousDayBaseline
from src.train import fit_model, predict
from src.windows import build_dataset, walk_forward_datasets


def parse() -> Config:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default=Config.ticker)
    ap.add_argument("--source", default=Config.source, choices=["yfinance", "csv", "synthetic"])
    ap.add_argument("--csv", default=None)
    ap.add_argument("--start", default=Config.start)
    ap.add_argument("--end", default=Config.end)
    ap.add_argument("--seed", type=int, default=Config.seed)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    cfg = Config(ticker=a.ticker, source=a.source, csv_path=a.csv, start=a.start, end=a.end, seed=a.seed)
    cfg.quick = a.quick  # type: ignore[attr-defined]
    return cfg


def ds_for(prices, cfg, use_volume, **over):
    kw = dict(window=cfg.window, feature_set=cfg.feature_set, scaler_kind=cfg.scaler,
              train_end=cfg.train_end, val_end=cfg.val_end, use_volume=use_volume)
    kw.update(over)
    return build_dataset(prices, **kw)


def score(name, p, split, cfg, extra=None):
    m = ev.metrics(split.y, p, cfg.threshold)
    m["model"] = name
    m.update(extra or {})
    return m


def main():
    t0 = time.time()
    cfg = parse()
    quick = getattr(cfg, "quick", False)
    if quick:
        cfg = replace(cfg, epochs=15, patience=4)
    seeds = [cfg.seed] if quick else [cfg.seed, cfg.seed + 1, cfg.seed + 2]
    out = {"config": cfg.to_dict(), "data_note": "SYNTHETIC DATA - NOT REAL MARKET DATA" if cfg.source == "synthetic" else "real"}

    # ============================================================ 1-3: data, cleaning, EDA
    raw = load_prices(cfg.ticker, cfg.start, cfg.end, cfg.source, cfg.csv_path)
    prices, report = clean_prices(raw)
    out["cleaning_report"] = report
    use_vol = report["has_volume"]
    print("cleaning:", {k: v for k, v in report.items() if k != "outlier_dates"})

    ds = ds_for(prices, cfg, use_vol)
    print(f"samples train/val/test = {len(ds.train.y)}/{len(ds.val.y)}/{len(ds.test.y)}  features={len(ds.feature_names)}")
    ev.fig_price_split(prices, cfg.train_end, cfg.val_end, cfg.ticker)
    ev.fig_class_balance(ds.train.y, ds.val.y, ds.test.y)
    ev.fig_returns_eda(prices)
    ev.fig_corr(build_features(prices, "engineered", use_vol).dropna())
    out["class_balance"] = {k: float(getattr(ds, k).y.mean()) for k in ("train", "val", "test")}
    majority_rate_test = max(ds.test.y.mean(), 1 - ds.test.y.mean())
    out["majority_rate_test"] = float(majority_rate_test)

    rows, probs_test = [], {}

    # ============================================================ 6: baselines
    for B in (MajorityBaseline, PreviousDayBaseline, LogisticBaseline):
        b = B().fit(ds.train)
        p = b.predict_proba(ds.test)
        probs_test[b.name] = p
        rows.append(score(b.name, p, ds.test, cfg, {"seed": 0}))

    # ============================================================ 7: neural models x seeds
    hist_for_plot, models_keep = {}, {}
    for kind in ("mlp", "rnn", "lstm", "gru"):
        ps = []
        for s in seeds:
            c2 = replace(cfg, seed=s)
            model, h = fit_model(kind, ds, c2, tag=f"{kind}_s{s}")
            p = predict(model, ds.test)
            ps.append(p)
            rows.append(score(kind.upper(), p, ds.test, cfg, {"seed": s, "epochs_run": len(h["loss"])}))
            if s == seeds[0]:
                hist_for_plot[kind.upper()] = h
                models_keep[kind] = model
        probs_test[kind.upper()] = np.mean(ps, axis=0)  # seed-ensemble used for ROC / tests / backtest
        print(f"  {kind.upper()} done  ({time.time() - t0:.0f}s)")

    res = pd.DataFrame(rows)
    summary = (res.groupby("model", sort=False)[["accuracy", "precision", "recall", "f1", "roc_auc"]]
               .agg(["mean", "std"]).round(4))
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.fillna(0.0)
    out["model_comparison"] = summary.reset_index().to_dict(orient="records")
    comp = summary.reset_index().rename(columns={"accuracy_mean": "accuracy_mean", "accuracy_std": "accuracy_std"})
    comp["baseline"] = majority_rate_test
    ev.fig_model_comparison(comp[["model", "accuracy_mean", "accuracy_std", "baseline"]])
    ev.fig_history({k: hist_for_plot[k] for k in ("LSTM", "GRU", "MLP")})

    # ============================================================ 11: small hyper-parameter search (LSTM, validation only)
    grid = list(itertools.product([(32, 16), (64, 32)], [0.2, 0.4], [1e-3, 5e-4]))
    if quick:
        grid = grid[:2]
    tune = []
    for units, drop, lr in grid:
        model, h = fit_model("lstm", ds, cfg, units=units, dropout=drop, lr=lr, tag="tune")
        pv = predict(model, ds.val)
        tune.append({"units": str(units), "dropout": drop, "lr": lr, "val_loss": min(h["val_loss"]),
                     "val_auc": ev.metrics(ds.val.y, pv)["roc_auc"]})
    tune = pd.DataFrame(tune).sort_values("val_loss")
    out["tuning"] = tune.round(4).to_dict(orient="records")
    best = tune.iloc[0]
    best_units, best_drop, best_lr = eval(best["units"]), float(best["dropout"]), float(best["lr"])
    print("best LSTM hyper-parameters (by validation loss):", best.to_dict())

    # tuned LSTM, seed-ensemble -> the headline model
    tuned_ps, tuned_models = [], []
    for s in seeds:
        model, h = fit_model("lstm", ds, replace(cfg, seed=s), units=best_units, dropout=best_drop, lr=best_lr,
                             tag=f"lstm_tuned_s{s}")
        tuned_ps.append(predict(model, ds.test))
        tuned_models.append(model)
        rows.append(score("LSTM (tuned)", tuned_ps[-1], ds.test, cfg, {"seed": s, "epochs_run": len(h["loss"])}))
        if s == seeds[0]:
            tuned_hist = h
    p_lstm = np.mean(tuned_ps, axis=0)
    probs_test["LSTM (tuned)"] = p_lstm
    res = pd.DataFrame(rows)
    summary = (res.groupby("model", sort=False)[["accuracy", "precision", "recall", "f1", "roc_auc"]]
               .agg(["mean", "std"]).round(4))
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.fillna(0.0)
    out["model_comparison"] = summary.reset_index().to_dict(orient="records")
    comp = summary.reset_index()
    comp["baseline"] = majority_rate_test
    ev.fig_model_comparison(comp[["model", "accuracy_mean", "accuracy_std", "baseline"]])
    ev.fig_history({"LSTM (tuned)": tuned_hist, "GRU": hist_for_plot["GRU"], "MLP": hist_for_plot["MLP"]})

    # ============================================================ 12: evaluation of headline model
    thr = cfg.threshold
    ev.fig_confusion(ds.test.y, p_lstm, thr, "LSTM (tuned, seed-ensemble)")
    ev.fig_roc(ds.test.y, {k: probs_test[k] for k in ("Logistic Regression", "MLP", "LSTM (tuned)", "GRU")})
    out["tests"] = {
        "binom_p_vs_coinflip": ev.binomial_vs_baseline(ds.test.y, p_lstm, thr, 0.5),
        "binom_p_vs_majority": ev.binomial_vs_baseline(ds.test.y, p_lstm, thr, majority_rate_test),
        "mcnemar_p_lstm_vs_mlp": ev.mcnemar_exact(ds.test.y, p_lstm, probs_test["MLP"], thr),
        "mcnemar_p_lstm_vs_logreg": ev.mcnemar_exact(ds.test.y, p_lstm, probs_test["Logistic Regression"], thr),
    }

    # ============================================================ window sweep + feature-set + scaler ablations
    ab = []
    for w in ([20, 30] if quick else [10, 20, 30, 60]):
        d = ds_for(prices, cfg, use_vol, window=w)
        m, h = fit_model("lstm", d, cfg, units=best_units, dropout=best_drop, lr=best_lr, tag="abl")
        mm = ev.metrics(d.test.y, predict(m, d.test), thr)
        ab.append({"experiment": f"window={w}", "window": w, "accuracy": mm["accuracy"], "auc": mm["roc_auc"]})
    sweep = pd.DataFrame([a for a in ab])
    ev.fig_window_sweep(sweep)
    for label, over in [("features=raw OHLCV+MA", dict(feature_set="raw")),
                        ("scaler=minmax", dict(scaler_kind="minmax")),
                        ("scaler=standard", dict(scaler_kind="standard")),
                        ("features=engineered", dict(feature_set="engineered"))]:
        d = ds_for(prices, cfg, use_vol, **over)
        m, h = fit_model("lstm", d, cfg, units=best_units, dropout=best_drop, lr=best_lr, tag="abl")
        mm = ev.metrics(d.test.y, predict(m, d.test), thr)
        ab.append({"experiment": label, "window": cfg.window, "accuracy": mm["accuracy"], "auc": mm["roc_auc"]})
    out["ablations"] = pd.DataFrame(ab).round(4).to_dict(orient="records")
    print("ablations done", f"({time.time() - t0:.0f}s)")

    # ============================================================ walk-forward by year
    years = list(range(2021, 2026)) if not quick else [2024, 2025]
    wf = []
    last_year = prices.index[-1].year
    for yr, d in walk_forward_datasets(prices, [y for y in years if y <= last_year], window=cfg.window,
                                       feature_set=cfg.feature_set, scaler_kind=cfg.scaler, use_volume=use_vol):
        if len(d.test.y) < 50:
            continue
        m, h = fit_model("lstm", d, cfg, units=best_units, dropout=best_drop, lr=best_lr, tag="wf")
        p = predict(m, d.test)
        mm = ev.metrics(d.test.y, p, thr)
        wf.append({"year": yr, "n_test": len(d.test.y), "lstm_acc": mm["accuracy"], "lstm_auc": mm["roc_auc"],
                   "majority_acc": max(d.test.y.mean(), 1 - d.test.y.mean())})
    wf = pd.DataFrame(wf)
    if len(wf):
        ev.fig_walk_forward(wf)
        out["walk_forward"] = wf.round(4).to_dict(orient="records")
        out["walk_forward_summary"] = {"lstm_acc_mean": float(wf.lstm_acc.mean()), "lstm_acc_std": float(wf.lstm_acc.std(ddof=0)),
                                       "folds_beating_majority": int((wf.lstm_acc > wf.majority_acc).sum()), "folds": len(wf)}
    print("walk-forward done", f"({time.time() - t0:.0f}s)")

    # ============================================================ 13: interpretation
    imp = ev.permutation_importance(tuned_models[0], ds.test, ds.feature_names, n_repeats=3 if quick else 5)
    ev.fig_importance(imp)
    out["importance_top5"] = imp.head(5).round(4).to_dict(orient="records")
    ev.fig_calibration(ds.test.y, p_lstm)
    vol20 = build_features(prices, "engineered", use_vol)["vol_20"].reindex(ds.test.dates).values
    out["accuracy_by_vol_regime"] = ev.fig_regime(ds.test.dates, ds.test.y, p_lstm, thr, vol20)

    # ============================================================ 14: backtest
    bt = ev.backtest(ds.test.next_ret, p_lstm, thr, cfg.cost)
    ev.fig_equity(bt, ds.test.dates, f"{cfg.ticker}: LSTM long/cash vs buy-and-hold (test period, 0.1% cost)")
    out["backtest"] = ev.perf_summary(bt)
    out["pct_days_long"] = float((p_lstm >= thr).mean())

    out["runtime_seconds"] = round(time.time() - t0)
    (ROOT / "reports").mkdir(exist_ok=True)
    with open(ROOT / "reports" / "results.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    write_markdown(out, summary, sweep, pd.DataFrame(out["ablations"]), wf, tune, imp)
    print(f"finished in {time.time() - t0:.0f}s -> reports/results.md")


def write_markdown(out, summary, sweep, abl, wf, tune, imp):
    cfg = out["config"]
    md = [f"# Results: {cfg['ticker']} next-day direction ({cfg['start']} to {cfg['end']})\n"]
    if out["data_note"].startswith("SYNTHETIC"):
        md.append("> **SYNTHETIC DATA. These numbers only prove the pipeline runs; they say nothing about real markets.**\n")
    cb = out["class_balance"]
    md.append(f"Class balance (share of UP days): train {cb['train']:.1%}, val {cb['val']:.1%}, test {cb['test']:.1%}. "
              f"Best constant guess on test (majority class, known only in hindsight) = {out['majority_rate_test']:.1%}; this is the strictest baseline. The 'Majority class' row is the achievable version that uses the train-period majority.\n")
    md.append("## Model comparison (test set, mean over seeds)\n")
    md.append(summary.reset_index().to_markdown(index=False))
    md.append("\n## Statistical tests (LSTM tuned, seed-ensemble)\n")
    for k, v in out["tests"].items():
        md.append(f"- {k}: p = {v:.4f}")
    md.append("\n## Ablations\n")
    md.append(abl.to_markdown(index=False))
    md.append("\n## Walk-forward by year\n")
    if len(wf):
        md.append(wf.round(4).to_markdown(index=False))
    md.append("\n## Hyper-parameter search (validation only)\n")
    md.append(tune.round(4).to_markdown(index=False))
    md.append("\n## Permutation importance (top 8)\n")
    md.append(imp.head(8).round(4).to_markdown(index=False))
    md.append("\n## Backtest (test period)\n")
    md.append(pd.DataFrame(out["backtest"]).T.round(4).to_markdown())
    md.append(f"\nShare of days the model was long: {out['pct_days_long']:.1%}\n")
    (ROOT / "reports" / "results.md").write_text("\n".join(md))


if __name__ == "__main__":
    main()
