"""Predict whether the NEXT trading session closes UP or DOWN, for any stock.

    python predict_tomorrow.py --ticker INFY.NS --train    # one time per stock (5-10 min): trains its final models
    python predict_tomorrow.py --ticker INFY.NS            # any time after: fresh data -> new prediction (seconds)
    python predict_tomorrow.py --ticker AAPL --train --seeds 1     # faster: 1 model instead of 3 (~2-3 min)
    python predict_tomorrow.py --train --source synthetic --quick  # offline smoke test on FAKE data

Without --ticker it uses the stock of your last full study. Machine-learning exercise, not financial advice.
"""
import argparse
import os
from dataclasses import replace

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from src.predict import config_from_results, predict_latest, train_final  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--train", action="store_true", help="(re)train this stock's final models first")
ap.add_argument("--ticker", help="e.g. TCS.NS, INFY.NS, RELIANCE.NS, ^NSEI, AAPL")
ap.add_argument("--source", choices=["yfinance", "csv", "synthetic"])
ap.add_argument("--csv")
ap.add_argument("--seeds", type=int, default=3, help="number of models in the ensemble (default 3)")
ap.add_argument("--quick", action="store_true", help="1 seed, few epochs (smoke test only)")
a = ap.parse_args()

cfg = config_from_results()
if a.ticker:
    cfg = replace(cfg, ticker=a.ticker.strip())
if a.source:
    cfg = replace(cfg, source=a.source)
if a.csv:
    cfg = replace(cfg, csv_path=a.csv)

if a.train:
    n = 1 if a.quick else max(1, a.seeds)
    if a.quick:
        cfg = replace(cfg, epochs=8, patience=3)
    train_final(cfg, seeds=tuple(range(42, 42 + n)))
else:
    predict_latest(cfg.ticker)
print("\nOpen the dashboard to see it:  streamlit run dashboard.py")
