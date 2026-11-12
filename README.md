<div align="center">

# 📈 Stock Direction AI

**An LSTM-based next-day stock direction predictor, built leakage-safe and judged honestly against baselines, with a Streamlit dashboard.**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-2ee59d)

</div>

> **Educational machine-learning project. Not financial or trading advice.**

## Overview
Given the last 30 trading days of price and volume features, the model predicts whether tomorrow's close will be **UP or DOWN**.
The point is not a magic accuracy number. Daily stock direction is close to a random walk, so the project is built to answer one
question properly: *does a neural network beat simple baselines, and is the difference statistically real?*

It compares an **MLP** against **SimpleRNN / LSTM / GRU** classifiers, evaluates them on years the models never saw, and reports
the result honestly, including when there is no edge.

## Highlights
- **No look-ahead:** chronological train / validation / test split, scaler fitted on training rows only, causal features, 6 automated leakage tests.
- **Baselines first:** majority class, previous-day direction, logistic regression, then MLP, RNN, LSTM, GRU (3 seeds each).
- **Real statistics:** accuracy, precision, recall, F1, ROC-AUC, exact binomial and McNemar tests, walk-forward validation by year.
- **Robustness checks:** window-length, feature-set and scaler ablations; permutation importance; calibration; volatility regimes; cost-aware backtest.
- **Live prediction:** an ensemble gives the next-session call with probability and signal strength, plus its past out-of-sample calls.
- **Streamlit dashboard:** intro page, stock picker (18 presets or any Yahoo Finance ticker), one-click training, charts and a results gallery.

## Quick start
```bash
git clone https://github.com/RSaranShyam/stock-direction-ai.git
cd stock-direction-ai
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements-full.txt
streamlit run dashboard.py        # opens http://localhost:8501
```
Run the whole pipeline (tests, study, final models, prediction, dashboard) with `python run_all.py`.

## Author
**R Saran Shyam**, B.Tech CSE (Artificial Intelligence & Data Engineering), Lovely Professional University.
[LinkedIn](https://www.linkedin.com/in/saranshyam/) · [GitHub](https://github.com/RSaranShyam)

## What makes this more than a notebook
| Practice | How it is implemented |
|---|---|
| No look-ahead | Chronological split (train to 2021, validation 2022-23, test 2024-25); never shuffled across time |
| No scaling leakage | Scaler fitted on **train rows only** (`src/windows.py`) |
| Causal features | Every feature at day t uses data up to t; proven by `tests/test_leakage.py` |
| Stationary inputs | Log returns, MA ratios, RSI, MACD, volatility; raw OHLCV compared as an ablation |
| Baselines first | Majority class, previous-day direction, Logistic Regression, MLP, then RNN / LSTM / GRU |
| Honest evaluation | Accuracy, precision, recall, F1, ROC-AUC, confusion matrix, exact binomial and McNemar tests, 3 seeds |
| Robustness | Walk-forward validation (test years 2021-2025), window-length / feature-set / scaler ablations |
| Interpretation | Permutation importance, calibration curve, accuracy by volatility regime, cost-aware backtest |
| Reproducibility | Fixed seeds, config in one file, cached raw data, one command to rerun |

## Just open the dashboard (Windows)
Double-click `start_dashboard.bat`. Your browser opens at http://localhost:8501. Keep the black window open while you use it.

## Run everything with one command
```bash
python run_all.py            # or double-click run_all.bat on Windows
```
Installs missing packages, runs the leakage tests, the full study, trains the final models, makes the next-session
prediction and opens the dashboard at http://localhost:8501. Finished steps are skipped on later runs.
Options: `--force` redo all, `--ticker ^NSEI` another stock, `--quick` fast smoke test, `--no-dashboard`.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-full.txt
pytest -q                                              # 6 leakage / alignment tests must pass
python run_experiments.py                              # real data (TCS.NS) - needs internet, ~10-20 min on CPU
python run_experiments.py --ticker ^NSEI               # NIFTY 50 index
python run_experiments.py --ticker RELIANCE.NS
python run_experiments.py --source synthetic --quick   # offline smoke test on FAKE data (~2 min)
jupyter notebook notebooks/stock_lstm_walkthrough.ipynb
```
### Choose any stock
In the dashboard, pick a stock from the list at the top (or type any Yahoo Finance ticker such as `MARUTI.NS` or `AMD`).
✅ = already trained; ⬜ = press **Train and predict** once (2-12 min), then it refreshes in seconds. Each stock keeps its own
models and track record in `stocks/<TICKER>/`. From the command line:
```bash
python run_all.py --also INFY.NS RELIANCE.NS AAPL      # pre-train several stocks
python predict_tomorrow.py --ticker AAPL --train       # or one at a time (--seeds 1 for a faster 1-model version)
```
The full model study (comparison, ablations, walk-forward) is run for one main stock (`run_all.py --ticker X`).

### Predict the next session
```bash
python predict_tomorrow.py --train   # one time, 5-10 min: trains the final ensemble + out-of-sample history
python predict_tomorrow.py           # any time later: downloads the newest prices and predicts the next session
```
The dashboard's Prediction tab shows the call (UP/DOWN, P(up), signal strength), the model's past out-of-sample calls
and a refresh button. A 'Weak' signal means P(up) is within 3 points of 50%. Not financial advice.

### Dashboard (localhost)
```bash
pip install streamlit        # already in requirements-full.txt
streamlit run dashboard.py   # opens http://localhost:8501
```
It reads `reports/results.json` (created by `run_experiments.py`) and shows headline metrics, model comparison,
walk-forward and ablation charts, feature importance, backtest and all figures. Until you have run the experiments it
shows the clearly-flagged synthetic demo instead.

Outputs: `reports/results.md`, `reports/results.json`, `reports/figures/*.png`.
If Yahoo is unreachable, download a CSV with columns `Date,Open,High,Low,Close,Volume` and run
`python run_experiments.py --source csv --csv path/to/file.csv --ticker MYSTOCK`.

## Project layout
```
src/config.py     all settings (ticker, dates, window, hyper-parameters, cost)
src/data.py       download + cache + auditable cleaning (+ synthetic generator for smoke tests only)
src/features.py   stationary features and the next-day-direction target
src/windows.py    chronological split, train-only scaling, sliding windows, walk-forward folds
src/models.py     MLP / SimpleRNN / LSTM / GRU in Keras, plus classical baselines
src/train.py      training with early stopping, LR scheduling, checkpointing, seeding
src/evaluate.py   metrics, statistical tests, backtest, permutation importance, all figures
run_all.py        ONE command: tests -> study -> final models -> prediction -> dashboard
run_all.bat       double-click version for Windows
run_experiments.py   the full study, end to end
dashboard.py      Streamlit dashboard: next-session prediction + results (localhost:8501)
predict_tomorrow.py  train final models / predict the next trading session
src/predict.py    final-model training and live prediction (any ticker, stored in stocks/<TICKER>/)
tests/            leakage and alignment tests
notebooks/        guided walkthrough (one seed) for submission / viva
```

## Mapping to the 10-mark rubric
| Rubric component | Marks | Where |
|---|---|---|
| Problem definition and dataset | 1 | Notebook section 1, `src/data.py` |
| Data preprocessing | 1 | `clean_prices`, `src/features.py`, scaler comparison (standard vs minmax) |
| Neural-network implementation | 3 | `src/models.py` (LSTM, GRU, RNN, MLP) |
| Training and optimisation | 1 | `src/train.py`, grid search in `run_experiments.py` |
| Evaluation | 2 | `src/evaluate.py`, `reports/results.md` |
| Graphs / visualisation | 1 | `reports/figures/` (loss/accuracy per epoch, ROC, confusion matrix, equity curve ...) |
| Interpretation and conclusion | 1 | Notebook section 9, permutation importance, regime analysis |

## Reading the results
Daily direction is close to a random walk. A realistic outcome is **51-56 % test accuracy and ROC-AUC 0.50-0.57**.
Compare against the majority-class accuracy printed in `results.md`, not against 50 %. If you ever see 70-80 %,
suspect leakage. Report what you actually get and explain why the margin is small; that is a strong submission.

## Limitations
Price history only (no news, fundamentals or macro data); one or two instruments; regime changes; the backtest is a
simulation with a flat cost assumption and ignores slippage and taxes.
