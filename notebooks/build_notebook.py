"""Generates notebooks/stock_lstm_walkthrough.ipynb (run once: python notebooks/build_notebook.py)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = lambda s: nb.cells.append(nbf.v4.new_code_cell(s.strip("\n")))
M = lambda s: nb.cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))

M("""
# Stock Price Direction Prediction using LSTM
**Task:** predict whether tomorrow's closing price will move **UP or DOWN** (binary classification).
This is a machine-learning exercise, **not** a trading recommendation.

| Section | Rubric component |
|---|---|
| 1 | Problem definition and dataset |
| 2-3 | Data preprocessing |
| 4-6 | Neural-network implementation |
| 7 | Training and optimisation |
| 8 | Evaluation |
| 4, 7, 8 | Graphs / visualisation |
| 9 | Interpretation and conclusion |

The notebook is a guided, single-seed walkthrough. The full study (3 seeds, hyper-parameter search, ablations,
walk-forward validation) is `python run_experiments.py`.
""")

C("""
import sys, warnings, os
sys.path.insert(0, "..")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from IPython.display import Image, display
from src.config import Config, FIG_DIR
from src import evaluate as ev
from src.data import load_prices, clean_prices
from src.features import build_features, make_target
from src.windows import build_dataset
from src.models import MajorityBaseline, PreviousDayBaseline, LogisticBaseline
from src.train import fit_model, predict

# ---- choose the data here ----
cfg = Config(ticker="TCS.NS", source="yfinance")      # real data (needs internet)
# cfg = Config(ticker="TCS.NS", source="synthetic")   # offline smoke test on FAKE data
show = lambda name: display(Image(str(FIG_DIR / f"{name}.png")))
cfg
""")

M("""
## 1. Problem definition and dataset
* **Input:** the last `window` (30) trading days of features for one stock.
* **Target:** `y = 1` if `Close(t+1) > Close(t)` else `0`.
* **Why direction, not price?** Predicting the price level looks accurate only because tomorrow's price is close to today's;
  direction has an honest baseline (about 50-53%) so skill is measurable.
* **Data:** daily OHLCV from Yahoo Finance (`yfinance`), 2015-2025, cached untouched in `data/raw/`.
* **Split (chronological, never random):** train to 2021, validation 2022-23, test 2024-25.
""")
C("""
raw = load_prices(cfg.ticker, cfg.start, cfg.end, cfg.source, cfg.csv_path)
prices, report = clean_prices(raw)
print({k: v for k, v in report.items() if k != "outlier_dates"})
print("flagged outlier dates (kept, they are real events):", report["outlier_dates"][:10])
prices.tail()
""")

M("""
## 2. Exploratory analysis
What to look for: fat-tailed returns, volatility clustering, and **almost no autocorrelation** in returns
(the reason this problem is hard).
""")
C("""
ev.fig_price_split(prices, cfg.train_end, cfg.val_end, cfg.ticker); show("01_price_split")
ev.fig_returns_eda(prices); show("03_returns_eda")
""")

M("""
## 3. Preprocessing: features, target, split, scaling
* **Missing values / duplicates / non-positive prices** handled in `clean_prices`; holidays are not forward-filled.
* **Outliers:** flagged, not removed (crashes are genuine information).
* **Stationary features** (log returns, MA ratios, RSI, MACD, Bollinger width, volatility, volume change) instead of raw prices.
* **Standardisation vs normalisation:** `StandardScaler` (zero mean, unit variance) vs `MinMaxScaler` (0-1). We default to
  standardisation because out-of-sample values can fall outside the training range, which MinMax squeezes badly.
  Both are compared in `run_experiments.py`. **The scaler is fitted on training rows only.**
""")
C("""
ds = build_dataset(prices, window=cfg.window, feature_set=cfg.feature_set, scaler_kind=cfg.scaler,
                   train_end=cfg.train_end, val_end=cfg.val_end, use_volume=report["has_volume"])
print("features:", ds.feature_names)
for n in ("train", "val", "test"):
    s = getattr(ds, n); print(f"{n:5s} X={s.X.shape}  UP share={s.y.mean():.3f}  {s.dates[0].date()} -> {s.dates[-1].date()}")
ev.fig_class_balance(ds.train.y, ds.val.y, ds.test.y); show("02_class_balance")
ev.fig_corr(build_features(prices, "engineered", report["has_volume"]).dropna()); show("04_feature_corr")
""")

M("""
## 4. Baselines (the numbers the LSTM must beat)
""")
C("""
rows, probs = [], {}
for B in (MajorityBaseline, PreviousDayBaseline, LogisticBaseline):
    b = B().fit(ds.train); p = b.predict_proba(ds.test); probs[b.name] = p
    rows.append({"model": b.name, **ev.metrics(ds.test.y, p)})
pd.DataFrame(rows).round(4)
""")

M("""
## 5. Neural networks: MLP, RNN, LSTM, GRU
```
window (30 x features) -> LSTM(64, return_sequences) -> Dropout(0.3) -> LSTM(32) -> Dropout(0.3)
                       -> Dense(16, ReLU) -> Dense(1, Sigmoid)  -> P(UP)
```
Loss = binary cross-entropy, optimiser = Adam. **Why LSTM over a plain RNN:** the forget/input/output gates and the cell state
let gradients flow over many steps, which reduces the vanishing-gradient problem of a simple RNN.
""")
C("""
from src.models import build_model
build_model("lstm", cfg.window, len(ds.feature_names)).summary()
""")

M("""
## 6. Training and optimisation
Early stopping and learning-rate reduction both watch **validation** loss; the best epoch's weights are restored.
""")
C("""
results, hists, trained = {}, {}, {}
for kind in ("mlp", "rnn", "lstm", "gru"):
    model, h = fit_model(kind, ds, cfg, tag=f"nb_{kind}")
    p = predict(model, ds.test)
    results[kind.upper()] = p; hists[kind.upper()] = h; trained[kind] = model
    print(kind.upper(), "epochs run:", len(h["loss"]), " best val_loss:", round(min(h["val_loss"]), 4))
ev.fig_history({k: hists[k] for k in ("LSTM", "GRU", "MLP")}); show("05_training_curves")
""")

M("""
## 7. Evaluation on the untouched test set
Accuracy alone hides class imbalance, so we report precision, recall, F1 and ROC-AUC, and test whether the edge over
baselines is statistically real.
""")
C("""
probs.update(results)
table = pd.DataFrame([{"model": k, **ev.metrics(ds.test.y, p)} for k, p in probs.items()]).round(4)
display(table)
p_l = probs["LSTM"]
ev.fig_confusion(ds.test.y, p_l, 0.5, "LSTM"); show("06_confusion_matrix")
ev.fig_roc(ds.test.y, {k: probs[k] for k in ("Logistic Regression", "MLP", "LSTM", "GRU")}); show("07_roc")
maj = max(ds.test.y.mean(), 1 - ds.test.y.mean())
print("majority-class accuracy:", round(maj, 4))
print("p-value, LSTM accuracy > coin flip     :", round(ev.binomial_vs_baseline(ds.test.y, p_l, 0.5, 0.5), 4))
print("p-value, LSTM accuracy > majority class:", round(ev.binomial_vs_baseline(ds.test.y, p_l, 0.5, maj), 4))
print("McNemar p-value, LSTM vs MLP           :", round(ev.mcnemar_exact(ds.test.y, p_l, probs["MLP"]), 4))
""")

M("""
## 8. Backtest (simulation only)
Go long when P(UP) >= 0.5, otherwise stay in cash; 0.1% cost on every position change. Compared with buy-and-hold.
""")
C("""
bt = ev.backtest(ds.test.next_ret, p_l, 0.5, cfg.cost)
ev.fig_equity(bt, ds.test.dates, f"{cfg.ticker}: LSTM long/cash vs buy-and-hold"); show("11_equity_curve")
pd.DataFrame(ev.perf_summary(bt)).T.round(4)
""")

M("""
## 9. Interpretation and conclusion
""")
C("""
imp = ev.permutation_importance(trained["lstm"], ds.test, ds.feature_names, n_repeats=3)
ev.fig_importance(imp); show("12_feature_importance")
ev.fig_calibration(ds.test.y, p_l); show("13_calibration")
vol = build_features(prices, "engineered", report["has_volume"])["vol_20"].reindex(ds.test.dates).values
print(ev.fig_regime(ds.test.dates, ds.test.y, p_l, 0.5, vol)); show("14_regime_accuracy")
imp.head(8).round(4)
""")
M("""
### Conclusion (write this AFTER you have seen your real-data results)
Answer in your own words, using your numbers from the tables above and `reports/results.md`:
1. Did the LSTM beat the majority-class and previous-day baselines? By how much, and is the p-value convincing?
2. Did LSTM/GRU beat the plain RNN and MLP? Does that support the "gated memory" argument?
3. Which features mattered most (permutation importance), and does that make financial sense?
4. Was the result stable across years (walk-forward) and across volatility regimes?
5. Limitations: near-efficient markets, regime change, price-only inputs, transaction costs, one stock.
6. Future work: attention/Transformer, news-sentiment features, multi-stock pooled training, probability thresholding.
""")

nbf.write(nb, "notebooks/stock_lstm_walkthrough.ipynb")
print("notebook written")
