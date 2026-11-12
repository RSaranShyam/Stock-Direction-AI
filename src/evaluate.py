"""Metrics, statistical tests, backtest, interpretation and all report figures."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import binomtest
from sklearn.calibration import calibration_curve
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)

from .config import FIG_DIR

sns.set_theme(style="whitegrid", context="notebook")
C_MAIN, C_ALT, C_GREY = "#2563eb", "#dc2626", "#6b7280"


def _save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ metrics & tests
def metrics(y, p, thr=0.5) -> dict:
    yhat = (p >= thr).astype(int)
    try:
        auc = roc_auc_score(y, p) if len(np.unique(p)) > 1 else 0.5
    except ValueError:
        auc = float("nan")
    return {"accuracy": accuracy_score(y, yhat), "precision": precision_score(y, yhat, zero_division=0),
            "recall": recall_score(y, yhat, zero_division=0), "f1": f1_score(y, yhat, zero_division=0),
            "roc_auc": auc, "pct_pred_up": float(yhat.mean())}


def binomial_vs_baseline(y, p, thr=0.5, baseline_rate=0.5) -> float:
    """One-sided exact p-value: is accuracy better than a fixed baseline accuracy?"""
    hits = int(((p >= thr).astype(int) == y).sum())
    return binomtest(hits, len(y), baseline_rate, alternative="greater").pvalue


def mcnemar_exact(y, p_a, p_b, thr=0.5) -> float:
    """Exact McNemar test (two-sided): do models A and B make different errors?"""
    ca = (p_a >= thr).astype(int) == y
    cb = (p_b >= thr).astype(int) == y
    b, c = int((ca & ~cb).sum()), int((~ca & cb).sum())
    return 1.0 if b + c == 0 else binomtest(b, b + c, 0.5).pvalue


# ------------------------------------------------------------------ backtest
def backtest(next_ret, p, thr=0.5, cost=0.001) -> pd.DataFrame:
    """Long when P(up) >= thr else cash. Signal at close t earns next_ret[t]; cost on turnover."""
    pos = (p >= thr).astype(float)
    turnover = np.abs(np.diff(pos, prepend=0.0))
    strat = pos * next_ret - cost * turnover
    return pd.DataFrame({"strategy": strat, "buy_hold": next_ret})


def perf_summary(bt: pd.DataFrame) -> dict:
    out = {}
    for col in bt:
        r = bt[col].values
        eq = np.cumprod(1 + r)
        dd = (eq / np.maximum.accumulate(eq) - 1).min()
        out[col] = {"total_return": eq[-1] - 1,
                    "sharpe": float(np.sqrt(252) * r.mean() / r.std()) if r.std() > 0 else 0.0,
                    "max_drawdown": float(dd)}
    return out


# ------------------------------------------------------------------ interpretation
def permutation_importance(model, split, feature_names, n_repeats=5, seed=0) -> pd.DataFrame:
    """Drop in test ROC-AUC when one feature is shuffled across samples (breaks its link to y)."""
    rng = np.random.default_rng(seed)
    base = roc_auc_score(split.y, model.predict(split.X, verbose=0).ravel())
    rows = []
    for j, name in enumerate(feature_names):
        drops = []
        for _ in range(n_repeats):
            Xp = split.X.copy()
            Xp[:, :, j] = Xp[rng.permutation(len(Xp)), :, j]
            drops.append(base - roc_auc_score(split.y, model.predict(Xp, verbose=0).ravel()))
        rows.append((name, np.mean(drops), np.std(drops)))
    return pd.DataFrame(rows, columns=["feature", "auc_drop", "std"]).sort_values("auc_drop", ascending=False)


# ------------------------------------------------------------------ figures
def fig_price_split(prices, train_end, val_end, ticker):
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(prices.index, prices["Close"], color=C_GREY, lw=1)
    t1, t2 = pd.Timestamp(train_end), pd.Timestamp(val_end)
    ax.axvspan(prices.index[0], t1, color=C_MAIN, alpha=0.08, label="train")
    ax.axvspan(t1, t2, color="#f59e0b", alpha=0.15, label="validation")
    ax.axvspan(t2, prices.index[-1], color=C_ALT, alpha=0.10, label="test (touched once)")
    ax.set(title=f"{ticker} close price and chronological split", ylabel="Close")
    ax.legend(loc="upper left")
    _save(fig, "01_price_split")


def fig_class_balance(y_train, y_val, y_test):
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    vals = [np.mean(y) * 100 for y in (y_train, y_val, y_test)]
    ax.bar(["train", "validation", "test"], vals, color=[C_MAIN, "#f59e0b", C_ALT])
    ax.axhline(50, color="k", ls="--", lw=1)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.5, f"{v:.1f}%", ha="center")
    ax.set(ylabel="% of days that close UP", ylim=(0, 70), title="Class balance by period")
    _save(fig, "02_class_balance")


def fig_returns_eda(prices):
    r = np.log(prices["Close"]).diff().dropna()
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    ax[0].hist(r, bins=80, color=C_MAIN, alpha=0.8)
    ax[0].set(title="Daily log-return distribution (fat tails)", xlabel="log return")
    ax[1].plot(r.rolling(20).std() * np.sqrt(252), color=C_ALT, lw=1)
    ax[1].set(title="Rolling 20-day volatility (annualised)")
    rc = r.values - r.values.mean()
    a = np.array([np.sum(rc[k:] * rc[:-k]) / np.sum(rc ** 2) for k in range(1, 21)])
    ax[2].bar(range(1, 21), a, color=C_MAIN)
    ax[2].axhline(1.96 / np.sqrt(len(r)), color="k", ls="--", lw=1)
    ax[2].axhline(-1.96 / np.sqrt(len(r)), color="k", ls="--", lw=1)
    ax[2].set(title="Return autocorrelation (dashed = 95% noise band)", xlabel="lag (days)")
    _save(fig, "03_returns_eda")


def fig_corr(feats: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(feats.corr(), cmap="coolwarm", center=0, annot=False, ax=ax)
    ax.set_title("Feature correlation (engineered set)")
    _save(fig, "04_feature_corr")


def fig_history(histories: dict):
    """histories: {model_name: keras history dict}. Loss + accuracy per epoch, train vs validation."""
    n = len(histories)
    fig, axes = plt.subplots(2, n, figsize=(4.6 * n, 7), squeeze=False)
    for j, (name, h) in enumerate(histories.items()):
        ep = np.arange(1, len(h["loss"]) + 1)
        axes[0, j].plot(ep, h["loss"], color=C_MAIN, label="train")
        axes[0, j].plot(ep, h["val_loss"], color=C_ALT, label="validation")
        axes[0, j].axvline(int(np.argmin(h["val_loss"])) + 1, color=C_GREY, ls=":", label="best epoch")
        axes[0, j].set(title=f"{name}: loss", xlabel="epoch", ylabel="binary cross-entropy")
        axes[1, j].plot(ep, h["accuracy"], color=C_MAIN)
        axes[1, j].plot(ep, h["val_accuracy"], color=C_ALT)
        axes[1, j].axhline(0.5, color="k", ls="--", lw=0.8)
        axes[1, j].set(title=f"{name}: accuracy", xlabel="epoch")
    axes[0, 0].legend()
    _save(fig, "05_training_curves")


def fig_confusion(y, p, thr, name):
    cm = confusion_matrix(y, (p >= thr).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Pred DOWN", "Pred UP"], yticklabels=["True DOWN", "True UP"])
    ax.set_title(f"{name}: test confusion matrix")
    _save(fig, "06_confusion_matrix")


def fig_roc(y, probs: dict):
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    for name, p in probs.items():
        if len(np.unique(p)) > 2:
            fpr, tpr, _ = roc_curve(y, p)
            ax.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y, p):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="coin flip")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="Test ROC curves")
    ax.legend(fontsize=8)
    _save(fig, "07_roc")


def fig_model_comparison(df: pd.DataFrame):
    """df columns: model, accuracy_mean, accuracy_std, baseline (float)."""
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.bar(df["model"], df["accuracy_mean"] * 100, yerr=df["accuracy_std"] * 100, color=C_MAIN, alpha=0.85, capsize=4)
    ax.axhline(df["baseline"].iloc[0] * 100, color=C_ALT, ls="--", label="best constant guess (majority, in hindsight)")
    ax.axhline(50, color="k", ls=":", lw=1, label="coin flip")
    ax.set(ylabel="Test accuracy (%)", title="Model comparison (mean ± std over seeds)", ylim=(40, 65))
    ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    _save(fig, "08_model_comparison")


def fig_walk_forward(wf: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8.5, 4))
    w = 0.38
    x = np.arange(len(wf))
    ax.bar(x - w / 2, wf["lstm_acc"] * 100, w, label="LSTM", color=C_MAIN)
    ax.bar(x + w / 2, wf["majority_acc"] * 100, w, label="majority class", color=C_GREY)
    ax.axhline(50, color="k", ls=":", lw=1)
    ax.set_xticks(x, wf["year"].astype(str))
    ax.set(ylim=(40, 65), ylabel="Accuracy (%)", title="Walk-forward test accuracy by year")
    ax.legend()
    _save(fig, "09_walk_forward")


def fig_window_sweep(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.errorbar(df["window"], df["auc"], fmt="o-", color=C_MAIN)
    ax.axhline(0.5, color="k", ls="--", lw=1)
    ax.set(xlabel="window length (days)", ylabel="test ROC-AUC", title="Effect of history length (LSTM)")
    _save(fig, "10_window_sweep")


def fig_equity(bt: pd.DataFrame, dates, title):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dates, np.cumprod(1 + bt["strategy"].values), color=C_MAIN, label="LSTM long/cash (after costs)")
    ax.plot(dates, np.cumprod(1 + bt["buy_hold"].values), color=C_GREY, label="buy and hold")
    ax.set(title=title, ylabel="growth of 1.0")
    ax.legend()
    _save(fig, "11_equity_curve")


def fig_importance(imp: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(imp["feature"][::-1], imp["auc_drop"][::-1], xerr=imp["std"][::-1], color=C_MAIN)
    ax.axvline(0, color="k", lw=1)
    ax.set(xlabel="drop in test ROC-AUC when shuffled", title="Permutation feature importance (LSTM)")
    _save(fig, "12_feature_importance")


def fig_calibration(y, p):
    frac, mean_p = calibration_curve(y, p, n_bins=8, strategy="quantile")
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.plot(mean_p, frac, "o-", color=C_MAIN, label="LSTM")
    ax.plot([0.3, 0.7], [0.3, 0.7], "k--", lw=1, label="perfect calibration")
    ax.set(xlabel="predicted P(up)", ylabel="observed frequency of up", title="Calibration (test)")
    ax.legend()
    _save(fig, "13_calibration")


def fig_regime(dates, y, p, thr, vol):
    """Accuracy split by volatility tercile: where does the model fail?"""
    df = pd.DataFrame({"hit": ((p >= thr).astype(int) == y).astype(float), "vol": vol})
    df["regime"] = pd.qcut(df["vol"], 3, labels=["low vol", "mid vol", "high vol"])
    g = df.groupby("regime", observed=True)["hit"].mean() * 100
    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    ax.bar(g.index.astype(str), g.values, color=C_MAIN)
    ax.axhline(50, color="k", ls="--", lw=1)
    ax.set(ylim=(40, 65), ylabel="accuracy (%)", title="Accuracy by volatility regime (test)")
    _save(fig, "14_regime_accuracy")
    return g.to_dict()
