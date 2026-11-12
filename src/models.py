"""Model zoo: baselines (no skill, previous-day, logistic regression), MLP, SimpleRNN, LSTM, GRU."""
from __future__ import annotations

import numpy as np
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from tensorflow import keras
from tensorflow.keras import layers


def _compile(model, lr):
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="binary_crossentropy",
                  metrics=["accuracy", keras.metrics.AUC(name="auc")])
    return model


def build_recurrent(kind: str, window: int, n_features: int, units=(64, 32), dropout=0.3, lr=1e-3):
    """Input -> RNN(units[0], seq) -> Dropout -> RNN(units[1]) -> Dense(16, ReLU) -> Dense(1, Sigmoid)."""
    cell = {"lstm": layers.LSTM, "gru": layers.GRU, "rnn": layers.SimpleRNN}[kind]
    m = keras.Sequential([
        layers.Input((window, n_features)),
        cell(units[0], return_sequences=True),
        layers.Dropout(dropout),
        cell(units[1]),
        layers.Dropout(dropout),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ], name=kind.upper())
    return _compile(m, lr)


def build_mlp(window: int, n_features: int, dropout=0.3, lr=1e-3):
    """Flatten the window -> Dense(64) -> Dense(32) -> Sigmoid. No memory, just a non-linear map."""
    m = keras.Sequential([
        layers.Input((window, n_features)),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(dropout),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ], name="MLP")
    return _compile(m, lr)


def build_model(kind: str, window: int, n_features: int, units=(64, 32), dropout=0.3, lr=1e-3):
    if kind == "mlp":
        return build_mlp(window, n_features, dropout, lr)
    return build_recurrent(kind, window, n_features, units, dropout, lr)


# ---------- classical baselines (return probabilities so every model is scored the same way) ----------
class MajorityBaseline:
    name = "Majority class"

    def fit(self, split):
        self.p = float(split.y.mean() >= 0.5)
        return self

    def predict_proba(self, split):
        return np.full(len(split.y), self.p)


class PreviousDayBaseline:
    """Predict that tomorrow moves in the same direction as today."""
    name = "Previous-day direction"

    def fit(self, split):
        return self

    def predict_proba(self, split):
        return (split.ret_t > 0).astype(float)


class LogisticBaseline:
    name = "Logistic Regression"

    def fit(self, split):
        self.clf = LogisticRegression(max_iter=2000, C=0.1).fit(split.X.reshape(len(split.X), -1), split.y)
        return self

    def predict_proba(self, split):
        return self.clf.predict_proba(split.X.reshape(len(split.X), -1))[:, 1]
