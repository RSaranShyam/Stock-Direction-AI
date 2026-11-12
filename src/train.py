"""Training loop with the standard regularisation toolkit: early stopping, LR scheduling, checkpointing."""
from __future__ import annotations

import numpy as np
from tensorflow import keras

from .config import MODEL_DIR
from .models import build_model


def set_seed(seed: int):
    keras.utils.set_random_seed(seed)  # python, numpy and TF in one call


def fit_model(kind: str, ds, cfg, *, units=None, dropout=None, lr=None, tag: str | None = None, verbose=0):
    """Train one model; returns (model, history_dict). Selection uses VALIDATION loss only."""
    set_seed(cfg.seed)
    keras.backend.clear_session()
    units, dropout, lr = units or cfg.units, cfg.dropout if dropout is None else dropout, lr or cfg.lr
    n_feat = ds.train.X.shape[-1]
    model = build_model(kind, ds.window, n_feat, units, dropout, lr)
    MODEL_DIR.mkdir(exist_ok=True, parents=True)
    ckpt = MODEL_DIR / f"{tag or kind}.weights.h5"
    cbs = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=cfg.patience, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5),
        keras.callbacks.ModelCheckpoint(ckpt, monitor="val_loss", save_best_only=True, save_weights_only=True),
    ]
    h = model.fit(ds.train.X, ds.train.y, validation_data=(ds.val.X, ds.val.y), epochs=cfg.epochs,
                  batch_size=cfg.batch_size, shuffle=True,  # shuffling WINDOWS is fine; each window is intact
                  callbacks=cbs, verbose=verbose)
    return model, h.history


def predict(model, split) -> np.ndarray:
    return model.predict(split.X, verbose=0).ravel()
