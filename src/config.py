"""Central experiment configuration. One place to change ticker, dates and hyper-parameters."""
from dataclasses import dataclass, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Config:
    # --- data ---
    ticker: str = "TCS.NS"            # try "^NSEI" (NIFTY 50), "RELIANCE.NS", "AAPL"
    source: str = "yfinance"          # "yfinance" | "csv" | "synthetic"
    csv_path: str | None = None
    start: str = "2015-01-01"
    end: str = "2025-12-31"
    # --- chronological split (by date of the last day in each window) ---
    train_end: str = "2021-12-31"
    val_end: str = "2023-12-31"       # test = everything after val_end
    # --- features / windows ---
    feature_set: str = "engineered"   # "engineered" | "raw"
    scaler: str = "standard"          # "standard" | "minmax"
    window: int = 30
    # --- training ---
    seed: int = 42
    epochs: int = 60
    batch_size: int = 64
    patience: int = 10
    lr: float = 1e-3
    units: tuple = (64, 32)
    dropout: float = 0.3
    # --- backtest ---
    threshold: float = 0.5
    cost: float = 0.001               # 0.1 % per unit of turnover

    def to_dict(self):
        return asdict(self)


FIG_DIR = ROOT / "reports" / "figures"
MODEL_DIR = ROOT / "models"
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
