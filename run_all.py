"""ONE command to run the whole project.

    python run_all.py                  # everything, skipping steps that are already done
    python run_all.py --force          # redo everything from scratch
    python run_all.py --ticker ^NSEI   # a different main stock / index (redoes the study for it)
    python run_all.py --also INFY.NS AAPL RELIANCE.NS   # also train these stocks for the dashboard picker
    python run_all.py --quick          # fast smoke test (1 seed, few epochs; numbers are NOT for submission)
    python run_all.py --no-dashboard   # run the pipeline but don't open the dashboard

Steps: 1 install missing packages -> 2 leakage tests -> 3 full study (run_experiments.py) ->
       4 final models + prediction (predict_tomorrow.py) -> 5 dashboard at http://localhost:8501
First full run takes roughly 20-30 minutes on CPU; later runs take seconds.
"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
ENV = {**os.environ, "TF_CPP_MIN_LOG_LEVEL": "2", "TF_ENABLE_ONEDNN_OPTS": "0", "PYTHONIOENCODING": "utf-8"}

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--force", action="store_true", help="redo every step")
ap.add_argument("--ticker")
ap.add_argument("--source", choices=["yfinance", "csv", "synthetic"])
ap.add_argument("--csv")
ap.add_argument("--also", nargs="+", default=[], metavar="TICKER", help="extra stocks to train for the dashboard picker")
ap.add_argument("--quick", action="store_true")
ap.add_argument("--skip-tests", action="store_true")
ap.add_argument("--no-dashboard", action="store_true")
a = ap.parse_args()

T0 = time.time()


def banner(n, text):
    print(f"\n{'=' * 70}\n  STEP {n}/5  {text}\n{'=' * 70}", flush=True)


def run(cmd, what):
    r = subprocess.run([PY, *cmd], cwd=ROOT, env=ENV)
    if r.returncode != 0:
        print(f"\n[FAILED] {what} (exit code {r.returncode}). Fix the error above and run  python run_all.py  again;"
              f" finished steps are skipped.", flush=True)
        sys.exit(r.returncode)


# ---------------------------------------------------------------- 1. packages
banner(1, "Checking packages")
need = {"numpy": "numpy", "pandas": "pandas", "sklearn": "scikit-learn", "scipy": "scipy", "matplotlib": "matplotlib",
        "seaborn": "seaborn", "tensorflow": "tensorflow", "yfinance": "yfinance", "streamlit": "streamlit",
        "plotly": "plotly", "joblib": "joblib", "tabulate": "tabulate", "pytest": "pytest"}
missing = [pip for mod, pip in need.items() if importlib.util.find_spec(mod) is None]
if missing:
    print("Installing missing packages:", ", ".join(missing), flush=True)
    run(["-m", "pip", "install", "-r", "requirements.txt"], "package installation")
else:
    print("All packages present.")

# ---------------------------------------------------------------- 2. tests
banner(2, "Leakage and alignment tests")
if a.skip_tests:
    print("skipped (--skip-tests)")
else:
    run(["-m", "pytest", "-q", "tests"], "leakage tests (results would not be trustworthy)")

# ---------------------------------------------------------------- 3. study
results = ROOT / "reports" / "results.json"
force = a.force


def safe(t):
    import re
    return re.sub(r"[^A-Za-z0-9._-]", "_", t.strip())


def model_exists(ticker):
    if (ROOT / "stocks" / safe(ticker) / "model" / "meta.json").exists():
        return True
    legacy = ROOT / "models" / "final" / "meta.json"
    return legacy.exists() and json.loads(legacy.read_text())["config"]["ticker"] == ticker


main_ticker = a.ticker or (json.loads(results.read_text())["config"]["ticker"] if results.exists() else "TCS.NS")
if results.exists() and (a.ticker or a.source):
    old = json.loads(results.read_text())["config"]
    if (a.ticker and a.ticker != old["ticker"]) or (a.source and a.source != old["source"]):
        print(f"\nRequested {a.ticker or old['ticker']} / {a.source or old['source']} differs from existing results "
              f"({old['ticker']} / {old['source']}): redoing the study and models.")
        force = True

banner(3, "Full study: baselines, MLP/RNN/LSTM/GRU, tuning, walk-forward, figures")
if results.exists() and not force:
    print("reports/results.json already exists: skipped (use --force to redo)")
else:
    cmd = ["run_experiments.py"]
    for flag, val in (("--ticker", a.ticker), ("--source", a.source), ("--csv", a.csv)):
        if val:
            cmd += [flag, val]
    if a.quick:
        cmd.append("--quick")
    print("This is the long step (about 10-20 min on CPU).", flush=True)
    run(cmd, "the full study")

# ---------------------------------------------------------------- 4. prediction
banner(4, "Final models + next-session prediction (main stock + any --also stocks)")
cmd = ["predict_tomorrow.py", "--ticker", main_ticker]
if force or not model_exists(main_ticker):
    cmd.append("--train")
    print("Training the final ensemble (about 5-10 min on CPU).", flush=True)
else:
    print("Final models exist: only refreshing the prediction with the newest prices.", flush=True)
if a.quick:
    cmd.append("--quick")
if a.source:
    cmd += ["--source", a.source]
if a.csv:
    cmd += ["--csv", a.csv]
run(cmd, "the prediction step")

for extra in a.also:
    if extra == main_ticker:
        continue
    c2 = ["predict_tomorrow.py", "--ticker", extra]
    if force or not model_exists(extra):
        c2.append("--train")
        print(f"\nTraining extra stock {extra} (about 5-10 min)...", flush=True)
    else:
        print(f"\nRefreshing {extra}...", flush=True)
    if a.quick:
        c2.append("--quick")
    if a.source:
        c2 += ["--source", a.source]
    run(c2, f"training {extra}")

print(f"\nPipeline finished in {(time.time() - T0) / 60:.1f} min.", flush=True)

# ---------------------------------------------------------------- 5. dashboard
banner(5, "Dashboard")
if a.no_dashboard:
    print("skipped (--no-dashboard). Start it later with:  streamlit run dashboard.py")
else:
    print("Opening http://localhost:8501   (press Ctrl+C in this window to stop)\n", flush=True)
    try:
        subprocess.run([PY, "-m", "streamlit", "run", "dashboard.py"], cwd=ROOT, env=ENV)
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
