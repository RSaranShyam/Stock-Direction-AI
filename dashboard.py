"""Stock direction dashboard.   Run:  streamlit run dashboard.py     (opens http://localhost:8501)

Pick any stock at the top. Trained stocks show their next-session call, past out-of-sample calls and a backtest;
untrained ones offer a one-click "Train this stock" button. The full model study (comparison, ablations,
walk-forward) belongs to the stock used in run_experiments.py.
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import importlib.util

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
CAN_TRAIN = importlib.util.find_spec("tensorflow") is not None  # False on the lightweight hosted demo
st.set_page_config(page_title="Stock Direction AI", page_icon="📈", layout="wide", initial_sidebar_state="collapsed")

UP, DOWN, IND, CYAN, AMB = "#2ee59d", "#fb7185", "#a3e635", "#2dd4bf", "#fbbf24"
INK, MUTED, GRID = "#e8f5ef", "#8aa398", "rgba(255,255,255,0.07)"
ACC = "#2ee59d"
AUTHOR, LINKEDIN = "R Saran Shyam", "https://www.linkedin.com/in/saranshyam/"
GITHUB = "https://github.com/RSaranShyam"

PRESETS = {
    "TCS.NS": "TCS", "RELIANCE.NS": "Reliance", "INFY.NS": "Infosys", "HDFCBANK.NS": "HDFC Bank", "ICICIBANK.NS": "ICICI Bank",
    "SBIN.NS": "SBI", "ITC.NS": "ITC", "WIPRO.NS": "Wipro", "BHARTIARTL.NS": "Bharti Airtel", "^NSEI": "NIFTY 50", "^BSESN": "SENSEX",
    "AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Alphabet", "AMZN": "Amazon", "TSLA": "Tesla", "NVDA": "Nvidia", "^GSPC": "S&P 500",
}
CUSTOM = "✏️ Custom ticker…"

# ============================================================ styling
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stApp {{font-family: 'Space Grotesk', system-ui, sans-serif;}}
.stApp {{background:
    radial-gradient(900px 520px at 50% -10%, rgba(46,229,157,.22) 0%, transparent 62%),
    radial-gradient(700px 420px at 100% 35%, rgba(16,185,129,.10) 0%, transparent 60%),
    linear-gradient(rgba(255,255,255,.028) 1px, transparent 1px) 0 0/48px 48px,
    linear-gradient(90deg, rgba(255,255,255,.028) 1px, transparent 1px) 0 0/48px 48px, #050a08;}}
[data-testid="stToolbar"], #MainMenu, footer {{display:none !important;}}
header[data-testid="stHeader"] {{background: transparent;}}
.block-container {{padding-top: 1.4rem; max-width: 1240px;}}
h1,h2,h3 {{letter-spacing:-0.02em;}}
.stApp a, .stApp a:visited {{color:{ACC} !important;}}
.stApp a.btn-primary, .stApp a.btn-primary:visited {{color:#04130c !important;}}
.stApp a.btn-ghost {{color:{INK} !important;}}
.hero {{padding: 6px 2px 10px 2px;}}
.hero h1 {{font-size: 2.3rem; font-weight: 700; margin: 0 0 6px 0;
  background: linear-gradient(90deg, #d1fae5 0%, {ACC} 60%, #5eead4 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;}}
.pill {{display:inline-block; padding: 4px 12px; margin: 4px 8px 0 0; border-radius: 999px; font-size: .78rem;
  color:{INK}; background: rgba(46,229,157,.07); border: 1px solid rgba(46,229,157,.22);}}
.pill.warn {{background: rgba(251,191,36,.13); border-color: rgba(251,191,36,.5); color:#fde68a;}}
.card {{background: linear-gradient(180deg, rgba(46,229,157,.07), rgba(255,255,255,.015));
  border: 1px solid rgba(46,229,157,.16); border-radius: 20px; padding: 18px 22px; height: 100%;
  box-shadow: 0 10px 34px rgba(0,0,0,.4), inset 0 1px 0 rgba(255,255,255,.04);}}
.kpi .label {{font-size:.74rem; text-transform:uppercase; letter-spacing:.09em; color:{MUTED};}}
.kpi .value {{font-size:2rem; font-weight:700; line-height:1.15; margin-top:4px;}}
.kpi .sub {{font-size:.8rem; color:{MUTED}; margin-top:2px;}}
.verdict {{border-left: 4px solid {AMB}; background: rgba(251,191,36,.08); border-radius: 14px; padding: 14px 18px;
  color:#fde68a; margin: 14px 0 6px 0; font-size:.95rem;}}
.verdict.good {{border-color:{UP}; background: rgba(46,229,157,.08); color:#bbf7d0;}}
.signal {{text-align:center;}}
.signal .eyebrow {{font-size:.74rem; letter-spacing:.14em; color:{MUTED}; text-transform:uppercase;}}
.signal .big {{font-size:4.4rem; font-weight:700; line-height:1.0; margin:6px 0 2px 0;}}
.signal .date {{font-size:1.02rem; color:{INK}; margin-bottom:8px;}}
.chip {{display:inline-block; padding:5px 14px; border-radius:999px; font-size:.82rem; font-weight:650;}}
.small {{font-size:.8rem; color:{MUTED};}}
div[data-baseweb="tab-list"] {{gap: 6px;}}
button[data-baseweb="tab"] {{border-radius: 12px !important; padding: 8px 16px !important;}}
[data-testid="stDataFrame"] {{border-radius: 14px; overflow:hidden;}}
/* buttons */
button[kind="primary"], .btn-primary {{background: linear-gradient(180deg, #5af0b4, #1fd68b) !important; color:#04130c !important;
  border:0 !important; border-radius: 12px !important; font-weight:650 !important; box-shadow: 0 0 24px rgba(46,229,157,.35);}}
button[kind="primary"]:hover {{filter: brightness(1.08);}}
button[kind="secondary"] {{border-radius: 12px !important; border:1px solid rgba(46,229,157,.3) !important; background: rgba(46,229,157,.05) !important;}}
/* landing page */
.nav {{display:flex; align-items:center; justify-content:space-between; padding:10px 18px; border-radius:16px;
  background: rgba(10,22,17,.7); border:1px solid rgba(46,229,157,.15); backdrop-filter: blur(8px); margin-bottom: 8px;}}
.brand {{display:flex; align-items:center; gap:10px; font-weight:700; font-size:1.02rem; color:{INK};}}
.logo {{width:26px; height:26px; border-radius:8px; background: linear-gradient(135deg,#5af0b4,#0e9f6e); box-shadow:0 0 18px rgba(46,229,157,.6);}}
.navlinks a {{color:{MUTED}; text-decoration:none; margin-left:22px; font-size:.9rem;}}
.navlinks a:hover {{color:{ACC};}}
.btn {{display:inline-block; padding:9px 18px; border-radius:12px; font-weight:650; font-size:.92rem; text-decoration:none !important;}}
.btn-ghost {{color:{INK} !important; border:1px solid rgba(46,229,157,.35); background: rgba(46,229,157,.06);}}
.btn-ghost:hover {{background: rgba(46,229,157,.14);}}
.land {{text-align:center; padding: 52px 8px 18px 8px;}}
.badge {{display:inline-block; padding:6px 14px; border-radius:999px; font-size:.8rem; color:{ACC};
  border:1px solid rgba(46,229,157,.35); background: rgba(46,229,157,.07);}}
.land h1 {{font-size: clamp(2.3rem, 5.4vw, 4.2rem); line-height:1.04; font-weight:700; margin: 18px 0 14px 0; color:#f2fbf7;}}
.land h1 span {{background: linear-gradient(90deg, #8ff7cb, {ACC} 45%, #14b8a6); -webkit-background-clip:text; -webkit-text-fill-color:transparent;}}
.land p.lead {{max-width: 720px; margin: 0 auto; color:{MUTED}; font-size:1.08rem; line-height:1.6;}}
.stats {{display:flex; justify-content:center; gap:46px; flex-wrap:wrap; margin: 30px 0 8px 0;}}
.stats .n {{font-size:2.1rem; font-weight:700; color:#f2fbf7;}}
.stats .n b {{color:{ACC};}}
.stats .l {{font-size:.8rem; color:{MUTED}; text-transform:uppercase; letter-spacing:.1em;}}
.sec {{font-size:1.9rem; font-weight:700; margin: 46px 0 6px 0; color:#f2fbf7;}}
.sec-sub {{color:{MUTED}; margin-bottom:18px;}}
.fcard {{height:100%; padding:22px; border-radius:20px; border:1px solid rgba(46,229,157,.16);
  background: linear-gradient(160deg, rgba(46,229,157,.09), rgba(255,255,255,.012) 60%); box-shadow: 0 10px 34px rgba(0,0,0,.4);}}
.fcard .ic {{width:42px; height:42px; border-radius:12px; display:flex; align-items:center; justify-content:center; font-size:1.3rem;
  background: rgba(46,229,157,.12); border:1px solid rgba(46,229,157,.3); margin-bottom:14px;}}
.fcard h3 {{margin:0 0 6px 0; font-size:1.15rem; color:#f2fbf7;}}
.fcard p {{margin:0; color:{MUTED}; font-size:.93rem; line-height:1.55;}}
.step {{display:flex; gap:14px; align-items:flex-start; padding:14px 0; border-bottom:1px solid rgba(255,255,255,.06);}}
.step .no {{min-width:34px; height:34px; border-radius:10px; background:rgba(46,229,157,.12); border:1px solid rgba(46,229,157,.35);
  color:{ACC}; display:flex; align-items:center; justify-content:center; font-weight:700;}}
.step b {{color:#f2fbf7;}} .step span {{color:{MUTED}; font-size:.92rem;}}
.flow {{display:flex; flex-wrap:wrap; justify-content:center; align-items:center; gap:10px; margin-top:6px;}}
.flow .node {{padding:10px 16px; border-radius:12px; border:1px solid rgba(46,229,157,.3); background:rgba(46,229,157,.06); font-size:.9rem;}}
.flow .arrow {{color:{ACC};}}
.author {{display:flex; align-items:center; gap:20px; flex-wrap:wrap;}}
.avatar {{width:64px; height:64px; border-radius:18px; background:linear-gradient(135deg,#5af0b4,#0e9f6e); color:#04130c;
  display:flex; align-items:center; justify-content:center; font-weight:800; font-size:1.4rem; box-shadow:0 0 28px rgba(46,229,157,.4);}}
.author .nm {{font-size:1.35rem; font-weight:700; color:#f2fbf7;}}
.foot {{text-align:center; color:{MUTED}; font-size:.82rem; padding: 34px 0 10px 0;}}
</style>
""", unsafe_allow_html=True)

if "page" not in st.session_state:
    st.session_state.page = "intro"


def landing():
    st.markdown(f"""
    <div class="nav"><div class="brand"><div class="logo"></div>Stock Direction AI</div>
      <div class="navlinks"><a href="#what">What it does</a><a href="#how">How it works</a><a href="#author">Author</a>
      <a class="btn btn-ghost" style="margin-left:22px" href="{LINKEDIN}" target="_blank" rel="noopener">LinkedIn ↗</a></div></div>
    <div class="land">
      <span class="badge">● Machine-learning project · not trading advice</span>
      <h1>An AI co-pilot for<br><span>stock direction</span></h1>
      <p class="lead">A neural network reads the last 30 trading days of prices and volume and calls tomorrow's close <b>UP</b> or <b>DOWN</b>.
      Then it tells you, honestly and with statistics, whether that call is better than a coin flip.</p>
      <div class="stats">
        <div><div class="n">18<b>+</b></div><div class="l">stocks ready to pick</div></div>
        <div><div class="n">30<b>d</b></div><div class="l">LSTM input window</div></div>
        <div><div class="n">0</div><div class="l">look-ahead leaks (6 tests)</div></div>
        <div><div class="n">LSTM <b>vs</b> MLP</div><div class="l">compared head to head</div></div>
      </div>
    </div>""", unsafe_allow_html=True)

    _, mid, _ = st.columns([2, 2.2, 2])
    if mid.button("🚀  Start predicting", type="primary", width="stretch"):
        st.session_state.page = "app"
        st.rerun()
    st.markdown(f'<div style="text-align:center;margin-top:10px"><a class="btn btn-ghost" href="{LINKEDIN}" target="_blank" rel="noopener">Connect with the author on LinkedIn ↗</a></div>',
                unsafe_allow_html=True)

    st.markdown('<a id="what"></a><div class="sec" style="text-align:center">What it does</div>'
                '<div class="sec-sub" style="text-align:center">Three things, done carefully</div>', unsafe_allow_html=True)
    f = st.columns(3)
    feats = [("🔮", "Next-session call", "Pick a stock and get UP or DOWN, the probability of up, a signal-strength label and the ensemble's votes."),
             ("🧪", "Honest evaluation", "Compared against the best constant guess with significance tests, walk-forward years and ablations. If there is no edge, it says so."),
             ("🗂️", "Any stock you like", "18 presets (India and US) or type any Yahoo Finance ticker. Each stock keeps its own model and track record.")]
    for col, (ic, h, t) in zip(f, feats):
        col.markdown(f'<div class="fcard"><div class="ic">{ic}</div><h3>{h}</h3><p>{t}</p></div>', unsafe_allow_html=True)

    st.markdown('<a id="how"></a><div class="sec" style="text-align:center">How it works</div>'
                '<div class="sec-sub" style="text-align:center">A leakage-safe pipeline, start to finish</div>', unsafe_allow_html=True)
    nodes = ["Yahoo Finance data", "Clean + features", "30-day windows", "LSTM / MLP", "Evaluate vs baselines", "Tomorrow's call"]
    st.markdown('<div class="flow">' + '<span class="arrow">→</span>'.join(f'<span class="node">{n}</span>' for n in nodes) + '</div>',
                unsafe_allow_html=True)
    l, r = st.columns(2)
    steps = [("Chronological split", "Train on the past, validate, then test on years the model never saw. Never shuffled."),
             ("Train-only scaling", "The scaler learns from training rows only, so no future information leaks in."),
             ("Stationary features", "Returns, moving-average ratios, RSI, MACD, volatility, volume change."),
             ("Beat the baseline", "Majority class, previous-day direction and logistic regression must be beaten first.")]
    for i, (col, chunk) in enumerate(zip((l, r), (steps[:2], steps[2:]))):
        col.markdown("".join(f'<div class="step"><div class="no">{i * 2 + j + 1}</div><div><b>{h}</b><br><span>{t}</span></div></div>'
                             for j, (h, t) in enumerate(chunk)), unsafe_allow_html=True)

    st.markdown(f"""<a id="author"></a><div class="sec" style="text-align:center">Built by</div><div class="sec-sub"></div>
    <div class="fcard"><div class="author"><div class="avatar">RS</div>
      <div style="flex:1;min-width:240px"><div class="nm">{AUTHOR}</div>
      <div class="small" style="font-size:.92rem">B.Tech CSE · Artificial Intelligence &amp; Data Engineering · Lovely Professional University</div></div>
      <div><a class="btn btn-primary" href="{LINKEDIN}" target="_blank" rel="noopener">View LinkedIn profile ↗</a>
      <a class="btn btn-ghost" style="margin-left:8px" href="{GITHUB}" target="_blank" rel="noopener">GitHub ↗</a></div></div></div>
    <div class="foot">Educational ML exercise by {AUTHOR}. Predictions are not financial advice. Daily stock direction is close to a random walk.</div>""",
                unsafe_allow_html=True)


if st.session_state.page == "intro":
    landing()
    st.stop()


def kpi(label, value, sub="", color=INK):
    return f'<div class="card kpi"><div class="label">{label}</div><div class="value" style="color:{color}">{value}</div><div class="sub">{sub}</div></div>'


def style(fig, h=340, legend=True):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color=INK, family="Inter, system-ui, sans-serif", size=13), showlegend=legend,
                      legend=dict(orientation="h", y=1.08, x=0, bgcolor="rgba(0,0,0,0)"), hoverlabel=dict(bgcolor="#0e1d17"))
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def plot(fig, **kw):
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, **kw)


@st.cache_data
def load_json(path, mtime):
    return json.loads(Path(path).read_text())


def read_json(p: Path):
    return load_json(str(p), p.stat().st_mtime) if p.exists() else None


def safe(t):
    return re.sub(r"[^A-Za-z0-9._-]", "_", t.strip())


STOCKS = ROOT / "stocks"
rep = ROOT / "reports"


def stock_files(t, study_ticker):
    d = STOCKS / safe(t)
    p, h = d / "latest_prediction.json", d / "test_predictions.csv"
    if t == study_ticker:  # first version of the project wrote these into reports/
        p = p if p.exists() else rep / "latest_prediction.json"
        h = h if h.exists() else rep / "test_predictions.csv"
    return p, h


def is_trained(t, study_ticker):
    if (STOCKS / safe(t) / "model" / "meta.json").exists():
        return True
    legacy = ROOT / "models" / "final" / "meta.json"
    return legacy.exists() and json.loads(legacy.read_text())["config"]["ticker"] == t


def trained_list(study_ticker):
    out = set()
    if STOCKS.exists():
        for d in STOCKS.iterdir():
            m = d / "model" / "meta.json"
            if m.exists():
                out.add(json.loads(m.read_text())["config"]["ticker"])
    if is_trained(study_ticker, study_ticker):
        out.add(study_ticker)
    return out


# ============================================================ study results (one stock: the one run_experiments.py used)
real, demo = rep / "results.json", ROOT / "examples" / "synthetic_demo" / "results.json"
src = real if real.exists() else demo
if not src.exists():
    st.error("No results found. Run `python run_all.py` first.")
    st.stop()
R = read_json(src)
FIG = src.parent / "figures"
synthetic = str(R.get("data_note", "")).startswith("SYNTHETIC")
cfg, clean = R["config"], R["cleaning_report"]
study_ticker = cfg["ticker"]
mc = pd.DataFrame(R["model_comparison"])
maj = R["majority_rate_test"]
t_, bt, wfs = R["tests"], R["backtest"], R.get("walk_forward_summary", {})
head = mc[mc["model"] == "LSTM (tuned)"].iloc[0] if (mc["model"] == "LSTM (tuned)").any() else mc.iloc[-1]

# ============================================================ stock picker
done = trained_list(study_ticker)
options = list(PRESETS) + sorted(done - set(PRESETS)) + [CUSTOM]


def label(t):
    if t == CUSTOM:
        return t
    return f"{'✅' if t in done else '⬜'}  {PRESETS.get(t, t)}  ·  {t}"


hero_slot = st.container()
sc = st.columns([2.2, 2.2, 3])
choice = sc[0].selectbox("Choose a stock", options, index=options.index(study_ticker) if study_ticker in options else 0,
                         format_func=label, key="stock_choice",
                         help="✅ = already trained. ⬜ = needs a one-time training run (a few minutes).")
if choice == CUSTOM:
    T = sc[1].text_input("Yahoo Finance ticker", value="", placeholder="e.g. MARUTI.NS, AMD, BTC-USD", key="custom_ticker").strip().upper() or study_ticker
else:
    T = choice
    sc[1].text_input("Yahoo Finance ticker", value=T, disabled=True)
sc[2].markdown(f'<div class="small" style="padding-top:34px">{len(done)} stock(s) trained: {", ".join(sorted(done)) or "none yet"}</div>',
               unsafe_allow_html=True)

is_study = T == study_ticker
trained = is_trained(T, study_ticker)
pred_p, hist_p = stock_files(T, study_ticker)
pred = read_json(pred_p)
hist = pd.read_csv(hist_p, parse_dates=["date"]) if hist_p.exists() else None


def hist_kpis(h, cost):
    n = len(h)
    up = h.actual_up.mean()
    m = max(up, 1 - up)
    auc = roc_auc_score(h.actual_up, h.p_up) if h.p_up.nunique() > 1 and h.actual_up.nunique() > 1 else 0.5
    pv = binomtest(int(h.correct.sum()), n, m, alternative="greater").pvalue
    pos = (h.p_up.values >= 0.5).astype(float)
    r = pos * h.next_ret.values - cost * np.abs(np.diff(pos, prepend=0.0))
    sh = float(np.sqrt(252) * r.mean() / r.std()) if r.std() > 0 else 0.0
    bh = h.next_ret.values
    shb = float(np.sqrt(252) * bh.mean() / bh.std()) if bh.std() > 0 else 0.0
    return dict(acc=h.correct.mean(), maj=m, auc=auc, p=pv, sharpe=sh, sharpe_bh=shb, n=n)


# ============================================================ hero
with hero_slot:
    hb = st.columns([1, 5])
    if hb[0].button("← Home"):
        st.session_state.page = "intro"
        st.rerun()
    hb[1].markdown(f'<div class="small" style="text-align:right;padding-top:8px">By <a href="{LINKEDIN}" target="_blank" rel="noopener">{AUTHOR}</a></div>',
                   unsafe_allow_html=True)
    pills = [f"🏷️ {T}", f"🗓️ {clean['first_date']} → {clean['last_date']}" if is_study else "🗓️ live data from 2015",
             f"🪟 {cfg['window']}-day window", "🧠 LSTM ensemble"]
    ph = "".join(f'<span class="pill">{p}</span>' for p in pills)
    if synthetic and is_study:
        ph += '<span class="pill warn">⚠ SYNTHETIC DEMO DATA — run the pipeline on real data</span>'
    st.markdown(f'<div class="hero"><h1>Stock Direction AI</h1><div class="small">Will the next session close UP or DOWN? '
                f'Pick a stock below.</div><div>{ph}</div></div>', unsafe_allow_html=True)

# ============================================================ headline KPIs (study numbers for the study stock, history-derived otherwise)
if is_study:
    acc, base, auc, pv = head.accuracy_mean, maj, head.roc_auc_mean, t_["binom_p_vs_majority"]
    cards = [("Test accuracy", f"{acc:.1%}", f"{(acc - base) * 100:+.1f} pts vs best constant guess", UP if acc > base else DOWN),
             ("ROC-AUC", f"{auc:.3f}", "0.500 = coin flip", CYAN),
             ("p-value vs majority", f"{pv:.3f}", "real edge" if pv < .05 else "not significant", UP if pv < .05 else AMB),
             ("Years beating baseline", f"{wfs.get('folds_beating_majority', 0)} / {wfs.get('folds', 0)}", f"walk-forward mean {wfs.get('lstm_acc_mean', 0):.1%}", IND),
             ("Backtest Sharpe", f"{bt['strategy']['sharpe']:.2f}", f"buy & hold {bt['buy_hold']['sharpe']:.2f}", CYAN)]
    beats = acc > base and pv < 0.05
elif hist is not None:
    kk = hist_kpis(hist, cfg["cost"])
    acc, base, auc, pv = kk["acc"], kk["maj"], kk["auc"], kk["p"]
    cards = [("Out-of-sample accuracy", f"{acc:.1%}", f"{(acc - base) * 100:+.1f} pts vs best constant guess", UP if acc > base else DOWN),
             ("ROC-AUC", f"{auc:.3f}", "0.500 = coin flip", CYAN),
             ("p-value vs majority", f"{pv:.3f}", "real edge" if pv < .05 else "not significant", UP if pv < .05 else AMB),
             ("Days tested", f"{kk['n']}", "never seen during training", IND),
             ("Backtest Sharpe", f"{kk['sharpe']:.2f}", f"buy & hold {kk['sharpe_bh']:.2f}", CYAN)]
    beats = acc > base and pv < 0.05
else:
    cards, beats = None, False

if cards:
    for col, (lab, val, sub, c) in zip(st.columns(5), cards):
        col.markdown(kpi(lab, val, sub, c), unsafe_allow_html=True)
    msg = ("The model shows a statistically significant edge over the best constant guess here. Confirm it holds across time before trusting it."
           if beats else "Honest read: for this stock the model does <b>not</b> show a statistically significant edge over the best constant guess. "
           "That is the common outcome for daily direction, because markets are close to a random walk. Treat the call below as a "
           "modelling exercise, not advice.")
    st.markdown(f'<div class="verdict {"good" if beats else ""}">{msg}</div>', unsafe_allow_html=True)

tabs = st.tabs(["🔮 Prediction", "📊 Performance", "🧪 Robustness", "🔍 Interpretation", "💰 Backtest", "🖼️ Gallery"])


def study_only(fn):
    """Model study tabs describe ONE stock (the one run_experiments.py used)."""
    if is_study:
        fn()
    else:
        st.markdown(f'<div class="card">The full study (model comparison, ablations, walk-forward, feature importance) was run for '
                    f'<b>{study_ticker}</b>. For <b>{T}</b> you get the live prediction, its out-of-sample history and the backtest. '
                    f'To run the full study for {T} instead, use <code>python run_all.py --ticker {T} --force</code>.</div>', unsafe_allow_html=True)


# ============================================================ training UI
def train_ui():
    st.markdown(f'<div class="card"><div class="eyebrow small" style="letter-spacing:.14em">NOT TRAINED YET</div>'
                f'<div style="font-size:1.5rem;font-weight:750;margin:6px 0">{PRESETS.get(T, T)} <span class="small">({T})</span></div>'
                f'<div class="small">One-time setup: downloads the price history from Yahoo Finance, trains the LSTM ensemble and builds its '
                f'out-of-sample track record. Then the prediction appears here and refreshes in seconds.</div></div>', unsafe_allow_html=True)
    if not CAN_TRAIN:
        st.info("Training is switched off in this hosted demo (it needs TensorFlow). Pick a stock marked ✅, or run the project locally "
                "to train any stock: see the README.")
        return
    mode = st.radio("Training size", ["Fast · 1 model (about 2–4 min)", "Full · 3 models (about 6–12 min)"], horizontal=True)
    if st.button(f"🚀 Train {T} and predict", type="primary"):
        cmd = [sys.executable, str(ROOT / "predict_tomorrow.py"), "--ticker", T, "--train"] + (["--seeds", "1"] if mode.startswith("Fast") else [])
        env = {**os.environ, "TF_CPP_MIN_LOG_LEVEL": "3", "TF_ENABLE_ONEDNN_OPTS": "0", "PYTHONIOENCODING": "utf-8"}
        lines, allout = [], []
        with st.status(f"Training {T}…", expanded=True) as status:
            box = st.empty()
            proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            for ln in proc.stdout:
                ln = ln.rstrip()
                allout.append(ln)
                if ln.startswith("[") or "Error" in ln or "Traceback" in ln or "RuntimeError" in ln:
                    lines.append(ln)
                    box.code("\n".join(lines[-10:]))
            proc.wait()
            if proc.returncode == 0:
                status.update(label=f"{T} is ready", state="complete")
            else:
                status.update(label="Training failed", state="error")
        if proc.returncode == 0:
            st.cache_data.clear()
            st.rerun()
        else:
            st.error("Training failed. The last lines of the log are below. Common causes: ticker spelling (Yahoo format, e.g. INFY.NS), "
                     "Yahoo rate-limiting (wait a minute and retry) or too little price history.")
            st.code("\n".join([x for x in allout if x.strip()][-25:]) or "(no output)")


# ============================================================ prediction
with tabs[0]:
    if not trained and pred is None:
        train_ui()
    else:
        top = st.columns([4, 1])
        top[1].write("")
        if CAN_TRAIN and top[1].button("↻ Refresh with latest data", width="stretch"):
            try:
                with st.spinner("Downloading the latest prices and running the model..."):
                    from src.predict import predict_latest
                    pred = predict_latest(T, log=lambda *_: None)
                st.cache_data.clear()
            except Exception as e:
                st.error(f"Could not refresh: {e}")
        if not pred:
            st.info(f"{T} is trained but has no saved prediction yet. Press Refresh.")
        else:
            p, up = pred["p_up"], pred["p_up"] >= 0.5
            col = UP if up else DOWN
            strength_col = {"Weak": AMB, "Moderate": CYAN, "Strong": UP}[pred["strength"]]
            c1, c2 = st.columns([1, 2])
            with c1:
                st.markdown(f"""<div class="card signal"><div class="eyebrow">{PRESETS.get(T, T)} · call for next session</div>
                  <div class="big" style="color:{col}">{'▲ UP' if up else '▼ DOWN'}</div>
                  <div class="date">{pd.Timestamp(pred['next_session']).strftime('%A, %d %b %Y')}</div>
                  <span class="chip" style="background:{strength_col}22;color:{strength_col};border:1px solid {strength_col}66">
                  {pred['strength']} signal · P(up) {p:.1%}</span>
                  <div class="small" style="margin-top:10px">Based on the close of {pd.Timestamp(pred['as_of']).strftime('%d %b %Y')} ·
                  last close {pred['last_close']:,.2f}</div></div>""", unsafe_allow_html=True)
                g = go.Figure(go.Indicator(mode="gauge+number", value=p * 100, number=dict(suffix="%", font=dict(size=34, color=INK)),
                              gauge=dict(axis=dict(range=[0, 100], tickcolor=MUTED, tickfont=dict(color=MUTED)), bar=dict(color=col, thickness=0.28),
                                         bgcolor="rgba(0,0,0,0)", borderwidth=0,
                                         steps=[dict(range=[0, 50], color="rgba(244,63,94,.18)"), dict(range=[50, 100], color="rgba(34,197,94,.18)")],
                                         threshold=dict(line=dict(color="white", width=3), thickness=0.85, value=50))))
                plot(style(g, 210, False).update_layout(margin=dict(l=30, r=40, t=10, b=0)))
            with c2:
                tl = pd.DataFrame(pred["tail"]).tail(90)
                tl["date"] = pd.to_datetime(tl["date"])
                f = go.Figure()
                f.add_trace(go.Scatter(x=tl.date, y=tl.close, name="Close", line=dict(color=CYAN, width=2.4), fill="tozeroy", fillcolor="rgba(34,211,238,.06)"))
                f.add_trace(go.Scatter(x=tl.date, y=tl.ma20, name="20-day average", line=dict(color=AMB, width=1.6, dash="dot")))
                f.add_trace(go.Scatter(x=[tl.date.iloc[-1]], y=[tl.close.iloc[-1]], name="Model call", mode="markers",
                                       marker=dict(symbol="triangle-up" if up else "triangle-down", size=20, color=col, line=dict(color="white", width=1.5))))
                f.add_annotation(x=tl.date.iloc[-1], y=tl.close.iloc[-1], text=f"<b>{'UP' if up else 'DOWN'}</b>", showarrow=False,
                                 yshift=30 if up else -30, font=dict(color=col, size=15))
                lo, hi = tl.close.min(), tl.close.max()
                f.update_yaxes(range=[lo - (hi - lo) * 0.25, hi + (hi - lo) * 0.2])
                st.markdown(f'<div class="small" style="margin:2px 0 -6px 4px">{pred["ticker"]} · last 90 sessions</div>', unsafe_allow_html=True)
                plot(style(f, 470))
            v = pred["votes"]
            h = hist["strength"].eq(pred["strength"]) if hist is not None else None
            hr = (hist.loc[h, "correct"].mean() if h is not None and h.sum() > 20 else None)
            m = st.columns(4)
            m[0].markdown(kpi("Ensemble votes", f"{sum(x >= .5 for x in v)} of {len(v)} say UP", "independent models, averaged", IND), unsafe_allow_html=True)
            m[1].markdown(kpi("Past hit rate, same strength", f"{hr:.1%}" if hr else "n/a",
                              f"{int(h.sum())} out-of-sample days" if h is not None else "needs history", CYAN), unsafe_allow_html=True)
            m[2].markdown(kpi("Overall out-of-sample hit rate", f"{hist.correct.mean():.1%}" if hist is not None else "n/a",
                              "all days after validation period", UP if hist is not None and hist.correct.mean() > .5 else AMB), unsafe_allow_html=True)
            m[3].markdown(kpi("Models trained through", pd.Timestamp(pred["trained_through"]).strftime("%d %b %Y"),
                              f"prediction made {pred['generated_at']}", MUTED), unsafe_allow_html=True)
            st.caption("A 'Weak' signal means P(up) is within 3 points of 50%: the model is basically undecided. Machine-learning exercise, not financial advice. "
                       "If the market is still open, the latest bar is incomplete.")

            if hist is not None:
                st.subheader("How the model's past calls did")
                hh = hist.copy()
                f = go.Figure()
                f.add_trace(go.Scatter(x=hh.date, y=hh.close, name="Close", line=dict(color="rgba(255,255,255,.35)", width=1.4)))
                for ok, name, c in ((1, "Correct call", UP), (0, "Wrong call", DOWN)):
                    s = hh[hh.correct == ok]
                    f.add_trace(go.Scatter(x=s.date, y=s.close, mode="markers", name=name, marker=dict(size=5, color=c, opacity=.75),
                                           customdata=np.c_[s.p_up * 100, np.where(s.actual_up == 1, "UP", "DOWN")],
                                           hovertemplate="%{x|%d %b %Y}<br>P(up) %{customdata[0]:.0f}% · actual %{customdata[1]}<extra></extra>"))
                plot(style(f, 330))
                a, b = st.columns(2)
                with a:
                    roll = hh.set_index("date")["correct"].rolling(60).mean().dropna() * 100
                    f = go.Figure(go.Scatter(x=roll.index, y=roll.values, line=dict(color=IND, width=2.4), fill="tozeroy", fillcolor="rgba(129,140,248,.10)"))
                    f.add_hline(y=50, line=dict(color="white", dash="dot", width=1))
                    f.update_yaxes(range=[25, 75], title="rolling 60-day hit rate (%)")
                    plot(style(f, 280, False))
                with b:
                    by = hh.groupby("strength").agg(days=("correct", "size"), hit_rate=("correct", "mean")).reindex(["Weak", "Moderate", "Strong"]).dropna()
                    small = by.days < 20  # a hit rate from a handful of days is noise: grey it out and say so
                    f = go.Figure(go.Bar(x=by.index, y=by.hit_rate * 100,
                                         marker_color=["rgba(255,255,255,.18)" if sm else c for sm, c in zip(small, [AMB, CYAN, UP])],
                                         text=[f"{r:.0%}<br>{int(n)} days" + ("<br>too few to trust" if sm else "")
                                               for r, n, sm in zip(by.hit_rate, by.days, small)], textposition="outside"))
                    f.add_hline(y=50, line=dict(color="white", dash="dot", width=1))
                    f.update_yaxes(range=[20, 80], title="hit rate (%)")
                    plot(style(f, 280, False).update_layout(title=dict(text="Does a stronger signal mean more accuracy?", font=dict(size=14))))
                rec = hh.tail(12).iloc[::-1].copy()
                rec["Date"] = rec.date.dt.strftime("%d %b %Y")
                rec["Model said"] = np.where(rec.pred_up == 1, "▲ UP", "▼ DOWN")
                rec["P(up)"] = (rec.p_up * 100).round(1).astype(str) + "%"
                rec["Actual"] = np.where(rec.actual_up == 1, "▲ UP", "▼ DOWN")
                rec["Result"] = np.where(rec.correct == 1, "✅ correct", "❌ wrong")
                st.dataframe(rec[["Date", "Model said", "P(up)", "Actual", "Result"]], width="stretch", hide_index=True)

    # ---- all trained stocks at a glance
    rows = []
    for tk in sorted(trained_list(study_ticker)):
        pp = read_json(stock_files(tk, study_ticker)[0])
        if pp:
            rows.append({"Stock": f"{PRESETS.get(tk, tk)} ({tk})", "Call": "▲ UP" if pp["p_up"] >= .5 else "▼ DOWN", "P(up)": f"{pp['p_up']:.1%}",
                         "Signal": pp["strength"], "Based on close of": pp["as_of"], "For session": pp["next_session"]})
    if len(rows) > 1:
        st.subheader("All trained stocks")
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


# ============================================================ study tabs
def tab_performance():
    d = mc.assign(c=[IND if m_ != "LSTM (tuned)" else UP for m_ in mc.model])
    f = go.Figure(go.Bar(x=d.model, y=d.accuracy_mean * 100, marker_color=d.c, error_y=dict(type="data", array=d.accuracy_std * 100, color=INK),
                         text=[f"{v:.1f}%" for v in d.accuracy_mean * 100], textposition="outside", textfont=dict(color=INK),
                         customdata=d.roc_auc_mean, hovertemplate="%{x}<br>accuracy %{y:.2f}%<br>AUC %{customdata:.3f}<extra></extra>"))
    f.add_hline(y=maj * 100, line=dict(color=DOWN, dash="dash"), annotation_text=f"best constant guess {maj:.1%}", annotation_font_color=DOWN)
    f.add_hline(y=50, line=dict(color=MUTED, dash="dot"), annotation_text="coin flip", annotation_font_color=MUTED, annotation_position="bottom right")
    f.update_yaxes(range=[40, 62], title="test accuracy (%)")
    st.subheader(f"{study_ticker}: test accuracy by model (mean ± std over seeds)")
    plot(style(f, 400, False))
    st.dataframe(mc[["model", "accuracy_mean", "accuracy_std", "precision_mean", "recall_mean", "f1_mean", "roc_auc_mean", "roc_auc_std"]].round(4),
                 width="stretch", hide_index=True)
    st.subheader("Is the edge real?")
    tt = pd.DataFrame({"Test": ["Accuracy > coin flip (binomial)", "Accuracy > best constant guess (binomial)", "LSTM vs MLP (McNemar)", "LSTM vs Logistic Regression (McNemar)"],
                       "p-value": [t_["binom_p_vs_coinflip"], t_["binom_p_vs_majority"], t_["mcnemar_p_lstm_vs_mlp"], t_["mcnemar_p_lstm_vs_logreg"]]})
    tt["Significant at 5%"] = np.where(tt["p-value"] < .05, "✅ yes", "❌ no")
    st.dataframe(tt.round(4), width="stretch", hide_index=True)


def tab_robust():
    a, b = st.columns(2)
    with a:
        st.subheader("Walk-forward accuracy by year")
        wf = pd.DataFrame(R.get("walk_forward", []))
        if len(wf):
            f = go.Figure()
            f.add_trace(go.Bar(x=wf.year.astype(str), y=wf.lstm_acc * 100, name="LSTM", marker_color=IND))
            f.add_trace(go.Bar(x=wf.year.astype(str), y=wf.majority_acc * 100, name="Best constant guess", marker_color="rgba(255,255,255,.28)"))
            f.add_hline(y=50, line=dict(color=MUTED, dash="dot"))
            f.update_layout(barmode="group").update_yaxes(range=[40, 62], title="accuracy (%)")
            plot(style(f, 340))
    with b:
        st.subheader("Ablations (test ROC-AUC)")
        ab = pd.DataFrame(R["ablations"])
        f = go.Figure(go.Bar(y=ab.experiment, x=ab.auc, orientation="h", marker_color=CYAN, text=ab.auc.round(3), textposition="outside"))
        f.add_vline(x=0.5, line=dict(color=MUTED, dash="dot"))
        f.update_xaxes(range=[0.4, 0.62]).update_yaxes(autorange="reversed")
        plot(style(f, 340, False).update_layout(margin=dict(l=170, r=30, t=30, b=10)))
    st.subheader("Hyper-parameter search (validation only)")
    st.dataframe(pd.DataFrame(R["tuning"]).round(4), width="stretch", hide_index=True)


def tab_interp():
    a, b = st.columns(2)
    with a:
        st.subheader("What the model relies on")
        imp = pd.DataFrame(R["importance_top5"])
        f = go.Figure(go.Bar(y=imp.feature, x=imp.auc_drop, orientation="h", error_x=dict(type="data", array=imp["std"], color=MUTED), marker_color=IND))
        f.update_yaxes(autorange="reversed").update_xaxes(title="drop in test ROC-AUC when feature is shuffled")
        plot(style(f, 320, False))
        st.caption("Bars that are tiny or smaller than their error whisker are not reliably useful.")
    with b:
        st.subheader("Accuracy by volatility regime")
        reg = R["accuracy_by_vol_regime"]
        f = go.Figure(go.Bar(x=list(reg.keys()), y=list(reg.values()), marker_color=[CYAN, IND, AMB], text=[f"{v:.1f}%" for v in reg.values()], textposition="outside"))
        f.add_hline(y=50, line=dict(color=MUTED, dash="dot"))
        f.update_yaxes(range=[40, 62], title="accuracy (%)")
        plot(style(f, 320, False))
    q = st.columns(4)
    q[0].markdown(kpi("Missing values", clean["missing_values"], "after download"), unsafe_allow_html=True)
    q[1].markdown(kpi("Duplicate dates", clean["duplicate_dates"], "removed"), unsafe_allow_html=True)
    q[2].markdown(kpi("Outlier days", clean["outlier_days_flagged"], "flagged, kept (real events)", AMB), unsafe_allow_html=True)
    cb = R["class_balance"]
    q[3].markdown(kpi("UP days (train/val/test)", f"{cb['train']:.0%} / {cb['val']:.0%} / {cb['test']:.0%}", "class balance", CYAN), unsafe_allow_html=True)


def tab_gallery():
    imgs = sorted(FIG.glob("*.png")) if FIG.exists() else []
    if not imgs:
        st.write("No figures found. Run `python run_all.py`.")
    cols = st.columns(2)
    for i, p_ in enumerate(imgs):
        with cols[i % 2]:
            st.markdown(f'<div class="small">{p_.stem[3:].replace("_", " ").title()}</div>', unsafe_allow_html=True)
            st.image(str(p_), width="stretch")


with tabs[1]:
    study_only(tab_performance)
with tabs[2]:
    study_only(tab_robust)
with tabs[3]:
    study_only(tab_interp)

# ============================================================ backtest (any stock with history)
with tabs[4]:
    if hist is None and not (is_study and not trained):
        st.info(f"No prediction history for {T} yet. Train it on the Prediction tab first.")
    else:
        s, h_ = bt["strategy"], bt["buy_hold"]
        if hist is not None:  # same data as the chart below so cards and curve always agree
            def _perf(r):
                eq = np.cumprod(1 + r)
                return {"total_return": eq[-1] - 1, "sharpe": float(np.sqrt(252) * r.mean() / r.std()) if r.std() > 0 else 0.0,
                        "max_drawdown": float((eq / np.maximum.accumulate(eq) - 1).min())}
            _pos = (hist.p_up.values >= cfg["threshold"]).astype(float)
            _ret = _pos * hist.next_ret.values - cfg["cost"] * np.abs(np.diff(_pos, prepend=0.0))
            s, h_ = _perf(_ret), _perf(hist.next_ret.values)
            days_long = float(_pos.mean())
        else:
            days_long = R["pct_days_long"]
        q = st.columns(4)
        q[0].markdown(kpi("Strategy return", f"{s['total_return']:.1%}", f"buy & hold {h_['total_return']:.1%}", UP if s["total_return"] > h_["total_return"] else AMB), unsafe_allow_html=True)
        q[1].markdown(kpi("Sharpe ratio", f"{s['sharpe']:.2f}", f"buy & hold {h_['sharpe']:.2f}", CYAN), unsafe_allow_html=True)
        q[2].markdown(kpi("Max drawdown", f"{s['max_drawdown']:.1%}", f"buy & hold {h_['max_drawdown']:.1%}", IND), unsafe_allow_html=True)
        q[3].markdown(kpi("Days invested", f"{days_long:.0%}", f"cost {cfg['cost']:.1%} per trade", MUTED), unsafe_allow_html=True)
        if hist is not None:
            f = go.Figure()
            f.add_trace(go.Scatter(x=hist.date, y=np.cumprod(1 + _ret), name="Model long/cash", line=dict(color=UP, width=2.6)))
            f.add_trace(go.Scatter(x=hist.date, y=np.cumprod(1 + hist.next_ret.values), name="Buy and hold", line=dict(color="rgba(255,255,255,.5)", width=2)))
            f.update_yaxes(title="growth of 1.0")
            st.subheader(f"{PRESETS.get(T, T)}: growth of 1.0 over the out-of-sample period")
            plot(style(f, 380))
        elif (FIG / "11_equity_curve.png").exists():
            st.image(str(FIG / "11_equity_curve.png"), width="stretch")
        st.caption("Simulation only: long when P(up) is above the threshold, otherwise cash; ignores slippage and taxes.")

with tabs[5]:
    study_only(tab_gallery)
