"""
Lightweight deployment script:
- Pull last 5 trading days of 1-min bars from yfinance (to allow z-normalization),
  then restrict decisions to the last 2 days of regular trading hours (RTH) using
  the Alpaca market calendar like in bar_retriever.py.
- Compute TA features with the same generate_features() used in preprocessing.
- Build 64-D embeddings using the trained MLP (best_*_model.pt).
- Load the trained Hoeffding Tree and choose a target leaf node_id (from env TARGET_NODE_ID,
  else the most populated leaf if available, else first leaf).
- For each symbol, evaluate the last available embedding. If the tree traversal for that
  embedding ends in the chosen node_id, place a simple market order (no SL/TP) via Alpaca.
- Finally, check all open positions; if a buy was filled >= 30 minutes ago, submit a market sell
  to close the position.

Notes:
- Requires packages: yfinance, requests, torch, river (for the tree object), pandas, PyYAML.
- Configure keys and params under ../../conf/keys.yaml and ../../conf/params.yaml.
- Tickers: by default uses the S&P500 CSV under ../../data/.

Run once (one-shot). You can schedule externally to run periodically.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
import pytz
import requests
import importlib.util
import pickle

import yfinance as yf

import torch
from torch import nn

# Load feature generator from preprocessing scripts via explicit path
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
CONF_DIR = os.path.join(EXP_DIR, "conf")
MODELS_DIR = os.path.join(EXP_DIR, "models")
DATA_DIR = os.path.join(EXP_DIR, "data")
FEATURES_PY_PATH = os.path.join(EXP_DIR, "scripts", "03_pre_split_prep", "features.py")

spec = importlib.util.spec_from_file_location("features_module", FEATURES_PY_PATH)
features_module = importlib.util.module_from_spec(spec) if spec else None
if spec and spec.loader:
    spec.loader.exec_module(features_module)  # type: ignore[attr-defined]
else:
    raise RuntimeError(f"Could not load features.py from {FEATURES_PY_PATH}")

generate_features = getattr(features_module, "generate_features")

# -----------------------------
# Config and paths
# -----------------------------
with open(os.path.join(CONF_DIR, "params.yaml"), "r") as f:
    params = yaml.safe_load(f)
with open(os.path.join(CONF_DIR, "keys.yaml"), "r") as f:
    keys = yaml.safe_load(f)

# Data prep params
ema_periods = params["DATA_PREP"]["EMA_PERIODS"]
slope_periods = params["DATA_PREP"]["SLOPE_PERIODS"]
z_norm_window = params["DATA_PREP"]["Z_NORM_WINDOW"]
feature_path = params["DATA_PREP"].get("FEATURE_PATH")

# Modeling params
hidden1 = params["MODELING"].get("HIDDEN1", 128)
hidden2 = params["MODELING"].get("HIDDEN2", 64)
dropout_p = params["MODELING"].get("DROPOUT", 0.1)
model_path_cfg = params["MODELING"].get("MODEL_PATH", "../../models")
model_path = os.path.abspath(os.path.join(THIS_DIR, model_path_cfg))

# Alpaca keys (Paper by default). You can override by setting ALPACA_KEY_ID / ALPACA_SECRET env vars.
ALPACA_KEY_ID = os.getenv("ALPACA_KEY_ID", keys["KEYS"].get("APCA-API-KEY-ID-Paper_v3") or keys["KEYS"].get("APCA-API-KEY-ID-Paper"))
ALPACA_SECRET = os.getenv("ALPACA_SECRET", keys["KEYS"].get("APCA-API-SECRET-KEY-Paper_v3") or keys["KEYS"].get("APCA-API-SECRET-KEY-Paper"))
ALPACA_BASE = os.getenv("ALPACA_BASE", "https://paper-api.alpaca.markets")

# Optional: Which node_id to target; if None we choose most populated leaf.
TARGET_NODE_ID_ENV = os.getenv("TARGET_NODE_ID")
TARGET_NODE_ID = int(TARGET_NODE_ID_ENV) if TARGET_NODE_ID_ENV else None

# Ticker universe: env var TICKERS="AAPL,MSFT" or load S&P500 CSV as fallback (Symbol column)
TICKERS_ENV = os.getenv("TICKERS")
if TICKERS_ENV:
    TICKERS = [t.strip().upper() for t in TICKERS_ENV.split(",") if t.strip()]
else:
    sp500_csv = os.path.join(DATA_DIR, "sp500_companies_as_of_jan_2025.csv")
    if os.path.exists(sp500_csv):
        try:
            _df_sp = pd.read_csv(sp500_csv)
            # Use the first 20 to avoid overloading yfinance; adjust as needed
            TICKERS = _df_sp["Symbol"].dropna().astype(str).str.upper().tolist()[:20]
        except Exception:
            TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "NFLX", "AMD", "INTC"]
    else:
        TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "NFLX", "AMD", "INTC"]

# Device selection for Torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Model definitions and loading
# -----------------------------
class MLP(nn.Module):
    """MLP architecture matching the training script (Sequential under `net`).

    net.0: Linear(in_dim -> h1)
    net.1: ReLU
    net.2: Dropout
    net.3: Linear(h1 -> h2)
    net.4: Dropout
    net.5: Linear(h2 -> 1)

    The embed() method returns the representation after net.4 (output of second hidden layer), shape (N, h2).
    """

    def __init__(self, in_dim: int, h1: int, h2: int, dropout_p: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(h1, h2),
            nn.Dropout(dropout_p),
            nn.Linear(h2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net[0](x)
        x = self.net[1](x)
        x = self.net[2](x)
        x = self.net[3](x)
        x = self.net[4](x)
        return x

# Load feature list
FEATURES: List[str] = []
if feature_path:
    # Resolve relative to this script first
    feat_abs = os.path.abspath(os.path.join(THIS_DIR, feature_path))
    if not os.path.exists(feat_abs):
        # Try relative to EXP_DIR
        feat_abs = os.path.abspath(os.path.join(EXP_DIR, os.path.normpath(feature_path)))
    if not os.path.exists(feat_abs):
        raise FileNotFoundError(f"FEATURE_PATH not found: {feature_path}")
    with open(feat_abs, "r") as f:
        for line in f:
            FEATURES.append(line.strip())
else:
    raise RuntimeError("FEATURE_PATH not set in params.yaml -> DATA_PREP.FEATURE_PATH")

IN_DIM = len(FEATURES)

# Load MLP checkpoint
ckpt_candidates = [
    os.path.join(model_path, "best_acc_model.pt"),
    os.path.join(model_path, "best_acc_model_temp.pt"),
]
_ckpt = None
for c in ckpt_candidates:
    if os.path.exists(c):
        _ckpt = c
        break
if _ckpt is None:
    raise FileNotFoundError("No model checkpoint found in models directory (best_acc_model*.pt)")

ckpt = torch.load(_ckpt, map_location=DEVICE)
model = MLP(IN_DIM, hidden1, hidden2, dropout_p).to(DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Load Hoeffding Tree
ht_path = os.path.join(model_path, "hoeffding_tree.pkl")
with open(ht_path, "rb") as f:
    ht = pickle.load(f)

df_tree = ht.to_dataframe().reset_index()
# River names the first column as the index; normalize to 'node_id'
if df_tree.columns[0] != "node_id":
    df_tree = df_tree.rename(columns={df_tree.columns[0]: "node_id"})
leaf_ids = df_tree.loc[df_tree["is_leaf"], "node_id"].tolist()

if TARGET_NODE_ID is None:
    # Try to choose most populated leaf; fall back to first leaf
    pop_col = None
    for c in ["n", "n_obs", "n_samples", "n_bytes"]:
        if c in df_tree.columns:
            pop_col = c
            break
    cand = df_tree[df_tree["is_leaf"]]
    if pop_col is not None:
        cand = cand.sort_values(pop_col, ascending=False)
    TARGET_NODE_ID = int(cand.iloc[0]["node_id"]) if not cand.empty else int(leaf_ids[0])

print(f"[init] Target node_id for entry: {TARGET_NODE_ID}")

# -----------------------------
# Helpers: Alpaca RTH calendar and trading via REST
# -----------------------------
EASTERN = pytz.timezone("US/Eastern")


def alpaca_headers() -> Dict[str, str]:
    if not ALPACA_KEY_ID or not ALPACA_SECRET:
        raise RuntimeError("Alpaca API keys are missing. Set env ALPACA_KEY_ID/ALPACA_SECRET or fill keys.yaml.")
    return {
        "APCA-API-KEY-ID": ALPACA_KEY_ID,
        "APCA-API-SECRET-KEY": ALPACA_SECRET,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def build_calendar_map(start_dt: datetime, end_dt: datetime) -> Dict[datetime.date, Tuple[datetime, datetime]]:
    # Alpaca calendar v2: /v2/calendar?start=YYYY-MM-DD&end=YYYY-MM-DD
    url = f"{ALPACA_BASE}/v2/calendar"
    params_q = {
        "start": start_dt.strftime("%Y-%m-%d"),
        "end": end_dt.strftime("%Y-%m-%d"),
    }
    r = requests.get(url, headers=alpaca_headers(), params=params_q, timeout=30)
    r.raise_for_status()
    days = r.json()
    cal_map: Dict[datetime.date, Tuple[datetime, datetime]] = {}
    for d in days:
        # d: {'date': '2025-09-24', 'open': '09:30', 'close': '16:00', ...}
        date_str = d.get("date")
        open_str = d.get("open")
        close_str = d.get("close")
        if not date_str or not open_str or not close_str:
            continue
        y, m, dd = map(int, date_str.split("-"))
        oh, om = map(int, open_str.split(":"))
        ch, cm = map(int, close_str.split(":"))
        open_dt = EASTERN.localize(datetime(y, m, dd, oh, om))
        close_dt = EASTERN.localize(datetime(y, m, dd, ch, cm))
        cal_map[open_dt.date()] = (open_dt, close_dt)
    return cal_map


def is_rth(ts: pd.Timestamp, cal_map: Dict[datetime.date, Tuple[datetime, datetime]]) -> bool:
    if ts.tzinfo is None:
        # Assume UTC if tz-naive
        ts_eastern = ts.tz_localize("UTC").astimezone(EASTERN)
    else:
        try:
            ts_eastern = ts.tz_convert(EASTERN)  # type: ignore[attr-defined]
        except Exception:
            ts_eastern = ts.tz_localize("UTC").astimezone(EASTERN)
    d = ts_eastern.date()
    if d not in cal_map:
        return False
    open_dt, close_dt = cal_map[d]
    return open_dt <= ts_eastern < close_dt


def get_positions() -> List[dict]:
    url = f"{ALPACA_BASE}/v2/positions"
    r = requests.get(url, headers=alpaca_headers(), timeout=30)
    if r.status_code == 404:
        return []
    r.raise_for_status()
    return r.json()


def get_filled_orders_for_symbol(symbol: str, limit: int = 50) -> List[dict]:
    # /v2/orders?status=closed will include filled orders; we also filter by symbol client-side to be safe
    url = f"{ALPACA_BASE}/v2/orders"
    params_q = {
        "status": "closed",
        "limit": str(limit),
        "nested": "false",
        "direction": "desc",
    }
    r = requests.get(url, headers=alpaca_headers(), params=params_q, timeout=30)
    r.raise_for_status()
    orders = r.json()
    # Filter filled and by symbol
    out = []
    for o in orders:
        if str(o.get("status", "")).lower() != "filled":
            continue
        if str(o.get("symbol", "")).upper() != symbol.upper():
            continue
        out.append(o)
    return out


def submit_market_order(symbol: str, side: str, qty: int = 1) -> dict | None:
    url = f"{ALPACA_BASE}/v2/orders"
    payload = {
        "symbol": symbol,
        "qty": qty,
        "side": side.lower(),
        "type": "market",
        "time_in_force": "day",
    }
    try:
        r = requests.post(url, headers=alpaca_headers(), json=payload, timeout=30)
        r.raise_for_status()
        od = r.json()
        print(f"[order] {side.upper()} {qty} {symbol}: submitted id={od.get('id')}")
        return od
    except Exception as e:
        print(f"[order] {side.upper()} {symbol} failed: {e}")
        return None


def close_positions_older_than_30m():
    """Check open positions and sell those whose most recent BUY fill is >= 30 minutes ago."""
    try:
        positions = get_positions()
    except Exception as e:
        print(f"[pos] Cannot fetch positions: {e}")
        return

    if not positions:
        print("[pos] No open positions.")
        return

    now_utc = datetime.now(timezone.utc)
    cutoff = now_utc - timedelta(minutes=30)

    for p in positions:
        symbol = p.get("symbol") or p.get("asset_symbol")
        if not symbol:
            continue
        try:
            qty_str = p.get("qty") or p.get("quantity")
            qty = int(float(qty_str)) if qty_str is not None else None
        except Exception:
            qty = None
        if not qty:
            continue

        # Find last filled BUY order for this symbol
        try:
            orders = get_filled_orders_for_symbol(symbol, limit=50)
        except Exception as e:
            print(f"[pos] get_orders failed for {symbol}: {e}")
            continue

        last_buy_fill = None
        for o in orders:
            side = str(o.get("side", "")).lower()
            if side != "buy":
                continue
            filled_at = o.get("filled_at")
            if not filled_at:
                continue
            try:
                # ISO8601 with Z
                dt = datetime.fromisoformat(str(filled_at).replace("Z", "+00:00"))
            except Exception:
                continue
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            if (last_buy_fill is None) or (dt > last_buy_fill):
                last_buy_fill = dt

        if last_buy_fill and last_buy_fill <= cutoff:
            print(f"[pos] Closing {symbol}: last BUY fill at {last_buy_fill.isoformat()} (<= {cutoff.isoformat()})")
            submit_market_order(symbol, side="sell", qty=qty)
        else:
            print(f"[pos] Keep {symbol}: last BUY fill at {last_buy_fill}")

# -----------------------------
# Data acquisition via yfinance
# -----------------------------

def download_minute_data(tickers: List[str], days_hist: int = 5) -> Dict[str, pd.DataFrame]:
    """Download 1m bars for last `days_hist` days for the given tickers.

    Returns a dict ticker -> DataFrame with UTC tz-aware DatetimeIndex and columns:
    Open, High, Low, Close, Adj Close, Volume
    """
    print(f"[yf] Downloading {days_hist}d of 1m data for {len(tickers)} tickers...")
    data: Dict[str, pd.DataFrame] = {}

    # yfinance supports multi-ticker download, but to avoid rate limits and simplify, iterate.
    for t in tickers:
        try:
            df = yf.download(t, period=f"{days_hist}d", interval="1m", auto_adjust=True, prepost=True, progress=False)
            if df is None or df.empty:
                print(f"[yf] {t}: no data returned")
                continue
            # Ensure tz-aware UTC index
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            else:
                df.index = df.index.tz_convert("UTC")
            data[t] = df
            time.sleep(0.2)  # small backoff
        except Exception as e:
            print(f"[yf] {t}: error {e}")
    return data

# -----------------------------
# Feature and embedding computation
# -----------------------------

def compute_latest_embedding(df_raw: pd.DataFrame) -> Tuple[pd.Timestamp | None, np.ndarray | None]:
    """Given raw 1m OHLCV DataFrame (UTC index),
    - build approximate vwap and lowercase volume
    - compute features with training params
    - align to FEATURES, fill missing with 0
    - return (last_ts, embedding_vector) where last_ts is the timestamp of the last row used
    """
    if df_raw is None or df_raw.empty:
        return None, None

    df = df_raw.copy()
    # VWAP approximation via typical price
    df["vwap"] = (df["High"] + df["Low"] + df["Close"]) / 3.0
    df["volume"] = df["Volume"]

    # Compute features
    df_feat, _ = generate_features(
        df=df[["vwap", "volume"]],
        ema_periods=ema_periods,
        slope_periods=slope_periods,
        z_norm_window=z_norm_window,
    )

    # Build feature frame and align to training FEATURES list
    # Some features may be missing in live due to rolling windows; fill with 0
    X = pd.DataFrame(index=df_feat.index)
    for col in FEATURES:
        if col in df_feat.columns:
            X[col] = df_feat[col]
        else:
            X[col] = 0.0

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Take the latest timestamp row
    if X.empty:
        return None, None

    last_ts = X.index[-1]
    x = X.iloc[[-1]].astype(np.float32).values

    with torch.no_grad():
        emb = model.embed(torch.tensor(x, dtype=torch.float32, device=DEVICE)).cpu().numpy()[0]
    return last_ts, emb

# -----------------------------
# Entry rule via target leaf
# -----------------------------

def embedding_matches_target_leaf(emb: np.ndarray, target_leaf_id: int) -> bool:
    sample = {f"emb_{i}": float(emb[i]) for i in range(len(emb))}
    try:
        # River's internal traversal returns the leaf node id
        leaf_id = ht._root.traverse(sample)
    except Exception:
        leaf_id = None
    return leaf_id == target_leaf_id

# -----------------------------
# Main flow
# -----------------------------

while True:
    # Build calendar map for RTH filtering covering the last ~10 calendar days
    end_dt = datetime.now(tz=EASTERN)
    start_dt = end_dt - timedelta(days=10)
    cal_map = build_calendar_map(start_dt=start_dt, end_dt=end_dt)

    # Download 5 days of 1m bars to have enough context; we will only act on last 2 days.
    data = download_minute_data(TICKERS, days_hist=5)

    # Restrict decisions to last 2 days
    decision_cutoff_utc = datetime.now(timezone.utc) - timedelta(days=2)

    actions = []

    for sym, df in data.items():
        # Filter to RTH using Alpaca calendar
        if df is None or df.empty:
            continue
        # Keep only rows within RTH
        mask_rth = df.index.to_series().map(lambda ts: is_rth(ts, cal_map))
        df_rth = df.loc[mask_rth]
        if df_rth.empty:
            print(f"[eval] {sym}: no RTH rows")
            continue

        # Apply 2-day decision window
        df_rth = df_rth[df_rth.index >= decision_cutoff_utc]
        if df_rth.empty:
            print(f"[eval] {sym}: no rows within last 2 days")
            continue

        # Compute latest embedding
        last_ts, emb = compute_latest_embedding(df_rth)
        if emb is None:
            print(f"[eval] {sym}: could not compute embedding (insufficient features)")
            continue

        # Entry rule: does the sample land in the target leaf?
        if embedding_matches_target_leaf(emb, TARGET_NODE_ID):
            print(f"[signal] {sym} @ {last_ts}: matches target leaf {TARGET_NODE_ID} -> BUY")
            order = submit_market_order(sym, side="buy", qty=1)
            actions.append((sym, last_ts, "BUY", order.get("id") if isinstance(order, dict) else None))
        else:
            print(f"[signal] {sym} @ {last_ts}: no entry")

    # Post-trade management: close positions >= 30 minutes old
    close_positions_older_than_30m()

    print("[done] Actions:")
    for a in actions:
        print(a)
