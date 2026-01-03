from __future__ import annotations

import os
from collections import deque

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from alpaca.data.live import StockDataStream
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest

EPS = 1e-12

# -------------------------------------------------
# Trading-Regel
# -------------------------------------------------
EMA_SPAN = 20
REBALANCE_N = 10
DEADBAND = 0.10
DELAY_STEPS = 1

MIN_QTY_TRADE = 10

# -------------------------------------------------
# 1) Zeit Debug (tz-safe)
# -------------------------------------------------
now_utc = pd.Timestamp.utcnow()
print("NOW UTC:", now_utc)
print("NOW ET :", now_utc.tz_convert("US/Eastern"))
print("NOW BER:", now_utc.tz_convert("Europe/Berlin"))

# -------------------------------------------------
# 2) Keys laden
# -------------------------------------------------
with open("../../conf/keys.yaml", "r", encoding="utf-8") as f:
    keys = yaml.safe_load(f)

API_KEY = keys["KEYS"]["APCA-API-KEY-ID-Data"]
SECRET = keys["KEYS"]["APCA-API-SECRET-KEY-Data"]

print("API KEY geladen:", API_KEY[:6], "...")

# -------------------------------------------------
# 3) Alpaca Clients (Paper Trading + Live Data Stream)
# -------------------------------------------------
trade = TradingClient(api_key=API_KEY, secret_key=SECRET, paper=True)
account = trade.get_account()
print("Paper account OK | Equity:", account.equity)

stream = StockDataStream(API_KEY, SECRET)

# -------------------------------------------------
# 4) Modell + Featureliste laden
# -------------------------------------------------
MODEL_PATH = "../../models/gbt_vol_label_30m.txt"
model = lgb.Booster(model_file=MODEL_PATH)
print("Model loaded:", MODEL_PATH)

FEATURES_TXT = "../03_pre_split_prep/features.txt"
with open(FEATURES_TXT, "r", encoding="utf-8") as f:
    FEATURE_COLS = [ln.strip() for ln in f if ln.strip()]

print("Features loaded:", len(FEATURE_COLS), FEATURE_COLS)

# -------------------------------------------------
# 5) Deployment-Konfig
# -------------------------------------------------
SYMBOLS = ["AAPL", "MSFT", "NVDA"]

LOG_DIR = "live_logs"
os.makedirs(LOG_DIR, exist_ok=True)

SIGNALS_CSV = os.path.join(LOG_DIR, "signals.csv")  # jede Minute p,w,action
ORDERS_CSV = os.path.join(LOG_DIR, "orders.csv")  # Order submissions
FILLS_CSV = os.path.join(LOG_DIR, "fills.csv")  # Filled/Closed Orders Snapshots


# -------------------------------------------------
# 6) RTH check
# -------------------------------------------------
def is_rth(ts_utc: pd.Timestamp) -> bool:
    ts_et = ts_utc.tz_convert("US/Eastern")
    t = ts_et.time()
    return (t >= pd.Timestamp("09:30").time()) and (t < pd.Timestamp("16:00").time())


def minute_of_session(ts_utc: pd.Timestamp) -> int:
    ts_et = ts_utc.tz_convert("US/Eastern")
    return int((ts_et.hour - 9) * 60 + (ts_et.minute - 30))


# -------------------------------------------------
# 7) Buffer pro Symbol (nur aktueller ET-Tag)
# -------------------------------------------------
buffers: dict[str, pd.DataFrame] = {s: pd.DataFrame() for s in SYMBOLS}


def keep_only_today_et(df: pd.DataFrame, ts_utc: pd.Timestamp) -> pd.DataFrame:
    if df.empty:
        return df
    cur_date = ts_utc.tz_convert("US/Eastern").date()
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["_date_et"] = df["timestamp"].dt.tz_convert("US/Eastern").dt.date
    df = df[df["_date_et"] == cur_date].drop(columns=["_date_et"]).reset_index(drop=True)
    return df


# -------------------------------------------------
# State pro Symbol (EMA + deadband + delay)
# -------------------------------------------------
state = {
    s: {
        "ema_p": None,
        "last_w": 0.0,
        "delay": deque([0.0] * (DELAY_STEPS + 1), maxlen=DELAY_STEPS + 1),
    }
    for s in SYMBOLS
}


def ema_update(prev: float | None, x: float, span: int) -> float:
    alpha = 2.0 / (span + 1.0)
    return x if prev is None else alpha * x + (1.0 - alpha) * prev


# -------------------------------------------------
# 8) Features wie in features.py (pro Tag)
# -------------------------------------------------
def compute_features_last_row(df_day: pd.DataFrame) -> dict | None:
    if df_day is None or len(df_day) < 31:
        return None

    d = df_day.sort_values("timestamp").reset_index(drop=True)

    close = d["Close"].astype(float)
    high = d["High"].astype(float)
    low = d["Low"].astype(float)
    vol = d["Volume"].astype(float)

    # VWAP intraday
    dv = close * vol
    vwap = dv.cumsum() / (vol.cumsum() + EPS)

    # Preisfeatures
    log_ret_1m = np.log(close / close.shift(1))
    roll_ret_5m = log_ret_1m.rolling(5, min_periods=5).sum()
    roll_vol_15m = (log_ret_1m ** 2).rolling(15, min_periods=15).mean().pow(0.5)

    # VWAP Features
    vwap_dev = (close - vwap) / (vwap + EPS)
    vwap_roll_mean = vwap.rolling(30, min_periods=30).mean()
    vwap_roll_std = vwap.rolling(30, min_periods=30).std(ddof=0)
    vwap_z_30m = (vwap - vwap_roll_mean) / (vwap_roll_std + EPS)

    # Volume Features
    vol_roll_mean = vol.rolling(30, min_periods=30).mean()
    vol_roll_std = vol.rolling(30, min_periods=30).std(ddof=0)
    volume_z_30m = (vol - vol_roll_mean) / (vol_roll_std + EPS)
    roll_vol_15m_volume = vol.rolling(15, min_periods=15).mean()

    # Range Features
    hl_spread = (high - low) / (close + EPS)
    high_max_15 = high.rolling(15, min_periods=15).max()
    low_min_15 = low.rolling(15, min_periods=15).min()
    hl_range_15m = (high_max_15 - low_min_15) / (close + EPS)

    # Zeitfeatures (ET)
    ts = pd.to_datetime(d["timestamp"], utc=True, errors="coerce")
    ts_et = ts.dt.tz_convert("US/Eastern")
    minute_of_session_series = (ts_et.dt.hour - 9) * 60 + (ts_et.dt.minute - 30)
    minute_clip = minute_of_session_series.clip(lower=0, upper=389)

    time_sin = np.sin(2 * np.pi * minute_clip / 390.0)
    time_cos = np.cos(2 * np.pi * minute_clip / 390.0)
    is_open30 = (minute_of_session_series < 30).astype(int)
    is_close30 = (minute_of_session_series >= 360).astype(int)

    i = len(d) - 1

    feat_all = {
        "log_ret_1m": float(log_ret_1m.iat[i]),
        "roll_ret_5m": float(roll_ret_5m.iat[i]),
        "roll_vol_15m": float(roll_vol_15m.iat[i]),
        "vwap_dev": float(vwap_dev.iat[i]),
        "vwap_z_30m": float(vwap_z_30m.iat[i]),
        "volume_z_30m": float(volume_z_30m.iat[i]),
        "roll_vol_15m_volume": float(roll_vol_15m_volume.iat[i]),
        "hl_spread": float(hl_spread.iat[i]),
        "hl_range_15m": float(hl_range_15m.iat[i]),
        "time_sin": float(time_sin.iat[i]),
        "time_cos": float(time_cos.iat[i]),
        "is_open30": int(is_open30.iat[i]),
        "is_close30": int(is_close30.iat[i]),
    }

    out = {}
    for c in FEATURE_COLS:
        if c not in feat_all:
            return None
        out[c] = feat_all[c]

    if any(pd.isna(list(out.values()))):
        return None

    return out


# -------------------------------------------------
# 9) Logging helper
# -------------------------------------------------
def append_csv(path: str, row: dict) -> None:
    pd.DataFrame([row]).to_csv(
        path,
        mode="a",
        header=not os.path.exists(path),
        index=False
    )


# -------------------------------------------------
# Orders + Fills helper
# -------------------------------------------------
def get_position_qty(symbol: str) -> int:
    try:
        pos = trade.get_open_position(symbol)
        return int(float(getattr(pos, "qty", 0) or 0))
    except Exception:
        return 0


def submit_market(side: str, symbol: str, qty: int) -> tuple[str | None, str | None]:
    req = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
        time_in_force=TimeInForce.DAY,
    )
    try:
        od = trade.submit_order(req)
        order_id = getattr(od, "id", None) or (od.get("id") if isinstance(od, dict) else None)
        status = getattr(od, "status", None) or (od.get("status") if isinstance(od, dict) else None)
        return order_id, status
    except Exception as e:
        print(f"[order] {side.upper()} {symbol} failed: {e}")
        return None, None


def snapshot_recent_closed_orders(symbol: str, limit: int = 20) -> None:
    """Schreibt eine Momentaufnahme der zuletzt geschlossenen Orders (inkl. Fills) in fills.csv."""
    try:
        req = GetOrdersRequest(status=QueryOrderStatus.CLOSED, limit=limit)
        orders = trade.get_orders(req)
    except Exception:
        return

    for o in orders:
        sym = (getattr(o, "symbol", "") or "").upper()
        if sym != symbol.upper():
            continue

        append_csv(FILLS_CSV, {
            "logged_at_utc": pd.Timestamp.utcnow().isoformat(),
            "order_id": getattr(o, "id", None),
            "symbol": sym,
            "side": str(getattr(o, "side", "")),
            "status": str(getattr(o, "status", "")),
            "qty": str(getattr(o, "qty", "")),
            "filled_qty": str(getattr(o, "filled_qty", "")),
            "filled_avg_price": str(getattr(o, "filled_avg_price", "")),
            "submitted_at": str(getattr(o, "submitted_at", "")),
            "filled_at": str(getattr(o, "filled_at", "")),
        })


# -------------------------------------------------
# 10) Async Bar Handler
# -------------------------------------------------
async def on_bar(bar):
    sym = bar.symbol
    ts = pd.Timestamp(bar.timestamp).tz_convert("UTC")

    # Debug: zeigt ob Bars grundsätzlich ankommen
    print("BAR (raw):", sym, ts, "CLOSE:", bar.close)

    if not is_rth(ts):
        print("  -> outside RTH, skipping")
        return

    row = {
        "timestamp": ts,
        "Open": float(bar.open),
        "High": float(bar.high),
        "Low": float(bar.low),
        "Close": float(bar.close),
        "Volume": float(bar.volume),
    }

    df = buffers[sym]
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df = keep_only_today_et(df, ts)
    buffers[sym] = df

    feats = compute_features_last_row(df)
    if feats is None:
        return

    X = pd.DataFrame([{"symbol": sym, **feats}])
    X["symbol"] = X["symbol"].astype("category")

    raw = float(model.predict(X)[0])

    # p_raw: falls logits, sigmoid; sonst direkt
    p_raw = raw if (0.0 <= raw <= 1.0) else float(1.0 / (1.0 + np.exp(-raw)))
    p_raw = float(np.clip(p_raw, 0.0, 1.0))

    # ----- Logik: EMA -> w -> rebalance -> deadband -> delay -----
    st = state[sym]

    # EMA smoothing
    p_ema = ema_update(st["ema_p"], p_raw, EMA_SPAN)
    st["ema_p"] = p_ema

    # exposure rule
    w_raw = float(np.clip(1.0 - p_ema, 0.0, 1.0))

    # rebalance only every N minutes (ab 09:30 ET)
    mos = minute_of_session(ts)
    can_reb = (mos >= 0) and (mos % REBALANCE_N == 0)
    w_candidate = w_raw if can_reb else st["last_w"]

    # deadband
    if abs(w_candidate - st["last_w"]) < DEADBAND:
        w_target = st["last_w"]
    else:
        w_target = w_candidate
        st["last_w"] = w_target

    # delay (execute previous target)
    st["delay"].append(w_target)
    w_exec = float(st["delay"][0])  # bei DELAY_STEPS=1: w von t-1

    # ----- Umsetzung als Target-Exposure (Position size) -----
    acc = trade.get_account()
    equity = float(acc.equity)
    n = max(1, len(SYMBOLS))

    target_dollars = (equity * w_exec) / n
    px = max(float(bar.close), EPS)
    target_qty = int(np.floor(target_dollars / px))

    cur_qty = get_position_qty(sym)
    delta = target_qty - cur_qty

    action = "HOLD"
    order_id = None
    order_status = None

    if abs(delta) >= MIN_QTY_TRADE:
        if delta > 0:
            action = "BUY"
            order_id, order_status = submit_market("buy", sym, int(delta))
        else:
            action = "SELL"
            order_id, order_status = submit_market("sell", sym, int(-delta))

    print(
        f"  -> p_raw={p_raw:.3f}, p_ema={p_ema:.3f}, w_tgt={w_target:.3f}, w_exec={w_exec:.3f}, qty {cur_qty}->{target_qty} -> {action}")

    append_csv(SIGNALS_CSV, {
        "timestamp_utc": ts.isoformat(),
        "symbol": sym,
        "close": float(bar.close),
        "p_raw": p_raw,
        "p_ema": float(p_ema),
        "w_target": float(w_target),
        "w_exec": float(w_exec),
        "cur_qty": int(cur_qty),
        "target_qty": int(target_qty),
        "delta_qty": int(delta),
        "account_equity": equity,
        "action": action,
        "order_id": order_id,
        "order_status": order_status,
    })

    if action in ("BUY", "SELL"):
        append_csv(ORDERS_CSV, {
            "timestamp_utc": ts.isoformat(),
            "symbol": sym,
            "action": action,
            "qty": int(abs(delta)),
            "order_id": order_id,
            "submit_status": order_status,
        })
        snapshot_recent_closed_orders(sym, limit=20)


# -------------------------------------------------
# 11) Subscribe & Run
# -------------------------------------------------
for s in SYMBOLS:
    stream.subscribe_bars(on_bar, s)

print("Subscribed to:", SYMBOLS)
print("Waiting for live bars... (US-RTH)")
stream.run()
