"""
Minimaler Step-9 Report:

Eingaben (live_logs/):
- signals.csv  (jede Minute: p/w/qty/action)
- orders.csv   (Order-Submissions, optional)
- fills.csv    (Closed orders inkl. filled_avg_price / filled_at, optional)

Outputs (live_logs/report_out/):
- summary.txt
- plot_p_w.png
- plot_account_equity.png (falls account_equity vorhanden)
- plot_price_with_actions.png
- plot_actions.png
- plot_trades_by_hour.png
- plot_trades_by_day.png
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd


@dataclass
class Paths:
    log_dir: str = "live_logs"
    signals: str = os.path.join("live_logs", "signals.csv")
    orders: str = os.path.join("live_logs", "orders.csv")
    fills: str = os.path.join("live_logs", "fills.csv")
    out_dir: str = os.path.join("live_logs", "report_out")


def _read_csv_if_exists(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def _parse_times(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["timestamp_utc", "timestamp", "logged_at_utc", "submitted_at", "filled_at"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    return df


def _ensure_out_dir(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)


def _save_plot(fig, out_path: str) -> None:
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _write_summary(out_path: str, lines: list[str]) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).strip() + "\n")


def _pick_time_col(df: pd.DataFrame) -> str | None:
    for c in ["timestamp_utc", "timestamp", "logged_at_utc", "submitted_at", "filled_at"]:
        if c in df.columns:
            return c
    return None


def build_report(paths: Paths) -> None:
    _ensure_out_dir(paths.out_dir)

    signals = _parse_times(_read_csv_if_exists(paths.signals))
    orders = _parse_times(_read_csv_if_exists(paths.orders))
    fills = _parse_times(_read_csv_if_exists(paths.fills))

    summary: list[str] = []
    summary.append("=== Paper Trading Report (Step 9) ===")

    if signals.empty:
        summary.append("signals.csv: NICHT gefunden oder leer.")
        _write_summary(os.path.join(paths.out_dir, "summary.txt"), summary)
        print("Done (keine signals.csv).")
        return

    tcol = _pick_time_col(signals)
    if tcol is None:
        summary.append("signals.csv: kein Zeitstempel-Feld gefunden (timestamp_utc/timestamp/...).")
        _write_summary(os.path.join(paths.out_dir, "summary.txt"), summary)
        print("Done (kein Zeitstempel).")
        return

    signals = signals.sort_values(tcol).dropna(subset=[tcol])

    start_ts = signals[tcol].min()
    end_ts = signals[tcol].max()
    summary.append(f"Signals Zeitraum: {start_ts}  bis  {end_ts}")
    summary.append(f"Anzahl Signal-Zeilen: {len(signals):,}")

    if "symbol" in signals.columns:
        syms = sorted(signals["symbol"].dropna().astype(str).unique().tolist())
        summary.append(f"Symbole: {len(syms)} (z.B. {syms[:10]})")

    if "action" in signals.columns:
        c = signals["action"].fillna("NA").value_counts()
        summary.append("Action-Verteilung (signals.csv):")
        for k, v in c.items():
            summary.append(f"  {k}: {int(v)}")

    # p/w stats (support old/new columns)
    for col in ["p_high_vol", "p_raw", "p_ema"]:
        if col in signals.columns:
            s = pd.to_numeric(signals[col], errors="coerce")
            if s.notna().any():
                summary.append(f"{col}: mean={s.mean():.4f}, min={s.min():.4f}, max={s.max():.4f}")

    for col in ["w_exposure", "w_target", "w_exec"]:
        if col in signals.columns:
            s = pd.to_numeric(signals[col], errors="coerce")
            if s.notna().any():
                summary.append(f"{col}: mean={s.mean():.4f}, min={s.min():.4f}, max={s.max():.4f}")

    # Orders/Fills summary (nur Counts)
    if not orders.empty:
        summary.append(f"orders.csv: {len(orders):,} Zeilen")
        if "action" in orders.columns:
            c = orders["action"].fillna("NA").value_counts()
            summary.append("Orders actions (orders.csv):")
            for k, v in c.items():
                summary.append(f"  {k}: {int(v)}")
    else:
        summary.append("orders.csv: nicht gefunden oder leer.")

    if not fills.empty:
        summary.append(f"fills.csv: {len(fills):,} Zeilen")
        if "status" in fills.columns:
            c = fills["status"].fillna("NA").value_counts()
            summary.append("Fills status (fills.csv):")
            for k, v in c.items():
                summary.append(f"  {k}: {int(v)}")
    else:
        summary.append("fills.csv: nicht gefunden oder leer.")

    _write_summary(os.path.join(paths.out_dir, "summary.txt"), summary)

    # ----------------- Plots -----------------
    plot_df = signals.copy()

    # Wenn mehrere Symbole, nimm das häufigste für lesbare Plots
    if "symbol" in plot_df.columns and plot_df["symbol"].nunique() > 1:
        top_sym = plot_df["symbol"].value_counts().index[0]
        plot_df = plot_df[plot_df["symbol"] == top_sym].copy()

    # numeric casting
    for col in ["close", "p_high_vol", "p_raw", "p_ema", "w_exposure", "w_target", "w_exec", "account_equity"]:
        if col in plot_df.columns:
            plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")

    # choose best columns available
    p_col = "p_high_vol" if "p_high_vol" in plot_df.columns else ("p_ema" if "p_ema" in plot_df.columns else ("p_raw" if "p_raw" in plot_df.columns else None))
    w_col = "w_exposure" if "w_exposure" in plot_df.columns else ("w_exec" if "w_exec" in plot_df.columns else ("w_target" if "w_target" in plot_df.columns else None))

    # p & w over time
    if p_col and w_col and plot_df[p_col].notna().any() and plot_df[w_col].notna().any():
        fig = plt.figure()
        plt.plot(plot_df[tcol], plot_df[p_col], label=p_col)
        plt.plot(plot_df[tcol], plot_df[w_col], label=w_col)
        plt.xlabel("Zeit (UTC)")
        plt.ylabel("Wert")
        plt.title("Model Output: p und w über Zeit")
        plt.legend()
        _save_plot(fig, os.path.join(paths.out_dir, "plot_p_w.png"))

    # account equity over time (optional)
    if "account_equity" in plot_df.columns and plot_df["account_equity"].notna().any():
        fig = plt.figure()
        plt.plot(plot_df[tcol], plot_df["account_equity"], label="account_equity")
        plt.xlabel("Zeit (UTC)")
        plt.ylabel("Equity")
        plt.title("Paper Trading: Account Equity über Zeit")
        plt.legend()
        _save_plot(fig, os.path.join(paths.out_dir, "plot_account_equity.png"))

    # price with actions
    if "close" in plot_df.columns and plot_df["close"].notna().any():
        fig = plt.figure()
        plt.plot(plot_df[tcol], plot_df["close"], label="Close")
        if "action" in plot_df.columns:
            buys = plot_df[plot_df["action"] == "BUY"]
            sells = plot_df[plot_df["action"] == "SELL"]
            if not buys.empty:
                plt.scatter(buys[tcol], buys["close"], marker="^", label="BUY")
            if not sells.empty:
                plt.scatter(sells[tcol], sells["close"], marker="v", label="SELL")
        plt.xlabel("Zeit (UTC)")
        plt.ylabel("Preis")
        plt.title("Preis + BUY/SELL Punkte")
        plt.legend()
        _save_plot(fig, os.path.join(paths.out_dir, "plot_price_with_actions.png"))

    # p with action markers
    if p_col and "action" in plot_df.columns and plot_df[p_col].notna().any():
        fig = plt.figure()
        plt.plot(plot_df[tcol], plot_df[p_col], label=p_col)
        buys = plot_df[plot_df["action"] == "BUY"]
        sells = plot_df[plot_df["action"] == "SELL"]
        if not buys.empty:
            plt.scatter(buys[tcol], buys[p_col], marker="^", label="BUY")
        if not sells.empty:
            plt.scatter(sells[tcol], sells[p_col], marker="v", label="SELL")
        plt.xlabel("Zeit (UTC)")
        plt.ylabel(p_col)
        plt.title(f"{p_col} + BUY/SELL Marker")
        plt.legend()
        _save_plot(fig, os.path.join(paths.out_dir, "plot_actions.png"))

    # trade distribution (prefer orders.csv; fallback to signals BUY/SELL)
    trade_events = pd.DataFrame()

    if not orders.empty:
        ocol = _pick_time_col(orders)
        if ocol:
            trade_events = orders.copy()
            trade_events["_t"] = trade_events[ocol]
            if "action" not in trade_events.columns:
                trade_events["action"] = "NA"
    else:
        if "action" in signals.columns:
            trade_events = signals[signals["action"].isin(["BUY", "SELL"])].copy()
            trade_events["_t"] = trade_events[tcol]

    if not trade_events.empty and trade_events["_t"].notna().any():
        t_et = trade_events["_t"].dt.tz_convert("US/Eastern")
        trade_events["_hour_et"] = t_et.dt.hour
        trade_events["_date_et"] = t_et.dt.date

        by_hour = trade_events.groupby(["_hour_et", "action"]).size().unstack(fill_value=0).sort_index()
        fig = plt.figure()
        for a in by_hour.columns:
            plt.plot(by_hour.index, by_hour[a], label=str(a))
        plt.xlabel("Stunde (US/Eastern)")
        plt.ylabel("Anzahl Trades")
        plt.title("Trades nach Stunde (ET)")
        plt.legend()
        _save_plot(fig, os.path.join(paths.out_dir, "plot_trades_by_hour.png"))

        by_day = trade_events.groupby(["_date_et", "action"]).size().unstack(fill_value=0).sort_index()
        fig = plt.figure()
        for a in by_day.columns:
            plt.plot(by_day.index.astype(str), by_day[a], label=str(a))
        plt.xlabel("Tag (ET)")
        plt.ylabel("Anzahl Trades")
        plt.title("Trades pro Tag (ET)")
        plt.xticks(rotation=45)
        plt.legend()
        _save_plot(fig, os.path.join(paths.out_dir, "plot_trades_by_day.png"))

    print("Report geschrieben nach:", paths.out_dir)


def main() -> None:
    build_report(Paths())


if __name__ == "__main__":
    main()
