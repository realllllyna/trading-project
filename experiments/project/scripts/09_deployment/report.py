"""
Liest die Logfiles aus live_logs/ und erzeugt eine einfache, gut erklärbare Auswertung.

Eingaben:
- live_logs/signals.csv  (jede Minute: p, w, action)
- live_logs/orders.csv   (Order-Submissions)
- live_logs/fills.csv    (Closed orders inkl. filled_avg_price / filled_at)

Outputs:
- summary.txt                 (kurze Kennzahlen)
- signals_overview.csv        (bereinigte Signals)
- orders_overview.csv         (bereinigte Orders)
- fills_overview.csv          (bereinigte Fills)
- plot_p_w.png                (p & w über Zeit)
- plot_actions.png            (BUY/SELL Marker über Zeit)
- plot_trades_by_hour.png     (Trade-Verteilung nach Stunde)
- plot_trades_by_day.png      (Trades pro Tag)
- plot_price_with_actions.png (Preis + BUY/SELL Marker)
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


def _save_df(df: pd.DataFrame, out_path: str) -> None:
    df.to_csv(out_path, index=False)


def _save_plot(fig, out_path: str) -> None:
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _write_summary(out_path: str, lines: list[str]) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).strip() + "\n")


def _pick_time_col(df: pd.DataFrame) -> str | None:
    for c in ["timestamp_utc", "timestamp", "logged_at_utc"]:
        if c in df.columns:
            return c
    return None


def build_report(paths: Paths) -> None:
    _ensure_out_dir(paths.out_dir)

    signals = _parse_times(_read_csv_if_exists(paths.signals))
    orders = _parse_times(_read_csv_if_exists(paths.orders))
    fills = _parse_times(_read_csv_if_exists(paths.fills))

    # Save cleaned overviews
    if not signals.empty:
        _save_df(signals, os.path.join(paths.out_dir, "signals_overview.csv"))
    if not orders.empty:
        _save_df(orders, os.path.join(paths.out_dir, "orders_overview.csv"))
    if not fills.empty:
        _save_df(fills, os.path.join(paths.out_dir, "fills_overview.csv"))

    summary_lines: list[str] = []
    summary_lines.append("=== Paper Trading Report (Step 9) ===")

    # ---- Signals summary ----
    if signals.empty:
        summary_lines.append("signals.csv: NICHT gefunden oder leer.")
        _write_summary(os.path.join(paths.out_dir, "summary.txt"), summary_lines)
        print("Done (keine signals.csv).")
        return

    tcol = _pick_time_col(signals)
    if tcol is None:
        summary_lines.append("signals.csv: kein Zeitstempel-Feld gefunden (timestamp_utc/timestamp).")
        _write_summary(os.path.join(paths.out_dir, "summary.txt"), summary_lines)
        print("Done (kein Zeitstempel).")
        return

    signals = signals.sort_values(tcol).dropna(subset=[tcol])

    # Ensure required columns exist
    for col in ["symbol", "close", "p_high_vol", "w_exposure", "action"]:
        if col not in signals.columns:
            summary_lines.append(f"signals.csv: Spalte fehlt: {col}")
    # Best effort: continue anyway

    start_ts = signals[tcol].min()
    end_ts = signals[tcol].max()
    summary_lines.append(f"Signals Zeitraum: {start_ts}  bis  {end_ts}")
    summary_lines.append(f"Anzahl Signal-Zeilen: {len(signals):,}")

    # Distinct symbols
    if "symbol" in signals.columns:
        syms = sorted(signals["symbol"].dropna().astype(str).unique().tolist())
        summary_lines.append(f"Symbole in signals.csv: {len(syms)} (z.B. {syms[:10]})")

    # Action counts
    if "action" in signals.columns:
        action_counts = signals["action"].fillna("NA").value_counts()
        summary_lines.append("Action-Verteilung:")
        for k, v in action_counts.items():
            summary_lines.append(f"  {k}: {int(v)}")

    # p/w stats
    if "p_high_vol" in signals.columns:
        summary_lines.append(f"p_high_vol: mean={signals['p_high_vol'].mean():.4f}, "
                             f"min={signals['p_high_vol'].min():.4f}, max={signals['p_high_vol'].max():.4f}")
    if "w_exposure" in signals.columns:
        summary_lines.append(f"w_exposure: mean={signals['w_exposure'].mean():.4f}, "
                             f"min={signals['w_exposure'].min():.4f}, max={signals['w_exposure'].max():.4f}")

    # ---- Orders / Fills summary ----
    if not orders.empty:
        otcol = _pick_time_col(orders)
        if otcol and otcol in orders.columns:
            orders = orders.sort_values(otcol)
        summary_lines.append(f"orders.csv: {len(orders):,} Zeilen")
        if "action" in orders.columns:
            oc = orders["action"].fillna("NA").value_counts()
            summary_lines.append("Orders actions:")
            for k, v in oc.items():
                summary_lines.append(f"  {k}: {int(v)}")
    else:
        summary_lines.append("orders.csv: nicht gefunden oder leer.")

    if not fills.empty:
        ftcol = _pick_time_col(fills)
        if ftcol and ftcol in fills.columns:
            fills = fills.sort_values(ftcol)
        summary_lines.append(f"fills.csv: {len(fills):,} Zeilen")
        if "status" in fills.columns:
            fc = fills["status"].fillna("NA").value_counts()
            summary_lines.append("Fills status:")
            for k, v in fc.items():
                summary_lines.append(f"  {k}: {int(v)}")
    else:
        summary_lines.append("fills.csv: nicht gefunden oder leer.")

    _write_summary(os.path.join(paths.out_dir, "summary.txt"), summary_lines)

    # ----------------- PLOTS -----------------
    # 1) p & w over time (for one symbol if multiple)
    plot_df = signals.copy()
    if "symbol" in plot_df.columns and plot_df["symbol"].nunique() > 1:
        # choose most frequent symbol to keep plots readable
        top_sym = plot_df["symbol"].value_counts().index[0]
        plot_df = plot_df[plot_df["symbol"] == top_sym].copy()

    # ensure numeric
    for col in ["p_high_vol", "w_exposure", "close"]:
        if col in plot_df.columns:
            plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")

    # Plot p and w
    if "p_high_vol" in plot_df.columns and "w_exposure" in plot_df.columns:
        fig = plt.figure()
        plt.plot(plot_df[tcol], plot_df["p_high_vol"], label="p_high_vol")
        plt.plot(plot_df[tcol], plot_df["w_exposure"], label="w_exposure")
        plt.xlabel("Zeit (UTC)")
        plt.ylabel("Wert")
        plt.title("Model Output: p und w über Zeit")
        plt.legend()
        _save_plot(fig, os.path.join(paths.out_dir, "plot_p_w.png"))

    # Plot price with actions
    if "close" in plot_df.columns:
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

    # Plot actions over time (p line + markers)
    if "p_high_vol" in plot_df.columns and "action" in plot_df.columns:
        fig = plt.figure()
        plt.plot(plot_df[tcol], plot_df["p_high_vol"], label="p_high_vol")
        buys = plot_df[plot_df["action"] == "BUY"]
        sells = plot_df[plot_df["action"] == "SELL"]
        if not buys.empty:
            plt.scatter(buys[tcol], buys["p_high_vol"], marker="^", label="BUY")
        if not sells.empty:
            plt.scatter(sells[tcol], sells["p_high_vol"], marker="v", label="SELL")
        plt.xlabel("Zeit (UTC)")
        plt.ylabel("p_high_vol")
        plt.title("p_high_vol + BUY/SELL Marker")
        plt.legend()
        _save_plot(fig, os.path.join(paths.out_dir, "plot_actions.png"))

    # Trade distribution plots (from orders if available; else derive from signals actions)
    trade_events = pd.DataFrame()
    if not orders.empty:
        ocol = _pick_time_col(orders)
        if ocol:
            trade_events = orders.copy()
            trade_events["_t"] = trade_events[ocol]
            trade_events["action"] = trade_events.get("action", "NA")
    else:
        # fallback: actions in signals
        if "action" in signals.columns:
            trade_events = signals[signals["action"].isin(["BUY", "SELL"])].copy()
            trade_events["_t"] = trade_events[tcol]

    if not trade_events.empty:
        # by hour (ET is nicer for trading)
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
