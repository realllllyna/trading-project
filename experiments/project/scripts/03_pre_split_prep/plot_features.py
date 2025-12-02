"""
Plot-Helfer zur Visualisierung der Beziehung zwischen wichtigen Features
und dem Volatilitäts-Target (vol_label_30m).

Dieses Skript:
- lädt ein Train-Parquet für ein S&P-500-Symbol,
- schneidet ein Fenster um einen Index,
- plottet

  1) speziell die Zeit-Features:
        time_sin, time_cos  +  vol_label_30m

  2) weitere wichtige Features einzeln zusammen mit dem Target
     (z.B. roll_vol_15m, hl_range_15m, volume_z_30m, ...).
"""

import os

import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import pandas as pd
import yaml

# Target-Spalte
TARGET_COL = "vol_label_30m"

# Wichtigste Features laut Random Forest
IMPORTANT_FEATURES = [
    "time_cos",
    "time_sin",
    "roll_vol_15m",
    "hl_range_15m",
    "roll_vol_15m_volume",
    "hl_spread",
    "volume_z_30m",
]


def plot_time_features_with_target(
        df: pd.DataFrame,
        index: int,
        window_before: int = 200,
        window_after: int = 200,
        symbol: str | None = None,
):
    """
    Separater Plot nur für:
    - time_sin
    - time_cos
    - vol_label_30m
    """

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    for col in ["time_sin", "time_cos", TARGET_COL]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fenster bauen
    start = max(0, index - window_before)
    end = min(len(df) - 1, index + window_after)
    subset = df.iloc[start: end + 1].copy().reset_index(drop=True)

    target_ts = df.loc[index, "timestamp"]
    target_idx = (subset["timestamp"] - target_ts).abs().idxmin()

    fig, ax1 = plt.subplots(figsize=(18, 9))

    # Zeit-Features (sin/cos)
    ax1.plot(subset.index, subset["time_sin"], label="time_sin", linewidth=1.5)
    ax1.plot(subset.index, subset["time_cos"], label="time_cos", linewidth=1.5)
    ax1.set_ylabel("time_sin / time_cos")
    ax1.grid(True)

    # Zweite Achse für Target (0/1)
    ax2 = ax1.twinx()
    ax2.step(
        subset.index,
        subset[TARGET_COL],
        where="mid",
        color="red",
        alpha=0.5,
        label=TARGET_COL,
    )
    ax2.set_ylabel("Volatilitäts-Label (0/1)")

    # Zielzeile
    ax1.axvline(target_idx, color="black", linestyle="--", linewidth=1.5)

    # neue Handelstage (grün)
    day_changes = subset["timestamp"].dt.date.ne(
        subset["timestamp"].dt.date.shift()
    ).fillna(False)
    for i, is_new in enumerate(day_changes):
        if is_new and i != 0:
            ax1.axvline(i, color="green", linestyle="--", alpha=0.6)

    # X-Achse formatieren
    tick_positions = subset.index[:: max(1, len(subset) // 10)]
    tick_labels = subset.loc[tick_positions, "timestamp"].dt.strftime("%Y-%m-%d %H:%M")
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(tick_labels, rotation=45, ha="right")

    title_prefix = f"{symbol} " if symbol else ""
    ax1.set_title(
        f"{title_prefix}time_sin / time_cos + {TARGET_COL} "
        f"(±{window_before} Zeilen um {target_ts.date()})"
    )

    # Legenden kombinieren
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.tight_layout()
    plt.show()


def plot_single_feature_with_target(
        df: pd.DataFrame,
        feature: str,
        index: int,
        window_before: int = 200,
        window_after: int = 200,
        symbol: str | None = None,
):
    """
    Allgemeiner Plot: EIN Feature + vol_label_30m in einem Zeitfenster.
    """

    if feature not in df.columns:
        print(f"Feature '{feature}' nicht im DataFrame – übersprungen.")
        return

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df[feature] = pd.to_numeric(df[feature], errors="coerce")
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")

    start = max(0, index - window_before)
    end = min(len(df) - 1, index + window_after)
    subset = df.iloc[start: end + 1].copy().reset_index(drop=True)

    target_ts = df.loc[index, "timestamp"]
    target_idx = (subset["timestamp"] - target_ts).abs().idxmin()

    fig, ax1 = plt.subplots(figsize=(18, 6))

    ax1.plot(subset.index, subset[feature], label=feature, linewidth=1.5)
    ax1.set_ylabel(feature)
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.step(
        subset.index,
        subset[TARGET_COL],
        where="mid",
        color="red",
        alpha=0.5,
        label=TARGET_COL,
    )
    ax2.set_ylabel("Volatilitäts-Label (0/1)")

    ax1.axvline(target_idx, color="black", linestyle="--", linewidth=1.2)

    day_changes = subset["timestamp"].dt.date.ne(
        subset["timestamp"].dt.date.shift()
    ).fillna(False)
    for i, is_new in enumerate(day_changes):
        if is_new and i != 0:
            ax1.axvline(i, color="green", linestyle="--", alpha=0.6)

    tick_positions = subset.index[:: max(1, len(subset) // 10)]
    tick_labels = subset.loc[tick_positions, "timestamp"].dt.strftime("%Y-%m-%d %H:%M")
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(tick_labels, rotation=45, ha="right")

    title_prefix = f"{symbol} " if symbol else ""
    ax1.set_title(
        f"{title_prefix}{feature} + {TARGET_COL} "
        f"(±{window_before} Zeilen um {target_ts.date()})"
    )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.tight_layout()
    plt.show()


def main():
    # Konfiguration laden
    params = yaml.safe_load(open("../../conf/params.yaml", "r"))

    symbols_csv = params["DATA_ACQUISITION"]["SYMBOLS_CSV"]
    processed_path = params["FEATURE_ENGINEERING"]["PROCESSED_PATH"]

    ticker_list_df = pd.read_csv(symbols_csv)
    symbols = ticker_list_df["Symbol"].dropna().tolist()
    if not symbols:
        raise ValueError("No symbols found in symbol CSV.")

    symbol = symbols[0]

    train_file = os.path.join(processed_path, f"{symbol}_train.parquet")
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Train file not found: {train_file}")

    df = pd.read_parquet(train_file)

    target_index = min(2500, len(df) - 1)

    # 1) Zeit-Features + Target
    plot_time_features_with_target(
        df,
        index=target_index,
        window_before=500,
        window_after=500,
        symbol=symbol,
    )

    # 2) Weitere wichtige Features + Target (ein Plot pro Feature)
    for feat in IMPORTANT_FEATURES:
        if feat in ["time_sin", "time_cos"]:
            continue  # die haben wir oben schon gemeinsam geplottet
        plot_single_feature_with_target(
            df,
            feature=feat,
            index=target_index,
            window_before=500,
            window_after=500,
            symbol=symbol,
        )


if __name__ == "__main__":
    main()
