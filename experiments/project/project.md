# Vorhersage kurzfristiger Volatilität von S&P-500-Aktien

## Problem Definition

### Zielsetzung

In diesem Experiment soll vorhergesagt werden, ob eine **S&P-500-Aktie** in den nächsten
t = [5, 10, 15, 30, 60] Minuten **ruhig bleibt** (Low Volatility) oder **stark schwankt** (High Volatility).

Für jede Aktie und jede Minute im Zeitraum **01.01.2020 bis 25.06.2025** wird untersucht, 
ob in den nächsten Minuten eine ungewöhnlich starke Preisbewegung kommt.

### Target

Um die künftige Schwankungsintensität zu bestimmen, wird eine Kennzahl berechnet.

**1. Log-Renditen**
- rₖ = ln(Pₖ / Pₖ₋₁)
- Das misst die Preisänderung von einer Minute zur nächsten.

**2. Realisierte Volatilität**
- RV(τ, t) = √( Σ rₖ² )  für k = τ+1 bis τ+t
- Alle Preisänderungen in einem t-Minuten-Fenster werden zusammengefasst. Je größer dieses Ergebnis, desto stärker schwankt die Aktie.

**3. Normalisierung**
- RV_norm(τ, t) = RV(τ, t) / durchschnittliche Tagesvolatilität
- Manche Aktien sind immer sehr volatil, manche sind immer ruhig.
Durch Division durch die eigene Tagesvolatilität wird alles fair.

**4. Label-Definition**
- Die Werte werden dann sortiert. Das Modell soll jetzt 0 oder 1 vorhersagen.
- Top 30 % → High Volatility → y(τ, t) = 1
- Untere 70 % → Low Volatility → y(τ, t) = 0

### Input Features

Das Modell sieht nur Informationen, die **vor Zeitpunkt τ** verfügbar sind (keine Zukunftsinfos).

**1. Preisbezogene Features**
- 1-Minuten-Log-Return
- Rolling Return (5 Minuten)
- Rolling Volatilität (15 Minuten)

**2. VWAP & Abweichung**
- Intraday-VWAP
- Relative Abweichung vom VWAP
- VWAP-Z-Score (30 Minuten)

**3. Volumen & Liquidität**
- Volume-Z-Score (30 Minuten)
- Rolling Volume (15 Minuten)

**4. Handelsbereich**
- High-Low-Spread
- Rolling-Range (15 Minuten)

**5. Zeitliche Merkmale**
- Minute des Handelstages (sin/cos)
- Dummy: erste 30 Minuten nach Markteröffnung
- Dummy: letzte 30 Minuten vor Handelsschluss

---

## Step 1 – Data Acquisition
- Historische 1-Minuten-Daten werden über die **Alpaca Market Data API** abgerufen.
- Für jede Aktie wird eine Datei {TICKER}.parquet gespeichert.

### Script
[`bar_retriever.py`](scripts/01_data_acquisition/bar_retriever.py)

![01_AAPL_bar_data.png.png](images/01_AAPL_bar_data.png.png)

---

## Step 2 – Data Understanding

Dieser Schritt visualisiert Intraday-1-Minuten-Open-Preise einzelner S&P-500-Aktien 
und untersucht ihr Verhalten um einen festen Zeitindex herum.

### Script
[`plotter.py`](scripts/02_data_understanding/plotter.py)

### Plots

<img src="images/02_AAPL_open.png" alt="AAPL open window" width="800"/>
<img src="images/02_NVDA_open.png" alt="NVDA open window" width="800"/>

*Die Volatilität ist eher hoch – besonders an den Tagesöffnungen.*

---

## Step 3 – Pre-Split Preparation

- **Targets berechnen**
  - Realisierte zukünftige Volatilität (mehrere Horizonte), Normalisierung und binäre Labels.

- **Features erzeugen**
  - Preis-, Volumen-, VWAP-, Range- und Zeit-Features.

- **Daten splitten**
  - Train / Validation / Test nach Datum (keine Zufallssplits).

### Main Script
[`main.py`](scripts/03_pre_split_prep/main.py)

### Feature Engineering Script
[`features.py`](scripts/03_pre_split_prep/features.py)

### Target Computation Script
[`targets.py`](scripts/03_pre_split_prep/targets.py)

### Plotting Script
[`plot_features.py`](scripts/03_pre_split_prep/plot_features.py)

### Plots

![03_feature_zeitreihe.png](images/03_feature_zeitreihe.png)
*Wie stark sich der Preis innerhalb von jeweils 15 Minuten verändert hat.*

![03_histogramm_des_targets.png](images/03_histogramm_des_targets.png)
*Es gibt nur sehr selten sehr hohe Volatilität.*

![03_intraday_volatility_pattern.png](images/03_intraday_volatility_pattern.png)
*Volatilität ist zum Tagesstart und Tagesende deutlich erhöht.*

![03_scatter_feature_vs_target.png](images/03_scatter_feature_vs_target.png)
*Höhere kurzfristige Volatilität geht meist mit höherer zukünftiger Volatilität einher.*

### Data after feature engineering
[`features_example.csv`](data/features_example.csv)

---

## Step 4 – Split Data

Pro Split (Train/Validation/Test) werden alle {symbol}_{split}.parquet Dateien gesammelt. 
Daten werden gemischt (shuffle), damit Training effizienter ist. 
Danach werden sie in mehrere Shards gespeichert, um RAM-schonend trainieren zu können.

### Script
[`shuffle.py`](scripts/04_split_data/shuffle.py)

---

## Step 5 – Post-Split Preparation

Nach Sharding sind keine weiteren Schritte notwendig.

---

## Step 6 – Feature Selection

Der Random Forest erkennt **nichtlineare Zusammenhänge** zwischen Features
und dem Volatilitäts-Label und liefert ein klares Ranking der Feature-Wichtigkeiten.

### Script
[`main.py`](scripts/06_feature_selection/main.py)

### Feature Importance

![06_random_forest.png](images/06_random_forest.png)
*Zeitbasierte Muster und kurzfristige Volatilitäts-Features sind am wichtigsten.*

---

## Step 7 – Model Training

Ein **Gradient Boosted Trees** Modell (LightGBM) wird trainiert:
- Input: Feature-Spalten aus `features.txt` (+ symbol als kategoriales Feature)
- Target: `vol_label_30m`
- Training auf Shards (RAM-schonend)
- Modell-Output pro Minute: **p(t) = Wahrscheinlichkeit für High Volatility**

### Script
- Training: [`01_gbt_train.py`](scripts/07_model_training/01_gbt_train.py)
- Metriken: [`metrics.py`](scripts/07_model_training/metrics.py)
- Baseline: [`baseline.py`](scripts/07_model_training/baseline.py)

---

## Step 8 – Model Testing 
Da das Modell keine Richtung (up/down), sondern Volatilitätsrisiko vorhersagt, 
wird daraus eine Risiko-/Exposure-Strategie abgeleitet:
- Modell liefert p(t) = P(High Vol)
- Strategie setzt Exposure (Investitionsgrad) als: w(t) = 1 - p(t) (clipped auf [0,1])
- Ausführung mit 1-Minuten Delay (kein Lookahead)
- Transaktionskosten über Turnover (|Δw|)

Auswertung:
- Equity Curve (Strategie vs Buy&Hold Benchmark)
- Sharpe, Max Drawdown, Turnover
- Beispielplots (Preis, p(t), w(t))
- Verteilung der “Trading Points” über Zeit (pro Tag / Stunde)

### Output 
...

---

## Step 9 – Deployment

### Deployment (Inference)

- Das trainierte Modell wird geladen und kann auf 
neue, bereits feature-engineerte Minuten-Daten angewendet werden.
- Output ist eine CSV mit p(t) pro Minute.
- Script: scripts/09_deployment/main.py

### Paper Trading (Simulation)

Zusätzlich wird eine Paper-Trading-Simulation 
auf historischen Testdaten durchgeführt (live-like):
- chronologisch Minute für Minute
- gleiche Trading-Regel wie Backtest (w(t)=1-p(t), Delay, Costs)

Reporting:
- Gesamtperformance (Zeitrahmen, Equity, Sharpe, DD)
- pro Aktie
- pro Woche (Time Frame)
- Vergleich Paper vs Backtest

Script: scripts/09_deployment/paper_trading.py
