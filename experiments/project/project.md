# Vorhersage kurzfristiger Volatilität von S&P-500-Aktien

## Problem Definition

### Zielsetzung

In diesem Experiment soll vorhergesagt werden, ob eine **S&P-500-Aktie** in den nächsten
30 Minuten **ruhig bleibt** (Low Volatility) oder **stark schwankt** (High Volatility).

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

Dieser Schritt visualisiert Intraday-1-Minuten-Open-Preise einer S&P-500-Aktie 
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

*Es gibt sehr selten sehr hohe Volatilität.*

![03_intraday_volatility_pattern.png](images/03_intraday_volatility_pattern.png)

*Volatilität ist zum Tagesstart und Tagesende deutlich erhöht.*

![03_scatter_feature_vs_target.png](images/03_scatter_feature_vs_target.png)

*Höhere kurzfristige Volatilität geht meist mit höherer zukünftiger Volatilität einher.*

### Data after feature engineering
[`features_example.csv`](data/features_example.csv)

---

## Step 4 – Split Data

Pro Split (Train/Validation/Test) werden alle Dateien gesammelt. 
Daten werden gemischt. Danach werden sie in mehrere Shards gespeichert.

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

Für das Modelltraining wurde **Gradient Boosted Trees (LightGBM)** verwendet.

### Warum GBT?
- Sehr gut geeignet für tabellarische Daten
- Kann nicht-lineare Zusammenhänge lernen
- Stabil und schnell im Training
- Keine komplexe Sequenzstruktur nötig (im Gegensatz zu LSTM)

### Input (Features):
- Preis-Features (Returns, Volatilität)
- Volumen-Features
- VWAP-Abweichung
- Handelsbereich (High-Low)
- Zeit-Features (Minute im Handelstag)
- Aktien-Symbol als kategoriales Feature

### Output:
- Wahrscheinlichkeit p(t), dass in den nächsten 30 Minuten hohe Volatilität auftritt

### Ergebnisse
- Das Modell erreicht eine AUC von über 0.90. 
- Das zeigt, dass das Modell sehr gut zwischen ruhigen und volatilen Phasen unterscheiden kann.

![07_model_result_30m.png](images/07_model_result_30m.png)

### Baseline
- Als Baseline wurde ein konstantes Modell, das immer die durchschnittliche Volatilitätswahrscheinlichkeit ausgibt, verwendet.
- Das Modell übertrifft diese Baseline deutlich.

![07_baseline.png](images/07_baseline.png)

---

## Step 8 – Model Testing 

In diesem Schritt wird geprüft, wie gut das trainierte Modell in einer realistischen Handelssituation funktioniert.

Dazu wird eine Trading-Strategie aus den Modellvorhersagen abgeleitet und auf historischen Daten getestet.

### Ableitung der Handelsstrategie
- Das Modell gibt eine Wahrscheinlichkeit `p(t)` für hohe Volatilität aus.
- Strategie setzt Exposure (Investitionsgrad) als: `w(t) = 1 - p(t)`
  - Niedrige Volatilität → Exposure erhöhen
  - Hohe Volatilität → Exposure reduzieren
- Das gleichgewichtes Portfolio ist nicht immer voll investiert. Es passt sich dem vorhergesagten Risiko an.

### Prozess
- Out-of-Sample-Test auf dem Testzeitraum
- Ausführung mit 1-Minute Verzögerung (kein Look-Ahead)
- Transaktionskosten werden berücksichtigt
- Vergleich mit einem Buy-and-Hold Portfolio
  - bis zu 50 S&P-500-Aktien
  - immer voll investiert
  - keine Risikoanpassung

### Ergebnisse
#### Equity Curve
![equity_curve.png](results/backtest/equity_curve.png)
- Die Strategie erzielt eine geringere Gesamtrendite als Buy-and-Hold.
- In starken Marktphasen sind die Verluste jedoch deutlich geringer.
- Die Strategie reduziert Risiko, verzichtet aber auf Rendite.

#### Performance-Tabelle
![08_performance_tabelle.png](results/backtest/08_performance_tabelle.png)
- Besonders der geringere Drawdown zeigt,
dass die Strategie in Stressphasen Kapital schützt.

#### Einzelne Aktie an einem Tag
![example_AAPL_2025-01-02_price.png](results/backtest/example_AAPL_2025-01-02_price.png)
![example_AAPL_2025-01-02_prob.png](results/backtest/example_AAPL_2025-01-02_prob.png)
![example_AAPL_2025-01-02_exposure.png](results/backtest/example_AAPL_2025-01-02_exposure.png)
- Das Modell erkennt erhöhte Volatilität.
- Die Strategie reduziert daraufhin die Investitionsgröße.

#### Verteilung der Trading-Aktivität
![trading_points_by_hour.png](results/backtest/trading_points_by_hour.png)

*Der Plot zeigt, zu welchen Uhrzeiten die Strategie handelt.*

![trading_points_per_day.png](results/backtest/trading_points_per_day.png)

*Der zweite Plot zeigt die Anzahl der Trades pro Tag.*

### Fazit
- Die Handelsstrategie reduziert Risiko in turbulenten Phasen und vermeidet große Drawdowns.
- Die Rendite liegt oft unter Buy-and-Hold, dafür stabilerer Verlauf und geringeres Risiko.

---

## Step 9 – Deployment

Dieses Deployment führt das trainierte 30-Minuten-Volatilitätsmodell in einem Paper-Trading Setup aus.

### Überblick
- Input: Live 1-Minuten-Bars (nur Informationen bis Zeitpunkt τ)
- Output: Modellscore p(τ) + Ziel-Exposure w(τ) + Orders/Fills + Logs + Report
  - Hohe Volatilität → Exposure reduzieren
  - Ruhige Phase → Exposure erhöhen

### Paper Trading Setup
- Umgebung: Alpaca Paper Trading (Orders gehen in Paper, nicht live).
- Gehandelte Symbole: S&P-500 Subset
- Handelszeiten: nur Regular Trading Hours

### Pipeline Ablauf
1. Bars laden (1-Minuten-Daten bis τ)
2. Features berechnen
3. Modell laden & scoren → p(τ) pro Symbol
4. Exposure ableiten → w(τ) = 1 − p(τ)
5. Paper Orders platzieren (falls Rebalance aktiv)
6. Logging (Scores, Exposure, Orders/Fills, Positionen, Equity)
7. Report erstellen (report.py)

### Performance Auswertung
`report.py` erstellt eine detaillierte Analyse der Paper-Runs.
