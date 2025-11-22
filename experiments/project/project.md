# Vorhersage kurzfristiger Volatilität von S&P-500-Aktien mittels LSTM

### Problem Definition:
**Zielsetzung**

In diesem Experiment soll vorhergesagt werden, ob eine S&P-500-Aktie in den nächsten
t = [5, 10, 15, 30, 60] Minuten ruhig bleibt (Low Volatility) oder stark schwankt (High Volatility).

Für jede Aktie und jede Minute im Zeitraum 01.01.2020 bis 25.06.2025 wird bestimmt, 
ob gleich eine ungewöhnlich starke Preisbewegung kommt?

**Target**

Um herauszufinden, ob die Aktie bald stark schwankt, wird eine Kennzahl berechnet, 
die beschreibt, wie heftig sich der Preis in den nächsten Minuten bewegt.

1. Log-Renditen
- rₖ = ln(Pₖ / Pₖ₋₁)
- Das misst, wie stark sich ein Preis von einer Minute zur nächsten verändert.

2. Realisierte Volatilität 
- RV(τ, t) = √( Σ rₖ² )  für k = τ+1 bis τ+t
- Alle Preisänderungen der nächsten t Minuten werden zu einer einzigen Zahl zusammengefasst.
- Je größer dieses Ergebnis, desto stärker schwankt die Aktie.
3. Normalisierung: 
- RV_norm(τ, t) = RV(τ, t) / durchschnittliche Tagesvolatilität
- Manche Aktien sind immer sehr volatil, manche sind immer ruhig.
Durch Division durch die eigene Tagesvolatilität wird alles fair.
4. Label-Definition:
- Die Werte werden dann soriert. Das Modell soll jetzt 0 oder 1 vorhersagen.
- Top 30 % der normalisierten Werte → High Volatility → y(τ, t) = 1
- Untere 70 % → Low Volatility → y(τ, t) = 0

**Input Features**

Das Modell sieht immer nur Informationen, die vor Zeitpunkt τ vorhanden sind.
Als Input wird eine 30-Minuten-Sequenz genutzt (z. B. die letzten 30 Minuten der Aktie).

1. Preisbezogene Features
- Normalisierter Schlusskurs
- 1-Minuten-Log-Return
- Rolling-Return der letzten 5 Minuten
- Rolling-Volatilität der letzten 15 Minuten
- Sie helfen dem Modell zu erkennen, ob der Markt gerade ruhig oder aktiv ist.

2. VWAP & Abweichung
- Intraday-VWAP bis zum Zeitpunkt τ
- Relative Abweichung vom VWAP
- So kann das Modell erkennen, ob der aktuelle Preis ungewöhnlich weit vom typischen Tagesniveau abweicht.

3. Volumen & Liquidität
- Normalisiertes Handelsvolumen der aktuellen Minute
- Durchschnittliches Volumen der letzten 15 Minuten
- Ein sprunghaft ansteigendes Volumen kann ein Hinweis auf kommende Volatilität sein.

4. Handelsbereich
- Normalisierter High-Low-Spread der aktuellen Minute
- Rolling-Range (Differenz zwischen Hoch und Tief) der letzten 15 Minuten
- Je größer diese Werte, desto unruhiger war der Markt bereits.

5. Zeitliche Merkmale
- Minute des Handelstages kodiert als Sinus/Cosinus
- Dummy-Variable: erste 30 Minuten nach Markteröffnung
- Dummy-Variable: letzte 30 Minuten vor Handelsschluss
- So kann das Modell lernen, dass bestimmte Zeiten generell volatiler sind als andere.

---

## Data Acquisition
- Die Daten werden über die yfinance-API abgerufen.
- Für jede Aktie wird eine Datei {TICKER}.parquet gespeichert.

**Script**

- (script)
- Dieses Skript lädt die Daten per yfinance.download(), berechnet VWAP und speichert sie im .parquet-Format ab.
- (image)

---
