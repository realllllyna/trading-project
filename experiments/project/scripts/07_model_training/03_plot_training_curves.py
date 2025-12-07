"""
Simple training-curve plotter.

Assumes you log train_loss & val_auc into CSV from main script.
"""

import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("training_log.csv")

plt.figure(figsize=(12,6))
plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
plt.plot(df["epoch"], df["val_auc"], label="Val AUC")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.title("Training Curves")
plt.legend()
plt.grid()
plt.show()
