"""
Task 4: Multi-Window Anomaly Detection on Power + Generation

- Loops over multiple window size and step combinations
- Handles missing values: 30-day same-time mean + window-internal interpolation + delete remaining NaN
- Detects anomalies using One-Class SVM and Isolation Forest
- Saves figures to results/task4_multi/
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from dataset.dataset import dataset_power

# -------------------------
# Parameters
# -------------------------
WINDOW_COMBINATIONS = [(7,1),(15,5),(30,10)]
RESULT_DIR = "results/task4_multi"
os.makedirs(RESULT_DIR, exist_ok=True)

# -------------------------
# Load original data
# -------------------------
df = pd.read_csv('dataset/pv.csv', parse_dates=['time'])
df.set_index('time', inplace=True)

# -------------------------
# Fill missing values using 30-day same-time mean
# -------------------------
def fill_30day(series, days=15):
    filled = series.copy()
    na_idx = series[series.isna()].index
    for idx in tqdm(na_idx, desc=f"30-day mean filling ({series.name})"):
        start = idx - pd.Timedelta(days=days)
        end = idx + pd.Timedelta(days=days)
        window_series = series[start:end]
        window_values = window_series[window_series.index.time == idx.time()].dropna()
        if len(window_values) > 0:
            filled.loc[idx] = window_values.mean()
    return filled

df['power_filled'] = fill_30day(df['power'], 15)
df['generation_filled'] = fill_30day(df['generation'], 15)
df.to_csv('dataset/pv_filled.csv', index=True)

# -------------------------
# Window-internal NaN handling
# -------------------------
def clean_windows(X):
    X_filled = []
    for window in tqdm(X, desc="Filling window-internal NaN"):
        w = pd.Series(window).interpolate(method='linear', limit_direction='both').bfill().ffill()
        if not w.isna().any():
            X_filled.append(w.values)
    return np.array(X_filled)

# -------------------------
# Loop over window combinations
# -------------------------
for SW_WIDTH, STEP in WINDOW_COMBINATIONS:
    print(f"\nProcessing window={SW_WIDTH}, step={STEP}")

    # Generate sliding windows
    Xp = dataset_power('dataset/pv_filled.csv', SW_WIDTH, STEP, 'power_filled')
    Xg = dataset_power('dataset/pv_filled.csv', SW_WIDTH, STEP, 'generation_filled')

    # Clean NaN within windows
    Xp_filled = clean_windows(Xp)
    Xg_filled = clean_windows(Xg)

    # Keep same number of samples
    min_len = min(len(Xp_filled), len(Xg_filled))
    X = np.hstack([Xp_filled[:min_len], Xg_filled[:min_len]])
    print(f"Final window shape (power+generation): {X.shape}")

    # Standardize
    X_scaled = StandardScaler().fit_transform(X)

    # One-Class SVM
    svm = OneClassSVM(nu=0.05, kernel='rbf')
    labels_svm = svm.fit_predict(X_scaled)

    # Isolation Forest
    iso = IsolationForest(contamination=0.01, random_state=42)
    labels_if = iso.fit_predict(X_scaled)

    # Statistics
    print("One-Class SVM anomalies:", (labels_svm==-1).sum())
    print("Isolation Forest anomalies:", (labels_if==-1).sum())

    # Visualization
    plt.figure(figsize=(14,6))
    # One-Class SVM
    plt.subplot(1,2,1)
    colors_svm = np.array(['red' if l==-1 else 'blue' for l in labels_svm])
    plt.scatter(range(len(X_scaled)), X_scaled[:,0], c=colors_svm, s=10, alpha=0.6)
    plt.title(f"One-Class SVM (SW={SW_WIDTH}, STEP={STEP})")
    plt.xlabel("Sample Index")
    plt.ylabel("Standardized Power (first element)")

    # Isolation Forest
    plt.subplot(1,2,2)
    colors_if = np.array(['red' if l==-1 else 'blue' for l in labels_if])
    plt.scatter(range(len(X_scaled)), X_scaled[:,0], c=colors_if, s=10, alpha=0.6)
    plt.title(f"Isolation Forest (SW={SW_WIDTH}, STEP={STEP})")
    plt.xlabel("Sample Index")
    plt.ylabel("Standardized Power (first element)")

    plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}/power_generation_SW{SW_WIDTH}_STEP{STEP}.png")
    plt.close()
    print(f"Saved figure: {RESULT_DIR}/power_generation_SW{SW_WIDTH}_STEP{STEP}.png")

print("\nAll window combinations processed for Task 4.")
