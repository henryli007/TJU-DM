"""
Task 3: Multi-Window Power Anomaly Detection

- Loops over multiple window size and step combinations
- Uses dataset.dataset_power to generate sliding windows
- Handles NaN: internal interpolation + bfill/ffill + delete remaining NaN windows
- Detects anomalies using One-Class SVM and Isolation Forest
- Saves visualization for each window combination
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
WINDOW_COMBINATIONS = [(7,1),(15,5),(30,10)]  # (sw_width, step)
ATTRIBUTE = 'power'
RESULT_DIR = "results/task3_multi"
os.makedirs(RESULT_DIR, exist_ok=True)

# -------------------------
# Loop over multiple window combinations
# -------------------------
for SW_WIDTH, STEP in WINDOW_COMBINATIONS:
    print(f"\nProcessing window={SW_WIDTH}, step={STEP}")

    # -------------------------
    # Generate sliding windows
    # -------------------------
    X = dataset_power(path='dataset/pv_filled.csv', sw_width=SW_WIDTH, step=STEP, attribute=ATTRIBUTE)
    print(f"Generated {X.shape[0]} windows, each with {X.shape[1]} features")

    # -------------------------
    # Window-internal interpolation + bfill/ffill
    # -------------------------
    X_filled = []
    for window in tqdm(X, desc="Filling window-internal NaN"):
        w = pd.Series(window).interpolate(method='linear', limit_direction='both').bfill().ffill()
        if not w.isna().any():
            X_filled.append(w.values)
    X_filled = np.array(X_filled)
    num_removed = X.shape[0] - X_filled.shape[0]
    print(f"Removed {num_removed} windows due to NaN, {X_filled.shape[0]} remain")

    if X_filled.shape[0] == 0:
        print("No valid windows remain, skip this combination")
        continue

    # -------------------------
    # Standardize
    # -------------------------
    X_scaled = StandardScaler().fit_transform(X_filled)

    # -------------------------
    # One-Class SVM
    # -------------------------
    svm = OneClassSVM(nu=0.05, kernel='rbf')
    labels_svm = svm.fit_predict(X_scaled)

    # -------------------------
    # Isolation Forest
    # -------------------------
    iso = IsolationForest(contamination=0.01, random_state=42)
    labels_if = iso.fit_predict(X_scaled)

    # -------------------------
    # Statistics
    # -------------------------
    print("One-Class SVM anomalies:", (labels_svm==-1).sum())
    print("Isolation Forest anomalies:", (labels_if==-1).sum())

    # -------------------------
    # Visualization
    # -------------------------
    plt.figure(figsize=(12,5))

    # One-Class SVM
    plt.subplot(1,2,1)
    colors_svm = np.array(['red' if l==-1 else 'blue' for l in labels_svm])
    plt.scatter(range(len(X_scaled)), X_scaled[:,0], c=colors_svm, s=10, alpha=0.6)
    plt.title(f"One-Class SVM (window={SW_WIDTH}, step={STEP})")
    plt.xlabel("Sample Index")
    plt.ylabel("Standardized Power (first element)")

    # Isolation Forest
    plt.subplot(1,2,2)
    colors_if = np.array(['red' if l==-1 else 'blue' for l in labels_if])
    plt.scatter(range(len(X_scaled)), X_scaled[:,0], c=colors_if, s=10, alpha=0.6)
    plt.title(f"Isolation Forest (window={SW_WIDTH}, step={STEP})")
    plt.xlabel("Sample Index")
    plt.ylabel("Standardized Power (first element)")

    plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}/power_window_SW{SW_WIDTH}_STEP{STEP}.png")
    plt.close()

    print(f"Saved figure: {RESULT_DIR}/power_window_SW{SW_WIDTH}_STEP{STEP}.png")

print("\nAll window combinations processed.")
