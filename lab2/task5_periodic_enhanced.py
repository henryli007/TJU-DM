"""
Task 5: Power Anomaly Detection Considering Short-Term Trend + Periodicity

- Uses dataset_power() to generate sliding windows
- Sliding window + step (local trend)
- Adds periodicity feature: mean of previous LOOKBACK_DAYS same-time points
- Handles missing values:
    1) Forward/backward 15-day same-time mean
    2) Window-internal linear interpolation + bfill/ffill
    3) Deletes windows that still contain NaN
    4) Periodicity feature NaN replaced with 0
- Detects anomalies using One-Class SVM
- Visualizes and saves results to results/task5/
- Supports multiple window/step combinations for analysis
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from dataset.dataset import dataset_power

# -------------------------
# Parameters
# -------------------------
WINDOWS = [(7,1),(15,5),(30,10)]  # (window size, step)
LOOKBACK_DAYS = 15
RESULT_DIR = "results/task5"
os.makedirs(RESULT_DIR, exist_ok=True)

# -------------------------
# Load original data
# -------------------------
df = pd.read_csv("dataset/pv.csv", parse_dates=["time"])
df.set_index("time", inplace=True)

# -------------------------
# Fill missing values: forward/backward 15-day same-time mean
# -------------------------
def fill_30day(series, days=LOOKBACK_DAYS):
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

df['power_filled'] = fill_30day(df['power'], LOOKBACK_DAYS)
df.to_csv("dataset/pv_filled.csv")  # 保存填充后的列

# -------------------------
# Process each window configuration
# -------------------------
for sw_width, step in WINDOWS:
    print(f"\nProcessing window={sw_width}, step={step}")
    
    # Generate sliding windows using dataset_power()
    X_win = dataset_power("dataset/pv_filled.csv", sw_width, step, attribute="power_filled")

    # Window-internal interpolation + bfill/ffill
    X_filled = []
    for window in tqdm(X_win, desc="Filling window-internal NaN"):
        w = pd.Series(window).interpolate(method='linear', limit_direction='both').bfill().ffill()
        if not w.isna().any():
            X_filled.append(w.values)
    X_filled = np.array(X_filled)
    if len(X_filled) == 0:
        print("All windows removed due to NaN. Skipping this configuration.")
        continue

    # -------------------------
    # Add periodicity feature: mean of previous LOOKBACK_DAYS same-time points
    # -------------------------
    periodic_feature = []
    n_samples = X_filled.shape[0]
    for i in range(n_samples):
        # end index in original series
        end_idx = i*step + sw_width -1
        t = df.index[end_idx]
        past_values = []
        for d in range(1, LOOKBACK_DAYS+1):
            prev_time = t - pd.Timedelta(days=d)
            if prev_time in df.index:
                past_values.append(df.loc[prev_time, "power_filled"])
        if len(past_values) > 0:
            periodic_feature.append(np.mean(past_values))
        else:
            periodic_feature.append(0.0)  # NaN replaced with 0

    # Align samples
    min_len = min(len(X_filled), len(periodic_feature))
    X_final = np.hstack([X_filled[:min_len], np.array(periodic_feature[:min_len]).reshape(-1,1)])

    # Final NaN check
    X_final = X_final[~np.isnan(X_final).any(axis=1)]
    if len(X_final) == 0:
        print("All samples removed due to NaN after adding periodicity. Skipping this configuration.")
        continue

    # Standardize
    X_scaled = StandardScaler().fit_transform(X_final)

    # One-Class SVM
    svm = OneClassSVM(nu=0.05, kernel="rbf")
    labels = svm.fit_predict(X_scaled)

    # Statistics
    print(f"Window {sw_width}, Step {step}: Total samples={len(X_scaled)}, Anomalies={np.sum(labels==-1)}")

    # Visualization
    plt.figure(figsize=(12,5))
    colors = np.array(['red' if l==-1 else 'blue' for l in labels])
    plt.scatter(range(len(X_scaled)), X_scaled[:,0], c=colors, s=10, alpha=0.6)
    plt.title(f"Task5: Short-Term Trend + {LOOKBACK_DAYS}-Day Periodicity (SW={sw_width}, STEP={step})")
    plt.xlabel("Sample Index")
    plt.ylabel("Standardized Power (first element of window)")
    plt.tight_layout()
    out_path = f"{RESULT_DIR}/power_shortterm_periodicity_SW{sw_width}_STEP{step}.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Figure saved to {out_path}")

print("\nTask 5 completed for all window/step configurations.")
