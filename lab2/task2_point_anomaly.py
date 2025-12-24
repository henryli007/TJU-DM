"""
Task 2: Single-time-point Anomaly Detection (Memory-Friendly)

Each time point is a sample with two attributes: power and generation.
Anomaly detection is performed using:
1) One-Class SVM
2) Isolation Forest

Figures are saved to results/task2/
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest

# -------------------------
# 1. Create output directory
# -------------------------
RESULT_DIR = "results/task2"
os.makedirs(RESULT_DIR, exist_ok=True)

# -------------------------
# 2. Load and preprocess
# -------------------------
df = pd.read_csv("dataset/pv.csv")
df = df[["power", "generation"]].dropna()
X = StandardScaler().fit_transform(df.values)

# -------------------------
# 3. One-Class SVM
# -------------------------
svm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
labels_svm = svm.fit_predict(X)

# -------------------------
# 4. Isolation Forest
# -------------------------
iso = IsolationForest(contamination=0.01, random_state=42)
labels_if = iso.fit_predict(X)

# -------------------------
# 5. Visualization
# -------------------------
plt.figure(figsize=(14,6))

# One-Class SVM
plt.subplot(1,2,1)
colors_svm = np.array(['red' if l==-1 else 'blue' for l in labels_svm])
plt.scatter(X[:,0], X[:,1], c=colors_svm, s=10, alpha=0.6)
plt.title("One-Class SVM Anomaly Detection")
plt.xlabel("Power (standardized)")
plt.ylabel("Generation (standardized)")

# Isolation Forest
plt.subplot(1,2,2)
colors_if = np.array(['red' if l==-1 else 'blue' for l in labels_if])
plt.scatter(X[:,0], X[:,1], c=colors_if, s=10, alpha=0.6)
plt.title("Isolation Forest Anomaly Detection")
plt.xlabel("Power (standardized)")
plt.ylabel("Generation (standardized)")

plt.tight_layout()
plt.savefig(f"{RESULT_DIR}/point_anomaly_comparison.png")
plt.close()

# -------------------------
# 6. Print anomaly counts
# -------------------------
print("One-Class SVM anomalies:", (labels_svm==-1).sum())
print("Isolation Forest anomalies:", (labels_if==-1).sum())
print(f"Task 2 completed. Figure saved to {RESULT_DIR}/point_anomaly_comparison.png")
