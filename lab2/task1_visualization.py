"""
Task 1: Statistical Analysis and Visualization of PV Data

Visualizations include:
1) Time series of power and generation;
2) Scatter plot of power vs generation;
3) Boxplots for outlier inspection;
4) Histograms of power and generation (numeric scale);
5) 2×2 data availability matrix (generation complete);
6) Time-scale analysis of POWER missing values only.

All figures are saved to results/task1/.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ===============================
# Global settings
# ===============================
sns.set_theme(style="whitegrid", font_scale=1.15)
plt.rcParams["figure.dpi"] = 120

RESULT_DIR = "results/task1"
os.makedirs(RESULT_DIR, exist_ok=True)

# ===============================
# Load data
# ===============================
df = pd.read_csv("dataset/pv.csv", parse_dates=["time"])
N = len(df)

# ===============================
# 1. Time series
# ===============================
plt.figure(figsize=(16, 4))
plt.plot(df["time"], df["power"], linewidth=0.6)
plt.title("Power Time Series")
plt.xlabel("Time")
plt.ylabel("Power")
plt.tight_layout()
plt.savefig(f"{RESULT_DIR}/power_time_series.png")
plt.close()

plt.figure(figsize=(16, 4))
plt.plot(df["time"], df["generation"], linewidth=0.6, color="orange")
plt.title("Generation Time Series")
plt.xlabel("Time")
plt.ylabel("Generation")
plt.tight_layout()
plt.savefig(f"{RESULT_DIR}/generation_time_series.png")
plt.close()

# ===============================
# 2. Scatter
# ===============================
plt.figure(figsize=(6.5, 5.5))
plt.scatter(df["power"], df["generation"], s=6, alpha=0.4)
plt.xlabel("Power")
plt.ylabel("Generation")
plt.title("Power vs Generation")
plt.tight_layout()
plt.savefig(f"{RESULT_DIR}/power_generation_scatter.png")
plt.close()

# ===============================
# 3. Boxplots
# ===============================
plt.figure(figsize=(5, 4.5))
sns.boxplot(y=df["power"])
plt.title("Boxplot of Power")
plt.tight_layout()
plt.savefig(f"{RESULT_DIR}/power_boxplot.png")
plt.close()

plt.figure(figsize=(5, 4.5))
sns.boxplot(y=df["generation"], color="orange")
plt.title("Boxplot of Generation")
plt.tight_layout()
plt.savefig(f"{RESULT_DIR}/generation_boxplot.png")
plt.close()

# ===============================
# 4. Histograms
# ===============================

# ---- Power: focus on 0–60 ----
plt.figure(figsize=(7, 5))
bins_power = np.linspace(0, 60, 61)
plt.hist(df["power"].dropna(), bins=bins_power, alpha=0.85)
plt.xlabel("Power (0–60)")
plt.ylabel("Number of Samples")
plt.title("Histogram of Power")
plt.tight_layout()
plt.savefig(f"{RESULT_DIR}/power_histogram.png")
plt.close()

# ---- Generation: numeric bins, 99% range ----
plt.figure(figsize=(7, 5))
gen = df["generation"]
upper = np.percentile(gen, 99)
bins_gen = np.linspace(0, upper, 50)
plt.hist(gen, bins=bins_gen, alpha=0.85, color="orange")
plt.xlabel("Generation")
plt.ylabel("Number of Samples")
plt.title("Histogram of Generation (99% Range)")
plt.tight_layout()
plt.savefig(f"{RESULT_DIR}/generation_histogram.png")
plt.close()

# ===============================
# 5. Data availability matrix
# ===============================
power_ok = df["power"].notna()
gen_ok = df["generation"].notna()   # 全 True

matrix = np.array([
    [(power_ok & gen_ok).sum(), (power_ok & ~gen_ok).sum()],
    [(~power_ok & gen_ok).sum(), (~power_ok & ~gen_ok).sum()]
])
ratio = matrix / N * 100

plt.figure(figsize=(6, 5))
ax = sns.heatmap(
    matrix,
    cmap="Blues",
    cbar=True,
    xticklabels=["Generation Present", "Generation Missing"],
    yticklabels=["Power Present", "Power Missing"]
)

for i in range(2):
    for j in range(2):
        ax.text(
            j + 0.5, i + 0.5,
            f"{matrix[i, j]}\n({ratio[i, j]:.2f}%)",
            ha="center", va="center", fontsize=11
        )

plt.title("Data Availability Matrix")
plt.tight_layout()
plt.savefig(f"{RESULT_DIR}/power_generation_presence_matrix.png")
plt.close()

# ===============================
# 6. POWER missing timeline (time-scale)
# ===============================

power_missing = df["power"].isna().astype(int)

plt.figure(figsize=(16, 2.8))
plt.imshow(
    power_missing.values.reshape(1, -1),
    aspect="auto",
    cmap="Reds",
    interpolation="nearest"
)
plt.yticks([0], ["Power Missing"])
plt.xlabel("Time Index")
plt.title("Power Missing Timeline (Red = Missing)")
plt.colorbar(label="Missing Indicator")
plt.tight_layout()
plt.savefig(f"{RESULT_DIR}/power_missing_timeline.png")
plt.close()

# ===============================
# 7. Daily POWER missing ratio
# ===============================

daily = df.copy()
daily["date"] = daily["time"].dt.date

daily_missing = daily.groupby("date")["power"].apply(
    lambda x: x.isna().mean()
)

plt.figure(figsize=(14, 4))
plt.plot(daily_missing.index, daily_missing.values)
plt.xlabel("Date")
plt.ylabel("Missing Ratio")
plt.title("Daily Missing Ratio of Power")
plt.tight_layout()
plt.savefig(f"{RESULT_DIR}/power_daily_missing_ratio.png")
plt.close()

print("Task 1 completed. All figures saved to results/task1/")

# ===============================
# 8. Short-term time series (GENERATION)
# ===============================

def plot_generation_timeseries(df_slice, title, filename):
    plt.figure(figsize=(16, 4))
    plt.plot(
        df_slice["time"],
        df_slice["generation"],
        linewidth=0.7,
        color="orange"
    )
    plt.xlabel("Time")
    plt.ylabel("Generation")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}/{filename}")
    plt.close()


# ---- 2022 H2 ----
df_gen_22_h2 = df[
    (df["time"] >= "2022-07-01") &
    (df["time"] <  "2023-01-01")
]

plot_generation_timeseries(
    df_gen_22_h2,
    "Generation Time Series (2022 H2)",
    "generation_timeseries_2022_H2.png"
)

# ---- 2023 H1 ----
df_gen_23_h1 = df[
    (df["time"] >= "2023-01-01") &
    (df["time"] <  "2023-07-01")
]

plot_generation_timeseries(
    df_gen_23_h1,
    "Generation Time Series (2023 H1)",
    "generation_timeseries_2023_H1.png"
)

# ===============================
# 9. Short-term time series (POWER)
# ===============================

def plot_power_timeseries(df_slice, title, filename):
    plt.figure(figsize=(16, 4))
    plt.plot(
        df_slice["time"],
        df_slice["power"],
        linewidth=0.7
    )
    plt.xlabel("Time")
    plt.ylabel("Power")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}/{filename}")
    plt.close()


# ---- 2021 full year ----
df_power_2021 = df[
    (df["time"] >= "2021-01-01") &
    (df["time"] <  "2022-01-01")
]

plot_power_timeseries(
    df_power_2021,
    "Power Time Series (Year 2021)",
    "power_timeseries_2021_full.png"
)

# ---- 2022 H1 ----
df_power_22_h1 = df[
    (df["time"] >= "2022-01-01") &
    (df["time"] <  "2022-07-01")
]

plot_power_timeseries(
    df_power_22_h1,
    "Power Time Series (2022 H1)",
    "power_timeseries_2022_H1.png"
)

# ---- 2023 July ----
df_power_23_july = df[
    (df["time"] >= "2023-07-01") &
    (df["time"] <  "2023-08-01")
]

plot_power_timeseries(
    df_power_23_july,
    "Power Time Series (July 2023)",
    "power_timeseries_2023_July.png"
)

# ===============================
# 10. Daily-scale time series (POWER)
# ===============================

df_power_23_0715 = df[
    (df["time"] >= "2023-07-15 00:00:00") &
    (df["time"] <  "2023-07-16 00:00:00")
]

plt.figure(figsize=(14, 4))
plt.plot(
    df_power_23_0715["time"],
    df_power_23_0715["power"],
    marker="o",
    markersize=3,
    linewidth=1.0
)
plt.xlabel("Time (15-min resolution)")
plt.ylabel("Power")
plt.title("Power Time Series (2023-07-15)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(f"{RESULT_DIR}/power_timeseries_2023_07_15.png")
plt.close()

print("Daily-scale power visualization (2023-07-15) saved.")
