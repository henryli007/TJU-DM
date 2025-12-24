# =========================
# step4_outlier_processing.py
# Step 4: 异常值处理（智能跨数量级检测）
# =========================

import pandas as pd
import numpy as np
from scipy import stats
from scipy.cluster.hierarchy import fcluster, linkage
import matplotlib.pyplot as plt
import os
import json

# =========================
# 配置中文显示
# =========================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# =========================
# 读取 Step3 处理后的 CSV
# =========================
file_path = "dataset/covid_after_step3.csv"
df = pd.read_csv(file_path, encoding="utf-8-sig")

# =========================
# 智能数量级检测函数
# =========================
def intelligent_scale_detection(series, min_cluster_size=0.01, plot_debug=False):
    positive_series = series[series > 0].copy()
    if len(positive_series) < 10 or positive_series.nunique() == 1:
        return pd.Series(True, index=series.index)

    log_data = np.log10(positive_series)

    try:
        kde = stats.gaussian_kde(log_data, bw_method=0.2)
    except np.linalg.LinAlgError:
        return pd.Series(True, index=series.index)

    x_grid = np.linspace(log_data.min() - 1, log_data.max() + 1, 1000)
    kde_vals = kde(x_grid)

    peaks = [x_grid[i] for i in range(1, len(x_grid)-1)
             if kde_vals[i] > kde_vals[i-1] and kde_vals[i] > kde_vals[i+1]]

    if len(peaks) <= 1:
        return pd.Series(True, index=series.index)

    peaks_array = np.array(peaks).reshape(-1, 1)
    Z = linkage(peaks_array, method='ward')
    cluster_labels = fcluster(Z, t=0.5, criterion='distance')

    cluster_centers = [float(np.mean(peaks_array[cluster_labels==c])) for c in np.unique(cluster_labels)]

    cluster_boundaries = []
    for i, center in enumerate(sorted(cluster_centers)):
        lower = center - 0.5
        upper = center + 0.5
        if i > 0:
            lower = max(lower, cluster_boundaries[i-1][1] + 0.1)
        cluster_boundaries.append((lower, upper))

    cluster_proportions = [( (log_data>=l)&(log_data<=u) ).mean()
                           for l, u in cluster_boundaries]

    suspicious_clusters = [i for i, prop in enumerate(cluster_proportions) if prop < min_cluster_size]

    result_mask = pd.Series(True, index=series.index)
    for idx in suspicious_clusters:
        l, u = cluster_boundaries[idx]
        in_cluster = (log_data >= l) & (log_data <= u)
        result_mask.loc[positive_series[in_cluster].index] = False

    if plot_debug:
        fig, axes = plt.subplots(2, 2, figsize=(12,10))
        axes[0,0].hist(series.dropna(), bins=50, alpha=0.7)
        axes[0,0].set_title("原始数据分布")
        axes[0,1].hist(log_data, bins=30, alpha=0.7)
        for l,u in cluster_boundaries:
            axes[0,1].axvspan(l, u, alpha=0.2, color='red')
        axes[0,1].set_title("对数数据分布")
        axes[1,0].plot(x_grid, kde_vals, 'b-', linewidth=2)
        for peak in peaks:
            axes[1,0].axvline(peak, color='r', linestyle='--', alpha=0.5)
        axes[1,0].set_title("核密度与峰值")
        cluster_labels_plot = [f'聚类{i+1}' for i in range(len(cluster_proportions))]
        colors = ['green' if i not in suspicious_clusters else 'red' for i in range(len(cluster_proportions))]
        axes[1,1].bar(cluster_labels_plot, cluster_proportions, color=colors)
        axes[1,1].axhline(min_cluster_size, color='blue', linestyle='--', label=f'阈值({min_cluster_size*100:.1f}%)')
        axes[1,1].set_title("聚类比例")
        axes[1,1].legend()
        plt.tight_layout()
        plt.show()

    return result_mask

# =========================
# Step4: 处理 DataFrame
# =========================
result_df = df.copy()
masks_to_combine = []
deleted_records = {}  # 用于记录被删除的样本

for col in result_df.select_dtypes(include=[np.number]).columns:
    mask = intelligent_scale_detection(result_df[col], min_cluster_size=0.01, plot_debug=False)
    masks_to_combine.append(mask)

    # 保存每列被标记异常的样本
    deleted_idx = result_df.index[~mask].tolist()
    if deleted_idx:
        deleted_records[col] = result_df.loc[deleted_idx, col].to_dict()

if masks_to_combine:
    combined_mask = pd.concat(masks_to_combine, axis=1).all(axis=1)
    filtered_df = result_df[combined_mask]
else:
    filtered_df = result_df

# =========================
# Step4: 保存处理后的 CSV
# =========================
output_dir_csv = "dataset"
os.makedirs(output_dir_csv, exist_ok=True)
output_file_csv = os.path.join(output_dir_csv, "covid_after_step4.csv")
filtered_df.to_csv(output_file_csv, index=False, encoding="utf-8-sig")

# =========================
# Step4: 可视化删除样本统计
# =========================
output_dir_img = "results/step4"
os.makedirs(output_dir_img, exist_ok=True)

plt.figure(figsize=(6,4))
plt.bar(["保留样本","删除异常样本"],
        [combined_mask.sum(), (~combined_mask).sum()],
        color=["green","red"])
plt.ylabel("样本数量")
plt.title("Step4: 异常样本统计")
for i, v in enumerate([combined_mask.sum(), (~combined_mask).sum()]):
    plt.text(i, v + 0.5, str(v), ha='center', fontsize=10)
plt.tight_layout()
output_file_img = os.path.join(output_dir_img, "outlier_stats.png")
plt.savefig(output_file_img, dpi=300)
plt.show()

# =========================
# Step4: 保存删除样本 JSON
# =========================
output_file_json = os.path.join(output_dir_img, "deleted_samples.json")
with open(output_file_json, "w", encoding="utf-8") as f:
    json.dump(deleted_records, f, ensure_ascii=False, indent=4)

# =========================
# 输出信息
# =========================
print(f"Step4: 异常值处理完成，文件已保存到 {output_file_csv}")
print(f"Step4: 异常样本统计图已保存到 {output_file_img}")
print(f"Step4: 删除样本 JSON 已保存到 {output_file_json}")
print(f"Step4: 原始样本数={len(df)}, 保留样本数={len(filtered_df)}, 删除样本数={len(df)-len(filtered_df)}")
