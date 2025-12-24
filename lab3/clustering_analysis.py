# =========================
# clustering_analysis.py
# 统一 KMeans 聚类 + 正多边形雷达图
# =========================

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# =========================
# 中文显示
# =========================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# =========================
# 读取数据
# =========================
df = pd.read_csv("dataset/covid_after_step5.csv", encoding="utf-8-sig")

# =========================
# 输出目录
# =========================
output_dir = "results/clustering"
os.makedirs(output_dir, exist_ok=True)

# =========================
# 特征划分
# =========================
features = df.drop(columns=['病人ID', '结局'])
feature_names = features.columns.tolist()

physio_cols = feature_names[
    feature_names.index('心率'):feature_names.index('氧饱和度(spO2)') + 1
]
symptom_cols = feature_names[
    feature_names.index('发热'):feature_names.index('出血') + 1
]
history_cols = feature_names[
    feature_names.index('慢性心脏病'):feature_names.index('风湿性疾病') + 1
]
lab_cols = feature_names[
    feature_names.index('白细胞计数(x10^9/L)'):feature_names.index('铁蛋白(ng/mL)') + 1
]
complication_cols = feature_names[
    feature_names.index('病毒性肺炎/肺炎'):feature_names.index('感染性休克') + 1
]

module_config = {
    '整体特征': {
        'cols': feature_names,
        'show_label': False
    },
    '生理指标': {
        'cols': physio_cols,
        'show_label': True
    },
    '实验室检查': {
        'cols': lab_cols,
        'show_label': True
    },
    '症状描述': {
        'cols': symptom_cols,
        'show_label': False
    },
    '既往史': {
        'cols': history_cols,
        'show_label': True
    },
    '并发症': {
        'cols': complication_cols,
        'show_label': True
    }
}

# =========================
# 核心函数
# =========================
def cluster_and_visualize(df_subset, module_name, show_label, n_clusters=3):
    X = df_subset.values.astype(float)

    module_dir = os.path.join(output_dir, module_name)
    os.makedirs(module_dir, exist_ok=True)

    # ---------- 标准化 ----------
    Xs = StandardScaler().fit_transform(X)

    # ---------- KMeans 聚类 ----------
    labels = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    ).fit_predict(Xs)

    # ---------- PCA 可视化 ----------
    if X.shape[1] > 2:
        X_vis = PCA(n_components=2).fit_transform(Xs)
    else:
        X_vis = Xs

    plt.figure(figsize=(6, 5))
    for lab in np.unique(labels):
        plt.scatter(
            X_vis[labels == lab, 0],
            X_vis[labels == lab, 1],
            label=f'簇 {lab}',
            alpha=0.6
        )
    plt.title(f"{module_name} KMeans 聚类结果")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{module_dir}/{module_name}_cluster.png", dpi=300)
    plt.close()

    # ---------- 雷达图（正 n 边形） ----------
    summary = pd.DataFrame(Xs, columns=df_subset.columns)
    summary['cluster'] = labels
    summary = summary.groupby('cluster').mean()

    categories = summary.columns.tolist()
    N = len(categories)

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])

    fig = plt.figure(figsize=(7, 7))
    ax = plt.subplot(111, polar=True)

    for idx, row in summary.iterrows():
        values = row.values.tolist()
        values = values + [values[0]]
        ax.plot(angles, values, linewidth=2, label=f'簇 {idx}')
        ax.fill(angles, values, alpha=0.15)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    if show_label:
        ax.set_thetagrids(
            angles[:-1] * 180 / np.pi,
            categories,
            fontsize=8
        )
    else:
        ax.set_thetagrids([])

    ax.set_title(f"{module_name} 雷达图", y=1.1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{module_dir}/{module_name}_radar.png", dpi=300)
    plt.close()

    print(f"✔ {module_name} 完成")

# =========================
# 执行
# =========================
for name, cfg in module_config.items():
    cluster_and_visualize(
        df[cfg['cols']],
        name,
        cfg['show_label'],
        n_clusters=3
    )

print("✅ 所有模块已使用 KMeans 完成聚类与雷达图绘制")
