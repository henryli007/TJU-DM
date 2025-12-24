# =========================
# association_analysis_subsets_final.py
# COVID 数据子集关联规则挖掘（稳定版）
# =========================

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib as mpl
import networkx as nx

from mlxtend.frequent_patterns import fpgrowth, association_rules

# =========================
# 中文字体配置（稳定方案）
# =========================
font_candidates = ['SimHei', 'Microsoft YaHei', 'PingFang SC']
available_fonts = [f.name for f in fm.fontManager.ttflist]

for font in font_candidates:
    if font in available_fonts:
        mpl.rcParams['font.sans-serif'] = [font]
        break

mpl.rcParams['axes.unicode_minus'] = False

# =========================
# 读取数据
# =========================
df = pd.read_csv("dataset/covid_after_step5.csv", encoding="utf-8-sig")

# =========================
# 输出目录
# =========================
OUTPUT_DIR = "results/association_subsets"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# 特征子集划分
# =========================
feature_names = df.drop(columns=['病人ID', '结局']).columns.tolist()

physio_cols = feature_names[feature_names.index('心率'):feature_names.index('氧饱和度(spO2)') + 1]
symptom_cols = feature_names[feature_names.index('发热'):feature_names.index('出血') + 1]
history_cols = feature_names[feature_names.index('慢性心脏病'):feature_names.index('风湿性疾病') + 1]
lab_cols = feature_names[feature_names.index('白细胞计数(x10^9/L)'):feature_names.index('铁蛋白(ng/mL)') + 1]
complication_cols = feature_names[feature_names.index('病毒性肺炎/肺炎'):feature_names.index('感染性休克') + 1]

MODULES = {
    '生理指标': physio_cols,
    '症状描述': symptom_cols,
    '既往史': history_cols,
    '实验室检查': lab_cols,
    '并发症': complication_cols
}

# =========================
# 二值化函数（bool，避免 DeprecationWarning）
# =========================
def binarize_features(df_subset):
    df_bin = df_subset.copy()
    for col in df_bin.columns:
        if df_bin[col].nunique() > 2:
            threshold = df_bin[col].median()
            df_bin[col] = (df_bin[col] > threshold)
        else:
            df_bin[col] = df_bin[col].astype(bool)
    return df_bin

# =========================
# 关联规则挖掘主函数
# =========================
def mine_association_rules(
    df_subset,
    module_name,
    min_support=0.12,
    min_confidence=0.7,
    min_lift=1.2,
    max_len=2,
    top_k=20
):
    print(f"\n[{module_name}] 开始处理...")
    module_dir = os.path.join(OUTPUT_DIR, module_name)
    os.makedirs(module_dir, exist_ok=True)

    # ---------- 二值化 ----------
    df_bin = binarize_features(df_subset)

    # ---------- FP-Growth ----------
    frequent_itemsets = fpgrowth(
        df_bin,
        min_support=min_support,
        use_colnames=True,
        max_len=max_len
    )

    frequent_itemsets.to_csv(
        os.path.join(module_dir, f"{module_name}_frequent_itemsets.csv"),
        index=False,
        encoding='utf-8-sig'
    )

    if frequent_itemsets.empty:
        print(f"[{module_name}] 无频繁项集")
        return

    # ---------- 关联规则 ----------
    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=min_confidence
    )

    rules = rules[
        (rules['lift'] >= min_lift) &
        (rules['support'] >= min_support)
    ]

    if rules.empty:
        print(f"[{module_name}] 无满足条件的规则")
        return

    # ---------- Top-K ----------
    rules = rules.sort_values(
        ['lift', 'confidence', 'support'],
        ascending=False
    ).head(top_k)

    rules.to_csv(
        os.path.join(module_dir, f"{module_name}_Top{top_k}_rules.csv"),
        index=False,
        encoding='utf-8-sig'
    )

    # ---------- 文本说明 ----------
    with open(
        os.path.join(module_dir, f"{module_name}_Top{top_k}_rules.txt"),
        "w", encoding="utf-8"
    ) as f:
        for _, row in rules.iterrows():
            f.write(
                f"{set(row['antecedents'])} -> {set(row['consequents'])}\n"
                f"  support={row['support']:.3f}, "
                f"confidence={row['confidence']:.3f}, "
                f"lift={row['lift']:.3f}\n\n"
            )

    # ---------- 构建规则网络 ----------
    G = nx.DiGraph()
    edge_labels = {}

    for _, row in rules.iterrows():
        for a in row['antecedents']:
            for c in row['consequents']:
                G.add_edge(a, c)
                edge_labels[(a, c)] = (
                    f"s={row['support']:.2f}\n"
                    f"c={row['confidence']:.2f}"
                )

    # ---------- 绘图 ----------
    plt.figure(figsize=(12, 9))
    pos = nx.spring_layout(G, seed=42, k=0.8)

    nx.draw_networkx_nodes(
        G, pos,
        node_size=1200,
        node_color="#AED6F1",
        edgecolors="black"
    )

    nx.draw_networkx_edges(
        G, pos,
        arrows=True,
        arrowstyle="->",
        arrowsize=15,
        width=2
    )

    nx.draw_networkx_labels(
        G, pos,
        font_size=9
    )

    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=edge_labels,
        font_size=8
    )

    plt.title(f"{module_name} Top-{top_k} 关联规则网络")
    plt.axis("off")
    plt.tight_layout()

    plt.savefig(
        os.path.join(module_dir, f"{module_name}_Top{top_k}_rules_network.png"),
        dpi=300
    )
    plt.close()

    print(
        f"[{module_name}] 完成："
        f"{len(frequent_itemsets)} 个频繁项集，"
        f"{len(rules)} 条 Top 规则"
    )

# =========================
# 执行
# =========================
for module_name, cols in MODULES.items():
    mine_association_rules(
        df_subset=df[cols],
        module_name=module_name
    )

print("\n✅ 所有关联规则挖掘与可视化完成")
