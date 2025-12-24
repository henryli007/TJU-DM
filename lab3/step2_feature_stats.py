# =========================
# step2_feature_stats.py
# Step 2: 属性统计、可视化，并导出变量类型定义
# =========================

import pandas as pd
import matplotlib.pyplot as plt
import os
import json

# =========================
# 配置中文显示
# =========================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# =========================
# 读取 CSV
# =========================
file_path = "dataset/covid_processed_chinese.csv"
df = pd.read_csv(file_path, encoding="utf-8-sig")

# =========================
# Step 2.1: 变量类型识别
# =========================

# 二值变量：非空唯一值数为 2（此处允许 1/2，后续再转 0/1）
binary_cols = [
    col for col in df.columns
    if df[col].dropna().nunique() == 2
]

# 数值型列
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# 连续变量：数值型且不是二值
continuous_cols = [
    col for col in numeric_cols
    if col not in binary_cols
]

# 离散变量：剩余的
discrete_cols = [
    col for col in df.columns
    if col not in binary_cols + continuous_cols
]

# =========================
# Step 2.2: 打印统计结果
# =========================
print("Step 2: 属性统计结果")
print(f"总属性数: {df.shape[1]}")
print(f"二值变量数: {len(binary_cols)}")
print(f"离散变量数: {len(discrete_cols)}")
print(f"连续变量数: {len(continuous_cols)}\n")

print("二值变量列名：")
print(binary_cols if binary_cols else "无")

print("\n离散变量列名：")
print(discrete_cols if discrete_cols else "无")

print("\n连续变量列名：")
print(continuous_cols if continuous_cols else "无")

# =========================
# Step 2.3: 可视化变量类型分布
# =========================

output_dir = "results/step2"
os.makedirs(output_dir, exist_ok=True)

plot_file = os.path.join(output_dir, "feature_types_distribution.png")

plt.figure(figsize=(6, 4))
plt.bar(
    ["二值变量", "离散变量", "连续变量"],
    [len(binary_cols), len(discrete_cols), len(continuous_cols)],
    color=["skyblue", "orange", "green"]
)
plt.title("属性类型分布")
plt.ylabel("数量")

for i, v in enumerate([len(binary_cols), len(discrete_cols), len(continuous_cols)]):
    plt.text(i, v + 0.5, str(v), ha="center", fontsize=10)

plt.tight_layout()
plt.savefig(plot_file, dpi=300)
plt.show()

print(f"\nStep 2: 可视化已保存到 {plot_file}")

# =========================
# Step 2.4: 导出变量类型定义（供后续步骤使用）
# =========================

variable_types = {
    "binary": binary_cols,
    "discrete": discrete_cols,
    "continuous": continuous_cols
}

json_file = os.path.join(output_dir, "variable_types.json")

with open(json_file, "w", encoding="utf-8") as f:
    json.dump(variable_types, f, ensure_ascii=False, indent=2)

print(f"Step 2: 变量类型定义已保存到 {json_file}")
