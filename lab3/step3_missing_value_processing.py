# =========================
# step3_missing_value_processing.py
# Step 3: 数据缺失值处理（填充）
# =========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# =========================
# 配置中文显示
# =========================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# =========================
# 读取 Step2 处理后的 CSV
# =========================
file_path = "dataset/covid_processed_chinese.csv"
df = pd.read_csv(file_path, encoding="utf-8-sig")

# =========================
# Step 3.1: 定义变量类别
# =========================
binary_cols = [col for col in df.columns if df[col].dropna().nunique() == 2]
discrete_cols = ['年龄']  # 假设这里只有年龄
numeric_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
continuous_cols = [col for col in numeric_cols if col not in binary_cols + discrete_cols]

# 记录删除的列
deleted_cols = []

# =========================
# Step 3.2: 填充统计（填充前统计缺失数量）
# =========================
filled_binary = sum(df[col].isna().sum() for col in binary_cols)
filled_discrete = sum(df[col].isna().sum() for col in discrete_cols)
filled_continuous = sum(df[col].isna().sum() for col in continuous_cols)

# =========================
# Step 3.3: 二值变量 1/2 -> 0/1，空缺保持 NaN
# =========================
for col in binary_cols:
    df[col] = df[col].replace({1:0, 2:1})

# =========================
# Step 3.4: 离散变量填充
# =========================
for col in discrete_cols:
    mode_value = df[col].mode()[0]
    df[col] = df[col].fillna(mode_value)

# =========================
# 离散变量编码（年龄区间 -> 数值）
# =========================
age_mapping = {
    "60-70": 0,
    "71-80": 1,
    ">80": 2
}
df['年龄'] = df['年龄'].map(age_mapping)

# =========================
# Step 3.5: 连续变量填充 & 删除有效数据过少列
# =========================
drop_continuous_cols = []
min_valid_samples = 5  # 有效数据少于5个就删除

for col in continuous_cols:
    valid_values = df[col].dropna()
    missing_ratio = df[col].isna().mean()

    # 删除缺失率过高或有效数据过少的列
    if missing_ratio > 0.5 or len(valid_values) < min_valid_samples:
        drop_continuous_cols.append(col)
        deleted_cols.append(col)
        continue

    # 使用直方图主峰填充缺失值
    counts, bin_edges = np.histogram(valid_values, bins='auto')
    max_bin_index = np.argmax(counts)
    bin_mask = (valid_values >= bin_edges[max_bin_index]) & (valid_values <= bin_edges[max_bin_index+1])
    fill_value = valid_values[bin_mask].mean()
    df[col] = df[col].fillna(fill_value).astype(float).round(2)

# 删除缺失率过高或数据过少的连续列
if drop_continuous_cols:
    df.drop(columns=drop_continuous_cols, inplace=True)

# =========================
# Step 3.6: 保存填充后的 CSV
# =========================
output_dir_csv = "dataset"
os.makedirs(output_dir_csv, exist_ok=True)
output_file_csv = os.path.join(output_dir_csv, "covid_after_step3.csv")
df.to_csv(output_file_csv, index=False, encoding="utf-8-sig")

# =========================
# Step 3.7: 可视化填充数量（填充前缺失数）
# =========================
output_dir_img = "results/step3"
os.makedirs(output_dir_img, exist_ok=True)

plt.figure(figsize=(6,4))
plt.bar(
    ["二值变量","离散变量","连续变量"],
    [filled_binary, filled_discrete, filled_continuous],
    color=["skyblue","orange","green"]
)
plt.ylabel("缺失样本数")
plt.title("Step3: 缺失值统计（填充前）")
for i, v in enumerate([filled_binary, filled_discrete, filled_continuous]):
    plt.text(i, v + 0.5, str(v), ha='center', fontsize=10)

plt.tight_layout()
output_file_img = os.path.join(output_dir_img, "fill_stats.png")
plt.savefig(output_file_img, dpi=300)
plt.show()

# =========================
# Step 3.8: 输出删除列和文件路径
# =========================
print(f"Step3: 填充完成，文件已保存到 {output_file_csv}")
print(f"Step3: 填充统计图已保存到 {output_file_img}")
print(f"Step3: 删除的连续列如下：{deleted_cols if deleted_cols else '无'}")
