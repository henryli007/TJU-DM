# =========================
# step5_normalization.py
# Step 5: 连续变量归一化（排除离散变量，例如年龄）
# =========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler

# =========================
# 配置中文显示
# =========================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# =========================
# 读取 Step4 处理后的 CSV
# =========================
file_path = "dataset/covid_after_step4.csv"
df = pd.read_csv(file_path, encoding="utf-8-sig")

# =========================
# Step 5.1: 确定连续变量
# =========================
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# 假设二值变量已经是0/1
binary_cols = [col for col in numeric_cols if df[col].dropna().nunique() == 2]

# 排除标识列和离散变量，例如病人ID和年龄
exclude_cols = ['病人ID', '年龄']
continuous_cols = [col for col in numeric_cols if col not in binary_cols + exclude_cols]

print(f"Step5: 连续变量将被归一化: {continuous_cols}")

# =========================
# Step 5.2: 连续变量归一化 (0-1)
# =========================
scaler = MinMaxScaler()
df[continuous_cols] = scaler.fit_transform(df[continuous_cols])

# =========================
# Step 5.3: 保存归一化后的 CSV
# =========================
output_dir_csv = "dataset"
os.makedirs(output_dir_csv, exist_ok=True)
output_file_csv = os.path.join(output_dir_csv, "covid_after_step5.csv")
df.to_csv(output_file_csv, index=False, encoding="utf-8-sig")

# =========================
# Step 5.4: 可视化归一化后的统计情况
# =========================
output_dir_img = "results/step5"
os.makedirs(output_dir_img, exist_ok=True)

plt.figure(figsize=(10, 5))
for col in continuous_cols:
    plt.hist(df[col], bins=30, alpha=0.6, label=col)

plt.title("Step5: 连续变量归一化后分布 (0-1)")
plt.xlabel("归一化值")
plt.ylabel("频数")
plt.legend(loc="upper right", fontsize=8)
plt.tight_layout()

output_file_img = os.path.join(output_dir_img, "continuous_normalized_hist.png")
plt.savefig(output_file_img, dpi=300)
plt.show()

# =========================
# Step 5.5: 输出提示信息
# =========================
print(f"Step5: 归一化完成，文件已保存到 {output_file_csv}")
print(f"Step5: 归一化统计图已保存到 {output_file_img}")
