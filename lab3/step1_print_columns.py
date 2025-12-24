# =========================
# Step 1: 列名清洗 + 中文映射 + 括号统一
# =========================

import pandas as pd
import json

# 文件路径
excel_file = "dataset/COVID_study_F1000_2.xlsx"
json_file = "dataset/column_mapping.json"
output_csv = "dataset/covid_processed_chinese.csv"

# 读取 Excel
df = pd.read_excel(excel_file, sheet_name="Data")

# 1. 列名清洗：去掉首尾空格
df.columns = df.columns.str.strip()

# 2. 加载列名映射
with open(json_file, "r", encoding="utf-8") as f:
    column_dict = json.load(f)

# 3. 映射中文列名
df = df.rename(columns=column_dict)

# 4. 括号统一：中文括号 → 英文括号
df.columns = df.columns.str.replace("（", "(", regex=False).str.replace("）", ")", regex=False)

# 5. 打印映射后的列名
print("Step 1: 中文列名 CSV 已生成：", output_csv)
print("Step 1: 中文列名如下（用于检查）：")
for col in df.columns:
    print(col)

# 6. 检查未映射列
unmapped_cols = [col for col in df.columns if col not in column_dict.values()]
if unmapped_cols:
    print("\n注意：以下列未映射，可能需要更新 JSON 映射或清洗数据：")
    for col in unmapped_cols:
        print(col)

# 7. 保存为 CSV
df.to_csv(output_csv, index=False, encoding="utf-8-sig")
