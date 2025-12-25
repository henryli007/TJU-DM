# 数据预处理报告

## 1. 处理概述
本报告总结了Step3数据预处理过程中采取的所有步骤和方法。

## 2. 处理步骤

### 2.1 人口统计数据 (demographic_data_split.csv)
- 检查并删除了关键列(Patient Code, Age Group, Gender)的缺失值
- 年龄组进行了编码转换
- 性别进行了0/1编码(Male=0, Female=1)
- 编码映射保存在 encoding_mappings.json 中

### 2.2 实验室数据 (laboratory_data.csv)
- 按缺失率分层处理特征：
  - 0-30%缺失: 峰值区间法分组填补
  - 30-60%缺失: 峰值区间法分组填补 + 创建缺失指示器
  - 60-80%缺失: -999填补 + 创建检测指示器
  - >80%缺失: 直接删除
- 对连续数值特征进行0-1标准化
- 创建了缺失指示器(Missing_*)和检测指示器(Tested_*)特征
- 特殊处理Troponin列：
  - 将'Negative'转换为0
  - 将空值转换为0
  - 创建Tested_Troponin指示器标记是否检测
  - 不进行标准化（已经是0/1编码）

### 2.3 患者结局数据 (patient_outcome.csv)
- Deceased列转换为0/1编码
- 按缺失率分层处理特征(方法与实验室数据类似)
- 特殊处理Troponin列（同实验室数据）
- 对连续数值特征进行0-1标准化

## 3. 文件输出

### 3.1 处理后的数据文件
- dataset/demographic_data_processed.csv
- dataset/laboratory_data_processed.csv
- dataset/patient_outcome_processed.csv

### 3.2 配置文件
- results/step3/encoding_mappings.json
- results/step3/missing_demographic_records.json
- results/step3/scaler_info_lab.json
- results/step3/scaler_info_outcome.json

### 3.3 可视化图表
- 人口统计数据分布图
- 实验室数据分布图
- 患者结局分布图
- Troponin分布与检测情况
- Troponin与患者结局的关系
- 缺失指示器分布图
- 特征相关性热图

## 4. 后续分析建议
- 聚类分析: 使用 laboratory_data_processed.csv
- 分类分析: 使用 patient_outcome_processed.csv
- 关联分析: 结合 demographic_data_processed.csv 和其他特征

## 5. 注意事项
1. 缺失指示器特征可用于分析缺失模式与临床结局的关联
2. 检测指示器特征可用于分析检测行为与病情的关联
3. 标准化后的数据适合用于距离计算和聚类分析
4. 极度不平衡的类别分布需要在分类模型中特别处理
5. Troponin已经是0/1编码，不参与标准化
