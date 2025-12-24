# =========================
# COVID 数据 XGBoost 分类（网格搜索 + 10折交叉验证 + 自动阈值 + 进度条）
# =========================
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, f1_score, roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from tqdm import tqdm

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
output_dir = "results/xgb_cv_progress"
os.makedirs(output_dir, exist_ok=True)

# =========================
# 特征划分
# =========================
features = df.drop(columns=['病人ID', '结局'])
feature_names = features.columns.tolist()

physio_cols = feature_names[feature_names.index('心率'):feature_names.index('氧饱和度(spO2)')+1]
symptom_cols = feature_names[feature_names.index('发热'):feature_names.index('出血')+1]
history_cols = feature_names[feature_names.index('慢性心脏病'):feature_names.index('风湿性疾病')+1]
lab_cols = feature_names[feature_names.index('白细胞计数(x10^9/L)'):feature_names.index('铁蛋白(ng/mL)')+1]
complication_cols = feature_names[feature_names.index('病毒性肺炎/肺炎'):feature_names.index('感染性休克')+1]

module_dict = {
    '整体特征': feature_names,
    '生理指标': physio_cols,
    '症状描述': symptom_cols,
    '既往史': history_cols,
    '实验室检查': lab_cols,
    '并发症': complication_cols
}

y_all = df['结局']

# =========================
# 超参数网格
# =========================
param_grid = {
    'max_depth': [2,4,6,8],
    'learning_rate': [0.01, 0.1],
    'n_estimators': [200, 500],
    'gamma': [0,0.5]
}

# =========================
# 核心训练函数
# =========================
def train_xgb_cv_progress(X_subset, y_all, module_name):
    module_dir = os.path.join(output_dir, module_name)
    os.makedirs(module_dir, exist_ok=True)
    
    X_scaled = StandardScaler().fit_transform(X_subset)
    
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    best_f1 = -1
    best_model = None
    best_threshold = 0.5
    
    grid_list = list(ParameterGrid(param_grid))
    # 外层进度条：网格搜索
    for params in tqdm(grid_list, desc=f"{module_name} 网格搜索"):
        f1_scores = []
        # 内层进度条：10折CV
        for train_idx, val_idx in skf.split(X_scaled, y_all):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y_all.iloc[train_idx], y_all.iloc[val_idx]
            
            model = xgb.XGBClassifier(
                max_depth=params['max_depth'],
                learning_rate=params['learning_rate'],
                n_estimators=params['n_estimators'],
                gamma=params['gamma'],
                eval_metric='logloss',
                use_label_encoder=False,
                random_state=42
            )
            model.fit(X_train, y_train)
            y_prob = model.predict_proba(X_val)[:,1]
            
            precisions, recalls, thresholds = precision_recall_curve(y_val, y_prob)
            f1s = 2*precisions*recalls/(precisions+recalls+1e-8)
            f1_scores.append(np.max(f1s))
        
        mean_f1 = np.mean(f1_scores)
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_model = model
            best_threshold = thresholds[np.argmax(f1s)]
    
    # 最终模型预测
    y_prob = best_model.predict_proba(X_scaled)[:,1]
    precisions, recalls, thresholds = precision_recall_curve(y_all, y_prob)
    f1s = 2*precisions*recalls/(precisions+recalls+1e-8)
    best_idx = np.argmax(f1s)
    y_pred = (y_prob >= thresholds[best_idx]).astype(int)
    
    # 保存预测结果
    result_df = X_subset.copy()
    result_df['真实'] = y_all
    result_df['预测概率'] = y_prob
    result_df['预测'] = y_pred
    result_df.to_csv(os.path.join(module_dir, f"{module_name}_pred.csv"), index=False, encoding='utf-8-sig')
    
    # ROC
    fpr, tpr, _ = roc_curve(y_all, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f'AUC={roc_auc:.3f}')
    plt.plot([0,1],[0,1],'--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{module_name} ROC曲线")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(module_dir, f"{module_name}_ROC.png"), dpi=300)
    plt.close()
    
    # PR
    plt.figure(figsize=(6,5))
    plt.plot(recalls, precisions, label=f'F1_max={f1s[best_idx]:.3f}')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{module_name} PR曲线")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(module_dir, f"{module_name}_PR.png"), dpi=300)
    plt.close()
    
    # 混淆矩阵
    cm = confusion_matrix(y_all, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel("预测")
    plt.ylabel("真实")
    plt.title(f"{module_name} 混淆矩阵")
    plt.tight_layout()
    plt.savefig(os.path.join(module_dir, f"{module_name}_confusion.png"), dpi=300)
    plt.close()
    
    print(f"{module_name} 完成，最佳 F1={f1s[best_idx]:.3f}")

# =========================
# 执行
# =========================
for module_name, cols in tqdm(module_dict.items(), desc="模块训练进度"):
    X_subset = df[cols]
    train_xgb_cv_progress(X_subset, y_all, module_name)

print("✅ 所有模块训练完成，结果保存在", output_dir)
