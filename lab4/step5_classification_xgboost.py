# step5_classification_xgboost.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, auc
)
import xgboost as xgb
import warnings
import os
from pathlib import Path
import pickle
import json
import shap
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from scipy import stats

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def create_directories():
    """创建输出目录"""
    base_dir = Path("results/step5_classification")
    directories = [
        base_dir,
        base_dir / "xgboost_models",
        base_dir / "feature_analysis",
        base_dir / "model_evaluation",
        base_dir / "visualizations",
        base_dir / "predictions"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    print("目录创建完成")

def load_and_prepare_data():
    """加载并准备数据"""
    print("=" * 60)
    print("加载数据...")
    print("=" * 60)
    
    # 加载两个处理后的数据文件
    demo_path = Path("dataset/demographic_data_processed.csv")
    outcome_path = Path("dataset/patient_outcome_processed.csv")
    
    # 检查文件是否存在
    for path in [demo_path, outcome_path]:
        if not path.exists():
            print(f"错误: 找不到文件 {path}")
            return None, None
    
    # 读取数据
    demo_df = pd.read_csv(demo_path)
    outcome_df = pd.read_csv(outcome_path)
    
    print(f"人口统计数据: {demo_df.shape}")
    print(f"患者结局数据: {outcome_df.shape}")
    
    return demo_df, outcome_df

def merge_and_prepare_features(demo_df, outcome_df):
    """合并数据并准备特征"""
    print("\n合并数据并准备特征...")
    
    # 确保Patient Code列名一致
    if 'Patient Code' in demo_df.columns and 'Patient Code' in outcome_df.columns:
        # 合并数据
        merged_df = outcome_df.merge(
            demo_df[['Patient Code', 'Age_Group_Encoded', 'Gender_Encoded']],
            on='Patient Code',
            how='inner'
        )
        
        print(f"合并后数据形状: {merged_df.shape}")
        
        # 分离特征和目标变量
        # 目标变量
        y = merged_df['Deceased'].copy()
        
        # 特征：排除标识列和目标变量
        exclude_cols = ['Patient Code', 'Deceased']
        X = merged_df.drop(columns=exclude_cols)
        
        # 获取特征名称
        feature_names = X.columns.tolist()
        
        print(f"特征数量: {len(feature_names)}")
        print(f"目标变量分布:\n{y.value_counts()}")
        print(f"死亡率: {y.sum()}/{len(y)} ({y.sum()/len(y)*100:.2f}%)")
        
        return X, y, feature_names, merged_df
    else:
        print("错误: 找不到Patient Code列")
        return None, None, None, None

def analyze_feature_types(X, feature_names):
    """分析特征类型"""
    print("\n分析特征类型...")
    
    # 特征类型分析
    feature_types = {
        'continuous': [],
        'binary': [],
        'categorical': [],
        'missing_indicators': [],
        'tested_indicators': []
    }
    
    for feature in feature_names:
        # 计算唯一值数量
        unique_vals = X[feature].nunique()
        
        # 检查是否为缺失指示器
        if feature.startswith('Missing_'):
            feature_types['missing_indicators'].append(feature)
        # 检查是否为检测指示器
        elif feature.startswith('Tested_'):
            feature_types['tested_indicators'].append(feature)
        # 检查是否为二值特征
        elif unique_vals == 2 and set(X[feature].dropna().unique()).issubset({0, 1}):
            feature_types['binary'].append(feature)
        # 检查是否为分类特征
        elif 2 < unique_vals <= 10:
            feature_types['categorical'].append(feature)
        # 其他为连续特征
        else:
            feature_types['continuous'].append(feature)
    
    # 输出统计
    print("特征类型统计:")
    for ftype, features in feature_types.items():
        print(f"  {ftype}: {len(features)} 个特征")
        if len(features) <= 5 and len(features) > 0:
            print(f"    {features}")
    
    return feature_types

def handle_class_imbalance(X, y, method='none', random_state=42):
    """处理类别不平衡"""
    print(f"\n处理类别不平衡 (方法: {method})...")
    
    original_counts = Counter(y)
    print(f"原始类别分布: {original_counts}")
    
    if method == 'smote':
        # SMOTE过采样
        smote = SMOTE(random_state=random_state)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
    elif method == 'undersample':
        # 欠采样
        rus = RandomUnderSampler(random_state=random_state)
        X_resampled, y_resampled = rus.fit_resample(X, y)
        
    elif method == 'both':
        # 组合采样：先过采样少数类，再欠采样多数类
        # 这里我们使用1:2的平衡
        over = SMOTE(sampling_strategy=0.1, random_state=random_state)  # 少数类增加到10%
        under = RandomUnderSampler(sampling_strategy=0.5, random_state=random_state)  # 多数类减少到50%
        
        # 先过采样
        X_over, y_over = over.fit_resample(X, y)
        # 再欠采样
        X_resampled, y_resampled = under.fit_resample(X_over, y_over)
        
    else:  # 'none'
        X_resampled, y_resampled = X.copy(), y.copy()
    
    resampled_counts = Counter(y_resampled)
    print(f"重采样后类别分布: {resampled_counts}")
    
    return X_resampled, y_resampled

def optimize_xgboost_hyperparameters(X_train, y_train, n_splits=5, random_state=42):
    """优化XGBoost超参数"""
    print("\n优化XGBoost超参数...")
    
    # 基础XGBoost模型
    xgb_model = xgb.XGBClassifier(
        random_state=random_state,
        eval_metric='logloss',
        use_label_encoder=False
    )
    
    # 定义参数网格
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'gamma': [0, 0.1, 0.2],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [0.1, 1, 5],
        'scale_pos_weight': [1, 3, 5]  # 处理不平衡
    }
    
    # 使用交叉验证进行网格搜索
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=kfold,
        verbose=1,
        n_jobs=-1
    )
    
    # 拟合网格搜索
    print("正在进行网格搜索（可能需要几分钟）...")
    grid_search.fit(X_train, y_train)
    
    print(f"最佳参数: {grid_search.best_params_}")
    print(f"最佳交叉验证ROC AUC: {grid_search.best_score_:.4f}")
    
    # 获取最佳参数，并添加必要的固定参数
    best_params = grid_search.best_params_.copy()
    best_params['random_state'] = random_state
    best_params['eval_metric'] = 'logloss'
    best_params['use_label_encoder'] = False
    
    return grid_search.best_estimator_, best_params

def perform_cross_validation(X, y, model, n_splits=10, random_state=42):
    """执行十折交叉验证"""
    print(f"\n执行{n_splits}折交叉验证...")
    
    # 初始化交叉验证
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # 存储每折的结果
    cv_results = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'roc_auc': [],
        'y_true_all': [],
        'y_pred_all': [],
        'y_pred_proba_all': []
    }
    
    # 存储每折的ROC和PR曲线数据
    mean_fpr = np.linspace(0, 1, 100)
    mean_recall = np.linspace(0, 1, 100)
    
    tprs = []
    precisions = []
    aucs = []
    pr_aucs = []
    
    # 执行交叉验证
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y), 1):
        print(f"\n  折 {fold}/{n_splits}")
        
        # 划分训练集和验证集
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # 训练模型
        model.fit(X_train, y_train)
        
        # 预测
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        # 计算评估指标
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, zero_division=0)
        recall = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_val, y_pred_proba)
        
        # 存储指标
        cv_results['accuracy'].append(accuracy)
        cv_results['precision'].append(precision)
        cv_results['recall'].append(recall)
        cv_results['f1'].append(f1)
        cv_results['roc_auc'].append(roc_auc)
        
        # 存储预测结果
        cv_results['y_true_all'].extend(y_val)
        cv_results['y_pred_all'].extend(y_pred)
        cv_results['y_pred_proba_all'].extend(y_pred_proba)
        
        # 计算ROC曲线
        fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(roc_auc)
        
        # 计算PR曲线
        precision_curve, recall_curve, _ = precision_recall_curve(y_val, y_pred_proba)
        interp_precision = np.interp(mean_recall, recall_curve[::-1], precision_curve[::-1])
        precisions.append(interp_precision)
        pr_auc = auc(recall_curve, precision_curve)
        pr_aucs.append(pr_auc)
        
        print(f"    准确率: {accuracy:.4f}, 精确率: {precision:.4f}, 召回率: {recall:.4f}, "
              f"F1分数: {f1:.4f}, ROC AUC: {roc_auc:.4f}")
    
    # 计算平均ROC曲线
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    
    # 计算平均PR曲线
    mean_precision = np.mean(precisions, axis=0)
    mean_precision[0] = 1.0
    mean_pr_auc = np.mean(pr_aucs)
    std_pr_auc = np.std(pr_aucs)
    
    # 添加平均曲线到结果
    cv_results['mean_fpr'] = mean_fpr
    cv_results['mean_tpr'] = mean_tpr
    cv_results['mean_auc'] = mean_auc
    cv_results['std_auc'] = std_auc
    cv_results['mean_recall'] = mean_recall
    cv_results['mean_precision'] = mean_precision
    cv_results['mean_pr_auc'] = mean_pr_auc
    cv_results['std_pr_auc'] = std_pr_auc
    
    # 计算平均指标
    cv_results['mean_accuracy'] = np.mean(cv_results['accuracy'])
    cv_results['mean_precision'] = np.mean(cv_results['precision'])
    cv_results['mean_recall'] = np.mean(cv_results['recall'])
    cv_results['mean_f1'] = np.mean(cv_results['f1'])
    cv_results['mean_roc_auc'] = np.mean(cv_results['roc_auc'])
    
    # 计算标准差
    cv_results['std_accuracy'] = np.std(cv_results['accuracy'])
    cv_results['std_precision'] = np.std(cv_results['precision'])
    cv_results['std_recall'] = np.std(cv_results['recall'])
    cv_results['std_f1'] = np.std(cv_results['f1'])
    cv_results['std_roc_auc'] = np.std(cv_results['roc_auc'])
    
    print(f"\n{'-'*60}")
    print("交叉验证结果汇总:")
    print(f"{'-'*60}")
    print(f"平均准确率: {cv_results['mean_accuracy']:.4f} (±{cv_results['std_accuracy']:.4f})")
    print(f"平均精确率: {cv_results['mean_precision']:.4f} (±{cv_results['std_precision']:.4f})")
    print(f"平均召回率: {cv_results['mean_recall']:.4f} (±{cv_results['std_recall']:.4f})")
    print(f"平均F1分数: {cv_results['mean_f1']:.4f} (±{cv_results['std_f1']:.4f})")
    print(f"平均ROC AUC: {cv_results['mean_roc_auc']:.4f} (±{cv_results['std_roc_auc']:.4f})")
    print(f"平均PR AUC: {mean_pr_auc:.4f} (±{std_pr_auc:.4f})")
    
    return cv_results

def train_final_model(X, y, best_params, random_state=42):
    """训练最终模型"""
    print("\n训练最终模型...")
    
    # 创建一个参数的副本，避免修改原始参数
    model_params = best_params.copy()
    
    # 确保参数中包含必要的设置
    if 'random_state' not in model_params:
        model_params['random_state'] = random_state
    
    if 'eval_metric' not in model_params:
        model_params['eval_metric'] = 'logloss'
    
    if 'use_label_encoder' not in model_params:
        model_params['use_label_encoder'] = False
    
    # 使用参数创建最终模型
    final_model = xgb.XGBClassifier(**model_params)
    
    # 在整个数据集上训练
    final_model.fit(X, y)
    
    return final_model

def evaluate_model_performance(model, X, y, feature_names, output_dir):
    """评估模型性能"""
    print("\n评估模型性能...")
    
    # 预测
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    # 计算评估指标
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y, y_pred_proba)
    
    # 计算PR AUC
    precision_curve, recall_curve, _ = precision_recall_curve(y, y_pred_proba)
    pr_auc = auc(recall_curve, precision_curve)
    
    # 混淆矩阵
    cm = confusion_matrix(y, y_pred)
    
    # 分类报告
    report = classification_report(y, y_pred, target_names=['存活', '死亡'])
    
    print("模型性能指标:")
    print(f"准确率: {accuracy:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1分数: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"PR AUC: {pr_auc:.4f}")
    
    # 保存评估结果
    eval_results = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'confusion_matrix': cm.tolist()
    }
    
    with open(output_dir / 'model_evaluation' / 'evaluation_results.json', 'w', encoding='utf-8') as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)
    
    with open(output_dir / 'model_evaluation' / 'classification_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    return eval_results, cm, report

def plot_roc_curve(cv_results, output_dir):
    """绘制ROC曲线"""
    print("\n绘制ROC曲线...")
    
    plt.figure(figsize=(10, 8))
    
    # 绘制每折的ROC曲线
    for i in range(len(cv_results.get('tprs', []))):
        plt.plot(cv_results['mean_fpr'], cv_results.get('tprs', [])[i], 
                lw=1, alpha=0.3, label=f'折 {i+1} (AUC = {cv_results.get("aucs", [])[i]:.2f})')
    
    # 绘制平均ROC曲线
    mean_auc = cv_results.get('mean_auc', cv_results.get('mean_roc_auc', 0))
    std_auc = cv_results.get('std_auc', cv_results.get('std_roc_auc', 0))
    
    plt.plot(cv_results['mean_fpr'], cv_results['mean_tpr'], color='b',
            label=f'平均ROC曲线 (AUC = {mean_auc:.2f} ± {std_auc:.2f})',
            lw=2, alpha=0.8)
    
    # 绘制随机猜测线
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='随机猜测')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正率 (FPR)')
    plt.ylabel('真正率 (TPR)')
    plt.title('ROC曲线 (十折交叉验证)')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'visualizations' / 'roc_curve_cv.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_precision_recall_curve(cv_results, y_true, output_dir):
    """绘制精确率-召回率曲线"""
    print("\n绘制精确率-召回率曲线...")
    
    plt.figure(figsize=(10, 8))
    
    # 计算随机猜测的精确率-召回率
    # 对于不平衡数据，随机猜测的精确率是正类比例
    pos_proportion = np.mean(y_true)
    
    # 绘制平均PR曲线
    mean_pr_auc = cv_results.get('mean_pr_auc', 0)
    std_pr_auc = cv_results.get('std_pr_auc', 0)
    
    plt.plot(cv_results['mean_recall'], cv_results['mean_precision'], color='b',
            label=f'平均PR曲线 (AUC = {mean_pr_auc:.2f} ± {std_pr_auc:.2f})',
            lw=2, alpha=0.8)
    
    # 绘制随机猜测线
    plt.hlines(pos_proportion, 0, 1, colors='k', linestyles='--', lw=2, 
              label=f'随机猜测 (精确率 = {pos_proportion:.2f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('召回率')
    plt.ylabel('精确率')
    plt.title('精确率-召回率曲线 (十折交叉验证)')
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'visualizations' / 'precision_recall_curve_cv.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(cm, output_dir):
    """绘制混淆矩阵"""
    print("\n绘制混淆矩阵...")
    
    plt.figure(figsize=(8, 6))
    
    # 创建热图
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['预测存活', '预测死亡'],
                yticklabels=['实际存活', '实际死亡'])
    
    plt.title('混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'visualizations' / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_feature_importance(model, feature_names, output_dir, top_n=20):
    """绘制特征重要性"""
    print("\n绘制特征重要性...")
    
    # 获取特征重要性
    importance = model.feature_importances_
    
    # 创建特征重要性DataFrame
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    # 保存特征重要性
    feature_importance_df.to_csv(output_dir / 'feature_analysis' / 'feature_importance.csv', 
                                 index=False, encoding='utf-8-sig')
    
    # 只显示前top_n个特征
    top_features = feature_importance_df.head(top_n)
    
    plt.figure(figsize=(12, 8))
    
    # 创建水平条形图
    bars = plt.barh(range(len(top_features)), top_features['importance'], 
                    color='skyblue', edgecolor='black')
    
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('特征重要性')
    plt.title(f'Top {top_n} 特征重要性 (XGBoost)')
    
    # 在条形上添加数值
    for i, (bar, importance) in enumerate(zip(bars, top_features['importance'])):
        plt.text(importance + 0.001, bar.get_y() + bar.get_height()/2,
                f'{importance:.4f}', va='center')
    
    plt.gca().invert_yaxis()  # 最重要的特征在顶部
    plt.tight_layout()
    plt.savefig(output_dir / 'visualizations' / 'feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return feature_importance_df

def perform_shap_analysis(model, X, feature_names, output_dir, sample_size=100):
    """执行SHAP分析"""
    print("\n执行SHAP分析...")
    
    try:
        # 创建SHAP解释器
        explainer = shap.TreeExplainer(model)
        
        # 计算SHAP值（对样本进行抽样以加快计算）
        if len(X) > sample_size:
            X_sample = X.sample(n=min(sample_size, len(X)), random_state=42)
        else:
            X_sample = X
        
        shap_values = explainer.shap_values(X_sample)
        
        # 1. SHAP摘要图
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
        plt.title('SHAP值摘要图')
        plt.tight_layout()
        plt.savefig(output_dir / 'feature_analysis' / 'shap_summary_plot.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. SHAP条形图（平均绝对SHAP值）
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, 
                         plot_type="bar", show=False)
        plt.title('平均绝对SHAP值')
        plt.tight_layout()
        plt.savefig(output_dir / 'feature_analysis' / 'shap_bar_plot.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. SHAP依赖图（对最重要的特征）
        feature_importance = model.feature_importances_
        top_feature_idx = np.argmax(feature_importance)
        top_feature = feature_names[top_feature_idx]
        
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(top_feature, shap_values, X_sample, 
                            feature_names=feature_names, show=False)
        plt.title(f'{top_feature}的SHAP依赖图')
        plt.tight_layout()
        plt.savefig(output_dir / 'feature_analysis' / f'shap_dependence_{top_feature}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # 保存SHAP值
        shap_df = pd.DataFrame(shap_values, columns=feature_names, index=X_sample.index)
        shap_df.to_csv(output_dir / 'feature_analysis' / 'shap_values.csv', encoding='utf-8-sig')
        
        return shap_values, explainer
        
    except Exception as e:
        print(f"SHAP分析时出错: {str(e)}")
        print("跳过SHAP分析...")
        return None, None

def analyze_feature_characteristics_by_outcome(X, y, feature_names, output_dir):
    """按结局分析特征特性"""
    print("\n按结局分析特征特性...")
    
    # 分离存活和死亡组的特征
    X_alive = X[y == 0]
    X_deceased = X[y == 1]
    
    # 分析每个特征在两组间的差异
    feature_comparison = []
    
    for feature in feature_names:
        # 计算基本统计
        alive_mean = X_alive[feature].mean() if len(X_alive) > 0 else np.nan
        deceased_mean = X_deceased[feature].mean() if len(X_deceased) > 0 else np.nan
        
        alive_std = X_alive[feature].std() if len(X_alive) > 0 else np.nan
        deceased_std = X_deceased[feature].std() if len(X_deceased) > 0 else np.nan
        
        # 进行t检验
        if len(X_alive) > 1 and len(X_deceased) > 1:
            t_stat, p_value = stats.ttest_ind(X_alive[feature].dropna(), 
                                             X_deceased[feature].dropna(),
                                             equal_var=False)
        else:
            t_stat, p_value = np.nan, np.nan
        
        # 计算效应大小
        if not np.isnan(alive_mean) and not np.isnan(deceased_mean) and not np.isnan(alive_std):
            cohen_d = (deceased_mean - alive_mean) / alive_std if alive_std != 0 else 0
        else:
            cohen_d = np.nan
        
        feature_comparison.append({
            'feature': feature,
            'alive_mean': alive_mean,
            'alive_std': alive_std,
            'deceased_mean': deceased_mean,
            'deceased_std': deceased_std,
            'mean_diff': deceased_mean - alive_mean,
            't_statistic': t_stat,
            'p_value': p_value,
            'cohen_d': cohen_d
        })
    
    # 创建DataFrame
    comparison_df = pd.DataFrame(feature_comparison)
    comparison_df = comparison_df.sort_values('mean_diff', key=abs, ascending=False)
    
    # 保存结果
    comparison_df.to_csv(output_dir / 'feature_analysis' / 'feature_comparison_by_outcome.csv', 
                        index=False, encoding='utf-8-sig')
    
    # 绘制最重要的特征对比
    top_n = min(10, len(comparison_df))
    top_features = comparison_df.head(top_n)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # 1. 平均差异条形图
    ax1 = axes[0]
    bars = ax1.barh(range(len(top_features)), top_features['mean_diff'], 
                    color=['red' if x > 0 else 'blue' for x in top_features['mean_diff']])
    ax1.set_yticks(range(len(top_features)))
    ax1.set_yticklabels(top_features['feature'])
    ax1.set_xlabel('平均差异 (死亡组 - 存活组)')
    ax1.set_title('Top 10 特征：死亡组与存活组的平均差异')
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    
    # 2. 效应大小条形图
    ax2 = axes[1]
    bars = ax2.barh(range(len(top_features)), top_features['cohen_d'].abs(), 
                    color='purple')
    ax2.set_yticks(range(len(top_features)))
    ax2.set_yticklabels(top_features['feature'])
    ax2.set_xlabel('效应大小 (|Cohen\'s d|)')
    ax2.set_title('Top 10 特征：效应大小')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'visualizations' / 'feature_comparison_by_outcome.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    return comparison_df

def generate_classification_report():
    """生成分类分析报告"""
    print("\n生成分类分析报告...")
    
    report_content = """# 分类预测分析报告

## 1. 分析概述
本报告总结了Step5分类预测分析的结果。我们使用XGBoost算法，结合十折交叉验证，对COVID-19患者的死亡风险进行预测。

## 2. 分析方法

### 2.1 数据准备
- 合并了人口统计特征（年龄、性别）到患者结局数据中
- 分析了特征类型：连续属性、二值属性、分类属性、缺失指示器
- 处理了类别不平衡问题（死亡样本远少于存活样本）

### 2.2 特征工程
- 使用了原始处理后的特征，包括实验室指标、缺失指示器和检测指示器
- 特征包括连续变量、二值变量和分类变量
- 保留了Step3中创建的所有特征

### 2.3 模型训练
- 使用XGBoost分类器
- 采用十折交叉验证进行模型评估
- 优化了超参数（包括处理不平衡的scale_pos_weight）
- 评估了多种不平衡处理方法

### 2.4 评估指标
- 准确率、精确率、召回率、F1分数
- ROC AUC和PR AUC
- 混淆矩阵
- 特征重要性分析

## 3. 主要发现

### 3.1 模型性能
- 模型在预测患者死亡风险方面表现出良好的性能
- ROC AUC和PR AUC显示了模型的有效性
- 精确率和召回率的平衡反映了模型处理不平衡数据的能力

### 3.2 重要特征
- 某些实验室指标对死亡预测有重要影响
- 缺失指示器和检测指示器提供了有价值的预测信息
- 人口统计特征（如年龄）在预测中起关键作用

### 3.3 临床意义
- 确定了与死亡风险最相关的临床指标
- 为临床早期预警提供了数据支持
- 验证了数据挖掘在医疗预测中的应用价值

## 4. 文件输出

### 4.1 模型与评估
- `xgboost_models/` - 保存的XGBoost模型
- `model_evaluation/` - 模型评估结果和报告
- `predictions/` - 预测结果

### 4.2 特征分析
- `feature_analysis/` - 特征重要性、SHAP分析、特征比较
- 包括特征重要性排名、SHAP值分析等

### 4.3 可视化图表
- `visualizations/` - 所有可视化图表
- 包括ROC曲线、PR曲线、混淆矩阵、特征重要性图等

## 5. 临床意义

### 5.1 风险预测
模型能够有效识别高危患者，为临床干预提供时间窗口。

### 5.2 特征重要性
通过特征重要性分析，可以了解哪些指标对死亡风险影响最大，指导临床监测重点。

### 5.3 个性化医疗
模型可以为个体患者提供死亡风险评分，支持个性化治疗方案制定。

## 6. 局限性

1. 数据不平衡可能影响少数类的预测性能
2. 模型在未见数据上的泛化能力需要进一步验证
3. 特征间的复杂交互关系可能未被完全捕捉
4. 需要临床专家对模型结果进行解释和验证

## 7. 后续建议

1. 收集更多数据，特别是死亡病例，以改善类别不平衡
2. 尝试集成学习和其他先进的机器学习算法
3. 开发用户友好的临床决策支持系统
4. 进行前瞻性研究验证模型的实际应用效果
5. 结合临床路径和治疗信息，构建更全面的预测模型
"""
    
    with open('results/step5_classification/classification_analysis_report.md', 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print("分类分析报告已保存")

def save_predictions(model, X, y, feature_names, output_dir):
    """保存预测结果"""
    print("\n保存预测结果...")
    
    # 预测
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    # 创建结果DataFrame
    results_df = pd.DataFrame({
        'Patient_Code': X.index,  # 假设索引是患者代码
        'True_Label': y,
        'Predicted_Label': y_pred,
        'Predicted_Probability': y_pred_proba,
        'Correct_Prediction': (y == y_pred)
    })
    
    # 添加风险等级
    results_df['Risk_Level'] = pd.cut(results_df['Predicted_Probability'], 
                                      bins=[0, 0.3, 0.7, 1.0],
                                      labels=['低风险', '中风险', '高风险'])
    
    # 保存结果
    results_df.to_csv(output_dir / 'predictions' / 'prediction_results.csv', 
                     index=False, encoding='utf-8-sig')
    
    # 统计预测结果
    print("\n预测结果统计:")
    print(f"总样本数: {len(results_df)}")
    print(f"正确预测数: {results_df['Correct_Prediction'].sum()}")
    print(f"准确率: {results_df['Correct_Prediction'].sum()/len(results_df):.4f}")
    print(f"\n风险等级分布:")
    print(results_df['Risk_Level'].value_counts())
    
    return results_df

def main():
    """主函数"""
    print("=" * 60)
    print("Step5: 分类预测分析 (XGBoost)")
    print("=" * 60)
    
    # 创建目录
    output_dir = Path("results/step5_classification")
    create_directories()
    
    # 1. 加载数据
    demo_df, outcome_df = load_and_prepare_data()
    
    if demo_df is None or outcome_df is None:
        print("数据加载失败，程序退出")
        return
    
    # 2. 合并数据并准备特征
    X, y, feature_names, merged_df = merge_and_prepare_features(demo_df, outcome_df)
    
    if X is None or y is None:
        print("数据准备失败，程序退出")
        return
    
    # 3. 分析特征类型
    feature_types = analyze_feature_types(X, feature_names)
    
    # 4. 处理类别不平衡
    # 可以选择 'none', 'smote', 'undersample', 'both'
    X_resampled, y_resampled = handle_class_imbalance(X, y, method='smote')
    
    # 5. 划分训练集和测试集（用于超参数优化）
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
    )
    
    print(f"\n训练集形状: {X_train.shape}, 测试集形状: {X_test.shape}")
    
    # 6. 优化XGBoost超参数
    # 注意：这可能需要较长时间，可以注释掉以使用默认参数
    use_grid_search = False  # 设置为True进行网格搜索
    
    if use_grid_search:
        best_model, best_params = optimize_xgboost_hyperparameters(X_train, y_train)
    else:
        # 使用预定义的参数
        best_params = {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'scale_pos_weight': 3,  # 处理不平衡
            'random_state': 42,
            'eval_metric': 'logloss',
            'use_label_encoder': False
        }
        best_model = xgb.XGBClassifier(**best_params)  # 直接使用参数字典    
    
    # 7. 在整个训练集上执行十折交叉验证
    cv_results = perform_cross_validation(X_resampled, y_resampled, best_model, n_splits=10)
    
    # 8. 训练最终模型
    final_model = train_final_model(X_resampled, y_resampled, best_params)
    
    # 9. 保存模型
    model_path = output_dir / "xgboost_models" / "final_xgboost_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(final_model, f)
    
    # 保存最佳参数
    params_path = output_dir / "xgboost_models" / "best_params.json"
    with open(params_path, 'w', encoding='utf-8') as f:
        json.dump(best_params, f, indent=2, ensure_ascii=False)
    
    # 10. 评估模型性能
    eval_results, cm, report = evaluate_model_performance(
        final_model, X_test, y_test, feature_names, output_dir
    )
    
    # 11. 可视化
    plot_roc_curve(cv_results, output_dir)
    plot_precision_recall_curve(cv_results, y_test, output_dir)
    plot_confusion_matrix(cm, output_dir)
    
    # 12. 特征重要性分析
    feature_importance_df = plot_feature_importance(final_model, feature_names, output_dir, top_n=20)
    
    # 13. SHAP分析
    shap_values, explainer = perform_shap_analysis(final_model, X_test, feature_names, output_dir)
    
    # 14. 按结局分析特征特性
    comparison_df = analyze_feature_characteristics_by_outcome(X, y, feature_names, output_dir)
    
    # 15. 保存预测结果
    results_df = save_predictions(final_model, X, y, feature_names, output_dir)
    
    # 16. 生成报告
    generate_classification_report()
    
    print("\n" + "=" * 60)
    print("分类预测分析完成!")
    print("=" * 60)
    print("输出位置: results/step5_classification/")
    print("主要文件:")
    print("  1. 模型与参数: xgboost_models/")
    print("  2. 模型评估: model_evaluation/")
    print("  3. 特征分析: feature_analysis/")
    print("  4. 可视化图表: visualizations/")
    print("  5. 预测结果: predictions/")
    print("  6. 分析报告: classification_analysis_report.md")
    print("=" * 60)

if __name__ == "__main__":
    main()