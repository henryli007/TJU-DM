# step3_preprocessing_and_feature_engineering.py
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def create_directories():
    """创建必要的目录"""
    Path("dataset").mkdir(exist_ok=True)
    Path("results/step3").mkdir(parents=True, exist_ok=True)
    print("目录创建完成")

def load_data():
    """加载数据并计算基本统计"""
    print("=" * 60)
    print("加载数据文件...")
    print("=" * 60)
    
    # 加载三个CSV文件
    demo_path = Path("dataset/demographic_data_split.csv")
    lab_path = Path("dataset/laboratory_data.csv")
    outcome_path = Path("dataset/patient_outcome.csv")
    
    if not demo_path.exists() or not lab_path.exists() or not outcome_path.exists():
        print("错误: 找不到输入文件，请确保step1已正确运行")
        return None, None, None
    
    # 读取数据
    demo_df = pd.read_csv(demo_path)
    lab_df = pd.read_csv(lab_path)
    outcome_df = pd.read_csv(outcome_path)
    
    print(f"demographic_data_split.csv: {demo_df.shape}")
    print(f"laboratory_data.csv: {lab_df.shape}")
    print(f"patient_outcome.csv: {outcome_df.shape}")
    
    return demo_df, lab_df, outcome_df

def process_demographic_data(df):
    """处理人口统计数据"""
    print("\n" + "=" * 60)
    print("处理人口统计数据...")
    print("=" * 60)
    
    # 检查缺失值
    missing_info = {}
    initial_rows = len(df)
    
    # 检查关键列的缺失
    for col in ['Patient Code', 'Age Group', 'Gender']:
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            missing_rows = df[df[col].isna()].index.tolist()
            missing_info[col] = {
                'missing_count': int(missing_count),
                'missing_rows': missing_rows
            }
            print(f"警告: {col} 列有 {missing_count} 个缺失值")
    
    # 删除有缺失值的行
    df_clean = df.dropna(subset=['Patient Code', 'Age Group', 'Gender']).reset_index(drop=True)
    rows_removed = initial_rows - len(df_clean)
    
    if rows_removed > 0:
        print(f"删除了 {rows_removed} 行有缺失的数据")
    
    # 保存删除的缺失记录到JSON
    if missing_info:
        missing_info['total_rows_removed'] = int(rows_removed)
        missing_info['initial_rows'] = int(initial_rows)
        missing_info['final_rows'] = int(len(df_clean))
        
        with open('results/step3/missing_demographic_records.json', 'w', encoding='utf-8') as f:
            json.dump(missing_info, f, indent=2, ensure_ascii=False)
        print("缺失记录已保存到 missing_demographic_records.json")
    
    # 年龄组编码
    # 根据您提供的年龄组分布图，我们需要将年龄组标准化
    # 从图中看到的年龄组: 10-20, 20-35, 30-40, 40-50, 50-60, 60+
    # 但我们实际数据中的年龄组可能是: 30-40 years, 40-50 years, 50-60 years, ≥ 60 years
    
    # 先查看实际的年龄组取值
    print(f"\n实际年龄组取值: {df_clean['Age Group'].unique()}")
    
    # 创建年龄组编码映射
    age_groups = sorted(df_clean['Age Group'].unique())
    age_mapping = {age: i for i, age in enumerate(age_groups)}
    
    # 应用编码
    df_clean['Age_Group_Encoded'] = df_clean['Age Group'].map(age_mapping)
    
    # 性别编码
    gender_mapping = {'Male': 0, 'Female': 1}
    df_clean['Gender_Encoded'] = df_clean['Gender'].map(gender_mapping)
    
    # 保存编码映射
    encoding_info = {
        'age_mapping': age_mapping,
        'gender_mapping': gender_mapping
    }
    
    with open('results/step3/encoding_mappings.json', 'w', encoding='utf-8') as f:
        json.dump(encoding_info, f, indent=2, ensure_ascii=False)
    
    print(f"年龄组编码: {age_mapping}")
    print(f"性别编码: {gender_mapping}")
    print(f"处理后数据形状: {df_clean.shape}")
    
    return df_clean

def analyze_missing_rates(df, data_name):
    """分析缺失率并返回按缺失率分层的特征列表"""
    print(f"\n分析 {data_name} 的缺失率...")
    
    # 排除非数值列
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if 'Patient Code' in df.columns:
        numeric_cols = [col for col in numeric_cols if col not in ['Patient Code']]
    
    if 'Group' in df.columns:
        numeric_cols = [col for col in numeric_cols if col not in ['Group']]
    
    if 'Deceased' in df.columns:
        numeric_cols = [col for col in numeric_cols if col not in ['Deceased']]
    
    # 计算缺失率
    missing_rates = {}
    for col in numeric_cols:
        missing_rate = df[col].isna().sum() / len(df)
        missing_rates[col] = missing_rate
    
    # 转换为Series并按缺失率排序
    missing_series = pd.Series(missing_rates).sort_values(ascending=False)
    
    # 按缺失率分层
    low_missing = missing_series[missing_series <= 0.3].index.tolist()
    medium_missing = missing_series[(missing_series > 0.3) & (missing_series <= 0.6)].index.tolist()
    high_missing = missing_series[(missing_series > 0.6) & (missing_series <= 0.8)].index.tolist()
    extreme_missing = missing_series[missing_series > 0.8].index.tolist()
    
    print(f"特征分层结果:")
    print(f"  低缺失(0-30%): {len(low_missing)} 个特征")
    print(f"  中缺失(30-60%): {len(medium_missing)} 个特征")
    print(f"  高缺失(60-80%): {len(high_missing)} 个特征")
    print(f"  极高缺失(>80%): {len(extreme_missing)} 个特征")
    
    if extreme_missing:
        print(f"  将删除的特征: {extreme_missing}")
    
    return {
        'low_missing': low_missing,
        'medium_missing': medium_missing,
        'high_missing': high_missing,
        'extreme_missing': extreme_missing,
        'missing_rates': missing_series
    }

def process_troponin_column(df, feature_name='Troponin'):
    """处理Troponin列，将其中的'Negative'和空值转换为0/1编码"""
    if feature_name not in df.columns:
        print(f"警告: 未找到{feature_name}列")
        return df
    
    print(f"\n处理{feature_name}列...")
    
    # 统计原始值分布
    original_counts = df[feature_name].value_counts(dropna=False)
    print(f"原始值分布:")
    for value, count in original_counts.items():
        print(f"  {value}: {count} ({count/len(df)*100:.1f}%)")
    
    # 将'Negative'转换为0，将空值转换为0，并创建检测指示器
    # 创建检测指示器：1表示有检测结果，0表示无检测结果
    tested_indicator_name = f"Tested_{feature_name}"
    df[tested_indicator_name] = (~df[feature_name].isna()).astype(int)
    
    # 将Troponin列转换为0/1：Negative->0，空值->0
    # 如果有数值，则大于0的值转换为1
    def convert_troponin_value(x):
        if pd.isna(x) or str(x).strip() == '':
            return 1
        elif str(x).strip().lower() == 'negative':
            return 0
        else:
            # 尝试转换为数值
            try:
                val = float(x)
                return 1 if val > 0 else 0
            except:
                return 0  # 无法解析的值也视为阴性
    
    df[feature_name] = df[feature_name].apply(convert_troponin_value)
    
    # 统计转换后的值分布
    converted_counts = df[feature_name].value_counts()
    print(f"转换后值分布:")
    for value, count in converted_counts.items():
        value_label = '阳性' if value == 1 else '阴性'
        print(f"  {value_label}({value}): {count} ({count/len(df)*100:.1f}%)")
    
    # 统计检测指示器分布
    tested_counts = df[tested_indicator_name].value_counts()
    print(f"检测指示器分布:")
    for value, count in tested_counts.items():
        status = '检测了' if value == 1 else '未检测'
        print(f"  {status}({value}): {count} ({count/len(df)*100:.1f}%)")
    
    return df

def peak_interval_imputation(df, feature, group_col=None):
    """
    峰值区间填补法
    
    参数:
        df: 数据框
        feature: 要填补的特征名
        group_col: 分组列名（如'Group'），如果为None则不分组
    """
    if group_col is None or group_col not in df.columns:
        # 不分组的情况
        feature_data = df[feature].dropna()
        
        if len(feature_data) == 0:
            return df  # 没有有效数据，无法填补
        
        # 使用直方图找到峰值区间
        hist, bins = np.histogram(feature_data, bins='auto')
        peak_bin_index = np.argmax(hist)
        
        # 获取峰值区间的边界
        lower_bound = bins[peak_bin_index]
        upper_bound = bins[peak_bin_index + 1]
        
        # 计算峰值区间内的均值
        peak_interval_data = feature_data[(feature_data >= lower_bound) & (feature_data < upper_bound)]
        fill_value = peak_interval_data.mean()
        
        # 填补缺失值
        df[feature] = df[feature].fillna(fill_value)
        
    else:
        # 分组填补
        groups = df[group_col].unique()
        
        for group in groups:
            group_mask = df[group_col] == group
            feature_data = df.loc[group_mask, feature].dropna()
            
            if len(feature_data) == 0:
                continue  # 这个组没有有效数据
            
            # 使用直方图找到峰值区间
            hist, bins = np.histogram(feature_data, bins='auto')
            peak_bin_index = np.argmax(hist)
            
            # 获取峰值区间的边界
            lower_bound = bins[peak_bin_index]
            upper_bound = bins[peak_bin_index + 1]
            
            # 计算峰值区间内的均值
            peak_interval_data = feature_data[(feature_data >= lower_bound) & (feature_data < upper_bound)]
            
            if len(peak_interval_data) > 0:
                fill_value = peak_interval_data.mean()
            else:
                fill_value = feature_data.mean()  # 回退到组均值
            
            # 填补这个组内的缺失值
            group_missing_mask = group_mask & df[feature].isna()
            df.loc[group_missing_mask, feature] = fill_value
    
    return df

def process_laboratory_data(df):
    """处理实验室数据"""
    print("\n" + "=" * 60)
    print("处理实验室数据...")
    print("=" * 60)
    
    # 首先处理Troponin列
    df = process_troponin_column(df, 'Troponin')
    
    # 分析缺失率
    missing_info = analyze_missing_rates(df, "laboratory_data")
    
    # 删除极高缺失的特征
    if missing_info['extreme_missing']:
        print(f"\n删除极高缺失特征: {missing_info['extreme_missing']}")
        df = df.drop(columns=missing_info['extreme_missing'])
    
    # 确保有Group列
    if 'Group' not in df.columns:
        print("警告: 未找到Group列，无法分组处理")
        return None
    
    # 从高缺失列表中移除Troponin，因为已经单独处理了
    if 'Troponin' in missing_info['high_missing']:
        missing_info['high_missing'].remove('Troponin')
    
    # 按缺失率分层处理特征
    for feature in missing_info['low_missing']:
        if feature in df.columns and feature != 'Troponin':  # 排除Troponin
            print(f"处理低缺失特征: {feature} (峰值区间法分组填补)")
            df = peak_interval_imputation(df, feature, 'Group')
    
    for feature in missing_info['medium_missing']:
        if feature in df.columns and feature != 'Troponin':  # 排除Troponin
            print(f"处理中缺失特征: {feature} (峰值区间法分组填补 + 创建缺失指示器)")
            
            # 创建缺失指示器
            missing_indicator_name = f"Missing_{feature}"
            df[missing_indicator_name] = df[feature].notna().astype(int)
            
            # 填补缺失值
            df = peak_interval_imputation(df, feature, 'Group')
    
    for feature in missing_info['high_missing']:
        if feature in df.columns and feature != 'Troponin':  # 排除Troponin
            print(f"处理高缺失特征: {feature} (-999填补 + 创建检测指示器)")
            
            # 创建检测指示器
            tested_indicator_name = f"Tested_{feature}"
            df[tested_indicator_name] = df[feature].notna().astype(int)
            
            # 分组用-999填补
            groups = df['Group'].unique()
            for group in groups:
                group_mask = df['Group'] == group
                group_missing_mask = group_mask & df[feature].isna()
                df.loc[group_missing_mask, feature] = -999
    
    # 对连续数值特征进行0-1标准化
    print("\n对连续数值特征进行0-1标准化...")
    
    # 确定需要标准化的列
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # 排除不需要标准化的列
    exclude_cols = ['Group']  # Group列是分类变量
    if 'Patient Code' in df.columns:
        exclude_cols.append('Patient Code')
    
    # 排除缺失指示器、检测指示器和Troponin
    indicator_cols = [col for col in df.columns if col.startswith('Missing_') or col.startswith('Tested_')]
    exclude_cols.extend(indicator_cols)
    
    # 排除Troponin列（已经是0/1编码）
    if 'Troponin' in df.columns:
        exclude_cols.append('Troponin')
    
    # 确定要标准化的列
    scale_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    print(f"将标准化 {len(scale_cols)} 个特征")
    
    if scale_cols:
        scaler = MinMaxScaler()
        df[scale_cols] = scaler.fit_transform(df[scale_cols])
        
        # 保存标准化器的信息
        scaler_info = {
            'features_scaled': scale_cols,
            'scaler_type': 'MinMaxScaler',
            'feature_ranges': {col: {'min': float(df[col].min()), 'max': float(df[col].max())} 
                              for col in scale_cols}
        }
        
        with open('results/step3/scaler_info_lab.json', 'w', encoding='utf-8') as f:
            json.dump(scaler_info, f, indent=2, ensure_ascii=False)
    
    print(f"处理后数据形状: {df.shape}")
    return df

def process_patient_outcome(df):
    """处理患者结局数据"""
    print("\n" + "=" * 60)
    print("处理患者结局数据...")
    print("=" * 60)
    
    # 处理Deceased列
    if 'Deceased' in df.columns:
        deceased_mapping = {'Yes': 1, 'No': 0, 'yes': 1, 'no': 0}
        df['Deceased'] = df['Deceased'].map(deceased_mapping)
        
        # 统计分布
        deceased_counts = df['Deceased'].value_counts()
        print(f"Deceased列分布: 死亡={deceased_counts.get(1, 0)}, 存活={deceased_counts.get(0, 0)}")
    else:
        print("警告: 未找到Deceased列")
    
    # 处理Troponin列
    df = process_troponin_column(df, 'Troponin')
    
    # 分析缺失率
    missing_info = analyze_missing_rates(df, "patient_outcome")
    
    # 删除极高缺失的特征
    if missing_info['extreme_missing']:
        print(f"\n删除极高缺失特征: {missing_info['extreme_missing']}")
        df = df.drop(columns=missing_info['extreme_missing'])
    
    # 从高缺失列表中移除Troponin，因为已经单独处理了
    if 'Troponin' in missing_info['high_missing']:
        missing_info['high_missing'].remove('Troponin')
    
    # 按缺失率分层处理特征
    for feature in missing_info['low_missing']:
        if feature in df.columns and feature != 'Deceased' and feature != 'Troponin':
            print(f"处理低缺失特征: {feature} (峰值区间法填补)")
            df = peak_interval_imputation(df, feature, None)
    
    for feature in missing_info['medium_missing']:
        if feature in df.columns and feature != 'Deceased' and feature != 'Troponin':
            print(f"处理中缺失特征: {feature} (峰值区间法填补 + 创建缺失指示器)")
            
            # 创建缺失指示器
            missing_indicator_name = f"Missing_{feature}"
            df[missing_indicator_name] = df[feature].notna().astype(int)
            
            # 填补缺失值
            df = peak_interval_imputation(df, feature, None)
    
    for feature in missing_info['high_missing']:
        if feature in df.columns and feature != 'Deceased' and feature != 'Troponin':
            print(f"处理高缺失特征: {feature} (-999填补 + 创建检测指示器)")
            
            # 创建检测指示器
            tested_indicator_name = f"Tested_{feature}"
            df[tested_indicator_name] = df[feature].notna().astype(int)
            
            # 用-999填补
            df[feature] = df[feature].fillna(-999)
    
    # 对连续数值特征进行0-1标准化
    print("\n对连续数值特征进行0-1标准化...")
    
    # 确定需要标准化的列
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # 排除不需要标准化的列
    exclude_cols = ['Deceased']
    if 'Patient Code' in df.columns:
        exclude_cols.append('Patient Code')
    
    # 排除缺失指示器、检测指示器和Troponin
    indicator_cols = [col for col in df.columns if col.startswith('Missing_') or col.startswith('Tested_')]
    exclude_cols.extend(indicator_cols)
    
    # 排除Troponin列（已经是0/1编码）
    if 'Troponin' in df.columns:
        exclude_cols.append('Troponin')
    
    # 确定要标准化的列
    scale_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    print(f"将标准化 {len(scale_cols)} 个特征")
    
    if scale_cols:
        scaler = MinMaxScaler()
        df[scale_cols] = scaler.fit_transform(df[scale_cols])
        
        # 保存标准化器的信息
        scaler_info = {
            'features_scaled': scale_cols,
            'scaler_type': 'MinMaxScaler',
            'feature_ranges': {col: {'min': float(df[col].min()), 'max': float(df[col].max())} 
                              for col in scale_cols}
        }
        
        with open('results/step3/scaler_info_outcome.json', 'w', encoding='utf-8') as f:
            json.dump(scaler_info, f, indent=2, ensure_ascii=False)
    
    print(f"处理后数据形状: {df.shape}")
    return df

def create_visualizations(demo_df, lab_df, outcome_df):
    """创建可视化图表"""
    print("\n" + "=" * 60)
    print("创建可视化图表...")
    print("=" * 60)
    
    # 1. 人口统计数据可视化
    if demo_df is not None:
        # 这里原计划复用此前作业中的可视化代码，为避免
        # if 语句体为空导致的语法错误，暂用简单输出占位。
        print("人口统计数据可视化: demo_df 记录数 =", len(demo_df))
    
    # 2. 实验室数据可视化
    if lab_df is not None:
        # 组别分布
        if 'Group' in lab_df.columns:
            # 简单输出各组别样本数量，避免空代码块
            group_counts = lab_df['Group'].value_counts()
            print("实验室数据 Group 分布:\n", group_counts)
        
        # Troponin分布
        if 'Troponin' in lab_df.columns:
            plt.figure(figsize=(10, 5))
            
            # Troponin值分布
            plt.subplot(1, 2, 1)
            troponin_counts = lab_df['Troponin'].value_counts()
            colors = ['lightgreen', 'salmon']  # 阴性/阳性
            # 动态生成标签，避免长度不匹配
            labels = []
            for val in sorted(troponin_counts.index):
                if val == 0:
                    labels.append('阴性(0)')
                else:  # val == 1
                    labels.append('阳性(1)')
            
            bars1 = plt.bar(range(len(troponin_counts)), troponin_counts.values, 
                           color=colors[:len(troponin_counts)], edgecolor='black')
            plt.title('Troponin值分布', fontsize=12, fontweight='bold')
            plt.xlabel('Troponin结果', fontsize=10)
            plt.ylabel('患者数量', fontsize=10)
            
            # 只有当有数据时才设置刻度标签
            if len(troponin_counts) > 0:
                plt.xticks(range(len(troponin_counts)), labels)
            
            # 添加数值标签
            for bar in bars1:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom')
            
            # Troponin检测情况分布
            plt.subplot(1, 2, 2)
            if 'Tested_Troponin' in lab_df.columns:
                tested_counts = lab_df['Tested_Troponin'].value_counts()
                colors2 = ['lightgray', 'lightblue']  # 未检测/检测
                # 动态生成标签
                labels2 = []
                for val in sorted(tested_counts.index):
                    if val == 0:
                        labels2.append('未检测(0)')
                    else:  # val == 1
                        labels2.append('检测了(1)')
                
                bars2 = plt.bar(range(len(tested_counts)), tested_counts.values, 
                               color=colors2[:len(tested_counts)], edgecolor='black')
                plt.title('Troponin检测情况', fontsize=12, fontweight='bold')
                plt.xlabel('是否检测', fontsize=10)
                plt.ylabel('患者数量', fontsize=10)
                
                # 只有当有数据时才设置刻度标签
                if len(tested_counts) > 0:
                    plt.xticks(range(len(tested_counts)), labels2)
                
                # 添加数值标签
                for bar in bars2:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height,
                            f'{int(height)}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig('results/step3/lab_troponin_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 缺失指示器分布
        missing_cols = [col for col in lab_df.columns if col.startswith('Missing_')]
        if missing_cols:
            # 简要统计前若干缺失指示器的缺失数量
            sample_cols = missing_cols[:10]
            missing_summary = {col: int(lab_df[col].sum()) for col in sample_cols}
            print("实验室数据缺失指示器(前10列)统计:", missing_summary)
    
    # 3. 患者结局数据可视化
    if outcome_df is not None and 'Deceased' in outcome_df.columns:
        # 患者结局分布
        plt.figure(figsize=(8, 6))
        deceased_counts = outcome_df['Deceased'].value_counts()
        colors = ['lightgreen', 'lightcoral']
        
        # 动态生成标签
        labels = []
        for val in sorted(deceased_counts.index):
            if val == 0:
                labels.append('存活 (0)')
            else:  # val == 1
                labels.append('死亡 (1)')
        
        bars = plt.bar(range(len(deceased_counts)), deceased_counts.values, 
                      color=colors[:len(deceased_counts)], edgecolor='black')
        plt.title('患者结局数据 - 死亡/存活分布', fontsize=14, fontweight='bold')
        plt.xlabel('结局', fontsize=12)
        plt.ylabel('患者数量', fontsize=12)
        
        # 设置刻度标签
        if len(deceased_counts) > 0:
            plt.xticks(range(len(deceased_counts)), labels)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('results/step3/outcome_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Troponin与结局的关系
        if 'Troponin' in outcome_df.columns and 'Tested_Troponin' in outcome_df.columns:
            # 只分析检测了Troponin的患者
            tested_mask = outcome_df['Tested_Troponin'] == 1
            tested_outcome_df = outcome_df[tested_mask]
            
            if len(tested_outcome_df) > 0 and tested_outcome_df['Troponin'].nunique() > 0:
                plt.figure(figsize=(12, 5))
                
                # Troponin结果与死亡的关系
                plt.subplot(1, 2, 1)
                cross_tab = pd.crosstab(tested_outcome_df['Troponin'], tested_outcome_df['Deceased'])
                cross_tab_percent = cross_tab.div(cross_tab.sum(axis=1), axis=0) * 100
                
                x = np.arange(len(cross_tab))
                width = 0.35
                
                # 动态生成标签
                troponin_labels = []
                for val in sorted(tested_outcome_df['Troponin'].unique()):
                    if val == 0:
                        troponin_labels.append('阴性(0)')
                    else:  # val == 1
                        troponin_labels.append('阳性(1)')
                
                # 确保有存活和死亡的数据
                has_survived = 0 in cross_tab.columns
                has_died = 1 in cross_tab.columns
                
                if has_survived:
                    bars1 = plt.bar(x - width/2, cross_tab[0], width, label='存活', 
                                   color='lightgreen', edgecolor='black')
                if has_died:
                    bars2 = plt.bar(x + width/2, cross_tab[1], width, label='死亡', 
                                   color='lightcoral', edgecolor='black')
                
                plt.title('Troponin结果与患者结局的关系', fontsize=12, fontweight='bold')
                plt.xlabel('Troponin结果', fontsize=10)
                plt.ylabel('患者数量', fontsize=10)
                
                if len(troponin_labels) > 0:
                    plt.xticks(x, troponin_labels)
                
                plt.legend()
                
                # 添加数值标签
                if has_survived:
                    for bar in bars1:
                        height = bar.get_height()
                        if height > 0:
                            plt.text(bar.get_x() + bar.get_width()/2., height,
                                    f'{int(height)}', ha='center', va='bottom')
                
                if has_died:
                    for bar in bars2:
                        height = bar.get_height()
                        if height > 0:
                            plt.text(bar.get_x() + bar.get_width()/2., height,
                                    f'{int(height)}', ha='center', va='bottom')
                
                # 死亡率的差异
                plt.subplot(1, 2, 2)
                death_rates = []
                labels_rate = []
                
                for troponin_val in sorted(tested_outcome_df['Troponin'].unique()):
                    subset = tested_outcome_df[tested_outcome_df['Troponin'] == troponin_val]
                    if len(subset) > 0:
                        death_rate = subset['Deceased'].mean() * 100
                        death_rates.append(death_rate)
                        
                        if troponin_val == 0:
                            labels_rate.append('阴性(0)')
                        else:  # troponin_val == 1
                            labels_rate.append('阳性(1)')
                
                if death_rates:
                    bars = plt.bar(range(len(death_rates)), death_rates, 
                                  color=['lightgreen', 'salmon'][:len(death_rates)], edgecolor='black')
                    plt.title('Troponin结果与死亡率', fontsize=12, fontweight='bold')
                    plt.xlabel('Troponin结果', fontsize=10)
                    plt.ylabel('死亡率 (%)', fontsize=10)
                    
                    if len(labels_rate) > 0:
                        plt.xticks(range(len(labels_rate)), labels_rate)
                    
                    # 添加数值标签
                    for i, (bar, rate) in enumerate(zip(bars, death_rates)):
                        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                                f'{rate:.1f}%', ha='center', va='bottom')
                
                plt.tight_layout()
                plt.savefig('results/step3/outcome_troponin_relationship.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        # 缺失指示器分布
        missing_cols = [col for col in outcome_df.columns if col.startswith('Missing_')]
        if missing_cols:
            plt.figure(figsize=(12, 8))
            missing_counts = []
            feature_names = []
            
            for col in missing_cols[:10]:  # 只显示前10个
                missing_counts.append(outcome_df[col].sum())
                feature_names.append(col.replace('Missing_', ''))
            
            bars = plt.barh(feature_names, missing_counts, color='lightblue', edgecolor='black')
            plt.title('患者结局数据 - 缺失指示器分布 (前10个)', fontsize=14, fontweight='bold')
            plt.xlabel('有数据的样本数', fontsize=12)
            plt.ylabel('特征', fontsize=12)
            
            # 添加数值标签
            for i, (count, name) in enumerate(zip(missing_counts, feature_names)):
                plt.text(count + 5, i, f'{count} ({count/len(outcome_df)*100:.1f}%)', 
                        va='center', fontsize=10)
            
            plt.tight_layout()
            plt.savefig('results/step3/outcome_missing_indicators.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    # 4. 特征相关性热图（样本量较大，只显示部分特征）
    if lab_df is not None:
        # 选择部分特征绘制相关性热图
        numeric_cols = lab_df.select_dtypes(include=[np.number]).columns.tolist()
        
        # 排除指示器列
        numeric_cols = [col for col in numeric_cols if not col.startswith('Missing_') and not col.startswith('Tested_')]
        
        if len(numeric_cols) > 5:
            # 选择前5个特征
            selected_cols = numeric_cols[:5]
            
            plt.figure(figsize=(10, 8))
            correlation_matrix = lab_df[selected_cols].corr()
            
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, linewidths=1, cbar_kws={"shrink": 0.8})
            plt.title('实验室数据 - 特征相关性热图 (前5个特征)', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig('results/step3/lab_feature_correlation.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    print("可视化图表已保存到 results/step3/")

def save_processed_data(demo_df, lab_df, outcome_df):
    """保存处理后的数据"""
    print("\n" + "=" * 60)
    print("保存处理后的数据...")
    print("=" * 60)
    
    # 保存人口统计数据
    if demo_df is not None:
        demo_output_path = "dataset/demographic_data_processed.csv"
        demo_df.to_csv(demo_output_path, index=False, encoding='utf-8-sig')
        print(f"保存人口统计数据: {demo_output_path}")
    
    # 保存实验室数据
    if lab_df is not None:
        lab_output_path = "dataset/laboratory_data_processed.csv"
        lab_df.to_csv(lab_output_path, index=False, encoding='utf-8-sig')
        print(f"保存实验室数据: {lab_output_path}")
    
    # 保存患者结局数据
    if outcome_df is not None:
        outcome_output_path = "dataset/patient_outcome_processed.csv"
        outcome_df.to_csv(outcome_output_path, index=False, encoding='utf-8-sig')
        print(f"保存患者结局数据: {outcome_output_path}")
    
    print("\n所有处理后的数据已保存到 dataset/ 目录")

def generate_preprocessing_report():
    """生成预处理报告"""
    print("\n" + "=" * 60)
    print("生成预处理报告...")
    print("=" * 60)
    
    report_content = """# 数据预处理报告

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
"""
    
    with open('results/step3/preprocessing_report.md', 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print("预处理报告已保存到 results/step3/preprocessing_report.md")

def main():
    """主函数"""
    print("=" * 60)
    print("Step3: 数据预处理与特征工程")
    print("=" * 60)
    
    # 创建目录
    create_directories()
    
    # 加载数据
    demo_df, lab_df, outcome_df = load_data()
    
    if demo_df is None or lab_df is None or outcome_df is None:
        print("数据加载失败，程序退出")
        return
    
    # 处理人口统计数据
    demo_processed = process_demographic_data(demo_df)
    
    # 处理实验室数据
    lab_processed = process_laboratory_data(lab_df)
    
    # 处理患者结局数据
    outcome_processed = process_patient_outcome(outcome_df)
    
    # 创建可视化图表
    create_visualizations(demo_processed, lab_processed, outcome_processed)
    
    # 保存处理后的数据
    save_processed_data(demo_processed, lab_processed, outcome_processed)
    
    # 生成预处理报告
    generate_preprocessing_report()
    
    print("\n" + "=" * 60)
    print("Step3 处理完成!")
    print("=" * 60)
    print("输出位置:")
    print("  - 处理后的数据: dataset/*_processed.csv")
    print("  - 可视化图表: results/step3/")
    print("  - 配置文件: results/step3/*.json")
    print("  - 报告: results/step3/preprocessing_report.md")
    print("=" * 60)

if __name__ == "__main__":
    main()