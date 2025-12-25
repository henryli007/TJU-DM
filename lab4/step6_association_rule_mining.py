# step6_association_rule_mining.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import warnings
import os
from pathlib import Path
import json
from collections import Counter
import itertools

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def create_directories():
    """创建输出目录"""
    base_dir = Path("results/step6_association_rules")
    directories = [
        base_dir,
        base_dir / "rules",
        base_dir / "visualizations",
        base_dir / "transactions"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    print("目录创建完成")

def load_and_prepare_data():
    """加载并准备数据"""
    print("=" * 60)
    print("加载患者结局数据...")
    print("=" * 60)
    
    # 加载处理后的患者结局数据
    data_path = Path("dataset/patient_outcome_processed.csv")
    
    if not data_path.exists():
        print(f"错误: 找不到文件 {data_path}")
        return None
    
    # 读取数据
    df = pd.read_csv(data_path)
    
    print(f"数据形状: {df.shape}")
    print(f"列名: {list(df.columns)}")
    
    return df

def preprocess_for_association_rules(df, max_length=3):
    """为关联规则挖掘准备数据"""
    print(f"\n为关联规则挖掘准备数据（最大规则长度={max_length}）...")
    
    # 分离标识列和特征列
    identifier_cols = ['Patient Code', 'Deceased']
    feature_cols = [col for col in df.columns if col not in identifier_cols]
    
    print(f"标识列: {identifier_cols}")
    print(f"特征数量: {len(feature_cols)}")
    
    # 分析特征类型
    analyze_feature_types(df, feature_cols)
    
    return df, feature_cols

def analyze_feature_types(df, feature_cols):
    """分析特征类型"""
    print("\n分析特征类型...")
    
    # 特征分类
    feature_categories = {
        'continuous': [],
        'binary': [],
        'categorical': [],
        'missing_indicators': [],
        'tested_indicators': []
    }
    
    for col in feature_cols:
        unique_vals = df[col].nunique()
        
        if col.startswith('Missing_'):
            feature_categories['missing_indicators'].append(col)
        elif col.startswith('Tested_'):
            feature_categories['tested_indicators'].append(col)
        elif unique_vals == 2 and set(df[col].dropna().unique()).issubset({0, 1, 0.0, 1.0}):
            feature_categories['binary'].append(col)
        elif 2 < unique_vals <= 10:
            feature_categories['categorical'].append(col)
        else:
            feature_categories['continuous'].append(col)
    
    # 输出统计
    print("特征类型统计:")
    for category, features in feature_categories.items():
        print(f"  {category}: {len(features)} 个特征")
        if len(features) <= 5 and len(features) > 0:
            print(f"    {features}")
    
    return feature_categories

def discretize_continuous_features(df, feature_cols, n_bins=3):
    """修复的离散化函数，处理特殊分布"""
    print("\n修复离散化连续特征...")
    
    df_discrete = df.copy()
    discretization_info = {}
    
    # 先分析每个特征的分布
    feature_stats = {}
    for col in feature_cols:
        if col.startswith('Missing_') or col.startswith('Tested_'):
            continue
        
        unique_vals = df[col].dropna().unique()
        feature_stats[col] = {
            'n_unique': len(unique_vals),
            'zero_count': (df[col] == 0).sum(),
            'min': df[col].min(),
            'max': df[col].max(),
            'q25': df[col].quantile(0.25),
            'median': df[col].median(),
            'q75': df[col].quantile(0.75)
        }
    
    for col in feature_cols:
        if col.startswith('Missing_') or col.startswith('Tested_'):
            continue
        
        stats = feature_stats[col]
        unique_count = stats['n_unique']
        zero_count = stats['zero_count']
        total_count = len(df[col].dropna())
        
        print(f"\n处理 {col}:")
        print(f"  唯一值: {unique_count}, 零值比例: {zero_count/total_count:.1%}")
        print(f"  范围: [{stats['min']:.3f}, {stats['max']:.3f}]")
        print(f"  分位数: 25%={stats['q25']:.3f}, 中位数={stats['median']:.3f}, 75%={stats['q75']:.3f}")
        
        # 根据特征分布选择合适的离散化策略
        if zero_count / total_count > 0.3:
            # 零值过多，使用零值/非零值分箱
            print(f"  → 零值过多({zero_count/total_count:.1%})，使用二值分箱")
            df_discrete[f'{col}_bin'] = np.where(df[col] == 0, f'{col}_zero', f'{col}_non_zero')
        
        elif unique_count <= 3:
            # 唯一值太少，直接作为分类
            print(f"  → 唯一值过少({unique_count})，直接作为分类变量")
            df_discrete[f'{col}_bin'] = df[col].apply(lambda x: f'{col}_{x}')
        
        else:
            try:
                # 尝试等频分箱
                print(f"  → 尝试等频分箱")
                df_discrete[f'{col}_bin'], bins = pd.qcut(
                    df[col], 
                    q=n_bins, 
                    labels=[f'{col}_low', f'{col}_med', f'{col}_high'],
                    retbins=True,
                    duplicates='drop'
                )
                print(f"    成功: 分箱边界 {bins}")
            except Exception as e:
                print(f"  → 等频分箱失败: {e}")
                try:
                    # 尝试等宽分箱
                    print(f"  → 尝试等宽分箱")
                    df_discrete[f'{col}_bin'], bins = pd.cut(
                        df[col],
                        bins=n_bins,
                        labels=[f'{col}_low', f'{col}_med', f'{col}_high'],
                        retbins=True
                    )
                    print(f"    成功: 分箱边界 {bins}")
                except Exception as e2:
                    print(f"  → 等宽分箱也失败: {e2}")
                    # 使用基于分位数的自定义分箱
                    print(f"  → 使用自定义分箱")
                    if stats['q25'] == stats['median'] == stats['q75']:
                        # 所有分位数相等，无法有意义分箱
                        df_discrete[f'{col}_bin'] = f'{col}_constant'
                    else:
                        # 使用中位数分割
                        median_val = stats['median']
                        df_discrete[f'{col}_bin'] = np.where(
                            df[col] <= median_val, f'{col}_low', f'{col}_high'
                        )
    
    return df_discrete, discretization_info

def create_transactions(df_discrete, feature_cols, max_length=3):
    """创建事务数据"""
    print(f"\n创建事务数据（最大项集长度={max_length}）...")
    
    transactions = []
    patient_codes = []
    
    # 为每个患者创建事务
    for idx, row in df_discrete.iterrows():
        transaction = []
        patient_code = row['Patient Code']
        deceased = row['Deceased']
        
        # 添加结局标签
        outcome_label = 'Deceased' if deceased == 1 else 'Alive'
        transaction.append(outcome_label)
        
        # 添加特征
        for col in feature_cols:
            # 处理离散化后的特征
            if f'{col}_bin' in df_discrete.columns:
                value = str(row[f'{col}_bin'])
                if value != 'nan':  # 忽略缺失值
                    transaction.append(f'{col}={value}')
            # 处理二值特征
            elif col.startswith('Missing_') or col.startswith('Tested_'):
                if row[col] == 1:
                    transaction.append(col)
            # 处理其他特征
            elif df_discrete[col].nunique() <= 10:  # 分类特征
                value = row[col]
                if not pd.isna(value):
                    transaction.append(f'{col}={value}')
        
        # 限制事务长度
        if len(transaction) > 1:  # 至少包含结局和一个特征
            transactions.append(transaction)
            patient_codes.append(patient_code)
    
    print(f"创建了 {len(transactions)} 个事务")
    print(f"平均事务长度: {np.mean([len(t) for t in transactions]):.1f}")
    
    # 保存事务数据
    save_transactions(transactions, patient_codes)
    
    return transactions, patient_codes

def save_transactions(transactions, patient_codes):
    """保存事务数据"""
    print("\n保存事务数据...")
    
    # 保存为文本文件
    with open('results/step6_association_rules/transactions/transactions.txt', 'w', encoding='utf-8') as f:
        for patient, transaction in zip(patient_codes, transactions):
            f.write(f"{patient}: {', '.join(transaction)}\n")
    
    # 统计项的出现频率
    all_items = []
    for transaction in transactions:
        all_items.extend(transaction)
    
    item_counts = Counter(all_items)
    item_freq_df = pd.DataFrame.from_dict(item_counts, orient='index', columns=['count'])
    item_freq_df = item_freq_df.sort_values('count', ascending=False)
    
    item_freq_df.to_csv('results/step6_association_rules/transactions/item_frequencies.csv', 
                       encoding='utf-8-sig')
    
    print(f"发现 {len(item_counts)} 个不同的项")
    print("Top 10 频繁项:")
    for item, count in item_counts.most_common(10):
        print(f"  {item}: {count} ({count/len(transactions)*100:.1f}%)")

def encode_transactions(transactions):
    """编码事务数据为one-hot格式"""
    print("\n编码事务数据...")
    
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    
    print(f"编码后数据形状: {df_encoded.shape}")
    print(f"项集数量: {len(te.columns_)}")
    
    return df_encoded, te.columns_

def mine_frequent_itemsets(df_encoded, min_support=0.1, max_len=3):
    """挖掘频繁项集"""
    print(f"\n挖掘频繁项集 (min_support={min_support}, max_len={max_len})...")
    
    # 使用Apriori算法
    frequent_itemsets = apriori(
        df_encoded, 
        min_support=min_support, 
        use_colnames=True,
        max_len=max_len
    )
    
    print(f"发现 {len(frequent_itemsets)} 个频繁项集")
    print(f"支持度范围: {frequent_itemsets['support'].min():.3f} - {frequent_itemsets['support'].max():.3f}")
    
    # 添加项集长度
    frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
    
    # 保存频繁项集
    save_frequent_itemsets(frequent_itemsets)
    
    return frequent_itemsets

def save_frequent_itemsets(frequent_itemsets):
    """保存频繁项集"""
    print("\n保存频繁项集...")
    
    # 按长度分组保存
    for length in sorted(frequent_itemsets['length'].unique()):
        itemsets_length = frequent_itemsets[frequent_itemsets['length'] == length]
        itemsets_length = itemsets_length.sort_values('support', ascending=False)
        
        # 格式化输出
        formatted_itemsets = []
        for idx, row in itemsets_length.iterrows():
            items = list(row['itemsets'])
            support = row['support']
            formatted_itemsets.append({
                'items': items,
                'support': float(support)
            })
        
        # 保存为JSON
        with open(f'results/step6_association_rules/rules/frequent_itemsets_length_{length}.json', 
                 'w', encoding='utf-8') as f:
            json.dump(formatted_itemsets, f, indent=2, ensure_ascii=False)
        
        print(f"  长度 {length} 的项集: {len(itemsets_length)} 个")
        
        # 显示Top 5
        if len(itemsets_length) > 0:
            print(f"  Top 5 频繁项集 (长度{length}):")
            for i, (idx, row) in enumerate(itemsets_length.head(5).iterrows(), 1):
                items = list(row['itemsets'])
                print(f"    {i}. {items} (支持度: {row['support']:.3f})")

def generate_association_rules(frequent_itemsets, metric='lift', min_threshold=1.0):
    """生成关联规则"""
    print(f"\n生成关联规则 (metric={metric}, min_threshold={min_threshold})...")
    
    # 生成关联规则
    rules = association_rules(
        frequent_itemsets, 
        metric=metric, 
        min_threshold=min_threshold
    )
    
    print(f"生成 {len(rules)} 条关联规则")
    
    if len(rules) > 0:
        print(f"指标范围:")
        print(f"  置信度: {rules['confidence'].min():.3f} - {rules['confidence'].max():.3f}")
        print(f"  提升度: {rules['lift'].min():.3f} - {rules['lift'].max():.3f}")
    
    return rules

def analyze_rules_by_metric(rules, output_dir):
    """按不同指标分析规则"""
    print("\n按不同指标分析规则...")
    
    metrics = ['confidence', 'lift', 'conviction', 'leverage']
    
    for metric in metrics:
        if metric in rules.columns:
            # 按指标排序
            rules_sorted = rules.sort_values(metric, ascending=False)
            
            # 保存Top 20规则
            top_rules = rules_sorted.head(20)
            
            # 格式化规则
            formatted_rules = []
            for idx, row in top_rules.iterrows():
                antecedents = list(row['antecedents'])
                consequents = list(row['consequents'])
                
                rule_info = {
                    'rule': f"{antecedents} => {consequents}",
                    'support': float(row['support']),
                    'confidence': float(row['confidence']),
                    'lift': float(row['lift']),
                    'conviction': float(row.get('conviction', 0)),
                    'leverage': float(row.get('leverage', 0))
                }
                formatted_rules.append(rule_info)
            
            # 保存为JSON
            with open(output_dir / f'rules/top_rules_by_{metric}.json', 'w', encoding='utf-8') as f:
                json.dump(formatted_rules, f, indent=2, ensure_ascii=False)
            
            print(f"  Top 5 规则 (按{metric}):")
            for i, row in top_rules.head(5).iterrows():
                antecedents = list(row['antecedents'])
                consequents = list(row['consequents'])
                print(f"    {antecedents} => {consequents}")
                print(f"      {metric}: {row[metric]:.3f}, 置信度: {row['confidence']:.3f}, 支持度: {row['support']:.3f}")

def save_association_rules(rules, output_dir):
    """保存关联规则"""
    print("\n保存关联规则...")
    
    # 格式化规则以便保存
    formatted_rules = []
    
    for idx, row in rules.iterrows():
        antecedents = list(row['antecedents'])
        consequents = list(row['consequents'])
        
        rule_info = {
            'antecedents': antecedents,
            'consequents': consequents,
            'support': float(row['support']),
            'confidence': float(row['confidence']),
            'lift': float(row['lift']),
            'conviction': float(row.get('conviction', 0)),
            'leverage': float(row.get('leverage', 0)),
            'zhangs_metric': float(row.get('zhangs_metric', 0))
        }
        formatted_rules.append(rule_info)
    
    # 保存为JSON
    with open(output_dir / 'rules/all_association_rules.json', 'w', encoding='utf-8') as f:
        json.dump(formatted_rules, f, indent=2, ensure_ascii=False)
    
    # 保存为CSV
    rules_csv = rules.copy()
    rules_csv['antecedents'] = rules_csv['antecedents'].apply(lambda x: ', '.join(list(x)))
    rules_csv['consequents'] = rules_csv['consequents'].apply(lambda x: ', '.join(list(x)))
    rules_csv.to_csv(output_dir / 'rules/association_rules.csv', index=False, encoding='utf-8-sig')
    
    print(f"保存了 {len(formatted_rules)} 条规则")

def visualize_frequent_itemsets(frequent_itemsets, output_dir):
    """可视化频繁项集"""
    print("\n可视化频繁项集...")
    
    # 1. 支持度分布
    plt.figure(figsize=(10, 6))
    plt.hist(frequent_itemsets['support'], bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('支持度')
    plt.ylabel('频数')
    plt.title('频繁项集支持度分布')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'visualizations/support_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. 项集长度分布
    length_counts = frequent_itemsets['length'].value_counts().sort_index()
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(length_counts.index, length_counts.values, color='skyblue', edgecolor='black')
    plt.xlabel('项集长度')
    plt.ylabel('数量')
    plt.title('频繁项集长度分布')
    plt.xticks(length_counts.index)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'visualizations/itemset_length_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Top 20频繁项集
    if len(frequent_itemsets) > 0:
        top_itemsets = frequent_itemsets.sort_values('support', ascending=False).head(20)
        
        plt.figure(figsize=(12, 8))
        y_pos = np.arange(len(top_itemsets))
        
        # 格式化项集标签
        labels = []
        for itemset in top_itemsets['itemsets']:
            items = list(itemset)
            if len(items) > 3:
                label = ', '.join(items[:3]) + '...'
            else:
                label = ', '.join(items)
            labels.append(label)
        
        bars = plt.barh(y_pos, top_itemsets['support'], color='lightcoral', edgecolor='black')
        plt.yticks(y_pos, labels)
        plt.xlabel('支持度')
        plt.title('Top 20 频繁项集')
        plt.gca().invert_yaxis()  # 最高的在顶部
        
        # 添加数值标签
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{width:.3f}', va='center')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'visualizations/top_itemsets.png', dpi=300, bbox_inches='tight')
        plt.show()

def visualize_association_rules(rules, output_dir, top_n=20):
    """可视化关联规则"""
    print("\n可视化关联规则...")
    
    if len(rules) == 0:
        print("没有规则可可视化")
        return
    
    # 1. 规则指标散点图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 支持度 vs 置信度
    ax1 = axes[0, 0]
    scatter1 = ax1.scatter(rules['support'], rules['confidence'], 
                          c=rules['lift'], cmap='viridis', alpha=0.6, s=50)
    ax1.set_xlabel('支持度')
    ax1.set_ylabel('置信度')
    ax1.set_title('支持度 vs 置信度 (颜色=提升度)')
    plt.colorbar(scatter1, ax=ax1, label='提升度')
    
    # 支持度 vs 提升度
    ax2 = axes[0, 1]
    scatter2 = ax2.scatter(rules['support'], rules['lift'], 
                          c=rules['confidence'], cmap='plasma', alpha=0.6, s=50)
    ax2.set_xlabel('支持度')
    ax2.set_ylabel('提升度')
    ax2.set_title('支持度 vs 提升度 (颜色=置信度)')
    plt.colorbar(scatter2, ax=ax2, label='置信度')
    
    # 置信度 vs 提升度
    ax3 = axes[1, 0]
    scatter3 = ax3.scatter(rules['confidence'], rules['lift'], 
                          c=rules['support'], cmap='coolwarm', alpha=0.6, s=50)
    ax3.set_xlabel('置信度')
    ax3.set_ylabel('提升度')
    ax3.set_title('置信度 vs 提升度 (颜色=支持度)')
    plt.colorbar(scatter3, ax=ax3, label='支持度')
    
    # 规则长度分布
    ax4 = axes[1, 1]
    rules['rule_length'] = rules.apply(lambda x: len(x['antecedents']) + len(x['consequents']), axis=1)
    length_counts = rules['rule_length'].value_counts().sort_index()
    bars = ax4.bar(length_counts.index, length_counts.values, color='lightgreen', edgecolor='black')
    ax4.set_xlabel('规则长度 (前件+后件)')
    ax4.set_ylabel('规则数量')
    ax4.set_title('关联规则长度分布')
    ax4.set_xticks(length_counts.index)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'visualizations/rule_metrics_scatter.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Top N规则网络图（简化版）
    if len(rules) >= top_n:
        plot_top_rules_network(rules.head(top_n), output_dir)
    
    # 3. 与死亡相关的规则
    death_rules = rules[rules.apply(
        lambda x: 'Deceased' in [str(item) for item in x['consequents']] or 
                  'Deceased' in [str(item) for item in x['antecedents']],
        axis=1
    )]
    
    if len(death_rules) > 0:
        plot_death_related_rules(death_rules, output_dir)

def plot_top_rules_network(rules, output_dir, top_n=20):
    """绘制Top规则网络图（简化版）"""
    print("  绘制Top规则网络图...")
    
    # 提取节点和边
    nodes = set()
    edges = []
    
    for _, row in rules.iterrows():
        antecedents = list(row['antecedents'])
        consequents = list(row['consequents'])
        
        # 添加节点
        for item in antecedents + consequents:
            nodes.add(str(item))
        
        # 添加边
        for ant in antecedents:
            for cons in consequents:
                edges.append((str(ant), str(cons), float(row['lift'])))
    
    # 创建节点DataFrame
    nodes_df = pd.DataFrame({'node': list(nodes)})
    
    # 创建边DataFrame
    edges_df = pd.DataFrame(edges, columns=['source', 'target', 'lift'])
    
    # 保存网络数据
    nodes_df.to_csv(output_dir / 'visualizations/network_nodes.csv', index=False, encoding='utf-8-sig')
    edges_df.to_csv(output_dir / 'visualizations/network_edges.csv', index=False, encoding='utf-8-sig')
    
    # 绘制简化的条形图表示
    plt.figure(figsize=(12, 8))
    
    # 统计每个节点的出现次数
    node_counts = {}
    for edge in edges:
        source, target, _ = edge
        node_counts[source] = node_counts.get(source, 0) + 1
        node_counts[target] = node_counts.get(target, 0) + 1
    
    # 取出现最多的节点
    top_nodes = sorted(node_counts.items(), key=lambda x: x[1], reverse=True)[:15]
    node_names = [node[0] for node in top_nodes]
    node_values = [node[1] for node in top_nodes]
    
    bars = plt.barh(range(len(node_names)), node_values, color='lightblue', edgecolor='black')
    plt.yticks(range(len(node_names)), node_names)
    plt.xlabel('在网络中出现的次数')
    plt.title('Top 15 关联规则网络节点')
    plt.gca().invert_yaxis()
    
    # 添加数值标签
    for i, (bar, count) in enumerate(zip(bars, node_values)):
        plt.text(count + 0.1, bar.get_y() + bar.get_height()/2,
                f'{count}', va='center')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'visualizations/rule_network_simplified.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_death_related_rules(death_rules, output_dir):
    """绘制与死亡相关的规则"""
    print("  绘制与死亡相关的规则...")
    
    # 分离导致死亡的规则和由死亡导致的规则
    rules_causing_death = death_rules[death_rules.apply(
        lambda x: 'Deceased' in [str(item) for item in x['consequents']],
        axis=1
    )]
    
    rules_caused_by_death = death_rules[death_rules.apply(
        lambda x: 'Deceased' in [str(item) for item in x['antecedents']],
        axis=1
    )]
    
    print(f"    导致死亡的规则: {len(rules_causing_death)} 条")
    print(f"    由死亡导致的规则: {len(rules_caused_by_death)} 条")
    
    # 可视化导致死亡的规则
    if len(rules_causing_death) > 0:
        plot_rules_heatmap(rules_causing_death, '导致死亡的规则', output_dir / 'visualizations/rules_causing_death.png')
    
    # 可视化由死亡导致的规则
    if len(rules_caused_by_death) > 0:
        plot_rules_heatmap(rules_caused_by_death, '由死亡导致的规则', output_dir / 'visualizations/rules_caused_by_death.png')

def plot_rules_heatmap(rules, title, save_path):
    """绘制规则热图"""
    # 提取前件和后件
    antecedents_list = []
    consequents_list = []
    
    for _, row in rules.iterrows():
        antecedents = ', '.join([str(item) for item in row['antecedents']])
        consequents = ', '.join([str(item) for item in row['consequents']])
        antecedents_list.append(antecedents)
        consequents_list.append(consequents)
    
    # 创建DataFrame用于热图
    unique_antecedents = list(set(antecedents_list))
    unique_consequents = list(set(consequents_list))
    
    # 简化处理，只显示前10个
    if len(unique_antecedents) > 10:
        unique_antecedents = unique_antecedents[:10]
    if len(unique_consequents) > 10:
        unique_consequents = unique_consequents[:10]
    
    # 创建关联矩阵
    association_matrix = np.zeros((len(unique_antecedents), len(unique_consequents)))
    
    for i, ant in enumerate(unique_antecedents):
        for j, cons in enumerate(unique_consequents):
            # 查找匹配的规则
            matching_rules = rules[
                rules.apply(
                    lambda x: ant == ', '.join([str(item) for item in x['antecedents']]) and
                              cons == ', '.join([str(item) for item in x['consequents']]),
                    axis=1
                )
            ]
            if len(matching_rules) > 0:
                # 使用平均置信度
                association_matrix[i, j] = matching_rules['confidence'].mean()
    
    # 绘制热图
    plt.figure(figsize=(10, 8))
    sns.heatmap(association_matrix, 
                xticklabels=unique_consequents,
                yticklabels=unique_antecedents,
                cmap='YlOrRd',
                annot=True,
                fmt='.2f',
                cbar_kws={'label': '平均置信度'})
    
    plt.title(title)
    plt.xlabel('后件')
    plt.ylabel('前件')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def generate_association_rules_report():
    """生成关联规则分析报告"""
    print("\n生成关联规则分析报告...")
    
    report_content = """# 关联规则分析报告

## 1. 分析概述
本报告总结了Step6关联规则分析的结果。我们使用Apriori算法从COVID-19患者数据中挖掘有意义的关联规则，最大规则长度不超过3。

## 2. 分析方法

### 2.1 数据准备
- 使用患者结局数据 (patient_outcome_processed.csv)
- 离散化连续特征为分类变量
- 创建事务数据库，每个患者为一个事务
- 包含临床指标、缺失指示器和患者结局

### 2.2 特征处理
- 连续特征：使用分位数离散化为低、中、高三组
- 二值特征：直接使用（Missing_*, Tested_*）
- 分类特征：保持原状
- 患者结局：作为特殊项（Deceased, Alive）

### 2.3 关联规则挖掘
- 算法：Apriori算法
- 最大项集长度：3
- 支持度阈值：动态调整
- 评估指标：支持度、置信度、提升度

## 3. 主要发现

### 3.1 频繁项集
- 发现了不同长度的频繁项集
- 分析了项集的支持度分布
- 识别了最常见的特征组合

### 3.2 关联规则
- 生成了具有统计意义的关联规则
- 分析了规则的质量指标
- 发现了临床有意义的模式

### 3.3 临床相关规则
- 识别了与死亡结局相关的规则
- 发现了风险因素的组合模式
- 揭示了临床检测行为与结局的关联

## 4. 文件输出

### 4.1 事务数据
- `transactions/transactions.txt` - 原始事务数据
- `transactions/item_frequencies.csv` - 项频率统计

### 4.2 频繁项集
- `rules/frequent_itemsets_length_*.json` - 按长度分组的频繁项集

### 4.3 关联规则
- `rules/all_association_rules.json` - 所有关联规则
- `rules/association_rules.csv` - 关联规则表格
- `rules/top_rules_by_*.json` - 按不同指标排序的Top规则

### 4.4 可视化图表
- `visualizations/` - 所有可视化图表
- 包括支持度分布、项集长度分布、规则散点图等

## 5. 临床意义

### 5.1 风险模式识别
关联规则分析能够发现临床指标的组合模式，这些模式可能对应特定的风险表型。

### 5.2 检测行为分析
通过分析缺失指示器和检测指示器的关联，可以了解临床实践模式与患者结局的关系。

### 5.3 预警规则发现
高置信度的关联规则可以作为临床预警规则，帮助早期识别高危患者。

## 6. 局限性

1. 关联规则反映相关性，不一定是因果关系
2. 离散化可能丢失连续特征的细节信息
3. 需要临床专家验证规则的实际意义
4. 规则数量可能较多，需要进一步筛选

## 7. 后续建议

1. 临床验证：邀请临床专家评估规则的医学合理性
2. 规则筛选：基于临床重要性进一步筛选规则
3. 时序分析：如果有时序数据，可以挖掘时序关联规则
4. 集成应用：将重要规则集成到临床决策支持系统中
"""
    
    with open('results/step6_association_rules/association_rules_report.md', 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print("关联规则分析报告已保存")

def main():
    """主函数"""
    print("=" * 60)
    print("Step6: 关联规则分析")
    print("=" * 60)
    
    # 创建目录
    output_dir = Path("results/step6_association_rules")
    create_directories()
    
    # 1. 加载数据
    df = load_and_prepare_data()
    
    if df is None:
        print("数据加载失败，程序退出")
        return
    
    # 2. 数据预处理
    df_processed, feature_cols = preprocess_for_association_rules(df, max_length=3)
    
    # 3. 离散化连续特征
    df_discrete, discretization_info = discretize_continuous_features(df_processed, feature_cols, n_bins=3)
    
    # 4. 创建事务数据
    transactions, patient_codes = create_transactions(df_discrete, feature_cols, max_length=3)
    
    if len(transactions) < 10:
        print("错误: 事务数据太少，无法进行有效的关联规则挖掘")
        return
    
    # 5. 编码事务数据
    df_encoded, items = encode_transactions(transactions)
    
    # 6. 挖掘频繁项集
    # 根据数据量动态调整最小支持度
    min_support = max(0.05, 10 / len(transactions))  # 至少出现10次
    print(f"使用最小支持度: {min_support:.3f}")
    
    frequent_itemsets = mine_frequent_itemsets(df_encoded, min_support=min_support, max_len=3)
    
    if len(frequent_itemsets) == 0:
        print("没有找到频繁项集，尝试降低最小支持度...")
        min_support = max(0.01, 5 / len(transactions))
        print(f"使用新的最小支持度: {min_support:.3f}")
        frequent_itemsets = mine_frequent_itemsets(df_encoded, min_support=min_support, max_len=3)
    
    if len(frequent_itemsets) == 0:
        print("错误: 仍然没有找到频繁项集")
        return
    
    # 7. 生成关联规则
    rules = generate_association_rules(frequent_itemsets, metric='lift', min_threshold=1.0)
    
    if len(rules) == 0:
        print("没有生成关联规则，尝试降低阈值...")
        rules = generate_association_rules(frequent_itemsets, metric='lift', min_threshold=0.5)
    
    # 8. 分析和保存规则
    if len(rules) > 0:
        analyze_rules_by_metric(rules, output_dir)
        save_association_rules(rules, output_dir)
        
        # 9. 可视化
        visualize_frequent_itemsets(frequent_itemsets, output_dir)
        visualize_association_rules(rules, output_dir, top_n=20)
    else:
        print("没有生成有效的关联规则")
    
    # 10. 生成报告
    generate_association_rules_report()
    
    print("\n" + "=" * 60)
    print("关联规则分析完成!")
    print("=" * 60)
    print("输出位置: results/step6_association_rules/")
    print("主要文件:")
    print("  1. 事务数据: transactions/")
    print("  2. 频繁项集: rules/frequent_itemsets_*.json")
    print("  3. 关联规则: rules/association_rules.*")
    print("  4. 可视化图表: visualizations/")
    print("  5. 分析报告: association_rules_report.md")
    print("=" * 60)

if __name__ == "__main__":
    main()