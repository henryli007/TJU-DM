import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
import time
import os
import json
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ==================== 中文字体解决方案 ====================
def setup_chinese_font():
    """设置中文字体支持"""
    try:
        # 尝试不同的中文字体
        font_options = [
            # Windows 字体
            'Microsoft YaHei', 'SimHei', 'KaiTi', 'SimSun',
            # macOS 字体  
            'PingFang SC', 'Hiragino Sans GB', 'Heiti SC',
            # Linux 字体
            'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'AR PL UMing CN'
        ]
        
        # 检查系统可用的中文字体
        available_fonts = []
        for font in font_options:
            try:
                # 尝试创建字体属性来检查字体是否存在
                test_font = plt.matplotlib.font_manager.FontProperties(family=font)
                available_fonts.append(font)
            except:
                pass
        
        if available_fonts:
            # 设置默认字体
            plt.rcParams['font.sans-serif'] = available_fonts
            plt.rcParams['axes.unicode_minus'] = False
            print(f"✓ 已设置中文字体: {available_fonts[0]}")
            return available_fonts[0]
        else:
            print("⚠ 警告: 未找到合适的中文字体，图表可能无法正确显示中文")
            return None
            
    except Exception as e:
        print(f"⚠ 字体设置失败: {e}")
        return None

# 初始化中文字体
chinese_font = setup_chinese_font()

# ==================== 主要功能函数 ====================
def load_standardized_data(standardized_xlsx_path):
    """加载标准化后的Excel数据"""
    df = pd.read_excel(standardized_xlsx_path)
    print(f"✓ 已加载标准化数据，样本数: {len(df)}")
    
    # 提取标准化关键词列
    standardized_keywords = df['标准化关键词'].tolist()
    
    # 将关键词字符串转换为列表
    transactions = [kw.split('、') for kw in standardized_keywords if kw]
    return transactions, df

def load_concept_dict(concept_dict_path):
    """加载概念词典"""
    with open(concept_dict_path, 'r', encoding='utf-8') as f:
        concept_dict = json.load(f)
    print(f"✓ 已加载概念词典，包含 {len(concept_dict)} 个标准概念")
    return concept_dict

def preprocess_transactions(transactions):
    """预处理事务数据，转换为适合关联分析的格式"""
    # 使用TransactionEncoder进行编码
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    print(f"✓ 事务数据预处理完成，特征数: {len(te.columns_)}")
    return df

def visualize_data_distribution(transactions, concept_dict, output_dir):
    """可视化数据分布（删除概念关系网络）"""
    print("开始数据可视化...")
    
    # 1. 概念频率分布
    all_concepts = [concept for concepts in transactions for concept in concepts]
    concept_counts = Counter(all_concepts)
    
    # 频率最高的20个概念
    top_concepts = concept_counts.most_common(20)
    concepts, counts = zip(*top_concepts)
    
    plt.figure(figsize=(15, 10))
    bars = plt.barh(range(len(concepts)), counts, color='skyblue')
    plt.yticks(range(len(concepts)), concepts, fontproperties=chinese_font)
    plt.xlabel('出现频率', fontproperties=chinese_font)
    plt.title('出现频率最高的20个概念', fontproperties=chinese_font)
    plt.gca().invert_yaxis()
    
    # 在条形上添加数值标签
    for i, bar in enumerate(bars):
        plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                str(counts[i]), ha='left', va='center', fontproperties=chinese_font)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_concepts.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 概念频率分布直方图
    plt.figure(figsize=(12, 6))
    plt.hist(list(concept_counts.values()), bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    plt.xlabel('概念出现频率', fontproperties=chinese_font)
    plt.ylabel('概念数量', fontproperties=chinese_font)
    plt.title('概念频率分布直方图', fontproperties=chinese_font)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'concept_frequency_distribution.png'), dpi=300)
    plt.close()
    
    # 3. 事务长度分布（每个样本包含的概念数量）
    transaction_lengths = [len(t) for t in transactions]
    
    plt.figure(figsize=(12, 6))
    plt.hist(transaction_lengths, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
    plt.xlabel('每个样本包含的概念数量', fontproperties=chinese_font)
    plt.ylabel('样本数量', fontproperties=chinese_font)
    plt.title('样本概念数量分布', fontproperties=chinese_font)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'transaction_length_distribution.png'), dpi=300)
    plt.close()
    
    # 4. 概念合并统计
    synonym_counts = [len(synonyms) for synonyms in concept_dict.values()]
    
    plt.figure(figsize=(12, 6))
    plt.hist(synonym_counts, bins=20, alpha=0.7, color='gold', edgecolor='black')
    plt.xlabel('每个标准概念包含的同义词数量', fontproperties=chinese_font)
    plt.ylabel('概念数量', fontproperties=chinese_font)
    plt.title('概念合并统计', fontproperties=chinese_font)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'concept_merge_statistics.png'), dpi=300)
    plt.close()
    
    print("✓ 数据可视化完成")

def run_apriori(df, min_support=0.1):
    """运行Apriori算法"""
    print(f"开始运行Apriori算法，最小支持度: {min_support}")
    start_time = time.time()
    
    # 挖掘频繁项集
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
    
    # 生成关联规则
    if len(frequent_itemsets) > 0:
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
        
        # 只保留正向规则（提升度>1）
        rules = rules[rules['lift'] > 1]
        
        # 排序规则（按支持度降序）
        rules = rules.sort_values(by=['support', 'confidence'], ascending=False)
    else:
        rules = pd.DataFrame()
    
    elapsed_time = time.time() - start_time
    print(f"✓ Apriori完成，耗时: {elapsed_time:.2f}秒")
    print(f"  发现频繁项集: {len(frequent_itemsets)}")
    print(f"  生成关联规则: {len(rules)}")
    
    return frequent_itemsets, rules, elapsed_time

def run_fpgrowth(df, min_support=0.1):
    """运行FP-Growth算法"""
    print(f"开始运行FP-Growth算法，最小支持度: {min_support}")
    start_time = time.time()
    
    # 挖掘频繁项集
    frequent_itemsets = fpgrowth(df, min_support=min_support, use_colnames=True)
    
    # 生成关联规则
    if len(frequent_itemsets) > 0:
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
        
        # 只保留正向规则（提升度>1）
        rules = rules[rules['lift'] > 1]
        
        # 排序规则（按支持度降序）
        rules = rules.sort_values(by=['support', 'confidence'], ascending=False)
    else:
        rules = pd.DataFrame()
    
    elapsed_time = time.time() - start_time
    print(f"✓ FP-Growth完成，耗时: {elapsed_time:.2f}秒")
    print(f"  发现频繁项集: {len(frequent_itemsets)}")
    print(f"  生成关联规则: {len(rules)}")
    
    return frequent_itemsets, rules, elapsed_time

def compare_thresholds(df, output_dir):
    """比较不同阈值参数设置的结果"""
    print("开始比较不同阈值参数设置...")
    
    # 定义不同的支持度阈值
    support_thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    
    # 存储结果
    results = []
    
    for min_support in support_thresholds:
        print(f"\n--- 测试支持度阈值: {min_support} ---")
        
        # 运行Apriori
        apriori_start = time.time()
        apriori_itemsets = apriori(df, min_support=min_support, use_colnames=True)
        apriori_time = time.time() - apriori_start
        
        # 运行FP-Growth
        fpgrowth_start = time.time()
        fpgrowth_itemsets = fpgrowth(df, min_support=min_support, use_colnames=True)
        fpgrowth_time = time.time() - fpgrowth_start
        
        # 统计规则数量
        apriori_rules_count = 0
        fpgrowth_rules_count = 0
        
        if len(apriori_itemsets) > 0:
            apriori_rules = association_rules(apriori_itemsets, metric="confidence", min_threshold=0.5)
            apriori_rules = apriori_rules[apriori_rules['lift'] > 1]
            apriori_rules_count = len(apriori_rules)
        
        if len(fpgrowth_itemsets) > 0:
            fpgrowth_rules = association_rules(fpgrowth_itemsets, metric="confidence", min_threshold=0.5)
            fpgrowth_rules = fpgrowth_rules[fpgrowth_rules['lift'] > 1]
            fpgrowth_rules_count = len(fpgrowth_rules)
        
        results.append({
            'support_threshold': min_support,
            'apriori_time': apriori_time,
            'fpgrowth_time': fpgrowth_time,
            'apriori_itemsets': len(apriori_itemsets),
            'fpgrowth_itemsets': len(fpgrowth_itemsets),
            'apriori_rules': apriori_rules_count,
            'fpgrowth_rules': fpgrowth_rules_count
        })
    
    # 转换为DataFrame
    results_df = pd.DataFrame(results)
    
    # 可视化比较结果
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 执行时间比较
    ax1.plot(results_df['support_threshold'], results_df['apriori_time'], 
             marker='o', linewidth=2, label='Apriori', color='blue')
    ax1.plot(results_df['support_threshold'], results_df['fpgrowth_time'], 
             marker='s', linewidth=2, label='FP-Growth', color='red')
    ax1.set_xlabel('最小支持度阈值', fontproperties=chinese_font)
    ax1.set_ylabel('执行时间 (秒)', fontproperties=chinese_font)
    ax1.set_title('不同支持度阈值下的算法执行时间', fontproperties=chinese_font)
    ax1.legend(prop=chinese_font)
    ax1.grid(True, alpha=0.3)
    
    # 2. 频繁项集数量比较
    ax2.plot(results_df['support_threshold'], results_df['apriori_itemsets'], 
             marker='o', linewidth=2, label='Apriori', color='blue')
    ax2.plot(results_df['support_threshold'], results_df['fpgrowth_itemsets'], 
             marker='s', linewidth=2, label='FP-Growth', color='red')
    ax2.set_xlabel('最小支持度阈值', fontproperties=chinese_font)
    ax2.set_ylabel('频繁项集数量', fontproperties=chinese_font)
    ax2.set_title('不同支持度阈值下的频繁项集数量', fontproperties=chinese_font)
    ax2.legend(prop=chinese_font)
    ax2.grid(True, alpha=0.3)
    
    # 3. 关联规则数量比较
    ax3.plot(results_df['support_threshold'], results_df['apriori_rules'], 
             marker='o', linewidth=2, label='Apriori', color='blue')
    ax3.plot(results_df['support_threshold'], results_df['fpgrowth_rules'], 
             marker='s', linewidth=2, label='FP-Growth', color='red')
    ax3.set_xlabel('最小支持度阈值', fontproperties=chinese_font)
    ax3.set_ylabel('关联规则数量', fontproperties=chinese_font)
    ax3.set_title('不同支持度阈值下的关联规则数量', fontproperties=chinese_font)
    ax3.legend(prop=chinese_font)
    ax3.grid(True, alpha=0.3)
    
    # 4. 性能提升比例
    results_df['speedup_ratio'] = (results_df['apriori_time'] - results_df['fpgrowth_time']) / results_df['apriori_time'] * 100
    ax4.bar(results_df['support_threshold'], results_df['speedup_ratio'], 
            width=0.02, color='green', alpha=0.7)
    ax4.set_xlabel('最小支持度阈值', fontproperties=chinese_font)
    ax4.set_ylabel('性能提升比例 (%)', fontproperties=chinese_font)
    ax4.set_title('FP-Growth相比Apriori的性能提升', fontproperties=chinese_font)
    ax4.grid(True, alpha=0.3)
    
    # 在柱状图上添加数值标签
    for i, ratio in enumerate(results_df['speedup_ratio']):
        ax4.text(results_df['support_threshold'][i], ratio + 1, 
                f'{ratio:.1f}%', ha='center', va='bottom', fontproperties=chinese_font)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'threshold_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存详细结果到Excel
    results_df.to_excel(os.path.join(output_dir, 'threshold_comparison_results.xlsx'), index=False)
    
    print("✓ 阈值参数比较完成")
    return results_df

def visualize_rules(rules, algorithm_name, output_dir):
    """可视化关联规则"""
    print(f"开始可视化{algorithm_name}规则...")
    
    if rules.empty:
        print(f"⚠ 警告: {algorithm_name}未生成有效规则")
        return
    
    # 1. 规则支持度-置信度散点图
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(rules['support'], rules['confidence'], 
                         c=rules['lift'], cmap='viridis', 
                         s=rules['lift']*20, alpha=0.6)
    plt.colorbar(scatter, label='提升度 (Lift)')
    plt.xlabel('支持度 (Support)')
    plt.ylabel('置信度 (Confidence)')
    plt.title(f'{algorithm_name}规则: 支持度 vs 置信度')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{algorithm_name.lower()}_support_confidence.png'), dpi=300)
    plt.close()
    
    # 2. 规则长度分布
    rules['antecedent_len'] = rules['antecedents'].apply(lambda x: len(x))
    rules['consequent_len'] = rules['consequents'].apply(lambda x: len(x))
    
    plt.figure(figsize=(12, 6))
    plt.hist(rules['antecedent_len'], bins=range(1, rules['antecedent_len'].max()+2), 
             alpha=0.7, color='lightblue', edgecolor='black')
    plt.xlabel('前件包含的概念数量')
    plt.ylabel('规则数量')
    plt.title(f'{algorithm_name}规则: 前件长度分布')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{algorithm_name.lower()}_antecedent_length.png'), dpi=300)
    plt.close()
    
    # 3. 提升度分布
    plt.figure(figsize=(12, 6))
    plt.hist(rules['lift'], bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
    plt.xlabel('提升度 (Lift)')
    plt.ylabel('规则数量')
    plt.title(f'{algorithm_name}规则: 提升度分布')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{algorithm_name.lower()}_lift_distribution.png'), dpi=300)
    plt.close()
    
    # 4. 前10条规则可视化
    top_rules = rules.head(10).copy()
    
    # 创建规则描述
    rule_descriptions = []
    for _, rule in top_rules.iterrows():
        antecedents = ', '.join(list(rule['antecedents']))
        consequents = ', '.join(list(rule['consequents']))
        rule_descriptions.append(f"{antecedents} → {consequents}")
    
    plt.figure(figsize=(14, 10))
    y_pos = np.arange(len(rule_descriptions))
    
    # 创建复合条形图
    width = 0.3
    plt.barh(y_pos - width/2, top_rules['support'], width, label='支持度', alpha=0.7)
    plt.barh(y_pos + width/2, top_rules['confidence'], width, label='置信度', alpha=0.7)
    
    plt.yticks(y_pos, rule_descriptions, fontsize=9)
    plt.xlabel('度量值')
    plt.title(f'Top 10 {algorithm_name} 规则')
    plt.legend()
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{algorithm_name.lower()}_top_rules.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ {algorithm_name}规则可视化完成")

def compare_algorithms(apriori_time, fpgrowth_time, apriori_rules, fpgrowth_rules, output_dir):
    """比较算法性能"""
    print("开始比较算法性能...")
    
    algorithms = ['Apriori', 'FP-Growth']
    times = [apriori_time, fpgrowth_time]
    rule_counts = [len(apriori_rules), len(fpgrowth_rules)]
    
    # 创建子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 执行时间比较
    bars1 = ax1.bar(algorithms, times, color=['lightblue', 'lightgreen'])
    ax1.set_ylabel('执行时间 (秒)', fontproperties=chinese_font)
    ax1.set_title('算法执行时间比较', fontproperties=chinese_font)
    ax1.grid(True, alpha=0.3)
    
    # 在条形上添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}s', ha='center', va='bottom', fontproperties=chinese_font)
    
    # 规则数量比较
    bars2 = ax2.bar(algorithms, rule_counts, color=['lightcoral', 'gold'])
    ax2.set_ylabel('规则数量', fontproperties=chinese_font)
    ax2.set_title('生成规则数量比较', fontproperties=chinese_font)
    ax2.grid(True, alpha=0.3)
    
    # 在条形上添加数值标签
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height}', ha='center', va='bottom', fontproperties=chinese_font)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'algorithm_comparison.png'), dpi=300)
    plt.close()
    
    # 计算性能提升
    if apriori_time > 0:
        speedup = (apriori_time - fpgrowth_time) / apriori_time * 100
        print(f"✓ FP-Growth 相比 Apriori 速度提升: {speedup:.1f}%")
    
    print("✓ 算法性能比较完成")

def save_results(apriori_itemsets, apriori_rules, fpgrowth_itemsets, fpgrowth_rules, output_dir):
    """保存结果到Excel"""
    print("保存结果到Excel文件...")
    
    # 创建Excel写入器
    with pd.ExcelWriter(os.path.join(output_dir, 'association_rules_results.xlsx'), engine='openpyxl') as writer:
        # 保存Apriori结果
        if not apriori_itemsets.empty:
            apriori_itemsets['itemsets'] = apriori_itemsets['itemsets'].apply(lambda x: ', '.join(list(x)))
            apriori_itemsets.to_excel(writer, sheet_name='Apriori_频繁项集', index=False)
        
        if not apriori_rules.empty:
            apriori_rules_copy = apriori_rules.copy()
            apriori_rules_copy['antecedents'] = apriori_rules_copy['antecedents'].apply(lambda x: ', '.join(list(x)))
            apriori_rules_copy['consequents'] = apriori_rules_copy['consequents'].apply(lambda x: ', '.join(list(x)))
            apriori_rules_copy.to_excel(writer, sheet_name='Apriori_关联规则', index=False)
        
        # 保存FP-Growth结果
        if not fpgrowth_itemsets.empty:
            fpgrowth_itemsets['itemsets'] = fpgrowth_itemsets['itemsets'].apply(lambda x: ', '.join(list(x)))
            fpgrowth_itemsets.to_excel(writer, sheet_name='FPGrowth_频繁项集', index=False)
        
        if not fpgrowth_rules.empty:
            fpgrowth_rules_copy = fpgrowth_rules.copy()
            fpgrowth_rules_copy['antecedents'] = fpgrowth_rules_copy['antecedents'].apply(lambda x: ', '.join(list(x)))
            fpgrowth_rules_copy['consequents'] = fpgrowth_rules_copy['consequents'].apply(lambda x: ', '.join(list(x)))
            fpgrowth_rules_copy.to_excel(writer, sheet_name='FPGrowth_关联规则', index=False)
    
    print("✓ 结果已保存到Excel文件")

def generate_analysis_report(transactions, concept_dict, apriori_rules, fpgrowth_rules, 
                           apriori_time, fpgrowth_time, threshold_results, output_dir):
    """生成分析报告"""
    print("生成分析报告...")
    
    report_content = f"""
关联规则分析报告
{'='*60}

数据概况:
• 总样本数: {len(transactions)}
• 总概念数: {len(concept_dict)}
• 平均每个样本的概念数: {np.mean([len(t) for t in transactions]):.2f}

算法性能比较:
• Apriori算法执行时间: {apriori_time:.3f}秒
• FP-Growth算法执行时间: {fpgrowth_time:.3f}秒
• 速度提升: {((apriori_time - fpgrowth_time) / apriori_time * 100):.1f}%

关联规则统计:
• Apriori生成规则数: {len(apriori_rules)}
• FP-Growth生成规则数: {len(fpgrowth_rules)}

不同阈值参数分析:
"""
    
    # 添加阈值分析结果
    if threshold_results is not None:
        for _, result in threshold_results.iterrows():
            report_content += (f"\n支持度阈值: {result['support_threshold']}")
            report_content += (f"\n  - Apriori: {result['apriori_time']:.3f}s, "
                            f"{result['apriori_itemsets']}个项集, {result['apriori_rules']}条规则")
            report_content += (f"\n  - FP-Growth: {result['fpgrowth_time']:.3f}s, "
                            f"{result['fpgrowth_itemsets']}个项集, {result['fpgrowth_rules']}条规则")
            report_content += (f"\n  - 性能提升: {result['speedup_ratio']:.1f}%")

    # 添加Apriori的top规则
    if not apriori_rules.empty:
        report_content += "\n\nApriori算法Top 5规则:\n"
        for i, (_, rule) in enumerate(apriori_rules.head(5).iterrows()):
            antecedents = ', '.join(list(rule['antecedents']))
            consequents = ', '.join(list(rule['consequents']))
            report_content += (f"{i+1}. {antecedents} → {consequents}\n"
                             f"   支持度: {rule['support']:.3f}, 置信度: {rule['confidence']:.3f}, "
                             f"提升度: {rule['lift']:.3f}\n")
    
    # 添加FP-Growth的top规则
    if not fpgrowth_rules.empty:
        report_content += "\nFP-Growth算法Top 5规则:\n"
        for i, (_, rule) in enumerate(fpgrowth_rules.head(5).iterrows()):
            antecedents = ', '.join(list(rule['antecedents']))
            consequents = ', '.join(list(rule['consequents']))
            report_content += (f"{i+1}. {antecedents} → {consequents}\n"
                             f"   支持度: {rule['support']:.3f}, 置信度: {rule['confidence']:.3f}, "
                             f"提升度: {rule['lift']:.3f}\n")
    
    # 算法对比分析
    report_content += "\n算法对比分析:\n"
    report_content += "1. 执行效率: FP-Growth算法在时间效率上明显优于Apriori算法\n"
    report_content += "2. 内存使用: Apriori算法需要多次扫描数据库，内存消耗较大\n"
    report_content += "3. 适用场景: 对于大规模数据集，FP-Growth更具优势\n"
    report_content += "4. 规则质量: 两种算法生成的规则质量基本一致\n"
    
    # 阈值选择建议
    report_content += "\n阈值选择建议:\n"
    report_content += "• 较低支持度(0.05-0.1): 发现更多规则，但可能包含噪声\n"
    report_content += "• 中等支持度(0.1-0.2): 平衡规则数量和质量的较好选择\n"
    report_content += "• 较高支持度(>0.2): 规则数量较少，但质量较高\n"
    
    # 保存报告
    with open(os.path.join(output_dir, 'analysis_report.txt'), 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print("✓ 分析报告已生成")

def association_analysis_main():
    """主函数"""
    # 设置输入文件路径
    base_dir = r"D:\大三上\2025-tju-数据挖掘\数据"
    standardized_xlsx = os.path.join(base_dir, "关键词调查_FINAL_standardized.xlsx")
    concept_dict_path = os.path.join(base_dir, "关键词调查_FINAL_concept_dict.json")
    
    # 检查文件是否存在
    if not os.path.exists(standardized_xlsx):
        print(f"❌ 错误: 文件不存在 - {standardized_xlsx}")
        return
    if not os.path.exists(concept_dict_path):
        print(f"❌ 错误: 文件不存在 - {concept_dict_path}")
        return
    
    # 创建输出目录（改为result文件夹）
    output_dir = os.path.join(base_dir, "result")
    os.makedirs(output_dir, exist_ok=True)
    
    print("=== 开始关联规则分析 ===")
    
    # 1. 加载数据
    transactions, original_df = load_standardized_data(standardized_xlsx)
    concept_dict = load_concept_dict(concept_dict_path)
    
    # 2. 数据可视化（删除概念关系网络）
    visualize_data_distribution(transactions, concept_dict, output_dir)
    
    # 3. 预处理事务数据
    df = preprocess_transactions(transactions)
    
    # 4. 比较不同阈值参数设置
    threshold_results = compare_thresholds(df, output_dir)
    
    # 5. 运行Apriori算法（使用默认阈值0.1）
    min_support = 0.1
    apriori_itemsets, apriori_rules, apriori_time = run_apriori(df, min_support)
    
    # 6. 运行FP-Growth算法（使用默认阈值0.1）
    fpgrowth_itemsets, fpgrowth_rules, fpgrowth_time = run_fpgrowth(df, min_support)
    
    # 7. 可视化规则
    visualize_rules(apriori_rules, "Apriori", output_dir)
    visualize_rules(fpgrowth_rules, "FP-Growth", output_dir)
    
    # 8. 比较算法性能
    compare_algorithms(apriori_time, fpgrowth_time, apriori_rules, fpgrowth_rules, output_dir)
    
    # 9. 保存结果
    save_results(apriori_itemsets, apriori_rules, fpgrowth_itemsets, fpgrowth_rules, output_dir)
    
    # 10. 生成分析报告
    generate_analysis_report(transactions, concept_dict, apriori_rules, fpgrowth_rules,
                           apriori_time, fpgrowth_time, threshold_results, output_dir)
    
    print("\n" + "="*70)
    print("关联分析完成！")
    print("="*70)
    print(f"生成结果保存在: {output_dir}")
    print(f"包含以下内容:")
    print(f"- 数据分布可视化图表 (4张)")
    print(f"- 不同阈值参数对比分析 (1张综合图表)")
    print(f"- Apriori和FP-Growth算法结果对比")
    print(f"- 关联规则可视化 (各4张)")
    print(f"- Excel格式的完整结果")
    print(f"- 详细的分析报告")
    
    # 显示关键统计信息
    if not apriori_rules.empty:
        print(f"\n关键统计信息:")
        print(f"- 最高支持度规则: {apriori_rules.iloc[0]['support']:.3f}")
        print(f"- 最高置信度规则: {apriori_rules['confidence'].max():.3f}")
        print(f"- 最高提升度规则: {apriori_rules['lift'].max():.3f}")
        print(f"- 平均规则长度: {apriori_rules['antecedent_len'].mean():.2f}个概念")

if __name__ == "__main__":
    association_analysis_main()