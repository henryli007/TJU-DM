# step2_data_preprocessing.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def create_output_dir():
    """创建输出目录"""
    output_dir = Path("results/step2")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def load_and_check_demographic_data(file_path):
    """加载并检查人口统计数据"""
    print("=" * 60)
    print("处理人口统计数据 (demographic_data_split.csv)")
    print("=" * 60)
    
    # 加载数据
    df = pd.read_csv(file_path)
    print(f"原始数据形状: {df.shape}")
    print(f"列名: {list(df.columns)}")
    print("\n前5行数据:")
    print(df.head())
    
    # 检查缺失值
    print("\n缺失值统计:")
    missing_stats = df[['Patient Code', 'Age Group', 'Gender']].isnull().sum()
    print(missing_stats)
    
    # 删除关键列有缺失值的行
    initial_count = len(df)
    df_clean = df.dropna(subset=['Patient Code', 'Age Group', 'Gender'])
    final_count = len(df_clean)
    
    print(f"\n删除缺失值后数据变化: {initial_count} -> {final_count}")
    print(f"删除了 {initial_count - final_count} 行数据")
    
    return df_clean

def plot_demographic_distributions(df, output_dir):
    """绘制人口统计数据分布图"""
    print("\n绘制人口统计数据分布图...")
    
    # 创建子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 年龄组分布
    age_counts = df['Age Group'].value_counts().sort_index()
    bars1 = ax1.bar(age_counts.index, age_counts.values, color='skyblue', edgecolor='black')
    ax1.set_title('年龄组分布', fontsize=14, fontweight='bold')
    ax1.set_xlabel('年龄组', fontsize=12)
    ax1.set_ylabel('患者数量', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    
    # 在柱子上添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')
    
    # 性别分布
    gender_counts = df['Gender'].value_counts()
    bars2 = ax2.bar(gender_counts.index, gender_counts.values, color=['lightcoral', 'lightblue'], edgecolor='black')
    ax2.set_title('性别分布', fontsize=14, fontweight='bold')
    ax2.set_xlabel('性别', fontsize=12)
    ax2.set_ylabel('患者数量', fontsize=12)
    
    # 在柱子上添加数值标签
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'demographic_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 保存统计信息
    stats_text = f"人口统计数据统计:\n\n"
    stats_text += f"总患者数: {len(df)}\n\n"
    stats_text += "年龄组分布:\n"
    for age_group, count in age_counts.items():
        stats_text += f"  {age_group}: {count} 人 ({count/len(df)*100:.1f}%)\n"
    
    stats_text += "\n性别分布:\n"
    for gender, count in gender_counts.items():
        stats_text += f"  {gender}: {count} 人 ({count/len(df)*100:.1f}%)\n"
    
    with open(output_dir / 'demographic_stats.txt', 'w', encoding='utf-8') as f:
        f.write(stats_text)
    
    print("人口统计数据分布图已保存")

def analyze_laboratory_data(file_path, output_dir):
    """分析实验室数据"""
    print("\n" + "=" * 60)
    print("处理实验室数据 (laboratory_data.csv)")
    print("=" * 60)
    
    # 加载数据
    df = pd.read_csv(file_path)
    print(f"原始数据形状: {df.shape}")
    print(f"列名: {list(df.columns)}")
    print("\n前5行数据:")
    print(df.head())
    
    # 检查Patient Code重复情况
    patient_code_counts = df['Patient Code'].value_counts()
    duplicate_counts = patient_code_counts.value_counts().sort_index()
    
    print(f"\nPatient Code重复情况:")
    print(f"唯一Patient Code数量: {len(patient_code_counts)}")
    print(f"总记录数: {len(df)}")
    print(f"重复记录分布:")
    for count, freq in duplicate_counts.items():
        print(f"  有 {count} 条记录的Patient Code: {freq} 个")
    
    # 绘制检测次数分布图
    plt.figure(figsize=(10, 6))
    bars = plt.bar(duplicate_counts.index.astype(str), duplicate_counts.values, 
                   color='lightgreen', edgecolor='black')
    plt.title('患者检测次数分布', fontsize=14, fontweight='bold')
    plt.xlabel('检测次数', fontsize=12)
    plt.ylabel('患者数量', fontsize=12)
    
    # 在柱子上添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'laboratory_test_count_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 记录重复情况
    duplicate_info = f"实验室数据重复情况分析:\n\n"
    duplicate_info += f"总记录数: {len(df)}\n"
    duplicate_info += f"唯一Patient Code数量: {len(patient_code_counts)}\n\n"
    duplicate_info += "检测次数分布:\n"
    
    for count, freq in duplicate_counts.items():
        duplicate_info += f"  有 {count} 条记录的Patient Code: {freq} 个\n"
        if count > 1:
            patients_with_duplicates = patient_code_counts[patient_code_counts == count].index.tolist()
            duplicate_info += f"    例如: {patients_with_duplicates[:5]}\n"  # 只显示前5个示例
    
    # 分析每个属性的缺失值比例
    missing_ratios = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    
    # 绘制缺失值比例柱状图
    plt.figure(figsize=(12, 8))
    bars = plt.barh(missing_ratios.index, missing_ratios.values, 
                   color=['red' if x > 50 else 'orange' if x > 10 else 'lightblue' for x in missing_ratios.values])
    plt.title('实验室数据各属性缺失值比例', fontsize=14, fontweight='bold')
    plt.xlabel('缺失值比例 (%)', fontsize=12)
    plt.xlim(0, 100)
    
    # 在柱子上添加数值标签
    for i, (index, value) in enumerate(missing_ratios.items()):
        plt.text(value + 1, i, f'{value:.1f}%', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'laboratory_missing_values.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 添加缺失值信息到报告
    duplicate_info += "\n各属性缺失值比例:\n"
    for col, ratio in missing_ratios.items():
        duplicate_info += f"  {col}: {ratio:.2f}%\n"
    
    with open(output_dir / 'laboratory_duplicate_analysis.txt', 'w', encoding='utf-8') as f:
        f.write(duplicate_info)
    
    print("实验室数据分析完成")
    
    return df, duplicate_info

def analyze_patient_outcome(file_path, output_dir):
    """分析患者结局数据"""
    print("\n" + "=" * 60)
    print("处理患者结局数据 (patient_outcome.csv)")
    print("=" * 60)
    
    # 加载数据
    df = pd.read_csv(file_path)
    print(f"原始数据形状: {df.shape}")
    print(f"列名: {list(df.columns)}")
    print("\n前5行数据:")
    print(df.head())
    
    # 检查Deceased列的分布
    if 'Deceased' in df.columns:
        deceased_counts = df['Deceased'].value_counts()
        print(f"\n患者结局分布:")
        for outcome, count in deceased_counts.items():
            print(f"  {outcome}: {count} 人")
        
        # 绘制患者结局分布图
        plt.figure(figsize=(8, 6))
        colors = ['lightgreen', 'lightcoral'] if 'Yes' in deceased_counts.index else ['lightblue', 'orange']
        bars = plt.bar(deceased_counts.index, deceased_counts.values, 
                       color=colors, edgecolor='black')
        plt.title('患者结局分布', fontsize=14, fontweight='bold')
        plt.xlabel('结局', fontsize=12)
        plt.ylabel('患者数量', fontsize=12)
        
        # 在柱子上添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'patient_outcome_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    else:
        print("警告: 数据中未找到'Deceased'列")
        deceased_counts = pd.Series()
    
    # 分析每个属性的缺失值比例
    missing_ratios = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    
    # 绘制缺失值比例柱状图
    plt.figure(figsize=(12, 8))
    bars = plt.barh(missing_ratios.index, missing_ratios.values, 
                   color=['red' if x > 50 else 'orange' if x > 10 else 'lightblue' for x in missing_ratios.values])
    plt.title('患者结局数据各属性缺失值比例', fontsize=14, fontweight='bold')
    plt.xlabel('缺失值比例 (%)', fontsize=12)
    plt.xlim(0, 100)
    
    # 在柱子上添加数值标签
    for i, (index, value) in enumerate(missing_ratios.items()):
        plt.text(value + 1, i, f'{value:.1f}%', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'outcome_missing_values.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 保存统计信息
    outcome_info = f"患者结局数据统计:\n\n"
    if not deceased_counts.empty:
        outcome_info += "患者结局分布:\n"
        for outcome, count in deceased_counts.items():
            outcome_info += f"  {outcome}: {count} 人 ({count/len(df)*100:.1f}%)\n"
    
    outcome_info += "\n各属性缺失值比例:\n"
    for col, ratio in missing_ratios.items():
        outcome_info += f"  {col}: {ratio:.2f}%\n"
    
    with open(output_dir / 'patient_outcome_analysis.txt', 'w', encoding='utf-8') as f:
        f.write(outcome_info)
    
    print("患者结局数据分析完成")
    
    return df, outcome_info

def generate_summary_report(output_dir, demo_stats, lab_info, outcome_info):
    """生成综合分析报告"""
    print("\n生成综合分析报告...")
    
    summary = "数据预处理与探索性分析报告\n"
    summary += "=" * 50 + "\n\n"
    
    # 人口统计数据摘要
    summary += "1. 人口统计数据摘要\n"
    summary += "-" * 30 + "\n"
    
    # 从保存的文件中读取人口统计信息
    try:
        with open(output_dir / 'demographic_stats.txt', 'r', encoding='utf-8') as f:
            summary += f.read() + "\n"
    except:
        summary += "人口统计数据信息无法读取\n\n"
    
    # 实验室数据摘要
    summary += "2. 实验室数据摘要\n"
    summary += "-" * 30 + "\n"
    
    # 从保存的文件中读取实验室数据信息
    try:
        with open(output_dir / 'laboratory_duplicate_analysis.txt', 'r', encoding='utf-8') as f:
            summary += f.read() + "\n"
    except:
        summary += "实验室数据信息无法读取\n\n"
    
    # 患者结局数据摘要
    summary += "3. 患者结局数据摘要\n"
    summary += "-" * 30 + "\n"
    
    # 从保存的文件中读取患者结局数据信息
    try:
        with open(output_dir / 'patient_outcome_analysis.txt', 'r', encoding='utf-8') as f:
            summary += f.read() + "\n"
    except:
        summary += "患者结局数据信息无法读取\n\n"
    
    # 数据质量评估
    summary += "4. 数据质量评估\n"
    summary += "-" * 30 + "\n"
    
    # 这里可以添加更详细的数据质量评估
    
    with open(output_dir / 'data_preprocessing_summary.txt', 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print("综合分析报告已生成")

def main():
    """主函数"""
    # 设置文件路径
    base_path = Path("dataset")
    demo_file = base_path / "demographic_data_split.csv"
    lab_file = base_path / "laboratory_data.csv"
    outcome_file = base_path / "patient_outcome.csv"
    
    # 创建输出目录
    output_dir = create_output_dir()
    
    print("开始数据预处理与探索性分析")
    print("=" * 60)
    
    # 检查文件是否存在
    if not demo_file.exists():
        print(f"错误: 找不到文件 {demo_file}")
        return
    
    if not lab_file.exists():
        print(f"错误: 找不到文件 {lab_file}")
        return
    
    if not outcome_file.exists():
        print(f"错误: 找不到文件 {outcome_file}")
        return
    
    try:
        # 1. 处理人口统计数据
        demo_df = load_and_check_demographic_data(demo_file)
        plot_demographic_distributions(demo_df, output_dir)
        
        # 2. 处理实验室数据
        lab_df, lab_info = analyze_laboratory_data(lab_file, output_dir)
        
        # 3. 处理患者结局数据
        outcome_df, outcome_info = analyze_patient_outcome(outcome_file, output_dir)
        
        # 4. 生成综合分析报告
        generate_summary_report(output_dir, None, lab_info, outcome_info)
        
        print("\n" + "=" * 60)
        print("数据预处理完成!")
        print(f"所有结果已保存到: {output_dir}")
        print("=" * 60)
        
    except Exception as e:
        print(f"处理过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()