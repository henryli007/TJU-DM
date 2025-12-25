# step1_clean_and_convert.py
import pandas as pd
import os
import numpy as np

def split_demographic_data(df):
    """
    将Sheet1的宽格式数据拆分为长格式，正确处理缺失的对照组数据
    """
    # 提取COVID-19患者数据
    covid_data = df[['Patient Code COVID', 'Age Group COVID', 'Gender COVID']].copy()
    
    # 重命名列
    covid_data = covid_data.rename(columns={
        'Patient Code COVID': 'Patient Code',
        'Age Group COVID': 'Age Group',
        'Gender COVID': 'Gender'
    })
    
    # 提取对照组数据
    control_data = df[['Patient Code CONTROL', 'Age Group CONTROL', 'Gender CONTROL']].copy()
    
    # 重命名列
    control_data = control_data.rename(columns={
        'Patient Code CONTROL': 'Patient Code',
        'Age Group CONTROL': 'Age Group',
        'Gender CONTROL': 'Gender'
    })
    
    # 过滤掉对照组中的缺失值 (n/a)
    # 先检查是否有n/a字符串
    mask_nan = control_data['Patient Code'].isna()
    mask_na_string = control_data['Patient Code'].astype(str).str.lower() == 'n/a'
    
    # 合并两个条件
    mask_valid = ~(mask_nan | mask_na_string)
    
    # 只保留有效的对照组数据
    control_data_valid = control_data[mask_valid].copy()
    
    print(f"COVID-19组数据行数: {len(covid_data)}")
    print(f"对照组原始行数: {len(control_data)}")
    print(f"对照组有效行数: {len(control_data_valid)}")
    print(f"过滤掉的对照组缺失行数: {len(control_data) - len(control_data_valid)}")
    
    # 合并两个数据集
    combined_data = pd.concat([covid_data, control_data_valid], ignore_index=True)
    
    return combined_data

def process_sheet2_and_sheet3(df):
    """
    处理Sheet2和Sheet3，删除第一行（表名）
    """
    # 检查第一行是否包含表名特征（通常是描述性文字，不是真正的列名）
    # 如果第一行看起来不像列名（包含空格、特殊字符等），则删除
    if len(df) > 0:
        # 检查第一行是否包含"Patient Code"或"Group"等关键词
        first_row_contains_patient_code = any('Patient' in str(cell) for cell in df.iloc[0].values)
        
        # 如果第一行不包含Patient Code，则可能是表名，需要删除
        if not first_row_contains_patient_code:
            print("检测到表名行，已删除第一行")
            df = df.iloc[1:].reset_index(drop=True)
        
        # 设置正确的列名（使用第二行作为列名）
        if len(df) > 1:
            # 检查第二行是否包含合理的列名
            second_row_contains_patient_code = any('Patient' in str(cell) for cell in df.iloc[0].values)
            
            if second_row_contains_patient_code:
                # 使用第二行作为列名
                df.columns = df.iloc[0]
                df = df.iloc[1:].reset_index(drop=True)
                print("已使用第二行作为列名")
    
    return df

def convert_excel_to_csv(excel_file_path, output_dir=None):
    """
    将Excel文件的三个sheet转换为CSV文件
    """
    if output_dir is None:
        output_dir = "processed_data"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 读取Excel文件
        print("正在读取Excel文件...")
        xls = pd.ExcelFile(excel_file_path)
        
        # 获取所有sheet名称
        sheet_names = xls.sheet_names
        print(f"找到 {len(sheet_names)} 个sheet: {sheet_names}")
        
        # 处理每个sheet
        for i, sheet_name in enumerate(sheet_names, 1):
            print(f"\n正在处理第 {i} 个sheet: {sheet_name}")
            
            # 读取sheet数据（不自动设置列名，以便我们手动处理）
            df = pd.read_excel(excel_file_path, sheet_name=sheet_name, header=None)
            print(f"原始数据形状: {df.shape}")
            
            # 根据不同的sheet进行不同的处理
            if i == 1:
                # Sheet1: 第一行是列名，不需要删除
                df.columns = df.iloc[0]  # 设置第一行为列名
                df = df.iloc[1:].reset_index(drop=True)  # 删除第一行（现在是列名）
                print("Sheet1: 已设置第一行为列名")
                
                # 对Sheet1进行数据拆分
                df_processed = split_demographic_data(df)
                csv_filename = "demographic_data_split.csv"
                print(f"拆分后数据形状: {df_processed.shape}")
                
                # 统计数据分布
                covid_count = df_processed['Patient Code'].astype(str).str.startswith('P').sum()
                control_count = df_processed['Patient Code'].astype(str).str.startswith('C').sum()
                print(f"\n数据分布:")
                print(f"  COVID-19组: {covid_count} 个样本")
                print(f"  对照组: {control_count} 个样本")
                
            else:
                # Sheet2和Sheet3: 第一行是表名，需要删除
                df_processed = process_sheet2_and_sheet3(df)
                
                if i == 2:
                    csv_filename = "laboratory_data.csv"
                elif i == 3:
                    csv_filename = "patient_outcome.csv"
                else:
                    csv_filename = f"sheet_{i}.csv"
            
            # 完整的输出路径
            csv_path = os.path.join(output_dir, csv_filename)
            
            # 保存为CSV文件
            df_processed.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"已保存: {csv_path}")
            
            if i == 1:
                print(f"\n前10行数据预览:")
                print(df_processed.head(10))
            else:
                print(f"前5行数据预览:")
                print(df_processed.head())
            
            print("-" * 50)
        
        print("\n所有sheet转换完成！")
        
        return True
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 {excel_file_path}")
        print("请确保文件路径正确，且文件存在于指定位置")
        return False
    except Exception as e:
        print(f"处理过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """
    主函数
    """
    # Excel文件路径
    excel_file_path = "dataset/Datasets.xlsx"
    
    # 输出目录
    output_directory = "dataset"
    
    print("=" * 60)
    print("Excel数据转换工具 - 修正版")
    print("=" * 60)
    print("修正内容:")
    print("1. Sheet1: 第一行是列名，保留")
    print("2. Sheet2和Sheet3: 第一行是表名，已删除")
    print("3. 正确处理所有数据格式")
    print("=" * 60)
    
    # 执行转换
    success = convert_excel_to_csv(excel_file_path, output_directory)
    
    if success:
        # 加载并显示拆分后的数据
        demo_path = os.path.join(output_directory, "demographic_data_split.csv")
        if os.path.exists(demo_path):
            df_demo = pd.read_csv(demo_path)
            
            print("\n" + "="*60)
            print("人口统计数据统计摘要:")
            print("="*60)
            
            # 按组别统计
            df_demo['Group'] = df_demo['Patient Code'].astype(str).str[0]
            
            group_stats = df_demo['Group'].value_counts()
            for group, count in group_stats.items():
                group_name = "COVID-19组" if group == 'P' else "对照组"
                print(f"{group_name} ({group}): {count} 个样本")
            
            # 年龄分布统计
            print(f"\n年龄组数量: {df_demo['Age Group'].nunique()}")
            print(f"性别种类: {df_demo['Gender'].nunique()}")
            
            print("\n前20行数据预览:")
            print(df_demo.head(20))
            
        print("\n生成的文件说明:")
        print("1. demographic_data_split.csv - 拆分后的人口统计数据")
        print("   - 每行一个患者，通过Patient Code前缀(P/C)区分组别")
        print("   - 正确处理了对照组数据缺失的情况")
        print("\n2. laboratory_data.csv - 实验室检验数据")
        print("   - 已删除第一行表名")
        print("\n3. patient_outcome.csv - 患者结局数据")
        print("   - 已删除第一行表名")
        
        print(f"\n所有文件保存位置: {os.path.abspath(output_directory)}")

if __name__ == "__main__":
    main()