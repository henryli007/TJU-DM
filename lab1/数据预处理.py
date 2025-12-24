import pandas as pd
import re
import os
import json
import time
from collections import Counter
from tqdm import tqdm
from volcenginesdkarkruntime import Ark

# 初始化火山方舟客户端
client = Ark(
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    api_key=os.environ.get("ARK_API_KEY") or "2db076aa-0671-41ca-be9c-d77cc3a2c48b"
)

def preliminary_clean_keyword(keyword):
    """初步清理：只删除空格、转为大写，保留括号等符号"""
    # 1. 去除所有空格（包括全角半角）
    keyword = re.sub(r'[\s\u3000]+', '', keyword)
    # 2. 转换为大写
    keyword = keyword.upper()
    return keyword

def extract_all_unique_keywords(df):
    """从数据框中提取所有唯一关键词（初步清理后）"""
    all_keywords = set()
    
    first_column = df.columns[0]
    for text in df[first_column]:
        if pd.isna(text):
            continue
            
        text_str = str(text).strip()
        # 分割关键词
        raw_keywords = re.split(r'[、,，]+', text_str)
        raw_keywords = [k.strip() for k in raw_keywords if k.strip()]
        
        # 初步清理每个关键词
        cleaned_keywords = [preliminary_clean_keyword(k) for k in raw_keywords]
        all_keywords.update(cleaned_keywords)
    
    return sorted(list(all_keywords))

def batch_standardize_keywords(all_keywords, batch_size=100):
    """批量标准化所有关键词（一次性处理）"""
    
    # 如果关键词太多，分批处理
    if len(all_keywords) > batch_size:
        print(f"关键词数量较多({len(all_keywords)})，将分批处理，每批{batch_size}个")
        return batch_process_large_keyword_set(all_keywords, batch_size)
    
    print(f"开始处理 {len(all_keywords)} 个唯一关键词...")
    
    prompt = f"""
    你是一个专业的数据标准化助手。请将以下关键词列表进行同义词标准化。
    
    任务要求：
    1. 识别表示相同概念的不同表达方式（如同义词、缩写、全称等）
    2. 为每个概念选择一个最标准、最常用的表达方式作为标准概念
    3. 特别注意括号内容，如"AI_FOR_SCIENCE(AI4S)"和"AI4S"应该识别为同一概念
    4. 输出格式必须是JSON，包含一个键为"concept_dict"的对象
    5. "concept_dict"的键是标准概念，值是该概念的所有同义词列表（包括标准概念本身）
    
    示例：
    输入关键词列表: ["AI4S", "AI_FOR_SCIENCE", "AI_FOR_SCIENCE(AI4S)", "人工智能4科学"]
    输出应为:
    {{
      "concept_dict": {{
        "AI4S": ["AI4S", "AI_FOR_SCIENCE", "AI_FOR_SCIENCE(AI4S)", "人工智能4科学"]
      }}
    }}
    
    现在请处理以下关键词列表：
    {json.dumps(all_keywords, ensure_ascii=False)}
    
    请返回JSON格式的标准化结果：
    """
    
    try:
        response = client.chat.completions.create(
            model="deepseek-v3-1-terminus",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=4000,
            response_format={"type": "json_object"}
        )
        
        result_text = response.choices[0].message.content.strip()
        result_data = json.loads(result_text)
        
        return result_data.get("concept_dict", {})
        
    except Exception as e:
        print(f"API调用失败: {e}")
        # 备用方案：每个关键词作为独立概念
        concept_dict = {}
        for keyword in all_keywords:
            concept_dict[keyword] = [keyword]
        return concept_dict

def batch_process_large_keyword_set(all_keywords, batch_size):
    """处理大量关键词的分批方案"""
    concept_dict = {}
    total_batches = (len(all_keywords) + batch_size - 1) // batch_size
    
    print(f"将分 {total_batches} 批处理关键词")
    
    for i in tqdm(range(0, len(all_keywords), batch_size), total=total_batches):
        batch = all_keywords[i:i+batch_size]
        batch_dict = batch_standardize_keywords(batch, batch_size=len(batch))
        
        # 合并批次结果
        for concept, synonyms in batch_dict.items():
            if concept in concept_dict:
                # 合并同义词列表
                concept_dict[concept].extend([s for s in synonyms if s not in concept_dict[concept]])
            else:
                concept_dict[concept] = synonyms
        
        time.sleep(2)  # API限流
    
    return concept_dict

def map_keywords_to_standardized(original_text, concept_dict):
    """将原始文本中的关键词映射到标准化概念"""
    if pd.isna(original_text):
        return ""
    
    text_str = str(original_text).strip()
    raw_keywords = re.split(r'[、,，]+', text_str)
    raw_keywords = [k.strip() for k in raw_keywords if k.strip()]
    
    standardized_keywords = []
    for raw_keyword in raw_keywords:
        # 对原始关键词进行同样的初步清理
        cleaned_keyword = preliminary_clean_keyword(raw_keyword)
        
        # 在概念词典中查找匹配
        found_concept = None
        for concept, synonyms in concept_dict.items():
            if cleaned_keyword in synonyms:
                found_concept = concept
                break
        
        # 如果找到匹配，使用标准概念；否则使用初步清理后的关键词
        standardized_keywords.append(found_concept if found_concept else cleaned_keyword)
    
    # 去重并排序
    return "、".join(sorted(set(standardized_keywords)))

def generate_sparse_arff(standardized_data, output_path):
    """生成稀疏格式的ARFF文件"""
    # 提取所有唯一概念
    all_concepts = set()
    transactions = []
    
    for text in standardized_data:
        if not text:
            transactions.append([])
            continue
            
        concepts = text.split('、')
        transactions.append(concepts)
        all_concepts.update(concepts)
    
    sorted_concepts = sorted(list(all_concepts))
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"% 关键词调查数据 - 关联规则分析\n")
        f.write(f"% 生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"% 样本数: {len(transactions)}, 概念数: {len(all_concepts)}\n")
        f.write("@RELATION keyword_survey\n\n")
        
        # 为每个概念创建属性（清理属性名）
        for concept in sorted_concepts:
            # 清理属性名中的特殊字符
            attr_name = re.sub(r'[^\w]', '_', concept)
            f.write(f"@ATTRIBUTE '{attr_name}' {{0,1}}\n")
        
        f.write("\n@DATA\n")
        
        # 写入稀疏格式数据
        for i, transaction in enumerate(transactions):
            if i > 0:
                f.write(",\n")
            
            sparse_entries = []
            for j, concept in enumerate(sorted_concepts):
                if concept in transaction:
                    sparse_entries.append(f"{j} 1")
            
            if sparse_entries:
                f.write("{" + ", ".join(sparse_entries) + "}")
            else:
                f.write("{ }")
    
    return len(all_concepts), len(transactions)

def process_keywords_final(xlsx_file_path):
    """最终处理流程：一次性标准化所有关键词"""
    
    file_dir = os.path.dirname(xlsx_file_path)
    file_name = os.path.basename(xlsx_file_path)
    base_name = os.path.splitext(file_name)[0]
    
    output_files = {
        'preliminary_report': os.path.join(file_dir, f"{base_name}_preliminary_report.txt"),
        'concept_dict': os.path.join(file_dir, f"{base_name}_FINAL_concept_dict.json"),
        'standardized_xlsx': os.path.join(file_dir, f"{base_name}_FINAL_standardized.xlsx"),
        'sparse_arff': os.path.join(file_dir, f"{base_name}_FINAL_sparse.arff"),
        'processing_summary': os.path.join(file_dir, f"{base_name}_FINAL_summary.txt")
    }
    
    print("=== 开始最终处理流程 ===")
    
    # 1. 读取原始数据
    try:
        df = pd.read_excel(xlsx_file_path)
        print("✓ 原始Excel文件读取成功")
    except Exception as e:
        print(f"✗ 读取文件失败: {e}")
        return None
    
    # 2. 提取所有唯一关键词（初步清理后）
    print("提取所有唯一关键词...")
    all_keywords = extract_all_unique_keywords(df)
    original_keyword_count = len(all_keywords)
    print(f"✓ 提取到 {original_keyword_count} 个唯一关键词（初步清理后）")
    
    # 保存初步处理报告
    with open(output_files['preliminary_report'], 'w', encoding='utf-8') as f:
        f.write("初步处理报告\n")
        f.write("=" * 50 + "\n")
        f.write(f"原始文件: {file_name}\n")
        f.write(f"提取时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"唯一关键词数量: {original_keyword_count}\n\n")
        f.write("前50个关键词示例:\n")
        f.write("-" * 30 + "\n")
        for i, keyword in enumerate(all_keywords[:50]):
            f.write(f"{i+1:2d}. {keyword}\n")
    
    # 3. 批量标准化所有关键词
    print("调用API进行关键词标准化...")
    concept_dict = batch_standardize_keywords(all_keywords)
    standardized_concept_count = len(concept_dict)
    
    print(f"✓ 标准化完成，原始 {original_keyword_count} 个关键词合并为 {standardized_concept_count} 个标准概念")
    compression_rate = 1 - standardized_concept_count / original_keyword_count
    print(f"✓ 概念压缩率: {compression_rate:.1%}")
    
    # 保存概念词典
    with open(output_files['concept_dict'], 'w', encoding='utf-8') as f:
        json.dump(concept_dict, f, ensure_ascii=False, indent=2)
    print(f"✓ 概念词典已保存")
    
    # 4. 应用概念词典映射原始文件
    print("应用概念词典映射原始文件...")
    standardized_data = []
    for original_text in tqdm(df.iloc[:, 0], desc="映射样本"):
        standardized_text = map_keywords_to_standardized(original_text, concept_dict)
        standardized_data.append(standardized_text)
    
    # 5. 生成标准化Excel文件
    df_final = df.copy()
    df_final['原始关键词'] = df.iloc[:, 0]
    df_final['标准化关键词'] = standardized_data
    df_final.to_excel(output_files['standardized_xlsx'], index=False)
    print(f"✓ 标准化Excel文件已生成")
    
    # 6. 生成稀疏ARFF文件
    print("生成稀疏ARFF文件...")
    final_concept_count, sample_count = generate_sparse_arff(standardized_data, output_files['sparse_arff'])
    print(f"✓ 稀疏ARFF文件已生成，包含 {sample_count} 个样本，{final_concept_count} 个概念")
    
    # 7. 生成处理摘要
    with open(output_files['processing_summary'], 'w', encoding='utf-8') as f:
        f.write("关键词标准化处理摘要\n")
        f.write("=" * 60 + "\n")
        f.write(f"处理文件: {file_name}\n")
        f.write(f"处理时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"样本数量: {len(df)}\n")
        f.write(f"原始唯一关键词数: {original_keyword_count}\n")
        f.write(f"标准化概念数: {standardized_concept_count}\n")
        f.write(f"概念压缩率: {compression_rate:.1%}\n")
        f.write(f"最终ARFF概念数: {final_concept_count}\n\n")
        
        f.write("概念合并统计:\n")
        f.write("-" * 20 + "\n")
        synonym_counts = Counter()
        for concept, synonyms in concept_dict.items():
            synonym_counts[len(synonyms)] += 1
        
        f.write("同义词数量分布:\n")
        for count_size, freq in sorted(synonym_counts.items()):
            f.write(f"  包含{count_size}个同义词的概念: {freq}个\n")
        
        f.write("\n合并效果最好的概念（同义词数量最多）:\n")
        f.write("-" * 40 + "\n")
        top_concepts = sorted(concept_dict.items(), key=lambda x: len(x[1]), reverse=True)[:10]
        for concept, synonyms in top_concepts:
            if len(synonyms) > 1:
                f.write(f"{concept}: {len(synonyms)}个同义词\n")
                f.write(f"  示例: {', '.join(synonyms[:5])}\n")
    
    print(f"✓ 处理摘要已生成")
    
    return {
        'original_keywords': original_keyword_count,
        'standardized_concepts': standardized_concept_count,
        'compression_rate': compression_rate,
        'final_concepts': final_concept_count,
        'files': output_files
    }

# 主程序
if __name__ == "__main__":
    input_xlsx = r"D:\大三上\2025-tju-数据挖掘\数据\关键词调查.xlsx"
    
    if not os.path.exists(input_xlsx):
        print(f"错误: 文件不存在 - {input_xlsx}")
    else:
        print("开始最终处理流程...")
        result = process_keywords_final(input_xlsx)
        
        if result:
            print("\n" + "="*70)
            print("处理完成！参数变化过程：")
            print("="*70)
            print(f"原始唯一关键词数量: {result['original_keywords']}")
            print(f"API标准化后概念数量: {result['standardized_concepts']}")
            print(f"概念压缩率: {result['compression_rate']:.1%}")
            print(f"最终ARFF文件概念数量: {result['final_concepts']}")
            
            print("\n生成的文件：")
            for key, path in result['files'].items():
                print(f"• {os.path.basename(path)}")
            
            print("\n下一步：")
            print("1. 查看 _FINAL_summary.txt 了解详细处理统计")
            print("2. 查看 _FINAL_concept_dict.json 了解概念合并详情")
            print("3. 在Weka中打开 _FINAL_sparse.arff 进行关联分析")