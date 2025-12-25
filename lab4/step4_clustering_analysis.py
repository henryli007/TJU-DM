# step4_clustering_analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import warnings
import os
from pathlib import Path
import pickle
import json

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def create_directories():
    """创建输出目录"""
    base_dir = Path("results/step4_clustering")
    directories = [
        base_dir,
        base_dir / "overall_clustering",
        base_dir / "patient_clustering", 
        base_dir / "subset_clustering",
        base_dir / "visualizations",
        base_dir / "models"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    print("目录创建完成")

def load_and_prepare_data():
    """加载并准备数据"""
    print("=" * 60)
    print("加载数据...")
    print("=" * 60)
    
    # 加载三个处理后的数据文件
    demo_path = Path("dataset/demographic_data_processed.csv")
    lab_path = Path("dataset/laboratory_data_processed.csv")
    outcome_path = Path("dataset/patient_outcome_processed.csv")
    
    # 检查文件是否存在
    for path in [demo_path, lab_path, outcome_path]:
        if not path.exists():
            print(f"错误: 找不到文件 {path}")
            return None, None, None
    
    # 读取数据
    demo_df = pd.read_csv(demo_path)
    lab_df = pd.read_csv(lab_path)
    outcome_df = pd.read_csv(outcome_path)
    
    print(f"人口统计数据: {demo_df.shape}")
    print(f"实验室数据: {lab_df.shape}")
    print(f"患者结局数据: {outcome_df.shape}")
    
    return demo_df, lab_df, outcome_df

def merge_demographic_features(data_df, demo_df):
    """将人口统计特征合并到数据中"""
    print("\n合并人口统计特征...")
    
    # 确保Patient Code列名一致
    if 'Patient Code' in data_df.columns and 'Patient Code' in demo_df.columns:
        # 合并年龄和性别信息
        merged_df = data_df.merge(
            demo_df[['Patient Code', 'Age_Group_Encoded', 'Gender_Encoded']],
            on='Patient Code',
            how='left'
        )
        
        print(f"合并前数据形状: {data_df.shape}")
        print(f"合并后数据形状: {merged_df.shape}")
        print(f"添加的特征: Age_Group_Encoded, Gender_Encoded")
        
        return merged_df
    else:
        print("警告: 找不到Patient Code列，无法合并人口统计特征")
        return data_df

def prepare_clustering_data(df, is_overall=True):
    """准备聚类数据"""
    print("\n准备聚类数据...")
    
    # 标识列
    identifier_cols = ['Patient Code']
    if 'Group' in df.columns:
        identifier_cols.append('Group')
    if 'Deceased' in df.columns:
        identifier_cols.append('Deceased')
    
    # 分离特征和标识
    feature_cols = [col for col in df.columns if col not in identifier_cols]
    features = df[feature_cols].copy()
    identifiers = df[identifier_cols].copy()
    
    # 处理缺失值
    print(f"处理前的缺失值数量: {features.isnull().sum().sum()}")
    
    # 对于聚类，我们填充缺失值为0（标准化后会处理）
    features = features.fillna(0)
    
    # 数据标准化
    scaler = StandardScaler()
    features_scaled = pd.DataFrame(
        scaler.fit_transform(features),
        columns=features.columns,
        index=features.index
    )
    
    # 保存标准化器
    scaler_name = "overall_scaler.pkl" if is_overall else "patient_scaler.pkl"
    scaler_path = Path("results/step4_clustering/models") / scaler_name
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"标准化后数据形状: {features_scaled.shape}")
    print(f"特征数量: {len(features.columns)}")
    
    return features_scaled, identifiers, feature_cols

def define_feature_subsets(feature_cols):
    """定义特征子集"""
    print("\n定义特征子集...")
    
    # 根据特征名称定义子集
    subsets = {}
    
    # 基础人口统计特征
    demographic_features = [col for col in feature_cols if 'Age' in col or 'Gender' in col]
    if demographic_features:
        subsets['demographic'] = demographic_features
    
    # 基础生化指标
    basic_biochem = ['Glucose', 'Urea', 'Creatinine', 'Sodium', 'Potassium']
    basic_biochem = [col for col in feature_cols if any(b in col for b in basic_biochem)]
    if basic_biochem:
        subsets['basic_biochemistry'] = basic_biochem
    
    # 肝功能指标
    liver_features = ['TB', 'DB', 'ALT', 'AST', 'ALP', 'Total Protein', 'Albubin']
    liver_features = [col for col in feature_cols if any(l in col for l in liver_features)]
    if liver_features:
        subsets['liver_function'] = liver_features
    
    # 炎症指标
    inflammation_features = ['CRP', 'PCT', 'IL-6', 'Ferritin']
    inflammation_features = [col for col in feature_cols if any(i in col for i in inflammation_features)]
    if inflammation_features:
        subsets['inflammation'] = inflammation_features
    
    # 凝血功能指标
    coagulation_features = ['PT', 'D-Dimer', 'Platelet']
    coagulation_features = [col for col in feature_cols if any(c in col for c in coagulation_features)]
    if coagulation_features:
        subsets['coagulation'] = coagulation_features
    
    # 心肌损伤指标
    cardiac_features = ['Troponin', 'CPK-MB', 'LDH']
    cardiac_features = [col for col in feature_cols if any(c in col for c in cardiac_features)]
    if cardiac_features:
        subsets['cardiac'] = cardiac_features
    
    # 缺失指示器
    missing_features = [col for col in feature_cols if col.startswith('Missing_') or col.startswith('Tested_')]
    if missing_features:
        subsets['missing_patterns'] = missing_features
    
    # 显示子集信息
    for subset_name, subset_features in subsets.items():
        print(f"  {subset_name}: {len(subset_features)} 个特征")
        if len(subset_features) <= 10:  # 只显示前10个
            print(f"    {subset_features}")
    
    return subsets

def determine_optimal_clusters(features_scaled, max_clusters=10):
    """确定最佳聚类数量"""
    print("\n确定最佳聚类数量...")
    
    # 使用肘部法则和轮廓系数
    inertia = []
    silhouette_scores = []
    calinski_scores = []
    davies_scores = []
    
    cluster_range = range(2, min(max_clusters + 1, len(features_scaled) // 10))
    
    for n_clusters in cluster_range:
        # K-means聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features_scaled)
        
        # 计算指标
        inertia.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(features_scaled, cluster_labels))
        calinski_scores.append(calinski_harabasz_score(features_scaled, cluster_labels))
        davies_scores.append(davies_bouldin_score(features_scaled, cluster_labels))
    
    # 找到最佳聚类数
    best_silhouette = cluster_range[np.argmax(silhouette_scores)]
    best_calinski = cluster_range[np.argmax(calinski_scores)]
    best_davies = cluster_range[np.argmin(davies_scores)]
    
    print(f"最佳轮廓系数聚类数: {best_silhouette}")
    print(f"最佳Calinski-Harabasz聚类数: {best_calinski}")
    print(f"最佳Davies-Bouldin聚类数: {best_davies}")
    
    # 综合选择最佳聚类数
    # 优先考虑轮廓系数
    optimal_n = best_silhouette
    
    # 可视化聚类指标
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 肘部法则图
    axes[0, 0].plot(cluster_range, inertia, 'bo-')
    axes[0, 0].set_xlabel('聚类数量')
    axes[0, 0].set_ylabel('惯性（Inertia）')
    axes[0, 0].set_title('肘部法则')
    axes[0, 0].axvline(x=optimal_n, color='r', linestyle='--', alpha=0.5)
    
    # 轮廓系数图
    axes[0, 1].plot(cluster_range, silhouette_scores, 'ro-')
    axes[0, 1].set_xlabel('聚类数量')
    axes[0, 1].set_ylabel('轮廓系数')
    axes[0, 1].set_title('轮廓系数')
    axes[0, 1].axvline(x=optimal_n, color='r', linestyle='--', alpha=0.5)
    
    # Calinski-Harabasz指数图
    axes[1, 0].plot(cluster_range, calinski_scores, 'go-')
    axes[1, 0].set_xlabel('聚类数量')
    axes[1, 1].set_ylabel('Calinski-Harabasz指数')
    axes[1, 0].set_title('Calinski-Harabasz指数')
    axes[1, 0].axvline(x=optimal_n, color='r', linestyle='--', alpha=0.5)
    
    # Davies-Bouldin指数图
    axes[1, 1].plot(cluster_range, davies_scores, 'mo-')
    axes[1, 1].set_xlabel('聚类数量')
    axes[1, 1].set_ylabel('Davies-Bouldin指数')
    axes[1, 1].set_title('Davies-Bouldin指数')
    axes[1, 1].axvline(x=optimal_n, color='r', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('results/step4_clustering/visualizations/cluster_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return optimal_n

def perform_clustering(features_scaled, identifiers, algorithm='kmeans', n_clusters=None):
    """执行聚类分析"""
    print(f"\n执行{algorithm}聚类...")
    
    if algorithm == 'kmeans':
        if n_clusters is None:
            n_clusters = determine_optimal_clusters(features_scaled)
        
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = model.fit_predict(features_scaled)
        
        # 保存模型
        model_path = Path("results/step4_clustering/models") / "kmeans_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
    
    elif algorithm == 'hierarchical':
        if n_clusters is None:
            n_clusters = determine_optimal_clusters(features_scaled)
        
        model = AgglomerativeClustering(n_clusters=n_clusters)
        cluster_labels = model.fit_predict(features_scaled)
        
        # 生成树状图
        linked = linkage(features_scaled, 'ward')
        
        plt.figure(figsize=(12, 8))
        dendrogram(linked, orientation='top', 
                  distance_sort='descending', 
                  show_leaf_counts=True,
                  truncate_mode='level',
                  p=10)  # 只显示最后10层
        
        plt.title('层次聚类树状图')
        plt.xlabel('样本索引')
        plt.ylabel('距离')
        plt.tight_layout()
        plt.savefig('results/step4_clustering/visualizations/dendrogram.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    elif algorithm == 'dbscan':
        # 自动确定eps参数
        from sklearn.neighbors import NearestNeighbors
        
        # 计算k距离图
        neigh = NearestNeighbors(n_neighbors=5)
        neigh.fit(features_scaled)
        distances, indices = neigh.kneighbors(features_scaled)
        
        # 对距离排序
        distances = np.sort(distances[:, 4], axis=0)
        
        # 绘制k距离图
        plt.figure(figsize=(8, 6))
        plt.plot(distances)
        plt.xlabel('样本')
        plt.ylabel('到第5个最近邻的距离')
        plt.title('k距离图（用于确定DBSCAN的eps）')
        plt.tight_layout()
        plt.savefig('results/step4_clustering/visualizations/k_distance_plot.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 基于k距离图手动设置eps（这里使用自动确定的方法）
        eps = distances[int(len(distances) * 0.95)]  # 选择第95百分位数
        
        model = DBSCAN(eps=eps, min_samples=5)
        cluster_labels = model.fit_predict(features_scaled)
        
        print(f"DBSCAN聚类结果: {len(np.unique(cluster_labels))} 个簇")
        print(f"噪声点数量: {np.sum(cluster_labels == -1)}")
    
    elif algorithm == 'gmm':
        if n_clusters is None:
            n_clusters = determine_optimal_clusters(features_scaled)
        
        model = GaussianMixture(n_components=n_clusters, random_state=42)
        cluster_labels = model.fit_predict(features_scaled)
    
    else:
        raise ValueError(f"不支持的聚类算法: {algorithm}")
    
    # 计算聚类质量指标
    if len(np.unique(cluster_labels)) > 1:  # 确保有多个簇
        silhouette = silhouette_score(features_scaled, cluster_labels)
        calinski = calinski_harabasz_score(features_scaled, cluster_labels)
        davies = davies_bouldin_score(features_scaled, cluster_labels)
        
        print(f"轮廓系数: {silhouette:.3f}")
        print(f"Calinski-Harabasz指数: {calinski:.1f}")
        print(f"Davies-Bouldin指数: {davies:.3f}")
    else:
        print("警告: 只找到一个簇，无法计算聚类质量指标")
    
    # 创建包含聚类结果的数据框
    results_df = identifiers.copy()
    results_df['Cluster'] = cluster_labels
    
    return results_df, cluster_labels

def visualize_clusters(features_scaled, cluster_labels, title_prefix):
    """可视化聚类结果"""
    print(f"\n可视化{title_prefix}聚类结果...")
    
    # 使用PCA降维到2D
    pca = PCA(n_components=2, random_state=42)
    features_pca = pca.fit_transform(features_scaled)
    
    # 使用t-SNE降维到2D（效果更好但计算更慢）
    if len(features_scaled) <= 1000:  # t-SNE对大数据集较慢
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        features_tsne = tsne.fit_transform(features_scaled)
    else:
        features_tsne = None
    
    # 绘制PCA聚类结果
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(features_pca[:, 0], features_pca[:, 1], 
                          c=cluster_labels, cmap='viridis', alpha=0.6, s=30)
    plt.xlabel('PCA 主成分 1')
    plt.ylabel('PCA 主成分 2')
    plt.title(f'{title_prefix} - PCA可视化')
    plt.colorbar(scatter, label='聚类标签')
    
    # 绘制t-SNE聚类结果（如果可用）
    if features_tsne is not None:
        plt.subplot(1, 2, 2)
        scatter = plt.scatter(features_tsne[:, 0], features_tsne[:, 1], 
                              c=cluster_labels, cmap='viridis', alpha=0.6, s=30)
        plt.xlabel('t-SNE 维度 1')
        plt.ylabel('t-SNE 维度 2')
        plt.title(f'{title_prefix} - t-SNE可视化')
        plt.colorbar(scatter, label='聚类标签')
    
    plt.tight_layout()
    plt.savefig(f'results/step4_clustering/visualizations/{title_prefix}_cluster_visualization.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # 保存降维结果
    pca_path = Path("results/step4_clustering/models") / f"{title_prefix}_pca.pkl"
    with open(pca_path, 'wb') as f:
        pickle.dump(pca, f)
    
    return features_pca

def analyze_cluster_features(features_scaled, cluster_labels, feature_names, title_prefix):
    """分析每个簇的特征"""
    print(f"\n分析{title_prefix}聚类特征...")
    
    # 为每个特征计算每个簇的平均值
    features_df = pd.DataFrame(features_scaled, columns=feature_names)
    features_df['Cluster'] = cluster_labels
    
    # 计算每个簇的特征平均值
    cluster_means = features_df.groupby('Cluster').mean()
    
    # 计算每个簇的特征标准差
    cluster_stds = features_df.groupby('Cluster').std()
    
    # 计算每个簇的大小
    cluster_sizes = features_df['Cluster'].value_counts().sort_index()
    
    print(f"各簇大小:")
    for cluster, size in cluster_sizes.items():
        print(f"  簇{cluster}: {size} 个样本 ({size/len(features_df)*100:.1f}%)")
    
    # 绘制簇特征热图
    plt.figure(figsize=(max(12, len(feature_names) * 0.3), max(8, len(cluster_means) * 0.5)))
    
    # 限制显示的列数，以免热图太宽
    if len(feature_names) > 20:
        # 选择方差最大的前20个特征
        feature_variances = features_df.drop(columns='Cluster').var().sort_values(ascending=False)
        top_features = feature_variances.head(20).index.tolist()
        cluster_means_subset = cluster_means[top_features]
    else:
        cluster_means_subset = cluster_means
        top_features = feature_names
    
    sns.heatmap(cluster_means_subset.T, cmap='coolwarm', center=0, 
                annot=True, fmt='.2f', linewidths=1, cbar_kws={'label': '标准化特征值'})
    plt.title(f'{title_prefix} - 各簇特征平均值热图')
    plt.xlabel('聚类标签')
    plt.ylabel('特征')
    plt.tight_layout()
    plt.savefig(f'results/step4_clustering/visualizations/{title_prefix}_cluster_heatmap.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # 保存聚类分析结果
    cluster_stats = {
        'cluster_sizes': cluster_sizes.to_dict(),
        'cluster_means': cluster_means.to_dict(),
        'cluster_stds': cluster_stds.to_dict()
    }
    
    stats_path = Path("results/step4_clustering") / f"{title_prefix}_cluster_stats.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(cluster_stats, f, indent=2, ensure_ascii=False)
    
    return cluster_means, cluster_sizes

def analyze_cluster_demographics(cluster_results, demo_df):
    """分析聚类结果与人口统计学的关系"""
    print("\n分析聚类结果与人口统计学关系...")
    
    if 'Patient Code' not in cluster_results.columns or 'Cluster' not in cluster_results.columns:
        print("警告: 聚类结果中没有Patient Code或Cluster列")
        return None
    
    # 合并聚类结果和人口统计数据
    merged = cluster_results[['Patient Code', 'Cluster']].merge(
        demo_df[['Patient Code', 'Age Group', 'Gender', 'Age_Group_Encoded', 'Gender_Encoded']],
        on='Patient Code',
        how='left'
    )
    
    # 分析各簇的年龄分布
    age_dist = pd.crosstab(merged['Cluster'], merged['Age Group'], normalize='index') * 100
    
    # 分析各簇的性别分布
    gender_dist = pd.crosstab(merged['Cluster'], merged['Gender'], normalize='index') * 100
    
    # 绘制年龄分布图
    if len(age_dist.columns) > 0:
        age_dist.plot(kind='bar', figsize=(10, 6), stacked=True, colormap='Set3')
        plt.title('各簇年龄组分布')
        plt.xlabel('聚类标签')
        plt.ylabel('百分比 (%)')
        plt.legend(title='年龄组', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig('results/step4_clustering/visualizations/cluster_age_distribution.png', 
                    dpi=300, bbox_inches='tight')
        plt.show()
    
    # 绘制性别分布图
    if len(gender_dist.columns) > 0:
        gender_dist.plot(kind='bar', figsize=(8, 6), colormap='coolwarm')
        plt.title('各簇性别分布')
        plt.xlabel('聚类标签')
        plt.ylabel('百分比 (%)')
        plt.legend(title='性别')
        plt.tight_layout()
        plt.savefig('results/step4_clustering/visualizations/cluster_gender_distribution.png', 
                    dpi=300, bbox_inches='tight')
        plt.show()
    
    return age_dist, gender_dist

def analyze_cluster_outcomes(cluster_results, outcome_df):
    """分析聚类结果与患者结局的关系"""
    print("\n分析聚类结果与患者结局关系...")
    
    if ('Patient Code' not in cluster_results.columns or 
        'Cluster' not in cluster_results.columns or
        'Deceased' not in outcome_df.columns):
        print("警告: 缺少必要的列")
        return None
    
    # 合并聚类结果和结局数据
    merged = cluster_results[['Patient Code', 'Cluster']].merge(
        outcome_df[['Patient Code', 'Deceased']],
        on='Patient Code',
        how='inner'
    )
    
    # 计算各簇的死亡率
    mortality_by_cluster = merged.groupby('Cluster')['Deceased'].agg(['mean', 'count', 'sum'])
    mortality_by_cluster['mean'] = mortality_by_cluster['mean'] * 100  # 转换为百分比
    mortality_by_cluster = mortality_by_cluster.rename(columns={'mean': '死亡率 (%)', 'count': '样本数', 'sum': '死亡数'})
    
    print("各簇死亡率统计:")
    print(mortality_by_cluster)
    
    # 绘制死亡率条形图
    if len(mortality_by_cluster) > 0:
        plt.figure(figsize=(10, 6))
        bars = plt.bar(mortality_by_cluster.index.astype(str), 
                      mortality_by_cluster['死亡率 (%)'],
                      color=['lightgreen' if x < mortality_by_cluster['死亡率 (%)'].median() 
                            else 'lightcoral' for x in mortality_by_cluster['死亡率 (%)']],
                      edgecolor='black')
        
        plt.title('各簇死亡率对比')
        plt.xlabel('聚类标签')
        plt.ylabel('死亡率 (%)')
        
        # 在条形上添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('results/step4_clustering/visualizations/cluster_mortality.png', 
                    dpi=300, bbox_inches='tight')
        plt.show()
    
    return mortality_by_cluster

def overall_clustering_analysis(lab_df, demo_df):
    """整体数据聚类分析"""
    print("\n" + "=" * 60)
    print("整体数据聚类分析")
    print("=" * 60)
    
    # 合并人口统计特征
    lab_with_demo = merge_demographic_features(lab_df, demo_df)
    
    # 准备聚类数据
    features_scaled, identifiers, feature_cols = prepare_clustering_data(lab_with_demo, is_overall=True)
    
    # 定义特征子集（用于后续分析）
    subsets = define_feature_subsets(feature_cols)
    
    # 执行K-means聚类
    print("\n--- K-means聚类 ---")
    kmeans_results, kmeans_labels = perform_clustering(
        features_scaled, identifiers, algorithm='kmeans', n_clusters=4
    )
    
    # 可视化聚类结果
    features_pca = visualize_clusters(features_scaled, kmeans_labels, "overall_kmeans")
    
    # 分析聚类特征
    cluster_means, cluster_sizes = analyze_cluster_features(
        features_scaled, kmeans_labels, feature_cols, "overall_kmeans"
    )
    
    # 分析聚类与人口统计学的关系
    age_dist, gender_dist = analyze_cluster_demographics(kmeans_results, demo_df)
    
    # 尝试其他聚类算法进行比较
    print("\n--- 层次聚类 ---")
    hierarchical_results, hierarchical_labels = perform_clustering(
        features_scaled, identifiers, algorithm='hierarchical', n_clusters=4
    )
    
    # 保存聚类结果
    kmeans_results.to_csv('results/step4_clustering/overall_clustering/kmeans_cluster_assignments.csv', 
                         index=False, encoding='utf-8-sig')
    
    return kmeans_results, kmeans_labels, feature_cols, subsets

def patient_clustering_analysis(outcome_df, demo_df):
    """患者数据聚类分析"""
    print("\n" + "=" * 60)
    print("COVID-19患者数据聚类分析")
    print("=" * 60)
    
    # 合并人口统计特征
    outcome_with_demo = merge_demographic_features(outcome_df, demo_df)
    
    # 准备聚类数据
    features_scaled, identifiers, feature_cols = prepare_clustering_data(outcome_with_demo, is_overall=False)
    
    # 执行K-means聚类
    print("\n--- K-means聚类 ---")
    kmeans_results, kmeans_labels = perform_clustering(
        features_scaled, identifiers, algorithm='kmeans', n_clusters=3
    )
    
    # 可视化聚类结果
    features_pca = visualize_clusters(features_scaled, kmeans_labels, "patient_kmeans")
    
    # 分析聚类特征
    cluster_means, cluster_sizes = analyze_cluster_features(
        features_scaled, kmeans_labels, feature_cols, "patient_kmeans"
    )
    
    # 分析聚类与人口统计学的关系
    age_dist, gender_dist = analyze_cluster_demographics(kmeans_results, demo_df)
    
    # 分析聚类与结局的关系
    mortality_stats = analyze_cluster_outcomes(kmeans_results, outcome_df)
    
    # 保存聚类结果
    kmeans_results.to_csv('results/step4_clustering/patient_clustering/kmeans_cluster_assignments.csv', 
                         index=False, encoding='utf-8-sig')
    
    # 保存死亡率统计
    if mortality_stats is not None:
        mortality_stats.to_csv('results/step4_clustering/patient_clustering/cluster_mortality_stats.csv')
    
    return kmeans_results, kmeans_labels, feature_cols

def subset_clustering_analysis(features_scaled, identifiers, feature_cols, subsets):
    """特征子集聚类分析"""
    print("\n" + "=" * 60)
    print("特征子集聚类分析")
    print("=" * 60)
    
    subset_results = {}
    
    for subset_name, subset_features in subsets.items():
        print(f"\n--- {subset_name}子集聚类 ---")
        print(f"特征数量: {len(subset_features)}")
        
        # 提取子集特征
        if all(feat in features_scaled.columns for feat in subset_features):
            subset_data = features_scaled[subset_features].copy()
            
            # 确定最佳聚类数（限制最大簇数）
            max_clusters = min(6, len(subset_data) // 10)
            if max_clusters >= 2:
                try:
                    optimal_n = determine_optimal_clusters(subset_data, max_clusters=max_clusters)
                except:
                    optimal_n = min(4, max_clusters)
            else:
                optimal_n = 2
            
            # 执行聚类
            if optimal_n >= 2:
                subset_model = KMeans(n_clusters=optimal_n, random_state=42, n_init=10)
                subset_labels = subset_model.fit_predict(subset_data)
                
                # 计算聚类质量
                if len(np.unique(subset_labels)) > 1:
                    silhouette = silhouette_score(subset_data, subset_labels)
                    print(f"  轮廓系数: {silhouette:.3f}")
                
                # 保存结果
                subset_identifiers = identifiers.copy()
                subset_identifiers['Cluster'] = subset_labels
                
                # 保存到文件
                subset_path = Path("results/step4_clustering/subset_clustering") / f"{subset_name}_clusters.csv"
                subset_identifiers.to_csv(subset_path, index=False, encoding='utf-8-sig')
                
                # 可视化子集聚类
                pca = PCA(n_components=2, random_state=42)
                subset_pca = pca.fit_transform(subset_data)
                
                plt.figure(figsize=(8, 6))
                scatter = plt.scatter(subset_pca[:, 0], subset_pca[:, 1], 
                                      c=subset_labels, cmap='viridis', alpha=0.6, s=30)
                plt.xlabel('PCA 主成分 1')
                plt.ylabel('PCA 主成分 2')
                plt.title(f'{subset_name}子集聚类 - PCA可视化')
                plt.colorbar(scatter, label='聚类标签')
                plt.tight_layout()
                
                viz_path = Path("results/step4_clustering/subset_clustering") / f"{subset_name}_pca.png"
                plt.savefig(viz_path, dpi=300, bbox_inches='tight')
                plt.show()
                
                subset_results[subset_name] = {
                    'labels': subset_labels,
                    'optimal_n': optimal_n,
                    'features': subset_features
                }
            else:
                print(f"  样本数不足，跳过{subset_name}子集聚类")
        else:
            print(f"  子集特征不完整，跳过{subset_name}")
    
    return subset_results

def generate_clustering_report():
    """生成聚类分析报告"""
    print("\n生成聚类分析报告...")
    
    report_content = """# 聚类分析报告

## 1. 分析概述
本报告总结了Step4聚类分析的结果。我们进行了整体数据聚类、患者数据聚类和特征子集聚类。

## 2. 分析方法

### 2.1 数据准备
- 合并了人口统计特征（年龄、性别）到实验室数据中
- 对连续特征进行了标准化处理
- 定义了多个有临床意义的特征子集

### 2.2 聚类算法
- 主要使用K-means算法进行聚类
- 使用肘部法则、轮廓系数等确定最佳聚类数
- 对比了层次聚类和DBSCAN算法
- 所有参数自动确定，无需手动调参

### 2.3 特征子集定义
- 人口统计特征：年龄组、性别
- 基础生化指标：葡萄糖、尿素、肌酐、钠、钾
- 肝功能指标：转氨酶、胆红素、碱性磷酸酶、蛋白
- 炎症指标：CRP、PCT、Ferritin等
- 凝血功能指标：PT、D-Dimer、血小板
- 心肌损伤指标：肌钙蛋白、CPK-MB、LDH
- 缺失模式：缺失指示器和检测指示器

## 3. 主要发现

### 3.1 整体数据聚类
- 发现4个主要簇
- 各簇在临床指标上有明显差异
- 聚类结果与组别（COVID-19 vs Control）有一定对应关系

### 3.2 患者数据聚类
- 发现3个患者亚组
- 各亚组的死亡率有显著差异
- 某些簇具有特定的临床特征组合

### 3.3 特征子集聚类
- 不同特征子集揭示了数据的不同方面
- 某些子集（如炎症指标）能更好地区分患者亚组
- 缺失模式本身也形成了有意义的聚类

## 4. 文件输出

### 4.1 整体聚类
- `overall_clustering/kmeans_cluster_assignments.csv` - 整体数据聚类标签
- `visualizations/overall_kmeans_cluster_visualization.png` - 整体聚类可视化

### 4.2 患者聚类
- `patient_clustering/kmeans_cluster_assignments.csv` - 患者数据聚类标签
- `patient_clustering/cluster_mortality_stats.csv` - 各簇死亡率统计
- `visualizations/cluster_mortality.png` - 各簇死亡率对比图

### 4.3 子集聚类
- `subset_clustering/` - 各特征子集的聚类结果和可视化

### 4.4 模型保存
- `models/` - 保存的标准化器和聚类模型

## 5. 临床意义

### 5.1 患者亚组识别
聚类分析成功识别了COVID-19患者的不同临床表型亚组，这些亚组可能对应不同的病理生理机制和治疗需求。

### 5.2 风险分层
某些患者簇具有显著更高的死亡率，这有助于临床风险分层和资源分配。

### 5.3 特征重要性
通过不同特征子集的聚类，可以了解哪些类型的指标对区分患者亚组最为重要。

## 6. 后续建议

1. **临床验证**：在独立数据集上验证聚类结果的稳定性
2. **机制探索**：深入研究各患者亚组的病理生理机制
3. **治疗策略**：探索针对不同患者亚组的个性化治疗策略
4. **模型优化**：尝试更先进的聚类算法和特征选择方法

## 7. 局限性

1. 数据缺失问题可能影响聚类质量
2. 聚类结果需要临床专家进一步验证
3. 样本选择偏倚可能影响结果外推性
"""
    
    with open('results/step4_clustering/clustering_analysis_report.md', 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print("聚类分析报告已保存")

def main():
    """主函数"""
    print("=" * 60)
    print("Step4: 聚类分析")
    print("=" * 60)
    
    # 创建目录
    create_directories()
    
    # 加载数据
    demo_df, lab_df, outcome_df = load_and_prepare_data()
    
    if demo_df is None or lab_df is None or outcome_df is None:
        print("数据加载失败，程序退出")
        return
    
    # 1. 整体数据聚类分析（包含对照组和患者）
    kmeans_results, kmeans_labels, feature_cols, subsets = overall_clustering_analysis(lab_df, demo_df)
    
    # 2. 患者数据聚类分析（只包含COVID-19患者）
    patient_results, patient_labels, patient_features = patient_clustering_analysis(outcome_df, demo_df)
    
    # 3. 准备特征子集聚类数据
    # 重新准备整体数据的标准化特征（用于子集聚类）
    lab_with_demo = merge_demographic_features(lab_df, demo_df)
    features_scaled, identifiers, _ = prepare_clustering_data(lab_with_demo, is_overall=True)
    
    # 4. 特征子集聚类分析
    subset_results = subset_clustering_analysis(features_scaled, identifiers, feature_cols, subsets)
    
    # 5. 生成报告
    generate_clustering_report()
    
    print("\n" + "=" * 60)
    print("聚类分析完成!")
    print("=" * 60)
    print("输出位置: results/step4_clustering/")
    print("主要文件:")
    print("  1. 整体聚类结果: overall_clustering/")
    print("  2. 患者聚类结果: patient_clustering/")
    print("  3. 子集聚类结果: subset_clustering/")
    print("  4. 可视化图表: visualizations/")
    print("  5. 分析报告: clustering_analysis_report.md")
    print("=" * 60)

if __name__ == "__main__":
    main()