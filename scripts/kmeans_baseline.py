#!/usr/bin/env python3
"""
K-Means Baseline for Topic Shift Detection
验证无监督聚类是否能提取出有语义可解释性的对话意图类别

步骤:
1. 加载三个数据集 (dialseg711, doc2dial, vhf)
2. 用 sentence-transformer 提取句子 embedding
3. K-Means (k=8) 聚类
4. 每个 cluster 提取 top-20 句子 + top-50 词
5. 输出到文件供人工检查
"""

import json
import os
import re
from collections import Counter
from typing import List, Dict, Tuple
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')  # headless rendering
import matplotlib.pyplot as plt

# 尝试导入 sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("请先安装: pip install sentence-transformers scikit-learn matplotlib numpy")
    raise

# 停用词列表
STOPWORDS = {
    'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'must', 'can', 'need', 'dare', 'ought',
    'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
    'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
    'between', 'under', 'again', 'further', 'then', 'once', 'here', 'there',
    'when', 'where', 'why', 'how', 'all', 'each', 'few', 'more', 'most',
    'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
    'so', 'than', 'too', 'very', 'just', 'and', 'but', 'if', 'or', 'because',
    'until', 'while', 'what', 'which', 'who', 'whom', 'this', 'that',
    'these', 'those', 'am', 'it', 'its', 'i', 'me', 'my', 'myself', 'we',
    'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself',
    'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
    'herself', 'they', 'them', 'their', 'theirs', 'themselves',
    's', 're', 'll', 've', 'd', 'm', 't', 'don', 'doesn', 'didn',
    'wasn', 'weren', 'won', 'wouldn', 'couldn', 'shouldn', 'isn', 'aren',
    'hasn', 'haven', 'hadn', 'let', 'us', 'say', 'said', 'also', 'get',
    'got', 'go', 'going', 'come', 'know', 'like', 'take', 'see', 'want',
    'yes', 'no', 'ok', 'okay', 'please', 'thank', 'thanks', 'hi', 'hello',
    'hey', 'please', 'right', 'well', 'ok', 'okay', 'alright', 'sure',
    'thing', 'things', 'something', 'anything', 'everything', 'nothing',
    'one', 'ones', 'two', 'three', 'first', 'second', 'third', 'now',
    'then', 'there', 'here', 'out', 'up', 'down', 'back', 'way', 'ways'
}


def load_dataset(filepath: str) -> List[Dict]:
    """加载 JSON 数据集"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_sentences(data: List[Dict], dataset_name: str) -> List[Dict]:
    """从对话数据中提取句子，保留上下文信息"""
    sentences = []
    for dialog in data:
        dial_id = dialog.get('dial_id', 0)
        utterances = dialog.get('utterances', [])

        for turn_idx, text in enumerate(utterances):
            sentences.append({
                'text': text.strip(),
                'dial_id': dial_id,
                'turn_idx': turn_idx,
                'dataset': dataset_name
            })
    return sentences


def get_embeddings(sentences: List[Dict], model_name: str = 'all-MiniLM-L6-v2') -> np.ndarray:
    """使用 sentence-transformer 获取句子 embedding"""
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    texts = [s['text'] for s in sentences]
    print(f"Encoding {len(texts)} sentences...")

    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
    return embeddings


def perform_clustering(embeddings: np.ndarray, n_clusters: int = 8) -> KMeans:
    """执行 K-Means 聚类"""
    print(f"\nPerforming K-Means clustering with k={n_clusters}...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(embeddings)
    return kmeans


def tokenize(text: str) -> List[str]:
    """简单的英文分词"""
    text = text.lower()
    words = re.findall(r'\b[a-z]+\b', text)
    words = [w for w in words if w not in STOPWORDS and len(w) > 2]
    return words


def analyze_cluster(
    sentences: List[Dict],
    embeddings: np.ndarray,
    labels: np.ndarray,
    cluster_id: int,
    top_n_sentences: int = 20,
    top_n_words: int = 50
) -> Dict:
    """分析单个 cluster，提取代表性句子和高频词"""
    cluster_indices = np.where(labels == cluster_id)[0]
    cluster_sentences = [sentences[i] for i in cluster_indices]

    # 按到聚类中心的距离排序，找最近的句子作为代表
    cluster_embeddings = embeddings[cluster_indices]
    centroid = cluster_embeddings.mean(axis=0)
    distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
    sorted_indices = np.argsort(distances)

    representative_sentences = [cluster_sentences[i] for i in sorted_indices[:top_n_sentences]]

    # 统计词频
    all_words = []
    for sent in cluster_sentences:
        all_words.extend(tokenize(sent['text']))

    word_freq = Counter(all_words)
    top_words = word_freq.most_common(top_n_words)

    return {
        'cluster_id': cluster_id,
        'size': len(cluster_sentences),
        'sentences': representative_sentences,
        'top_words': top_words,
        'datasets': Counter([s['dataset'] for s in cluster_sentences])
    }


def visualize_clusters(embeddings: np.ndarray, labels: np.ndarray, output_path: str):
    """使用 PCA 降维可视化聚类结果"""
    print("\nGenerating visualization...")
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    plt.figure(figsize=(14, 10))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                          c=labels, cmap='tab10', alpha=0.5, s=8)
    plt.colorbar(scatter, label='Cluster ID')
    plt.title('K-Means Clustering of Sentences (PCA Visualization)\nBaseline for Topic Shift Tag Discovery')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Visualization saved to: {output_path}")
    plt.close()


def save_cluster_results(clusters: List[Dict], output_path: str):
    """保存聚类分析结果到文本文件"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("K-Means Clustering Results for Topic Shift Detection\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total clusters: {len(clusters)}\n\n")

        for cluster in clusters:
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"CLUSTER {cluster['cluster_id']} (Size: {cluster['size']} sentences)\n")
            f.write("=" * 80 + "\n")

            # 数据集分布
            f.write("\n[Dataset Distribution]\n")
            for dataset, count in cluster['datasets'].most_common():
                f.write(f"  {dataset}: {count}\n")

            # 高频词
            f.write("\n[Top Words]\n")
            for word, freq in cluster['top_words'][:30]:
                f.write(f"  {word}: {freq}\n")

            # 代表性句子
            f.write("\n[Representative Sentences]\n")
            for i, sent in enumerate(cluster['sentences'], 1):
                f.write(f"\n  [{i}] [{sent['dataset']}] Dial {sent['dial_id']}, Turn {sent['turn_idx']}\n")
                f.write(f"  {sent['text']}\n")

            f.write("\n")

    print(f"Results saved to: {output_path}")


def main():
    # 配置
    DATA_DIR = "/home/sijin/maritime/dts/data/dataset"
    OUTPUT_DIR = "/home/sijin/maritime/dts/scripts/output"
    N_CLUSTERS = 8
    MODEL_NAME = 'all-MiniLM-L6-v2'  # 轻量级模型，384维

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 加载三个数据集
    print("=" * 60)
    print("Step 1: Loading datasets...")
    print("=" * 60)
    datasets = {
        'dialseg711': load_dataset(os.path.join(DATA_DIR, 'dialseg711.json')),
        'doc2dial':  load_dataset(os.path.join(DATA_DIR, 'doc2dial.json')),
        'vhf':       load_dataset(os.path.join(DATA_DIR, 'vhf.json'))
    }

    for name, data in datasets.items():
        print(f"  {name}: {len(data)} dialogs")

    # 2. 提取所有句子
    print("\n" + "=" * 60)
    print("Step 2: Extracting sentences...")
    print("=" * 60)
    all_sentences = []
    for name, data in datasets.items():
        sents = extract_sentences(data, name)
        all_sentences.extend(sents)
        print(f"  {name}: {len(sents)} sentences extracted")

    print(f"\n  Total sentences: {len(all_sentences)}")

    # 3. 获取 embeddings
    print("\n" + "=" * 60)
    print("Step 3: Computing embeddings...")
    print("=" * 60)
    embeddings = get_embeddings(all_sentences, model_name=MODEL_NAME)
    print(f"  Embedding shape: {embeddings.shape}")

    # 4. K-Means 聚类
    print("\n" + "=" * 60)
    print("Step 4: K-Means clustering...")
    print("=" * 60)
    kmeans = perform_clustering(embeddings, n_clusters=N_CLUSTERS)
    labels = kmeans.labels_

    # 打印每个 cluster 的大小
    print("\nCluster sizes:")
    for i in range(N_CLUSTERS):
        count = np.sum(labels == i)
        print(f"  Cluster {i}: {count} sentences ({count/len(labels)*100:.1f}%)")

    # 5. 分析每个 cluster
    print("\n" + "=" * 60)
    print("Step 5: Analyzing clusters...")
    print("=" * 60)
    clusters = []
    for i in range(N_CLUSTERS):
        cluster_info = analyze_cluster(
            all_sentences, embeddings, labels, i,
            top_n_sentences=20,
            top_n_words=50
        )
        clusters.append(cluster_info)
        print(f"  Cluster {i}: {cluster_info['size']} sentences")

    # 6. 保存结果
    results_path = os.path.join(OUTPUT_DIR, 'kmeans_cluster_results.txt')
    save_cluster_results(clusters, results_path)

    # 7. 可视化
    viz_path = os.path.join(OUTPUT_DIR, 'kmeans_visualization.png')
    visualize_clusters(embeddings, labels, viz_path)

    # 8. 额外：打印高频词的快速概览
    print("\n" + "=" * 60)
    print("Step 6: Quick Summary - Cluster Keywords")
    print("=" * 60)
    for cluster in clusters:
        top3 = [w for w, _ in cluster['top_words'][:5]]
        print(f"  Cluster {cluster['cluster_id']:2d} ({cluster['size']:4d}): {', '.join(top3)}")

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print(f"  Results : {results_path}")
    print(f"  Plot    : {viz_path}")
    print("=" * 80)
    print("\n>>> 打开 kmeans_cluster_results.txt 人工检查每个 cluster 的语义可解释性")


if __name__ == "__main__":
    main()
