import json
import os
import re
from collections import Counter, defaultdict
from typing import List, Dict
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

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
    'hey', 'right', 'well', 'alright', 'sure', 'thing', 'things',
    'something', 'anything', 'everything', 'nothing', 'one', 'ones',
    'two', 'three', 'first', 'second', 'third', 'now', 'back', 'way', 'ways',
    'oh', 'okay', 'ok', 'um', 'uh', 'ah', 'em', 'er', 'mm', 'hm', 'hmm',
    'll', 've', 'would', 'could', 'should', 'does', 'didn', 'don',
    'gonna', 'wanna', 'gotta', 'kinda', 'sorta', 'lot', 'lots',
    'really', 'actually', 'basically', 'literally', 'maybe', 'perhaps'
}


def load_dataset(filepath: str) -> List[Dict]:
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_sentences(data: List[Dict], dataset_name: str) -> List[Dict]:
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


def tokenize(text: str) -> List[str]:
    text = text.lower()
    words = re.findall(r'\b[a-z]+\b', text)
    return [w for w in words if w not in STOPWORDS and len(w) > 2]


def get_embeddings(sentences: List[Dict], model_name: str = 'all-MiniLM-L6-v2') -> np.ndarray:
    print(f"  Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    texts = [s['text'] for s in sentences]
    print(f"  Encoding {len(texts)} sentences...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
    return embeddings


def perform_clustering(embeddings: np.ndarray, n_clusters: int = 8) -> KMeans:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(embeddings)
    return kmeans


def analyze_cluster(
    sentences: List[Dict],
    embeddings: np.ndarray,
    labels: np.ndarray,
    cluster_id: int,
    top_n_sentences: int = 25,
    top_n_words: int = 60
) -> Dict:
    cluster_indices = np.where(labels == cluster_id)[0]
    cluster_sentences = [sentences[i] for i in cluster_indices]

    # 按到中心的距离选代表性句子
    cluster_embeddings = embeddings[cluster_indices]
    centroid = cluster_embeddings.mean(axis=0)
    distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
    sorted_indices = np.argsort(distances)

    representative_sentences = [cluster_sentences[i] for i in sorted_indices[:top_n_sentences]]

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
    }


def analyze_clusters_exclusive(
    sentences: List[Dict],
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
    top_n_sentences: int = 25,
    top_n_words_per_cluster: int = 60
) -> List[Dict]:
    cluster_word_freq = defaultdict(Counter)
    cluster_word_docs = defaultdict(lambda: defaultdict(set))
    cluster_total_words = defaultdict(int)

    for idx, sent in enumerate(sentences):
        cluster_id = labels[idx]
        words = tokenize(sent['text'])
        cluster_total_words[cluster_id] += len(words)
        for w in words:
            cluster_word_freq[cluster_id][w] += 1
            cluster_word_docs[cluster_id][w].add(sent['dial_id'])

    word_cluster_scores = defaultdict(dict)
    for cluster_id in range(n_clusters):
        cluster_size = np.sum(labels == cluster_id)
        if cluster_size == 0:
            continue
        total_words = cluster_total_words[cluster_id]
        if total_words == 0:
            continue
        for word, freq in cluster_word_freq[cluster_id].items():
            tf = freq / total_words
            doc_coverage = len(cluster_word_docs[cluster_id][word]) / cluster_size
            score = tf * (1 + np.log(doc_coverage + 1))
            word_cluster_scores[word][cluster_id] = score

    cluster_assigned_words = defaultdict(list)
    all_candidates = []
    for word, cluster_scores in word_cluster_scores.items():
        best_cluster = max(cluster_scores, key=cluster_scores.get)
        best_score = cluster_scores[best_cluster]
        best_freq = cluster_word_freq[best_cluster][word]
        all_candidates.append((best_score, word, best_cluster, best_freq))
    all_candidates.sort(reverse=True)

    for score, word, cluster_id, freq in all_candidates:
        cluster_assigned_words[cluster_id].append((word, score, freq))

    clusters = []
    for cluster_id in range(n_clusters):
        cluster_indices = np.where(labels == cluster_id)[0]
        cluster_sentences = [sentences[i] for i in cluster_indices]
        cluster_embeddings = embeddings[cluster_indices]
        centroid = cluster_embeddings.mean(axis=0)
        distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
        sorted_indices = np.argsort(distances)
        representative = [cluster_sentences[i] for i in sorted_indices[:top_n_sentences]]

        exclusive_words = cluster_assigned_words[cluster_id][:top_n_words_per_cluster]
        clusters.append({
            'cluster_id': cluster_id,
            'size': len(cluster_sentences),
            'sentences': representative,
            'top_words': [(w, f) for w, _, f in exclusive_words],
        })

    return clusters


def visualize_clusters(embeddings: np.ndarray, labels: np.ndarray,
                       dataset_name: str, output_path: str):
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    plt.figure(figsize=(14, 10))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                          c=labels, cmap='tab10', alpha=0.5, s=10)
    plt.colorbar(scatter, label='Cluster ID')
    plt.title(f'K-Means Clustering — {dataset_name} (PCA Visualization)')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"  Saved: {output_path}")
    plt.close()


def save_results(dataset_name: str, clusters: List[Dict], output_path: str):
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"K-Means Results — {dataset_name}  |  k=8\n")
        f.write("=" * 80 + "\n\n")

        for cluster in clusters:
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"CLUSTER {cluster['cluster_id']}  ({cluster['size']} sentences)\n")
            f.write("=" * 80 + "\n")

            # Top words
            f.write("\n[Top Words]\n")
            for word, freq in cluster['top_words'][:25]:
                bar = '█' * min(freq, 50)
                f.write(f"  {word:20s} {freq:5d}  {bar}\n")

            # Representative sentences
            f.write("\n[Representative Sentences]\n")
            for i, sent in enumerate(cluster['sentences'], 1):
                f.write(f"\n  [{i:2d}] [Dial {sent['dial_id']:3d}, Turn {sent['turn_idx']:2d}]\n")
                f.write(f"      {sent['text']}\n")

            f.write("\n")

    print(f"  Saved: {output_path}")


def export_topic_keywords(dataset_name: str, clusters: List[Dict], output_dir: str) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    payload = {
        "dataset": dataset_name,
        "clusters": []
    }
    for c in clusters:
        payload["clusters"].append({
            "cluster_id": c["cluster_id"],
            "size": c["size"],
            "top_words": [w for w, _ in c["top_words"][:25]]
        })
    payload["all_top_words"] = sorted(
        {w for cluster in payload["clusters"] for w in cluster["top_words"]}
    )
    out_path = os.path.join(output_dir, f"{dataset_name}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"  Saved: {out_path}")
    return payload


def analyze_dataset(name: str, data: List[Dict], model_name: str = 'all-MiniLM-L6-v2',
                    n_clusters: int = 8):
    """对单个数据集进行分析"""
    print(f"\n{'=' * 60}")
    print(f"Processing: {name}")
    print(f"{'=' * 60}")

    # 提取句子
    sentences = extract_sentences(data, name)
    print(f"  Dialogs: {len(data)}, Sentences: {len(sentences)}")

    # Embedding
    embeddings = get_embeddings(sentences, model_name)
    print(f"  Embedding shape: {embeddings.shape}")

    # K-Means
    print(f"  K-Means (k={n_clusters})...")
    kmeans = perform_clustering(embeddings, n_clusters=n_clusters)
    labels = kmeans.labels_

    print("\n  Cluster sizes:")
    for i in range(n_clusters):
        count = np.sum(labels == i)
        pct = count / len(labels) * 100
        bar = '▓' * int(pct / 2)
        print(f"    Cluster {i}: {count:5d} ({pct:5.1f}%) {bar}")

    # 分析每个 cluster（关键词排他分配）
    clusters = analyze_clusters_exclusive(
        sentences,
        embeddings,
        labels,
        n_clusters=n_clusters,
        top_n_sentences=25,
        top_n_words_per_cluster=60
    )

    # 快速关键词概览
    print(f"\n  Quick keyword summary:")
    for ci in clusters:
        top_words = [w for w, _ in ci['top_words'][:6]]
        print(f"    Cluster {ci['cluster_id']:2d} ({ci['size']:5d}): {', '.join(top_words)}")

    # 保存
    OUTPUT_DIR = "/home/sijin/maritime/dts/scripts/output"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    txt_path = os.path.join(OUTPUT_DIR, f'kmeans_{name}.txt')
    save_results(name, clusters, txt_path)

    viz_path = os.path.join(OUTPUT_DIR, f'kmeans_{name}_viz.png')
    visualize_clusters(embeddings, labels, name, viz_path)

    return clusters


def main():
    DATA_DIR = "/home/sijin/maritime/dts/data/dataset"
    N_CLUSTERS = 8
    MODEL_NAME = 'BAAI/bge-m3'

    datasets = {
        'dialseg711': load_dataset(os.path.join(DATA_DIR, 'dialseg711.json')),
        'doc2dial':  load_dataset(os.path.join(DATA_DIR, 'doc2dial.json')),
        'vhf':       load_dataset(os.path.join(DATA_DIR, 'vhf.json'))
    }

    print("=" * 70)
    print("  K-Means Baseline for Topic Shift Detection (Per-Dataset)")
    print(f"  Model: {MODEL_NAME}  |  Clusters: {N_CLUSTERS}")
    print("=" * 70)

    all_results = {}
    topic_export = {}
    for name, data in datasets.items():
        all_results[name] = analyze_dataset(name, data, MODEL_NAME, N_CLUSTERS)
        topic_export[name] = export_topic_keywords(
            name,
            all_results[name],
            output_dir="/home/sijin/maritime/dts/data/topic"
        )

    # 汇总对比
    print("\n" + "=" * 70)
    print("  Cross-Dataset Comparison Summary")
    print("=" * 70)
    for name, clusters in all_results.items():
        top_per_cluster = [','.join([w for w, _ in c['top_words'][:4]]) for c in clusters]
        print(f"\n  [{name}]")
        for i, kw in enumerate(top_per_cluster):
            print(f"    C{i}: {kw}")

    topic_all_path = "/home/sijin/maritime/dts/data/topic/topic_keywords.json"
    with open(topic_all_path, "w", encoding="utf-8") as f:
        json.dump(topic_export, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved: {topic_all_path}")

    print("\n" + "=" * 70)
    print("  All results saved to: /home/sijin/maritime/dts/scripts/output/")
    print("=" * 70)


if __name__ == "__main__":
    main()
