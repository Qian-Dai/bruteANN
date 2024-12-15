import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
import os
import util.read_ivecs
import util.read_fvecs


# PCA 降维
def apply_pca(data, query, n_components=128):
    """
    使用 PCA 将 data 和 query 降维到指定维度。
    """
    pca = PCA(n_components=n_components)
    data_reduced = pca.fit_transform(data)
    query_reduced = pca.transform(query)
    return data_reduced, query_reduced

# 暴力搜索
def brute_force_search(data_reduced, query_reduced, top_k=1000):
    """
    对每个 query，暴力计算到 data 所有向量的欧氏距离，返回 top_k 近邻。
    """
    all_top_k = []
    for query in query_reduced:
        distances = euclidean_distances(query.reshape(1, -1), data_reduced).flatten()
        top_k_indices = np.argsort(distances)[:top_k]
        all_top_k.append(top_k_indices)
    return np.array(all_top_k)

# 计算查准率
def compute_precision(top_k_results, groundtruth, top_n=100):
    """
    比较 brute-force 搜索的 top_k 结果和 groundtruth 的 top_n，计算查准率。
    """
    precision_scores = []
    allwin=0
    notallwin =0
    for i, top_k in enumerate(top_k_results):
        true_top_n = set(groundtruth[i, :top_n])
        retrieved_top_k = set(top_k)
        intersection = len(true_top_n & retrieved_top_k)
        scores = round(intersection / top_n, 8)
        if scores == 1.0:
          allwin = allwin+1
        else:
            notallwin = notallwin +1
            print(scores)
        precision_scores.append(scores)
    print(allwin)
    print(notallwin)
    return np.mean(precision_scores)

# 主函数
if __name__ == "__main__":
    # 定义文件路径
    base_path = '../dataset/gist/gist_base.fvecs'
    query_path = '../dataset/gist/gist_query.fvecs'
    ground_truth_path = '../dataset/gist/gist_groundtruth.ivecs'

    # 读取数据集、查询集、groundtruth集
    dataVector = util.read_fvecs.read_Fvecs(base_path)
    queryVector = util.read_fvecs.read_Fvecs(query_path)
    ground_truth_Vector = util.read_ivecs.read_Ivecs(ground_truth_path)

    # PCA 降维到 128 维
    data_reduced, query_reduced = apply_pca(dataVector, queryVector, n_components=128)


    # 暴力搜索，找到 top1000
    top_k_results = brute_force_search(data_reduced, query_reduced, top_k=1000)

    # 计算查准率
    mean_precision = compute_precision(top_k_results, ground_truth_Vector, top_n=100)
    print(f"平均查准率（top100 包含在 top1000 中的比例）: {mean_precision * 100:.8f}%")






















