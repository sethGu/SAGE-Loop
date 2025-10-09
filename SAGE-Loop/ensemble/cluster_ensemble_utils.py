from sklearn.cluster import SpectralClustering
from scipy.sparse import lil_matrix
from sklearn.decomposition import NMF


from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import AgglomerativeClustering
import numpy as np

def cluster_ensemble_default(X, clusterers, true_labels, n_clusters=None, min_valid_models=2, top_k_models=7):
    """
    聚类集成函数（增强版）：融合多个聚类器结果，主动丢弃异常模型，返回最终标签及 ARI/NMI。
    """
    n_samples = X.shape[0]
    valid_clusterers = []
    metrics_list = []

    for model in clusterers:
        # 跳过 NMF + 负数情况
        if is_model_nmf(model):
            print(f"[Skip] Detected NMF in model: {type(model)}")
            continue

        print(f"[Try] model: {type(model)}")

        try:
            labels = model.fit_predict(X)

            # 检查输出标签合法性
            if labels is None or len(labels) != n_samples:
                print(f"[Drop] Invalid label length from model {type(model)}")
                continue
            if len(set(labels)) <= 1:
                print(f"[Drop] Trivial clustering (only one cluster) in model {type(model)}")
                continue

            ari = adjusted_rand_score(true_labels, labels)
            nmi = normalized_mutual_info_score(true_labels, labels)

            # 异常值检查
            if ari == 0 or nmi == 0 or nmi > 1:
                print(f"[Drop] Low or invalid scores from model {type(model)}: ARI={ari:.3f}, NMI={nmi:.3f}")
                continue

            valid_clusterers.append((model, labels))
            metrics_list.append((ari, nmi))

        except Exception as e:
            print(f"[Error] Model {type(model)} failed during fit_predict: {e}")
            continue

    # 保底检查
    if len(valid_clusterers) < min_valid_models:
        print(f"[警告] 仅保留 {len(valid_clusterers)} 个有效模型，低于设定阈值 {min_valid_models}，不再筛选")
    elif top_k_models is not None and len(valid_clusterers) > top_k_models:
        # 选择得分前 top_k_models（默认平均 ari+nmi 得分）
        scores = [0.5 * ari + 0.5 * nmi for ari, nmi in metrics_list]
        top_indices = np.argsort(scores)[-top_k_models:][::-1]
        valid_clusterers = [valid_clusterers[i] for i in top_indices]

    n_models = len(valid_clusterers)
    labels_list = [labels for _, labels in valid_clusterers]

    if n_models == 0:
        raise RuntimeError("所有模型均被丢弃，无法进行集成。请检查模型定义或输入数据。")

    # 构建共识矩阵
    consensus_matrix = np.zeros((n_samples, n_samples))
    for labels in labels_list:
        for i in range(n_samples):
            for j in range(n_samples):
                if labels[i] == labels[j]:
                    consensus_matrix[i, j] += 1
    consensus_matrix /= n_models

    # 距离矩阵 + 层次聚类
    distance_matrix = 1 - consensus_matrix
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters, metric='precomputed', linkage='average'
    )
    final_labels = clustering.fit_predict(distance_matrix)

    # 评估
    ensemble_ari = adjusted_rand_score(true_labels, final_labels)
    ensemble_nmi = normalized_mutual_info_score(true_labels, final_labels)

    return {
        "ensemble_metrics": {
            "ensemble_ari": ensemble_ari,
            "ensemble_nmi": ensemble_nmi,
            "used_model_count": n_models
        },
        "used_models": [m for m, _ in valid_clusterers]
    }


def cluster_ensemble_mcla(X, clusterers, true_labels, n_clusters=None, min_valid_models=2):
    """
    使用 MCLA（Meta-Clustering Algorithm）进行聚类集成（使用余弦相似度）。

    参数:
    - X: ndarray, shape (n_samples, n_features)
    - clusterers: list of clusterer objects (must implement fit_predict)
    - true_labels: ndarray, shape (n_samples,)
    - n_clusters: 最终聚类簇数（Meta-Cluster 数），如果为 None 则使用有效模型数
    - min_valid_models: 最少保留的模型数阈值，低于此值将报错
    """
    n_samples = X.shape[0]
    base_labels = []

    # Step 1: 筛选有效聚类器
    for model in clusterers:
        if is_model_nmf(model) and np.any(X < 0):
            print(f"[Skip] NMF模型不支持负值，跳过：{type(model)}")
            continue

        print(f"[Try] model: {type(model)}")

        try:
            labels = model.fit_predict(X)

            # 合法性检查
            if labels is None or len(labels) != n_samples:
                print(f"[Drop] 无效标签长度，模型: {type(model)}")
                continue
            if len(np.unique(labels)) <= 1:
                print(f"[Drop] 聚类类别数太少（<=1），模型: {type(model)}")
                continue

            base_labels.append(labels)

        except Exception as e:
            print(f"[Error] 模型 {type(model)} 执行失败: {e}")
            continue

    if len(base_labels) < min_valid_models:
        raise RuntimeError(f"[错误] 有效模型数不足（仅 {len(base_labels)} 个），无法进行 MCLA 集成")

    # Step 2: 构造 cluster_vectors（每个簇一个 one-hot 向量）
    cluster_vectors = []
    for labels in base_labels:
        for cluster_id in np.unique(labels):
            vec = (labels == cluster_id).astype(float)  # shape: (n_samples,)
            cluster_vectors.append(vec)
    cluster_vectors = np.array(cluster_vectors)  # shape: (n_clusters_all, n_samples)
    n_base_clusters = cluster_vectors.shape[0]

    # Step 3: 构建 base cluster 之间的相似度矩阵（Cosine 相似度）
    sim_matrix = np.zeros((n_base_clusters, n_base_clusters))
    norms = np.linalg.norm(cluster_vectors, axis=1)
    for i in range(n_base_clusters):
        for j in range(i, n_base_clusters):
            denom = norms[i] * norms[j]
            sim = 0.0 if denom == 0 else np.dot(cluster_vectors[i], cluster_vectors[j]) / denom
            sim_matrix[i, j] = sim_matrix[j, i] = sim

    # Step 4: 对 base clusters 聚类，生成 meta-cluster（即最终聚类簇）
    meta_n_clusters = n_clusters if n_clusters is not None else len(base_labels)
    sc = SpectralClustering(n_clusters=meta_n_clusters, affinity='precomputed', random_state=42)
    meta_labels = sc.fit_predict(sim_matrix)  # shape: (n_base_clusters,)

    # Step 5: 将 base cluster 向量投票映射到 meta-cluster 上，生成每个样本的最终聚类标签
    membership = np.zeros((n_samples, meta_n_clusters))
    for idx, vec in enumerate(cluster_vectors):
        meta_id = meta_labels[idx]
        membership[:, meta_id] += vec  # 每个 base cluster 给对应 meta-cluster 投票

    final_labels = np.argmax(membership, axis=1)  # 最终标签由投票决定

    # Step 6: 评估性能
    ari = adjusted_rand_score(true_labels, final_labels)
    nmi = normalized_mutual_info_score(true_labels, final_labels)

    return {
        "ensemble_metrics": {
            "ensemble_ari": ari,
            "ensemble_nmi": nmi,
            "used_model_count": len(base_labels)
        },
        "ensemble_labels": final_labels
    }


def cluster_ensemble_eac(X, clusterers, true_labels, n_clusters=None):
    """
    使用 Evidence Accumulation Clustering (EAC) 进行聚类集成。
    """
    n_samples = X.shape[0]
    n_models = len(clusterers)
    labels_list = []

    # 第一步：收集每个聚类器的结果
    for model in clusterers:
        labels = model.fit_predict(X)
        labels_list.append(labels)

    # 第二步：构建稀疏共识矩阵
    consensus_matrix = lil_matrix((n_samples, n_samples), dtype=np.float32)
    for labels in labels_list:
        for cluster_id in np.unique(labels):
            idx = np.where(labels == cluster_id)[0]
            for i in idx:
                consensus_matrix[i, idx] += 1

    # 除以聚类器个数，转化为频率矩阵
    consensus_matrix = consensus_matrix / n_models
    consensus_matrix = consensus_matrix.tocsr()
    distance_matrix = 1 - consensus_matrix.toarray()

    # 第三步：层次聚类（平均连接）
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        linkage="average"
    )
    final_labels = clustering.fit_predict(distance_matrix)

    # 第四步：评估指标
    ari = adjusted_rand_score(true_labels, final_labels)
    nmi = normalized_mutual_info_score(true_labels, final_labels)

    return {
        "ensemble_metrics": {
            "ensemble_ari": ari,
            "ensemble_nmi": nmi
        }
        # "ensemble_labels": final_labels
    }



def select_top_models_with_adaptive_threshold(test_ari_list, test_nmi_list, base_models, top_k=5, threshold_ratio=0.5):
    ari_array = np.array(test_ari_list)
    nmi_array = np.array(test_nmi_list)

    # 排除负数
    non_negative_indices = [i for i, (ari, nmi) in enumerate(zip(ari_array, nmi_array))
                            if ari > 0 and nmi > 0]
    if not non_negative_indices:
        print("⚠️ 没有非负指标模型")
        return []

    # 获取最大值并计算自适应阈值
    max_ari = np.max(ari_array[non_negative_indices])
    max_nmi = np.max(nmi_array[non_negative_indices])
    min_ari = max_ari * threshold_ratio
    min_nmi = max_nmi * threshold_ratio

    # 根据自适应阈值筛选模型
    valid_indices = [i for i in non_negative_indices if ari_array[i] >= min_ari and nmi_array[i] >= min_nmi]

    if not valid_indices:
        print("⚠️ 没有满足自适应阈值的模型")
        return []

    # 在满足条件的模型中选 top_k（按 ARI 降序）
    valid_ari = ari_array[valid_indices]
    sorted_order = np.argsort(valid_ari)[::-1]
    top_indices = [valid_indices[i] for i in sorted_order[:top_k]]

    return [base_models[i] for i in top_indices]



def is_model_nmf(model):
    if isinstance(model, NMF):
        return True
    try:
        if hasattr(model, 'model') and isinstance(model.model, NMF):
            return True
    except:
        pass
    return False


import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


def build_stacking_features(base_labels_list):
    """
    将多个聚类器的标签转化为 stacking 的输入特征（使用 One-Hot 编码）
    """
    base_labels_array = np.vstack(base_labels_list).T  # shape: (n_samples, n_models)
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    stacking_features = encoder.fit_transform(base_labels_array)
    return stacking_features, encoder


def cluster_ensemble_stacking_unsupervised(X, clusterers, n_clusters, test_size=0.2, min_valid_models=2):
    """
    无监督聚类 stacking 集成流程：使用基础聚类器输出训练 meta-learner，伪标签由多数投票产生

    参数：
    - X: 输入数据
    - clusterers: 聚类模型列表
    - n_clusters: 最终聚类簇数
    - test_size: 拆分验证集比例（用于 meta-learner 训练）

    返回：
    - stacking meta-learner 模型
    - 预测标签
    """
    n_samples = X.shape[0]
    base_labels = []

    for model in clusterers:
        try:
            labels = model.fit_predict(X)
            if labels is None or len(np.unique(labels)) <= 1:
                continue
            base_labels.append(labels)
        except:
            continue

    if len(base_labels) < min_valid_models:
        raise RuntimeError("有效聚类器不足")

    # 构造 stacking 特征
    stacking_X, encoder = build_stacking_features(base_labels)

    # 生成伪标签（多数投票）
    base_labels_array = np.vstack(base_labels).T  # shape: (n_samples, n_models)
    pseudo_labels = []
    for row in base_labels_array:
        counts = np.bincount(row)
        pseudo_labels.append(np.argmax(counts))
    pseudo_labels = np.array(pseudo_labels)

    # 拆分训练集/测试集进行 meta-learner 训练
    X_train, X_test, y_train, y_test = train_test_split(
        stacking_X, pseudo_labels, test_size=test_size, random_state=42
    )

    # 使用逻辑回归作为 meta-learner
    meta_learner = LogisticRegression(max_iter=500, random_state=42)
    meta_learner.fit(X_train, y_train)
    y_pred = meta_learner.predict(X_test)

    return {
        "predicted_labels": y_pred,
        "meta_model": meta_learner,
        "stacking_encoder": encoder
    }

import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

def stacking_cluster(X_aug, selected_base_models, n_clusters, y_true, method='kmeans', use_onehot=True, random_state=42):
    """
    X_aug: shape (n_samples, n_features)
    selected_base_models: 经过筛选的聚类模型列表
    n_clusters: 聚类数
    y_true: 真实标签（用于评估）
    method: 'kmeans', 'gmm', 'spectral' 等
    use_onehot: 是否用独热编码方式
    """
    # Step 1: 生成所有 base model 的聚类标签
    base_labels = []
    for model in selected_base_models:
        labels = model.fit_predict(X_aug)
        base_labels.append(labels.reshape(-1, 1))  # shape: (n_samples, 1)
    base_labels = np.concatenate(base_labels, axis=1)  # shape: (n_samples, n_base_models)

    # Step 2: 可选独热编码
    import sklearn
    if use_onehot:
        if sklearn.__version__ >= '1.2':
            encoder = OneHotEncoder(sparse_output=False)
        else:
            encoder = OneHotEncoder(sparse=False)
        labels_feature = encoder.fit_transform(base_labels)
    else:
        labels_feature = base_labels

    # Step 3: 拼接到原始特征
    X_stack = np.concatenate([X_aug, labels_feature], axis=1)  # shape: (n_samples, n_features+...)

    # Step 4: 再做一次聚类
    if method == 'kmeans':
        from sklearn.cluster import KMeans
        final_model = KMeans(n_clusters=n_clusters, random_state=random_state)
    elif method == 'gmm':
        from sklearn.mixture import GaussianMixture
        final_model = GaussianMixture(n_components=n_clusters, random_state=random_state)
    elif method == 'spectral':
        from sklearn.cluster import SpectralClustering
        final_model = SpectralClustering(n_clusters=n_clusters, random_state=random_state)
    else:
        raise ValueError("Unknown clustering method: {}".format(method))

    # 注意 KMeans 和 SpectralClustering API 不同
    if hasattr(final_model, "fit_predict"):
        final_labels = final_model.fit_predict(X_stack)
    else:
        final_model.fit(X_stack)
        final_labels = final_model.predict(X_stack)

    # Step 5: 评估性能
    ari = adjusted_rand_score(y_true, final_labels)
    nmi = normalized_mutual_info_score(y_true, final_labels)

    return {
        "ensemble_metrics": {
            "ensemble_ari": ari,
            "ensemble_nmi": nmi,
            "used_model_count": len(selected_base_models)
        },
        "ensemble_labels": final_labels
    }



import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import AgglomerativeClustering

def eac_ensemble(X_aug, selected_base_models, n_clusters, y_true=None, linkage='average'):
    """
    EAC聚类集成：基于共聚矩阵再聚类
    X_aug: 原始/增强特征
    selected_base_models: 聚类器列表
    n_clusters: 最终簇数
    y_true: 真实标签（可选，仅评估用）
    linkage: 层次聚类链接方式，'average'推荐
    """
    n_samples = X_aug.shape[0]
    base_labels = []
    for model in selected_base_models:
        labels = model.fit_predict(X_aug)
        base_labels.append(labels)
    base_labels = np.array(base_labels)  # shape: (n_base_models, n_samples)

    # 1. 统计每对样本共聚频率
    co_matrix = np.zeros((n_samples, n_samples), dtype=np.float32)
    n_models = base_labels.shape[0]
    for labels in base_labels:
        for c in np.unique(labels):
            idx = np.where(labels == c)[0]
            for i in idx:
                co_matrix[i, idx] += 1

    # 2. 归一化（得到样本对共聚概率）
    co_matrix = co_matrix / n_models

    # 3. 距离矩阵（1-共聚概率）
    dist_matrix = 1.0 - co_matrix

    # 4. 用 AgglomerativeClustering 对距离矩阵做二次聚类
    agg = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='precomputed',
        linkage=linkage
    )
    final_labels = agg.fit_predict(dist_matrix)

    # 5. 性能评估
    result = {
        "ensemble_labels": final_labels,
        "ensemble_metrics": None
    }
    if y_true is not None:
        ari = adjusted_rand_score(y_true, final_labels)
        nmi = normalized_mutual_info_score(y_true, final_labels)
        result["ensemble_metrics"] = {
            "ensemble_ari": ari,
            "ensemble_nmi": nmi,
            "used_model_count": len(selected_base_models)
        }
    return result


# 1. 优化后的 get_param_prompt（提示词升级，LLM能自我纠错）
def get_param_prompt(
    best_code, best_ari, dataset_description,
    X_aug, feature_columns, dataset_name=None, max_rows=10
):
    """
    生成适用于 LLM 参数优化的提示词（支持mycluster包裹真实聚类器，参数透传版）
    """
    import pandas as pd
    if isinstance(X_aug, np.ndarray):
        df_show = pd.DataFrame(X_aug, columns=feature_columns)
    else:
        df_show = X_aug.copy()
    table = df_show.head(max_rows).to_string(index=False)
    data_shape = X_aug.shape if hasattr(X_aug, 'shape') else (len(X_aug), len(feature_columns))
    if dataset_name is None:
        dataset_name = "unknown"

    prompt = (
        f"Here is the best clustering model code so far, with its current ARI (Adjusted Rand Index) score:\n\n"
        f"Current best ARI: {best_ari:.4f}\n\n"
        f"Model code:\n"
        f"```python\n{best_code}\n```\n"
        f"The downstream clustering task is based on the following dataset.\n"
        f"Dataset name: {dataset_name}\n"
        f"Dataset description:\n{dataset_description}\n\n"
        f"Feature names:\n{', '.join(feature_columns)}\n\n"
        f"Dataset shape: {data_shape}\n"
        f"Here are the first {max_rows} rows of the dataset used for clustering:\n{table}\n\n"
        "Please ONLY optimize the hyperparameters\n"
        "in the given clustering model code to further improve the ARI value.\n"
        "DO NOT change the algorithm type or model structure.\n"
        "Output only a new optimized Python code block.\n"
        "No explanation, only code!\n"
        "IMPORTANT CONTEXT:\n"
        "You are writing clustering model code in Python using scikit-learn version 1.6.1.\n"
        "STRICT REQUIREMENT:\n"
        "ONLY use parameters that are supported by scikit-learn version 1.6.1.\n"
        "DO NOT use any parameters that are deprecated or were only available in versions prior to 1.2.\n"
        "Refer ONLY to the scikit-learn 1.6.1 documentation for valid parameters and their default values.\n"
    )

    return prompt

"""
"At the end of your code block, output two lines:\n"
        "_params = dict(...), containing only parameters needed for mycluster's __init__ (such as n_clusters);\n"
        "_params_gmm = dict(...), containing parameters to be passed to the underlying GaussianMixture (such as n_init, max_iter, etc.).\n"
        "In your mycluster class, make sure you pass **_params_gmm to GaussianMixture in the constructor.\n"
        "For example: self.model = GaussianMixture(n_components=self.n_clusters, **_params_gmm)\n"
        "This will ensure parameter optimization only applies the correct arguments to the correct objects."
"""


def get_param_prompt_oldSklearn(
    best_code, best_ari, dataset_description,
    X_aug, feature_columns, dataset_name=None, max_rows=10
):
    """
    生成适用于 LLM 参数优化的提示词（sklearn 1.1.3 兼容版）
    """
    import pandas as pd
    if isinstance(X_aug, np.ndarray):
        df_show = pd.DataFrame(X_aug, columns=feature_columns)
    else:
        df_show = X_aug.copy()
    table = df_show.head(max_rows).to_string(index=False)
    data_shape = X_aug.shape if hasattr(X_aug, 'shape') else (len(X_aug), len(feature_columns))
    if dataset_name is None:
        dataset_name = "unknown"

    prompt = (
        f"Here is the best clustering model code so far, with its current ARI (Adjusted Rand Index) score:\n\n"
        f"Current best ARI: {best_ari:.4f}\n\n"
        f"Model code:\n"
        f"```python\n{best_code}\n```\n\n"
        f"The downstream clustering task is based on the following dataset.\n"
        f"Dataset name: {dataset_name}\n"
        f"Dataset description:\n{dataset_description}\n\n"
        f"Feature names:\n{', '.join(feature_columns)}\n\n"
        f"Dataset shape: {data_shape}\n"
        f"Here are the first {max_rows} rows of the dataset used for clustering:\n{table}\n\n"
        "Please ONLY optimize the hyperparameters (such as n_init, max_iter, bandwidth, linkage, affinity, etc.)\n"
        "in the given clustering model code to further improve the ARI value.\n"
        "DO NOT change the algorithm type or model structure.\n"
        "Output only a new optimized Python code block.\n"
        "No explanation, only code!\n"
        "At the end of your code block, output a single line _params = dict(...) containing all parameters you used for mycluster.\n"
        "\nIMPORTANT CONTEXT:\n"
        "You are writing clustering model code in Python using scikit-learn.\n"
        "The current environment is using scikit-learn version 1.1.3.\n"
        "Please refer to the scikit-learn 1.1.3 documentation when determining which parameters are valid for a clustering class.\n"
        "The affinity parameter is still valid in many clustering estimators in this version, and metric is not required.\n"
        "Do NOT use parameters that were introduced after version 1.1.3.\n"
        "When using AgglomerativeClustering, you may use the affinity parameter, but ensure parameter combinations are valid for your sklearn version.\n"
    )
    return prompt


from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import SpectralClustering
import numpy as np

def cluster_ensemble_caps(X, clusterers, true_labels, n_clusters=None, min_valid_models=2):
    """
    CAPS（Cluster Aggregation with Pairwise Similarities）聚类集成方法。

    参数:
    - X: ndarray, shape (n_samples, n_features)
    - clusterers: list of clusterer objects (must implement fit_predict)
    - true_labels: ndarray, shape (n_samples,)
    - n_clusters: 最终聚类簇数，默认用最多的基础聚类簇数
    - min_valid_models: 至少多少个聚类器有效，否则报错
    """
    n_samples = X.shape[0]
    base_labels = []

    # Step 1: 筛选有效聚类器
    for model in clusterers:
        try:
            labels = model.fit_predict(X)
            # 合法性检查
            if labels is None or len(labels) != n_samples:
                continue
            if len(np.unique(labels)) <= 1:
                continue
            base_labels.append(labels)
        except Exception as e:
            continue

    if len(base_labels) < min_valid_models:
        raise RuntimeError(f"有效模型数不足，仅 {len(base_labels)} 个，无法集成")

    n_models = len(base_labels)
    base_labels = np.array(base_labels)  # shape: (n_models, n_samples)

    # Step 2: 构建共聚类概率矩阵（co-association matrix）
    # S[i, j] = base_labels的每一行中i和j属于同一簇的比例
    S = np.zeros((n_samples, n_samples))
    for m in range(n_models):
        labels = base_labels[m]
        for i in range(n_samples):
            for j in range(i, n_samples):
                same_cluster = int(labels[i] == labels[j])
                S[i, j] += same_cluster
                if i != j:
                    S[j, i] += same_cluster  # 保证对称

    S = S / n_models  # 归一化到[0,1]

    # Step 3: 用共聚类概率矩阵作为相似度做聚类
    # 如果未指定n_clusters，取所有基础聚类中簇数的众数
    if n_clusters is None:
        from scipy.stats import mode
        all_n_clusters = [len(np.unique(labels)) for labels in base_labels]
        n_clusters = int(mode(all_n_clusters, keepdims=True)[0][0])

    sc = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=42)
    final_labels = sc.fit_predict(S)

    # Step 4: 性能评估
    ari = adjusted_rand_score(true_labels, final_labels)
    nmi = normalized_mutual_info_score(true_labels, final_labels)

    return {
        "ensemble_metrics": {
            "ensemble_ari": ari,
            "ensemble_nmi": nmi,
            "used_model_count": n_models
        },
        "ensemble_labels": final_labels
    }



from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import normalize
import numpy as np
import scipy
from scipy.sparse.csgraph import laplacian

def estimate_optimal_clusters(sim_matrix, max_clusters=10):
    """
    使用 eigen-gap heuristic 自动估算最佳聚类数（2 ~ max_clusters）
    """
    L = laplacian(sim_matrix, normed=True)
    eigvals, _ = scipy.linalg.eigh(L)
    eigvals = np.sort(eigvals)

    # 计算 eigengap
    gaps = np.diff(eigvals[:max_clusters])
    best_k = np.argmax(gaps) + 1  # 因为 gap 在 k-1 和 k 之间，索引 +1 才是簇数
    return best_k

def cluster_ensemble_mcla_wocluater(X, clusterers, true_labels=None, min_valid_models=2, max_meta_clusters=10):
    n_samples = X.shape[0]
    base_labels = []

    for model in clusterers:
        if is_model_nmf(model) and np.any(X < 0):
            print(f"[Skip] NMF 不支持负值，跳过 {type(model)}")
            continue

        print(f"[Try] model: {type(model)}")
        try:
            labels = model.fit_predict(X)
            if labels is None or len(labels) != n_samples:
                print(f"[Drop] 无效输出: {type(model)}")
                continue
            if len(np.unique(labels)) <= 1:
                print(f"[Drop] 类别过少: {type(model)}")
                continue
            base_labels.append(labels)
        except Exception as e:
            print(f"[Error] 执行失败: {type(model)} -> {e}")
            continue

    if len(base_labels) < min_valid_models:
        raise RuntimeError(f"[错误] 有效模型数不足（仅 {len(base_labels)} 个）")

    cluster_vectors = []
    for labels in base_labels:
        for cluster_id in np.unique(labels):
            vec = (labels == cluster_id).astype(float)
            cluster_vectors.append(vec)
    cluster_vectors = np.array(cluster_vectors)
    n_base_clusters = cluster_vectors.shape[0]

    norms = np.linalg.norm(cluster_vectors, axis=1)
    sim_matrix = (cluster_vectors @ cluster_vectors.T) / (norms[:, None] * norms[None, :] + 1e-10)
    sim_matrix = np.nan_to_num(sim_matrix)

    # 自动估计 meta-cluster 数
    meta_n_clusters = estimate_optimal_clusters(sim_matrix, max_clusters=min(max_meta_clusters, n_base_clusters))

    print(f"[Info] 自动估计 Meta-Clusters 数量: {meta_n_clusters}")

    meta_labels = SpectralClustering(n_clusters=meta_n_clusters, affinity='precomputed', random_state=42).fit_predict(sim_matrix)

    membership = np.zeros((n_samples, meta_n_clusters))
    for idx, vec in enumerate(cluster_vectors):
        membership[:, meta_labels[idx]] += vec

    final_labels = np.argmax(membership, axis=1)

    metrics = {
        "used_model_count": len(base_labels),
        "auto_meta_clusters": meta_n_clusters
    }
    if true_labels is not None:
        metrics["ensemble_ari"] = adjusted_rand_score(true_labels, final_labels)
        metrics["ensemble_nmi"] = normalized_mutual_info_score(true_labels, final_labels)

    return {
        "ensemble_metrics": metrics,
        "ensemble_labels": final_labels
    }
