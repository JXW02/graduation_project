import math
import numpy as np
from scipy import sparse
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import os

def ConvertToDense(x, y, shape):
    """将稀疏数据转换为稠密矩阵"""
    row = x[:, 0]
    col = x[:, 1]
    data = y
    matrix_sparse = sparse.csr_matrix((data, (row, col)), shape=(shape[0] + 1, shape[1]))  # 添加占位符
    R = matrix_sparse.todense()
    R = R[1:, :]  # 去掉占位符
    R = np.asarray(R)
    return R


def LoadData(filename):
    """加载数据并生成用户-电影评分矩阵"""
    df = pd.read_csv(filename, sep=',')
    # 对电影ID重映射
    df['movieId'], item_mapping = pd.factorize(df['movieId'])
    id_to_original = dict(enumerate(item_mapping))  # 保存映射关系
    x = df[['userId', 'movieId']].values
    y = df['rating'].values
    n_users = len(df['userId'].unique())
    n_movies = len(df['movieId'].unique())
    R_shape = (n_users, n_movies)
    return x, y, R_shape, id_to_original


def LoadMovieGenres(filename, id_to_original):
    """加载电影类别信息"""
    # 检查文件是否存在
    if not os.path.exists(filename):
        print(f"警告: 找不到电影类别文件 {filename}")
        return {}
    
    try:
        movies_df = pd.read_csv(filename, sep=',')
        
        # 创建电影ID到类别的映射
        movie_to_genres = {}
        for _, row in movies_df.iterrows():
            movie_id = row['movieId']
            genres = row['genres'].split('|')
            movie_to_genres[movie_id] = genres
        
        # 转换为内部ID到类别的映射
        internal_id_to_genres = {}
        for internal_id, original_id in id_to_original.items():
            if original_id in movie_to_genres:
                internal_id_to_genres[internal_id] = movie_to_genres[original_id]
        
        print(f"成功加载了 {len(internal_id_to_genres)} 部电影的类别信息")
        return internal_id_to_genres
    except Exception as e:
        print(f"加载电影类别信息出错: {str(e)}")
        return {}


def fast_cosine_similarity(R, mask):
    """快速计算余弦相似度"""
    dot_product = np.dot(R, R.T)
    norms = np.sqrt(np.dot(mask * R ** 2, mask.T))
    norms[norms == 0] = np.inf  # 避免除以0
    similarity_matrix = dot_product / (norms * norms.T)
    np.fill_diagonal(similarity_matrix, 1)  # 确保对角线为1
    return similarity_matrix


def userBased(R_train, R_test, k=5, similarity_metric='cosine'):
    """基于用户的协同过滤推荐算法"""
    num_users, num_items = R_train.shape
    mask_train = R_train > 0
    mask_test = R_test > 0
    user_mean = np.true_divide(R_train.sum(1), mask_train.sum(1))
    user_mean = np.nan_to_num(user_mean)  # 分母为0时均值换成0

    # 计算用户相似度矩阵
    if similarity_metric == 'cosine':
        similarity = fast_cosine_similarity(R_train, mask_train)
    else:
        raise ValueError("Unsupported similarity metric. Choose 'cosine'.")

    # 为每个用户选择前k个相似用户作为邻居
    neighbors = np.argsort(-similarity, axis=1)[:, 1:k + 1]  # 排除自身

    # 预测评分
    R_pred = np.zeros((num_users, num_items))
    mask_union = mask_train | mask_test  # 取并集
    for u in range(num_users):
        for i in range(num_items):
            if mask_union[u, i]:
                neighbors_u = neighbors[u]
                mask_neighbors = R_train[neighbors_u, i] > 0
                if np.sum(mask_neighbors) == 0:
                    R_pred[u, i] = user_mean[u]
                else:
                    sim_u = similarity[u, neighbors_u][mask_neighbors]
                    ratings_u = R_train[neighbors_u, i][mask_neighbors]
                    R_pred[u, i] = np.dot(sim_u, ratings_u) / np.sum(np.abs(sim_u))

    # 裁剪评分范围
    R_pred[R_pred > 5] = 5.
    R_pred[R_pred < 1] = 1.

    # 计算训练集上的 RMSE
    preds_train = R_pred[mask_train]
    actual_train = R_train[mask_train]
    rmse_train = np.sqrt(mean_squared_error(actual_train, preds_train))
    print(f'训练集RMSE: {rmse_train:.4f}')

    # 计算物品相似度矩阵
    item_similarity = fast_cosine_similarity(R_train.T, mask_train.T)

    return R_pred, item_similarity


def mmr(candidate_items, item_similarity, k, theta):
    """MMR多样性重排序"""
    R_sorted = sorted(candidate_items, key=lambda x: x[1], reverse=True)
    S = [R_sorted.pop(0)]  # 初始化推荐列表

    for _ in range(k - 1):
        max_mmr = -np.inf
        best_item = None

        # 遍历剩余候选物品
        for idx, (item_id, rating) in enumerate(R_sorted):
            # 计算与已选物品的最大相似度
            sim_max = 0
            for s in S:
                sim = item_similarity[item_id, s[0]]
                if sim > sim_max:
                    sim_max = sim

            # 计算MMR得分
            mmr_score = theta * rating - (1 - theta) * sim_max

            # 选择最优项
            if mmr_score > max_mmr:
                max_mmr = mmr_score
                best_item = idx

        # 更新推荐列表
        S.append(R_sorted.pop(best_item))

    return S


def dpp_fast(kernel_matrix, max_length, epsilon=1E-10):
    """
    DPP快速贪婪算法实现
    :param kernel_matrix: 2-d array
    :param max_length: positive int
    :param epsilon: small positive scalar
    :return: list
    """
    item_size = kernel_matrix.shape[0]
    cis = np.zeros((max_length, item_size))
    di2s = np.copy(np.diag(kernel_matrix))
    selected_items = list()
    selected_item = np.argmax(di2s)
    selected_items.append(selected_item)
    while len(selected_items) < max_length:
        k = len(selected_items) - 1
        ci_optimal = cis[:k, selected_item]
        di_optimal = math.sqrt(di2s[selected_item])
        elements = kernel_matrix[selected_item, :]
        eis = (elements - np.dot(ci_optimal, cis[:k, :])) / di_optimal
        cis[k, :] = eis
        di2s -= np.square(eis)
        selected_item = np.argmax(di2s)
        if di2s[selected_item] < epsilon:
            break
        selected_items.append(selected_item)
    return selected_items


def dpp_with_theta(candidate_items, item_similarity, k, theta):
    """
    带theta参数的DPP多样性重排序，使用快速贪婪算法实现

    参数:
    - candidate_items: 候选物品列表，每项为(item_id, rating)
    - item_similarity: 物品相似度矩阵
    - k: 推荐列表长度
    - theta: 相关性与多样性的平衡参数 (0~1)
      - theta=0: 完全注重多样性
      - theta=1: 完全注重相关性

    返回:
    - 根据DPP算法选择的k个物品列表
    """
    # 按评分排序并取前100个候选项
    candidates = sorted(candidate_items, key=lambda x: x[1], reverse=True)[:min(100, len(candidate_items))]

    if len(candidates) < k:
        return candidates  # 如果候选项不足k个，直接返回所有候选

    # 提取物品ID和评分
    item_ids = [item[0] for item in candidates]
    ratings = np.array([item[1] for item in candidates])

    # 归一化评分
    if np.max(ratings) > np.min(ratings):
        quality = (ratings - np.min(ratings)) / (np.max(ratings) - np.min(ratings))
    else:
        quality = np.ones_like(ratings)

    # 构建核矩阵L
    n = len(candidates)
    L = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            # 计算相似度作为多样性的反面
            sim = item_similarity[item_ids[i], item_ids[j]]
            diversity = 1.0 - sim

            # 根据theta参数平衡相关性和多样性
            # 当theta=1时: 只考虑相关性
            # 当theta=0时: 只考虑多样性
            if i == j:  # 对角线: 物品的质量
                L[i, j] = theta * quality[i] + (1 - theta) * 0.1  # 确保对角元素非零
            else:  # 非对角线: 物品间的相关性与多样性
                # 使用核函数构建L矩阵
                quality_term = np.sqrt(quality[i] * quality[j])
                L[i, j] = (theta * quality_term + (1 - theta)) * diversity

    # 添加微小扰动确保正定性
    L = L + np.eye(n) * 1e-6

    # 使用快速DPP算法选择物品
    selected_indices = dpp_fast(L, k)

    # 返回选中的物品
    return [candidates[idx] for idx in selected_indices]


def calculate_ndcg(recommended_items, relevant_items, k=10):
    """
    计算NDCG@k

    参数:
    - recommended_items: 推荐项目列表，格式为[(item_id, rating),...]
    - relevant_items: 相关项目列表，格式为{item_id: rating, ...}，评分高于阈值的项目
    - k: 计算前k个推荐的NDCG

    返回:
    - NDCG@k值
    """
    # 提取推荐项目ID列表
    rec_ids = [item[0] for item in recommended_items[:k]]

    # 计算DCG
    dcg = 0
    for i, item_id in enumerate(rec_ids):
        if item_id in relevant_items:
            # 使用相关性评分作为增益
            relevance = relevant_items[item_id]
            # 使用2^rel - 1作为增益
            gain = 2 ** relevance - 1
            # 位置折损
            dcg += gain / np.log2(i + 2)  # i+2是因为log的底数从2开始(排名从1开始)

    # 按相关性排序的理想推荐列表
    ideal_items = sorted(relevant_items.items(), key=lambda x: x[1], reverse=True)

    # 计算IDCG
    idcg = 0
    for i, (_, relevance) in enumerate(ideal_items[:k]):
        gain = 2 ** relevance - 1
        idcg += gain / np.log2(i + 2)

    # 计算NDCG
    if idcg > 0:
        return dcg / idcg
    else:
        return 0  # 如果没有相关项目，返回0


def calculate_category_diversity(recommended_items, item_to_genres):
    """
    计算推荐列表中的类别多样性

    参数:
    - recommended_items: 推荐项目列表，格式为[(item_id, rating),...]
    - item_to_genres: 项目到类别的映射，格式为{item_id: [genre1, genre2, ...], ...}

    返回:
    - 类别多样性分数 (0-1之间，越高表示多样性越好)
    """
    if not recommended_items or not item_to_genres:
        return 0

    # 提取推荐项目ID列表
    rec_ids = [item[0] for item in recommended_items]

    # 统计所有出现的类别
    all_genres = set()
    valid_items = 0

    for item_id in rec_ids:
        if item_id in item_to_genres:
            genres = item_to_genres[item_id]
            all_genres.update(genres)
            valid_items += 1

    # 如果没有有效项目，返回0
    if valid_items == 0:
        return 0

    # 计算平均每个项目的唯一类别数
    return len(all_genres) / valid_items


def evaluate_with_method(R_pred, R_test, item_similarity, item_to_genres, theta=0.6, method='mmr', k=10, rating_threshold=3.5):
    """
    评估推荐系统的准确性和多样性

    参数:
    - R_pred: 预测评分矩阵
    - R_test: 测试评分矩阵
    - item_similarity: 物品相似度矩阵
    - item_to_genres: 项目到类别的映射
    - theta: MMR/DPP的平衡参数
    - method: 'mmr'或'dpp'
    - k: 推荐列表长度
    - rating_threshold: 将项目视为相关的评分阈值

    返回:
    - 评估指标元组: (rmse, ils_diversity, ndcg, category_diversity)
    """
    # 计算RMSE
    rmse = np.sqrt(mean_squared_error(R_pred[R_test > 0], R_test[R_test > 0]))

    # 收集各项指标
    ils_diversity_scores = []  # 列表内相似度多样性
    ndcg_scores = []           # NDCG
    category_diversity_scores = []  # 类别多样性

    for u in range(R_pred.shape[0]):
        # 获取候选项
        candidate = [(i, R_pred[u, i]) for i in np.argsort(-R_pred[u])[:100]]
        if len(candidate) < k:
            continue

        # 获取用户相关项目 (测试集中评分高于阈值的项目)
        relevant_items = {}
        for i in range(R_test.shape[1]):
            if R_test[u, i] >= rating_threshold:
                relevant_items[i] = R_test[u, i]

        # 如果没有相关项目，跳过此用户
        if not relevant_items:
            continue

        # 根据指定方法进行排序
        if method == 'mmr':
            top_k = mmr(candidate, item_similarity, k, theta)
        elif method == 'dpp':
            top_k = dpp_with_theta(candidate, item_similarity, k, theta)
        else:
            # 默认使用原始排序
            top_k = sorted(candidate, key=lambda x: x[1], reverse=True)[:k]

        # 确保top_k包含足够的元素
        if len(top_k) < 2:
            continue

        # 计算NDCG
        ndcg = calculate_ndcg(top_k, relevant_items, k)
        ndcg_scores.append(ndcg)

        # 计算类别多样性
        cat_diversity = calculate_category_diversity(top_k, item_to_genres)
        category_diversity_scores.append(cat_diversity)

        # 计算列表内相似度 (ILS) 多样性
        sim_sum = 0
        count = 0
        for i in range(len(top_k)):
            for j in range(i + 1, len(top_k)):
                id1 = top_k[i][0]
                id2 = top_k[j][0]
                sim_sum += item_similarity[id1, id2]
                count += 1

        # 多样性 = 1 - 平均相似度
        if count > 0:
            ils_diversity_scores.append(1 - sim_sum / count)

    # 确保有足够的用户以计算平均值
    if not ils_diversity_scores:
        ils_diversity = 0
    else:
        ils_diversity = np.mean(ils_diversity_scores)

    if not ndcg_scores:
        ndcg_avg = 0
    else:
        ndcg_avg = np.mean(ndcg_scores)

    if not category_diversity_scores:
        cat_diversity_avg = 0
    else:
        cat_diversity_avg = np.mean(category_diversity_scores)

    return rmse, ils_diversity, ndcg_avg, cat_diversity_avg


def main():
    """主函数"""
    # 加载完整数据集
    x, y, R_shape, id_to_original = LoadData('ml-latest-small/ratings.csv')

    # 加载电影类别信息
    movie_genres = LoadMovieGenres('ml-latest-small/movies.csv', id_to_original)

    # 直接使用全部数据作为训练集
    R = ConvertToDense(x, y, R_shape)
    R_pred, item_similarity = userBased(R, R.copy(), 5, 'cosine')  # 复制矩阵避免原地修改

    # 参数测试配置
    theta_values = [0.0, 0.2, 0.5, 0.7, 1.0]
    mmr_results = []
    dpp_results = []

    # 展示协同过滤原始推荐
    user_id = 1
    cf_candidates = [(i, R_pred[user_id-1, i]) for i in np.argsort(-R_pred[user_id-1])[:20]]
    print("\n协同过滤原始推荐Top20：")
    print([id_to_original[item[0]] for item in cf_candidates])  # 使用原始电影ID

    # 测试不同theta值 - MMR算法
    print("\n===== MMR算法结果 =====")
    for theta in theta_values:
        try:
            rmse, ils_div, ndcg, cat_div = evaluate_with_method(
                R_pred, R, item_similarity, movie_genres, theta, 'mmr')
            mmr_results.append((theta, rmse, ils_div, ndcg, cat_div))

            # 展示MMR重排结果
            mmr_top10 = mmr(cf_candidates.copy(), item_similarity, 10, theta)
            print(f"\nTheta={theta} MMR推荐Top10：")
            print([id_to_original[item[0]] for item in mmr_top10])
            print(f"RMSE: {rmse:.4f}, ILS多样性: {ils_div:.4f}, NDCG: {ndcg:.4f}, 类别多样性: {cat_div:.4f}")
        except Exception as e:
            print(f"MMR算法在theta={theta}时出错: {str(e)}")
            mmr_results.append((theta, 0, 0, 0, 0))

    # 测试不同theta值 - DPP算法
    print("\n===== DPP算法结果 =====")
    for theta in theta_values:
        try:
            rmse, ils_div, ndcg, cat_div = evaluate_with_method(
                R_pred, R, item_similarity, movie_genres, theta, 'dpp')
            dpp_results.append((theta, rmse, ils_div, ndcg, cat_div))

            # 展示DPP重排结果
            dpp_top10 = dpp_with_theta(cf_candidates.copy(), item_similarity, 10, theta)
            print(f"\nTheta={theta} DPP推荐Top10：")
            print([id_to_original[item[0]] for item in dpp_top10])
            print(f"RMSE: {rmse:.4f}, ILS多样性: {ils_div:.4f}, NDCG: {ndcg:.4f}, 类别多样性: {cat_div:.4f}")
        except Exception as e:
            print(f"DPP算法在theta={theta}时出错: {str(e)}")
            dpp_results.append((theta, 0, 0, 0, 0))

    # 绘制MMR和DPP的比较图
    plt.figure(figsize=(15, 10))  # 调整画布尺寸

    # RMSE比较
    plt.subplot(2, 2, 1)
    plt.plot([r[0] for r in mmr_results], [r[1] for r in mmr_results], 'bo-', label='MMR')
    plt.plot([r[0] for r in dpp_results], [r[1] for r in dpp_results], 'ro-', label='DPP')
    plt.xlabel('Parameter Value (Theta)')
    plt.ylabel('RMSE')
    plt.title('MMR vs DPP: RMSE Comparison')
    plt.legend()

    # ILS多样性比较
    plt.subplot(2, 2, 2)
    plt.plot([r[0] for r in mmr_results], [r[2] for r in mmr_results], 'bo-', label='MMR')
    plt.plot([r[0] for r in dpp_results], [r[2] for r in dpp_results], 'ro-', label='DPP')
    plt.xlabel('Parameter Value (Theta)')
    plt.ylabel('ILS Diversity')
    plt.title('MMR vs DPP: ILS Diversity Comparison')
    plt.legend()

    # NDCG比较
    plt.subplot(2, 2, 3)
    plt.plot([r[0] for r in mmr_results], [r[3] for r in mmr_results], 'bo-', label='MMR')
    plt.plot([r[0] for r in dpp_results], [r[3] for r in dpp_results], 'ro-', label='DPP')
    plt.xlabel('Parameter Value (Theta)')
    plt.ylabel('NDCG')
    plt.title('MMR vs DPP: NDCG Comparison')
    plt.legend()

    # 类别多样性比较
    plt.subplot(2, 2, 4)
    plt.plot([r[0] for r in mmr_results], [r[4] for r in mmr_results], 'bo-', label='MMR')
    plt.plot([r[0] for r in dpp_results], [r[4] for r in dpp_results], 'ro-', label='DPP')
    plt.xlabel('Parameter Value (Theta)')
    plt.ylabel('Category Diversity')
    plt.title('MMR vs DPP: Category Diversity Comparison')
    plt.legend()

    plt.tight_layout()
    plt.savefig('mmr_dpp_evaluation.png', dpi=300)
    plt.show()

    # 打印对比结果表格
    print("\n===== MMR vs DPP 对比结果 =====")
    print("| 参数值 | MMR-RMSE | MMR-ILS多样性 | MMR-NDCG | MMR-类别多样性 | DPP-RMSE | DPP-ILS多样性 | DPP-NDCG | DPP-类别多样性 |")
    print("|--------|----------|--------------|----------|--------------|----------|--------------|----------|--------------|")
    for i in range(len(theta_values)):
        mmr_data = mmr_results[i] if i < len(mmr_results) else (theta_values[i], 0, 0, 0, 0)
        dpp_data = dpp_results[i] if i < len(dpp_results) else (theta_values[i], 0, 0, 0, 0)

        theta, mmr_rmse, mmr_ils, mmr_ndcg, mmr_cat = mmr_data
        _, dpp_rmse, dpp_ils, dpp_ndcg, dpp_cat = dpp_data

        print(f"| {theta:.1f}    | {mmr_rmse:.4f}  | {mmr_ils:.4f}      | {mmr_ndcg:.4f}  | {mmr_cat:.4f}      | {dpp_rmse:.4f}  | {dpp_ils:.4f}      | {dpp_ndcg:.4f}  | {dpp_cat:.4f}      |")


if __name__ == '__main__':
    main()