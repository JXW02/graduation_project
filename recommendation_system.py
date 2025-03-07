import numpy as np
from scipy import sparse
import pandas as pd
from sklearn.model_selection import KFold
import time
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


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
    print(f'Training RMSE: {rmse_train:.4f}')

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


def evaluate(R_pred, R_test, item_similarity, theta=0.6):
    """评估推荐系统的准确性和多样性"""
    # 计算RMSE
    rmse = np.sqrt(mean_squared_error(R_pred[R_test > 0], R_test[R_test > 0]))

    # 计算多样性
    diversity_scores = []
    for u in range(R_pred.shape[0]):
        candidate = [(i, R_pred[u, i]) for i in np.argsort(-R_pred[u])[:100]]
        if len(candidate) < 10: continue
        top10 = mmr(candidate, item_similarity, 10, theta)

        # 计算列表内相似度
        sim_sum = 0
        count = 0
        for i in range(10):
            for j in range(i + 1, 10):
                id1 = top10[i][0]
                id2 = top10[j][0]
                sim_sum += item_similarity[id1, id2]
                count += 1
        diversity_scores.append(1 - sim_sum / count if count > 0 else 1)

    return rmse, np.mean(diversity_scores)


def main():
    """主函数"""
    # 加载完整数据集
    x, y, R_shape, id_to_original = LoadData('ml-latest-small/ratings.csv')
    
    # 直接使用全部数据作为训练集
    R = ConvertToDense(x, y, R_shape)
    R_pred, item_similarity = userBased(R, R.copy(), 5, 'cosine')  # 复制矩阵避免原地修改

    # 参数测试配置
    theta_values = [0.0, 0.2, 0.5, 0.7, 1.0]
    results = []
    
    # 展示协同过滤原始推荐
    user_id = 1
    cf_candidates = [(i, R_pred[user_id-1, i]) for i in np.argsort(-R_pred[user_id-1])[:20]]
    print("\n协同过滤原始推荐Top20：")
    print([id_to_original[item[0]] for item in cf_candidates])  # 使用原始电影ID

    # 测试不同theta值
    for theta in theta_values:
        rmse, diversity = evaluate(R_pred, R, item_similarity, theta)
        results.append((theta, rmse, diversity))
        
        # 展示MMR重排结果
        mmr_top10 = mmr(cf_candidates.copy(), item_similarity, 10, theta)
        print(f"\nTheta={theta} MMR推荐Top10：")
        print([id_to_original[item[0]] for item in mmr_top10])

    # 绘制指标变化图
    plt.figure(figsize=(12,6))
    
    # RMSE变化
    plt.subplot(1,2,1)
    plt.plot([r[0] for r in results], [r[1] for r in results], 'bo-')
    plt.xlabel('Theta')
    plt.ylabel('RMSE')
    
    # 多样性变化
    plt.subplot(1,2,2)
    plt.plot([r[0] for r in results], [r[2] for r in results], 'ro-')
    plt.xlabel('Theta')
    plt.ylabel('Diversity')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()