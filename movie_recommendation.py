import numpy as np
from scipy import sparse
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MultiLabelBinarizer


def ConvertToDense(x, y, shape):
    """将稀疏数据转换为稠密矩阵"""
    # 用户id 电影id 评分
    row = x[:, 0]
    col = x[:, 1]
    data = y
    matrix_sparse = sparse.csr_matrix((data, (row, col)), shape=(shape[0] + 1, shape[1]))  # 添加占位符
    R = matrix_sparse.todense()
    R = R[1:, :]  # 去掉占位符
    R = np.asarray(R)
    return R


def LoadMovieData():
    """加载数据并生成用户-电影评分矩阵"""
    movies_df = pd.read_csv('ml-latest-small/movies.csv')
    ratings_df = pd.read_csv('ml-latest-small/ratings.csv')
    # 为减少电影无评分而占用矩阵空间，对电影ID重映射
    ratings_df['movieId'], item_mapping = pd.factorize(ratings_df['movieId'])
    id_to_original = dict(enumerate(item_mapping))  # 保存映射关系
    x = ratings_df[['userId', 'movieId']].values
    y = ratings_df['rating'].values
    n_users = len(ratings_df['userId'].unique())
    n_movies = len(ratings_df['movieId'].unique())
    R_shape = (n_users, n_movies)
    print(f"数据集大小: {len(ratings_df)} 条评分, {n_users} 个用户, {n_movies} 部动漫")
    # 返回：[用户id，电影id](用户id从1开始，电影id从0开始)，评分，评分矩阵形状，电影id映射关系
    return x, y, R_shape, id_to_original, movies_df, ratings_df


def genre_one_hot(movies_df, movie_ids):
    """为每部电影创建种类的One-Hot编码"""
    # 将所有电影的种类合并成一个列表
    all_genres = []
    for genres in movies_df['genres'].str.split('|'):
        all_genres.extend(genres)

    # 获取种类集合
    unique_genres = sorted(list(set(all_genres)))

    # 将种类字符串转换为列表
    movies_df['genre_list'] = movies_df['genres'].str.split('|')

    # 使用MultiLabelBinarizer进行One-Hot编码
    mlb = MultiLabelBinarizer()
    # item_features = mlb.fit_transform(movies_df['genre_list'])

    # 创建电影ID到索引的映射
    id_to_index = {}
    filtered_movies = movies_df[movies_df['movieId'].isin(movie_ids)]

    # 为指定的电影生成特征
    item_features = mlb.fit_transform(filtered_movies['genre_list'])

    # 创建电影ID到特征索引的映射
    for idx, movie_id in enumerate(filtered_movies['movieId']):
        id_to_index[movie_id] = idx

    return item_features, unique_genres, id_to_index


def cosine_similarity(A, B):
    # 计算点积
    dot_product = np.dot(A, B.T)

    # 计算每个向量的范数
    norm_A = np.linalg.norm(A, axis=1, keepdims=True)
    norm_B = np.linalg.norm(B, axis=1, keepdims=True)

    # 避免除以零
    norm_A[norm_A == 0] = 1e-10
    norm_B[norm_B == 0] = 1e-10

    # 计算余弦相似度
    similarity_matrix = dot_product / np.dot(norm_A, norm_B.T)

    # 保证对角线为1
    np.fill_diagonal(similarity_matrix, 1)

    return similarity_matrix


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

    return R_pred


def mmr(candidate_items, item_similarity, k, theta):
    """MMR多样性重排序"""
    # 将预测分数由高到低排序，初始化推荐列表S
    R_sorted = sorted(candidate_items, key=lambda x: x[1], reverse=True)
    S = [R_sorted.pop(0)]

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


def dpp(item_features, rewards, k, theta):
    """
    基于Cholesky分解的k-DPP算法实现

    参数:
    - item_features: 物品特征矩阵，每行是一个物品的特征向量
    - rewards: 物品的精排分数
    - k: 要选择的物品数量
    - theta: 权衡相关性和多样性的参数，范围[0,1]

    返回:
    - 选中的k个物品的索引
    """
    n = len(rewards)  # 物品数量

    # 步骤2: 计算相似度矩阵A
    A = cosine_similarity(item_features, item_features)

    # 步骤3: 选择reward最高的物品作为初始物品
    i = np.argmax(rewards)
    S = [i]  # 初始化集合S
    L = np.array([[1.0]])  # 初始化Cholesky分解矩阵L

    # 剩余候选物品集合
    R = list(range(n))
    R.remove(i)

    # 步骤4: 迭代选择剩余的k-1个物品
    for t in range(1, k):
        if not R:  # 如果没有剩余物品可选
            break

        best_score = float('-inf')
        best_item = None
        best_c = None
        best_d = None

        # 步骤4(a): 对每个剩余物品计算分数
        for i in R:
            # I. 获取矩阵A_S∪{i}的最后一行[a_i, 1]
            a_i = np.array([A[i, j] for j in S])

            # II. 求解线性方程组a_i = L·c_i
            c_i = np.linalg.solve(L, a_i)

            # III. 计算d_i^2 = 1 - c_i^T·c_i
            d_i_squared = 1 - np.dot(c_i, c_i.T)

            # 确保数值稳定性
            if d_i_squared <= 0:
                d_i_squared = 1e-10

            # 步骤4(b): 计算分数并找到最佳物品
            score = theta * rewards[i] + (1 - theta) * np.log(d_i_squared)

            if score > best_score:
                best_score = score
                best_item = i
                best_c = c_i
                best_d = np.sqrt(d_i_squared)

        if best_item is None:
            break

        # 步骤4(c): 更新集合S
        S.append(best_item)
        R.remove(best_item)

        # 步骤4(d): 更新Cholesky分解矩阵L
        L_new = np.zeros((len(S), len(S)))
        L_new[:-1, :-1] = L
        L_new[-1, :-1] = best_c
        L_new[-1, -1] = best_d
        L = L_new

    # 步骤5: 返回选中的k个物品
    return S


def evaluate_cf_accuracy(R_pred, R_actual):
    """评估协同过滤预测结果的准确度"""
    # 只评估实际有评分的位置
    mask = R_actual > 0

    # 计算RMSE (Root Mean Squared Error)
    rmse = np.sqrt(mean_squared_error(R_actual[mask], R_pred[mask]))

    # 计算MAE (Mean Absolute Error)
    mae = np.mean(np.abs(R_actual[mask] - R_pred[mask]))

    # 计算预测评分与实际评分的相关系数
    correlation = np.corrcoef(R_actual[mask], R_pred[mask])[0, 1]

    # 计算预测准确率 (将评分四舍五入到最近的整数，然后计算准确匹配的比例)
    rounded_pred = np.round(R_pred[mask])
    rounded_pred[rounded_pred > 5] = 5
    rounded_pred[rounded_pred < 1] = 1
    accuracy = np.mean(rounded_pred == R_actual[mask])

    return {
        'rmse': rmse,
        'mae': mae,
        'correlation': correlation,
        'accuracy': accuracy
    }


def calculate_ndcg(recommended_items, ideal_items, k=10):
    """计算NDCG (Normalized Discounted Cumulative Gain)"""
    # 获取推荐项目的评分
    rec_ratings = [item[1] for item in recommended_items[:k]]
    # 获取理想排序的评分
    ideal_ratings = [item[1] for item in ideal_items[:k]]

    # 计算DCG
    dcg = 0
    for i, rating in enumerate(rec_ratings):
        dcg += rating / np.log2(i + 2)  # i+2是因为log2(1)=0

    # 计算IDCG
    idcg = 0
    for i, rating in enumerate(ideal_ratings):
        idcg += rating / np.log2(i + 2)

    # 计算NDCG
    if idcg == 0:
        return 0
    return dcg / idcg


def calculate_internal_similarity(items, similarity_matrix):
    """计算推荐列表的内部相似性"""
    if len(items) < 2:
        return 0

    total_sim = 0
    count = 0
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            id1 = items[i][0]
            id2 = items[j][0]
            total_sim += similarity_matrix[id1, id2]
            count += 1

    return total_sim / count if count > 0 else 0


def calculate_category_coverage(items, movies_df):
    """计算推荐列表的类别覆盖率"""
    # 获取所有可能的类别
    all_genres = set()
    for genres in movies_df['genres'].str.split('|'):
        all_genres.update(genres)

    # 获取推荐列表中的类别
    recommended_genres = set()
    for item in items:
        movie_id = item[0]
        genres = movies_df[movies_df['movieId'] == movie_id]['genres'].values
        if len(genres) > 0:
            recommended_genres.update(genres[0].split('|'))

    # 计算覆盖率
    return len(recommended_genres) / len(all_genres) if len(all_genres) > 0 else 0


def evaluate_diversity_relevance(recommended_items, ideal_items, similarity_matrix, movies_df):
    """评估推荐列表的相关性和多样性"""
    # 相关性指标
    avg_rating = np.mean([item[1] for item in recommended_items])
    ndcg = calculate_ndcg(recommended_items, ideal_items)

    # 多样性指标
    internal_similarity = calculate_internal_similarity(recommended_items, similarity_matrix)
    diversity = 1 - internal_similarity  # 多样性是相似性的补集
    category_coverage = calculate_category_coverage(recommended_items, movies_df)

    return {
        'avg_rating': avg_rating,
        'ndcg': ndcg,
        'internal_similarity': internal_similarity,
        'diversity': diversity,
        'category_coverage': category_coverage
    }


def main():
    # 加载完整数据集
    x, y, R_shape, id_to_original, movies_df, ratings_df = LoadMovieData()

    # 基于用户的协同过滤，生成用户对电影的预测评分
    R = ConvertToDense(x, y, R_shape)
    R_pred = userBased(R, R.copy(), 5, 'cosine')

    # 评估协同过滤预测结果的准确度
    accuracy_metrics = evaluate_cf_accuracy(R_pred, R)
    print("\n===== 协同过滤预测准确度评估 =====")
    print(f"RMSE: {accuracy_metrics['rmse']:.4f}")
    print(f"MAE: {accuracy_metrics['mae']:.4f}")
    print(f"相关系数: {accuracy_metrics['correlation']:.4f}")
    print(f"准确率: {accuracy_metrics['accuracy']:.4f}")

    # 参数测试配置
    theta_values = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    mmr_results = []
    dpp_results = []
    # 要选择的物品数量
    k = 10

    # 展示协同过滤原始推荐，协同过滤结果：（原始电影ID，用户预测评分）
    user_id = 1
    cf_candidates = [(id_to_original[i], R_pred[user_id - 1, i]) for i in np.argsort(-R_pred[user_id - 1])[:50]]
    print("\n协同过滤原始推荐Top50：")
    print([item[0] for item in cf_candidates])

    # 提取协同过滤结果中的电影ID和评分
    cf_movie_ids = [item[0] for item in cf_candidates]
    cf_ratings = [item[1] for item in cf_candidates]

    # 为协同过滤结果中的电影生成特征矩阵
    cf_item_features, unique_genres, cf_id_to_index = genre_one_hot(movies_df, cf_movie_ids)

    # 计算电影之间的相似度矩阵
    item_similarity_matrix = cosine_similarity(cf_item_features, cf_item_features)

    # 创建电影ID到相似度矩阵索引的映射关系
    id_to_sim_idx = {movie_id: idx for movie_id, idx in cf_id_to_index.items()}

    # 创建MMR使用的相似度矩阵
    mmr_similarity = np.zeros((max(cf_movie_ids) + 1, max(cf_movie_ids) + 1))
    for id1 in cf_movie_ids:
        for id2 in cf_movie_ids:
            if id1 in id_to_sim_idx and id2 in id_to_sim_idx:
                mmr_similarity[id1, id2] = item_similarity_matrix[id_to_sim_idx[id1], id_to_sim_idx[id2]]

    # 理想排序（按评分降序）
    ideal_items = sorted(cf_candidates, key=lambda x: x[1], reverse=True)

    # 测试不同theta值的MMR算法
    print("\n===== MMR算法结果 =====")
    for theta in theta_values:
        # 使用MMR算法重排序
        mmr_top10 = mmr(cf_candidates.copy(), mmr_similarity, k, theta)

        # 评估相关性和多样性
        metrics = evaluate_diversity_relevance(mmr_top10, ideal_items, mmr_similarity, movies_df)
        mmr_results.append((theta, metrics))

        print(f"\nTheta={theta} MMR推荐Top10：")
        print([item[0] for item in mmr_top10])
        print(f"平均评分: {metrics['avg_rating']:.4f}, NDCG: {metrics['ndcg']:.4f}")
        print(f"内部相似性: {metrics['internal_similarity']:.4f}, 多样性: {metrics['diversity']:.4f}")
        print(f"类别覆盖率: {metrics['category_coverage']:.4f}")

        # 获取电影详细信息
        movie_details = []
        for movie_id, rating in mmr_top10:
            movie_info = movies_df[movies_df['movieId'] == movie_id]
            if not movie_info.empty:
                title = movie_info['title'].values[0]
                genres = movie_info['genres'].values[0]
                movie_details.append(f"{title} ({genres}) - 预测评分: {rating:.2f}")

        print("电影详情:")
        for i, detail in enumerate(movie_details, 1):
            print(f"{i}. {detail}")

    # 测试不同theta值的DPP算法
    print("\n===== DPP算法结果 =====")
    for theta in theta_values:
        # 准备DPP算法的输入
        valid_indices = [i for i, movie_id in enumerate(cf_movie_ids) if movie_id in id_to_sim_idx]
        valid_features = cf_item_features[[id_to_sim_idx[cf_movie_ids[i]] for i in valid_indices]]
        valid_rewards = [cf_ratings[i] for i in valid_indices]

        # 使用DPP算法选择电影
        selected_indices = dpp(valid_features, valid_rewards, k, theta)

        # 获取选中的电影
        dpp_top10 = [(cf_movie_ids[valid_indices[i]], valid_rewards[i]) for i in selected_indices if
                     i < len(valid_indices)]

        # 评估相关性和多样性
        metrics = evaluate_diversity_relevance(dpp_top10, ideal_items, mmr_similarity, movies_df)
        dpp_results.append((theta, metrics))

        print(f"\nTheta={theta} DPP推荐Top10：")
        print([item[0] for item in dpp_top10])
        print(f"平均评分: {metrics['avg_rating']:.4f}, NDCG: {metrics['ndcg']:.4f}")
        print(f"内部相似性: {metrics['internal_similarity']:.4f}, 多样性: {metrics['diversity']:.4f}")
        print(f"类别覆盖率: {metrics['category_coverage']:.4f}")

        # 获取电影详细信息
        movie_details = []
        for movie_id, rating in dpp_top10:
            movie_info = movies_df[movies_df['movieId'] == movie_id]
            if not movie_info.empty:
                title = movie_info['title'].values[0]
                genres = movie_info['genres'].values[0]
                movie_details.append(f"{title} ({genres}) - 预测评分: {rating:.2f}")

        print("电影详情:")
        for i, detail in enumerate(movie_details, 1):
            print(f"{i}. {detail}")

    # 绘制评估指标对比图
    plt.figure(figsize=(15, 12))
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    # 提取评估指标
    mmr_avg_ratings = [r[1]['avg_rating'] for r in mmr_results]
    mmr_ndcg = [r[1]['ndcg'] for r in mmr_results]
    mmr_diversity = [r[1]['diversity'] for r in mmr_results]
    mmr_coverage = [r[1]['category_coverage'] for r in mmr_results]

    dpp_avg_ratings = [r[1]['avg_rating'] for r in dpp_results]
    dpp_ndcg = [r[1]['ndcg'] for r in dpp_results]
    dpp_diversity = [r[1]['diversity'] for r in dpp_results]
    dpp_coverage = [r[1]['category_coverage'] for r in dpp_results]

    # 绘制平均评分
    plt.subplot(2, 2, 1)
    plt.plot(theta_values, mmr_avg_ratings, 'bo-', linewidth=2, label='MMR')
    plt.plot(theta_values, dpp_avg_ratings, 'ro-', linewidth=2, label='DPP')
    plt.xlabel('Theta参数')
    plt.ylabel('平均评分')
    plt.title('平均评分随Theta变化')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # 绘制NDCG
    plt.subplot(2, 2, 2)
    plt.plot(theta_values, mmr_ndcg, 'bo-', linewidth=2, label='MMR')
    plt.plot(theta_values, dpp_ndcg, 'ro-', linewidth=2, label='DPP')
    plt.xlabel('Theta参数')
    plt.ylabel('NDCG')
    plt.title('NDCG随Theta变化')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # 绘制多样性
    plt.subplot(2, 2, 3)
    plt.plot(theta_values, mmr_diversity, 'bo-', linewidth=2, label='MMR')
    plt.plot(theta_values, dpp_diversity, 'ro-', linewidth=2, label='DPP')
    plt.xlabel('Theta参数')
    plt.ylabel('多样性')
    plt.title('多样性随Theta变化')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # 绘制类别覆盖率
    plt.subplot(2, 2, 4)
    plt.plot(theta_values, mmr_coverage, 'bo-', linewidth=2, label='MMR')
    plt.plot(theta_values, dpp_coverage, 'ro-', linewidth=2, label='DPP')
    plt.xlabel('Theta参数')
    plt.ylabel('类别覆盖率')
    plt.title('类别覆盖率随Theta变化')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('movie_diversity_relevance_metrics.png', dpi=300)
    plt.show()

    # 绘制相关性-多样性权衡图
    plt.figure(figsize=(10, 8))
    plt.plot(mmr_diversity, mmr_avg_ratings, 'bo-', linewidth=2, label='MMR')
    plt.plot(dpp_diversity, dpp_avg_ratings, 'ro-', linewidth=2, label='DPP')

    # 添加theta标注
    for i, theta in enumerate(theta_values):
        plt.annotate(f'θ={theta}', (mmr_diversity[i], mmr_avg_ratings[i]), textcoords="offset points",
                     xytext=(0, 10), ha='center')
        plt.annotate(f'θ={theta}', (dpp_diversity[i], dpp_avg_ratings[i]), textcoords="offset points",
                     xytext=(0, -15), ha='center')

    plt.xlabel('多样性')
    plt.ylabel('平均评分')
    plt.title('相关性-多样性权衡关系')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('movie_relevance_diversity_tradeoff.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
