import numpy as np
from scipy import sparse
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MultiLabelBinarizer
from movie_recommendation import mmr, dpp, cosine_similarity, calculate_ndcg, calculate_internal_similarity, evaluate_diversity_relevance


def ConvertToDense(x, y, shape):
    """将稀疏数据转换为稠密矩阵"""
    # 用户id 动漫id 评分
    row = x[:, 0]
    col = x[:, 1]
    data = y
    matrix_sparse = sparse.csr_matrix((data, (row, col)), shape=(shape[0] + 1, shape[1]))  # 添加占位符
    R = matrix_sparse.todense()
    R = R[1:, :]  # 去掉占位符
    R = np.asarray(R)
    return R


def LoadAnimeData():
    """加载动漫数据并生成用户-动漫评分矩阵"""
    anime_df = pd.read_csv('anime/anime.csv')
    rating_df = pd.read_csv('anime/rating.csv')

    # 过滤掉评分为-1的记录（用户观看但未评分）
    rating_df = rating_df[rating_df['rating'] > 0]

    # 数据集太大，采样部分用户和动漫进行测试
    # 选择评分数量最多的1000个用户和1000部动漫
    user_counts = rating_df['user_id'].value_counts()
    top_users = user_counts.nlargest(1000).index.tolist()

    anime_counts = rating_df['anime_id'].value_counts()
    top_anime = anime_counts.nlargest(1000).index.tolist()

    # 筛选数据
    rating_df = rating_df[rating_df['user_id'].isin(top_users) & rating_df['anime_id'].isin(top_anime)]

    # 为减少动漫无评分而占用矩阵空间，对动漫ID重映射
    rating_df['anime_id'], item_mapping = pd.factorize(rating_df['anime_id'])
    id_to_original = dict(enumerate(item_mapping))  # 保存映射关系

    # 对用户ID也进行重映射，确保从0开始
    rating_df['user_id'], user_mapping = pd.factorize(rating_df['user_id'])

    x = rating_df[['user_id', 'anime_id']].values
    y = rating_df['rating'].values
    n_users = len(rating_df['user_id'].unique())
    n_anime = len(rating_df['anime_id'].unique())
    R_shape = (n_users, n_anime)

    print(f"采样后的数据集大小: {len(rating_df)} 条评分, {n_users} 个用户, {n_anime} 部动漫")

    # 返回：[用户id，动漫id]，评分，评分矩阵形状，动漫id映射关系
    return x, y, R_shape, id_to_original, anime_df, rating_df


def genre_one_hot(anime_df, anime_ids):
    """为每部动漫创建类型的One-Hot编码"""
    # 将所有动漫的类型合并成一个列表
    all_genres = []
    for genres in anime_df['genre'].str.split(', '):
        if isinstance(genres, list):  # 确保genres是列表
            all_genres.extend(genres)
    
    # 获取类型集合
    unique_genres = sorted(list(set(all_genres)))
    
    # 将类型字符串转换为列表
    anime_df['genre_list'] = anime_df['genre'].str.split(', ')
    
    # 使用MultiLabelBinarizer进行One-Hot编码
    mlb = MultiLabelBinarizer()
    
    # 创建动漫ID到索引的映射
    id_to_index = {}
    filtered_anime = anime_df[anime_df['anime_id'].isin(anime_ids)]
    
    # 为指定的动漫生成特征
    item_features = mlb.fit_transform(filtered_anime['genre_list'])
    
    # 创建动漫ID到特征索引的映射
    for idx, anime_id in enumerate(filtered_anime['anime_id']):
        id_to_index[anime_id] = idx
    
    return item_features, unique_genres, id_to_index


def calculate_category_coverage(items, anime_df):
    """计算推荐列表的类别覆盖率"""
    # 获取所有可能的类别
    all_genres = set()
    for genres in anime_df['genre'].str.split(', '):
        if isinstance(genres, list):  # 确保genres是列表
            all_genres.update(genres)
    
    # 获取推荐列表中的类别
    recommended_genres = set()
    for item in items:
        anime_id = item[0]
        genres = anime_df[anime_df['anime_id'] == anime_id]['genre'].values
        if len(genres) > 0 and isinstance(genres[0], str):
            recommended_genres.update(genres[0].split(', '))
    
    # 计算覆盖率
    return len(recommended_genres) / len(all_genres) if len(all_genres) > 0 else 0


def userBased(R_train, R_test, k=5, similarity_metric='cosine'):
    """基于用户的协同过滤推荐算法"""
    num_users, num_items = R_train.shape
    mask_train = R_train > 0
    mask_test = R_test > 0

    # 计算用户平均评分，处理除零问题
    user_sum = R_train.sum(1)
    user_count = mask_train.sum(1)
    user_mean = np.zeros(num_users)
    for i in range(num_users):
        if user_count[i] > 0:
            user_mean[i] = user_sum[i] / user_count[i]
        else:
            # 对于没有评分的用户，使用全局平均评分
            user_mean[i] = np.mean(R_train[mask_train]) if np.any(mask_train) else 0

    # 使用更高效的方式计算用户相似度
    print("计算用户相似度矩阵...")
    similarity = np.zeros((num_users, num_users))

    # 分批计算相似度，避免内存溢出
    batch_size = 100  # 调整批次大小以适应内存
    for i in range(0, num_users, batch_size):
        end_i = min(i + batch_size, num_users)
        for j in range(0, num_users, batch_size):
            end_j = min(j + batch_size, num_users)

            # 计算批次间的相似度
            R_i = R_train[i:end_i]
            R_j = R_train[j:end_j]
            mask_i = mask_train[i:end_i]
            mask_j = mask_train[j:end_j]

            # 计算点积
            dot_product = np.dot(R_i, R_j.T)

            # 计算范数
            norm_i = np.sqrt(np.sum(R_i * R_i, axis=1)).reshape(-1, 1)
            norm_j = np.sqrt(np.sum(R_j * R_j, axis=1)).reshape(1, -1)

            # 避免除以零
            norm_i[norm_i == 0] = 1e-10
            norm_j[norm_j == 0] = 1e-10

            # 计算余弦相似度
            batch_similarity = dot_product / (norm_i * norm_j)

            # 存储结果
            similarity[i:end_i, j:end_j] = batch_similarity

    # 确保对角线为1
    np.fill_diagonal(similarity, 1)

    # 为每个用户选择前k个相似用户作为邻居
    print("为每个用户选择邻居...")
    neighbors = np.argsort(-similarity, axis=1)[:, 1:k + 1]  # 排除自身
    
    # 预测评分
    print("预测评分...")
    R_pred = np.zeros((num_users, num_items))
    mask_union = mask_train | mask_test  # 取并集
    
    # 分批处理用户，减少内存使用
    for u in range(num_users):
        if u % 100 == 0:
            print(f"处理用户 {u}/{num_users}")
        
        # 获取该用户的邻居
        neighbors_u = neighbors[u]
        
        # 只处理用户有评分或需要预测的物品
        for i in range(num_items):
            if mask_union[u, i]:
                # 检查邻居是否对该物品有评分
                mask_neighbors = R_train[neighbors_u, i] > 0
                if np.sum(mask_neighbors) == 0:
                    R_pred[u, i] = user_mean[u]
                else:
                    sim_u = similarity[u, neighbors_u][mask_neighbors]
                    ratings_u = R_train[neighbors_u, i][mask_neighbors]
                    R_pred[u, i] = np.dot(sim_u, ratings_u) / np.sum(np.abs(sim_u))
    
    # 裁剪评分范围（动漫评分范围是1-10）
    R_pred[R_pred > 10] = 10.
    R_pred[R_pred < 1] = 1.
    
    # 计算训练集上的 RMSE
    preds_train = R_pred[mask_train]
    actual_train = R_train[mask_train]
    rmse_train = np.sqrt(mean_squared_error(actual_train, preds_train))
    print(f'Training RMSE: {rmse_train:.4f}')
    
    return R_pred


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
    rounded_pred[rounded_pred > 10] = 10
    rounded_pred[rounded_pred < 1] = 1
    accuracy = np.mean(rounded_pred == R_actual[mask])
    
    return {
        'rmse': rmse,
        'mae': mae,
        'correlation': correlation,
        'accuracy': accuracy
    }


def evaluate_anime_diversity_relevance(recommended_items, ideal_items, similarity_matrix, anime_df):
    """评估推荐列表的相关性和多样性"""
    # 相关性指标
    avg_rating = np.mean([item[1] for item in recommended_items])
    ndcg = calculate_ndcg(recommended_items, ideal_items)
    
    # 多样性指标
    internal_similarity = calculate_internal_similarity(recommended_items, similarity_matrix)
    diversity = 1 - internal_similarity  # 多样性是相似性的补集
    category_coverage = calculate_category_coverage(recommended_items, anime_df)
    
    return {
        'avg_rating': avg_rating,
        'ndcg': ndcg,
        'internal_similarity': internal_similarity,
        'diversity': diversity,
        'category_coverage': category_coverage
    }


def main():
    # 加载动漫数据集
    x, y, R_shape, id_to_original, anime_df, rating_df = LoadAnimeData()
    
    # 基于用户的协同过滤，生成用户对动漫的预测评分
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
    theta_values = [0, 0.2, 0.5, 0.8, 1]
    mmr_results = []
    dpp_results = []
    # 要选择的物品数量
    k = 10
    
    # 展示协同过滤原始推荐，协同过滤结果：（原始动漫ID，用户预测评分）
    user_id = 1  # 选择一个用户
    cf_candidates = [(id_to_original[i], R_pred[user_id - 1, i]) for i in np.argsort(-R_pred[user_id - 1])[:50]]
    print("\n协同过滤原始推荐Top50：")
    print([item[0] for item in cf_candidates])
    
    # 提取协同过滤结果中的动漫ID和评分
    cf_anime_ids = [item[0] for item in cf_candidates]
    cf_ratings = [item[1] for item in cf_candidates]
    
    # 为协同过滤结果中的动漫生成特征矩阵
    cf_item_features, unique_genres, cf_id_to_index = genre_one_hot(anime_df, cf_anime_ids)
    
    # 计算动漫之间的相似度矩阵
    item_similarity_matrix = cosine_similarity(cf_item_features, cf_item_features)
    
    # 创建动漫ID到相似度矩阵索引的映射关系
    id_to_sim_idx = {anime_id: idx for anime_id, idx in cf_id_to_index.items()}
    
    # 创建MMR使用的相似度矩阵
    mmr_similarity = np.zeros((max(cf_anime_ids) + 1, max(cf_anime_ids) + 1))
    for id1 in cf_anime_ids:
        for id2 in cf_anime_ids:
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
        metrics = evaluate_anime_diversity_relevance(mmr_top10, ideal_items, mmr_similarity, anime_df)
        mmr_results.append((theta, metrics))
        
        print(f"\nTheta={theta} MMR推荐Top10：")
        print([item[0] for item in mmr_top10])
        print(f"平均评分: {metrics['avg_rating']:.4f}, NDCG: {metrics['ndcg']:.4f}")
        print(f"内部相似性: {metrics['internal_similarity']:.4f}, 多样性: {metrics['diversity']:.4f}")
        print(f"类别覆盖率: {metrics['category_coverage']:.4f}")
        
        # 获取动漫详细信息
        anime_details = []
        for anime_id, rating in mmr_top10:
            anime_info = anime_df[anime_df['anime_id'] == anime_id]
            if not anime_info.empty:
                name = anime_info['name'].values[0]
                genre = anime_info['genre'].values[0]
                anime_details.append(f"{name} ({genre}) - 预测评分: {rating:.2f}")
        
        print("动漫详情:")
        for i, detail in enumerate(anime_details, 1):
            print(f"{i}. {detail}")
    
    # 测试不同theta值的DPP算法
    print("\n===== DPP算法结果 =====")
    for theta in theta_values:
        # 准备DPP算法的输入
        valid_indices = [i for i, anime_id in enumerate(cf_anime_ids) if anime_id in id_to_sim_idx]
        valid_features = cf_item_features[[id_to_sim_idx[cf_anime_ids[i]] for i in valid_indices]]
        valid_rewards = [cf_ratings[i] for i in valid_indices]
        
        # 使用DPP算法选择动漫
        selected_indices = dpp(valid_features, valid_rewards, k, theta)
        
        # 获取选中的动漫
        dpp_top10 = [(cf_anime_ids[valid_indices[i]], valid_rewards[i]) for i in selected_indices if i < len(valid_indices)]
        
        # 评估相关性和多样性
        metrics = evaluate_anime_diversity_relevance(dpp_top10, ideal_items, mmr_similarity, anime_df)
        dpp_results.append((theta, metrics))
        
        print(f"\nTheta={theta} DPP推荐Top10：")
        print([item[0] for item in dpp_top10])
        print(f"平均评分: {metrics['avg_rating']:.4f}, NDCG: {metrics['ndcg']:.4f}")
        print(f"内部相似性: {metrics['internal_similarity']:.4f}, 多样性: {metrics['diversity']:.4f}")
        print(f"类别覆盖率: {metrics['category_coverage']:.4f}")
        
        # 获取动漫详细信息
        anime_details = []
        for anime_id, rating in dpp_top10:
            anime_info = anime_df[anime_df['anime_id'] == anime_id]
            if not anime_info.empty:
                name = anime_info['name'].values[0]
                genre = anime_info['genre'].values[0]
                anime_details.append(f"{name} ({genre}) - 预测评分: {rating:.2f}")
        
        print("动漫详情:")
        for i, detail in enumerate(anime_details, 1):
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
    plt.savefig('d:\\biyesheji\\anime_diversity_relevance_metrics.png', dpi=300)
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
    plt.savefig('d:\\biyesheji\\anime_relevance_diversity_tradeoff.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    main()