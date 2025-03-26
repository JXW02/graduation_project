import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt


def load_data():
    """加载电影和评分数据"""
    movies_df = pd.read_csv('ml-latest-small/movies.csv')
    ratings_df = pd.read_csv('ml-latest-small/ratings.csv')
    return movies_df, ratings_df


def genre_one_hot(movies_df):
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
    item_features = mlb.fit_transform(movies_df['genre_list'])

    return item_features, unique_genres


def genre_weighted_features(movies_df, ratings_df):
    """为每部电影创建加权种类特征"""
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
    item_features = mlb.fit_transform(movies_df['genre_list'])
    
    # 计算每个种类的平均评分
    genre_avg_ratings = {}
    
    # 为每部电影计算平均评分
    movie_avg_ratings = ratings_df.groupby('movieId')['rating'].mean().to_dict()
    
    # 计算每个种类的平均评分
    for genre_idx, genre in enumerate(unique_genres):
        genre_movies = []
        for idx, movie_row in movies_df.iterrows():
            if genre in movie_row['genre_list']:
                movie_id = movie_row['movieId']
                if movie_id in movie_avg_ratings:
                    genre_movies.append(movie_avg_ratings[movie_id])
        
        if genre_movies:
            genre_avg_ratings[genre] = np.mean(genre_movies)
        else:
            genre_avg_ratings[genre] = 0
    
    # 对每部电影的种类特征进行加权
    weighted_features = np.zeros_like(item_features, dtype=float)
    for idx, movie_row in movies_df.iterrows():
        for genre_idx, genre in enumerate(unique_genres):
            if item_features[idx, genre_idx] == 1:
                # 使用种类平均评分作为权重
                weighted_features[idx, genre_idx] = genre_avg_ratings[genre]
    
    # 归一化特征
    row_sums = weighted_features.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # 避免除以零
    weighted_features = weighted_features / row_sums
    
    return weighted_features, unique_genres


def calculate_avg_ratings(ratings_df):
    """计算每部电影的平均评分并返回字典形式"""
    # 按电影ID分组并计算平均评分
    avg_ratings = ratings_df.groupby('movieId')['rating'].mean()

    # 转换为字典形式
    avg_ratings_dict = avg_ratings.to_dict()

    print(f"计算完成，共有 {len(avg_ratings_dict)} 部电影的平均评分")
    return avg_ratings_dict


def calculate_bayesian_average(ratings_df, movies_df):
    """计算贝叶斯加权平均评分"""
    # 1. 计算全局平均评分C
    C = ratings_df['rating'].mean()

    # 计算每部电影的评分统计
    movie_stats = ratings_df.groupby('movieId').agg({
        'rating': ['count', 'mean']
    }).reset_index()
    movie_stats.columns = ['movieId', 'vote_count', 'vote_average']

    # 2. 确定基准数量m（评分数量的中位数）
    m = movie_stats['vote_count'].median()

    # 3. 计算贝叶斯加权平均分
    movie_stats['bayesian_average'] = (movie_stats['vote_count'] * movie_stats['vote_average'] +
                                       m * C) / (movie_stats['vote_count'] + m)

    # 添加电影信息
    result = pd.merge(movie_stats, movies_df[['movieId', 'title', 'genres']], on='movieId')

    # 按类别计算平均分
    genre_averages = {}
    for movie_row in result.itertuples():
        genres = movie_row.genres.split('|')
        for genre in genres:
            if genre not in genre_averages:
                genre_averages[genre] = []
            genre_averages[genre].append(movie_row.vote_average)

    for genre in genre_averages:
        genre_averages[genre] = np.mean(genre_averages[genre])

    # 使用类别平均分重新计算贝叶斯平均分
    def calculate_genre_bayesian_avg(row):
        genres = row['genres'].split('|')
        genre_C = np.mean([genre_averages[genre] for genre in genres])
        return (row['vote_count'] * row['vote_average'] + m * genre_C) / (row['vote_count'] + m)

    result['genre_bayesian_average'] = result.apply(calculate_genre_bayesian_avg, axis=1)

    return result, C, m, genre_averages


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


def cholesky_dpp(item_features, rewards, k, theta):
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


def main():
    # 要选择的物品数量
    k = 10

    #  加载电影及评分数据
    movies_df, ratings_df = load_data()

    # 使用one hot编码生成电影特征向量
    item_features, unique_genres = genre_one_hot(movies_df)
    # 加权种类one hot编码特征向量
    # item_features, unique_genres = genre_weighted_features(movies_df, ratings_df)

    # 生成电影均分
    # movie_avg_rating = calculate_avg_ratings(ratings_df)
    # rewards = list(movie_avg_rating.values())
    # 贝叶斯均分平衡均分与投票数量，减少少数用户打高分对整体评分影响
    result, C, m, genre_averages = calculate_bayesian_average(ratings_df, movies_df)
    rewards = result['bayesian_average'].tolist()

    # 测试不同theta值
    theta_values = [0.0, 0.2, 0.5, 0.8, 1.0]
    
    # 存储结果用于绘图
    avg_rewards = []
    avg_similarities = []
    diversity_scores = []
    selected_movies_info = []

    print("基于Cholesky分解的k-DPP算法测试:")
    for theta in theta_values:
        selected = cholesky_dpp(item_features.copy(), rewards, k, theta)

        # 计算所选物品的平均相似度（多样性的反面）
        similarity_sum = 0
        count = 0
        A = np.dot(item_features, item_features.T)
        for i in range(len(selected)):
            for j in range(i + 1, len(selected)):
                similarity_sum += A[selected[i], selected[j]]
                count += 1

        avg_similarity = similarity_sum / count if count > 0 else 0
        avg_reward = np.mean([rewards[i] for i in selected])
        diversity = 1 - avg_similarity
        
        # 存储结果
        avg_rewards.append(avg_reward)
        avg_similarities.append(avg_similarity)
        diversity_scores.append(diversity)
        
        # 获取选中电影的信息
        selected_movies = []
        for idx in selected:
            movie_id = result.iloc[idx]['movieId'] if idx < len(result) else "未知"
            movie_title = result.iloc[idx]['title'] if idx < len(result) else f"电影{idx}"
            selected_movies.append((movie_id, movie_title))
        selected_movies_info.append(selected_movies)

        print(f"\nTheta = {theta}:")
        print(f"  选中的物品索引: {selected}")
        print(f"  选中的电影:")
        for movie_id, movie_title in selected_movies:
            print(f"    - {movie_title} (ID: {movie_id})")
        print(f"  平均奖励值: {avg_reward:.4f}")
        print(f"  平均相似度: {avg_similarity:.4f}")
        print(f"  多样性指标: {diversity:.4f}")
    
    # 绘制比较图表
    plot_comparison(theta_values, avg_rewards, avg_similarities, diversity_scores)



def plot_comparison(theta_values, avg_rewards, avg_similarities, diversity_scores):
    """绘制不同theta值下的指标比较图"""
    plt.figure(figsize=(15, 10))
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
    # 创建子图
    plt.subplot(2, 2, 1)
    plt.plot(theta_values, avg_rewards, 'o-', linewidth=2, markersize=8)
    plt.title('不同theta值下的平均奖励值')
    plt.xlabel('Theta值')
    plt.ylabel('平均奖励值')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(2, 2, 2)
    plt.plot(theta_values, avg_similarities, 'o-', linewidth=2, markersize=8, color='orange')
    plt.title('不同theta值下的平均相似度')
    plt.xlabel('Theta值')
    plt.ylabel('平均相似度')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(2, 2, 3)
    plt.plot(theta_values, diversity_scores, 'o-', linewidth=2, markersize=8, color='green')
    plt.title('不同theta值下的多样性指标')
    plt.xlabel('Theta值')
    plt.ylabel('多样性指标')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 综合比较图
    plt.subplot(2, 2, 4)
    
    # 归一化数据以便于比较
    norm_rewards = [r/max(avg_rewards) for r in avg_rewards]
    norm_similarities = [s/max(avg_similarities) for s in avg_similarities]
    norm_diversity = [d/max(diversity_scores) for d in diversity_scores]
    
    plt.plot(theta_values, norm_rewards, 'o-', linewidth=2, markersize=8, label='归一化奖励值')
    plt.plot(theta_values, norm_similarities, 'o-', linewidth=2, markersize=8, label='归一化相似度')
    plt.plot(theta_values, norm_diversity, 'o-', linewidth=2, markersize=8, label='归一化多样性')
    plt.title('指标综合比较(归一化)')
    plt.xlabel('Theta值')
    plt.ylabel('归一化指标值')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('dpp_comparison.png', dpi=300)
    plt.show()




if __name__ == "__main__":
    main()