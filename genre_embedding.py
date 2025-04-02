import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def load_movie_data(file_path):
    """加载电影数据"""
    return pd.read_csv(file_path)

def extract_genres(movies_df):
    """提取所有电影种类"""
    # 将所有电影的种类合并成一个列表
    all_genres = []
    for genres in movies_df['genres'].str.split('|'):
        all_genres.extend(genres)
    
    # 获取唯一的种类
    unique_genres = sorted(list(set(all_genres)))
    return unique_genres

def create_genre_vectors(movies_df, unique_genres):
    """为每部电影创建种类向量"""
    # 创建一个矩阵，行是电影，列是种类
    genre_matrix = np.zeros((len(movies_df), len(unique_genres)))
    
    for i, genres in enumerate(movies_df['genres'].str.split('|')):
        for genre in genres:
            if genre in unique_genres:
                genre_idx = unique_genres.index(genre)
                genre_matrix[i, genre_idx] = 1
    
    return genre_matrix

def generate_genre_embeddings(genre_matrix, unique_genres, embedding_dim=20):
    """生成种类的嵌入向量"""
    # 计算种类共现矩阵
    genre_cooccurrence = np.dot(genre_matrix.T, genre_matrix)
    
    # 使用PCA降维生成嵌入向量
    pca = PCA(n_components=embedding_dim)
    genre_embeddings = pca.fit_transform(genre_cooccurrence)
    
    # 创建种类到嵌入向量的映射
    genre_to_embedding = {genre: embedding for genre, embedding in zip(unique_genres, genre_embeddings)}
    
    return genre_to_embedding, genre_embeddings

def visualize_genre_embeddings(genre_embeddings, unique_genres):
    """可视化种类嵌入向量（使用前两个主成分）"""
    plt.figure(figsize=(12, 10))
    
    # 只使用前两个维度进行可视化
    x = genre_embeddings[:, 0]
    y = genre_embeddings[:, 1]
    
    plt.scatter(x, y)
    
    # 添加标签
    for i, genre in enumerate(unique_genres):
        plt.annotate(genre, (x[i], y[i]), fontsize=9)
    
    plt.title('Genre Embeddings Visualization')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('d:/biyesheji/genre_embeddings.png')
    plt.show()


def main():
    # 加载电影数据
    movies_df = load_movie_data('d:/biyesheji/ml-latest-small/movies.csv')
    
    # 提取唯一的电影种类
    unique_genres = extract_genres(movies_df)
    print(f"发现 {len(unique_genres)} 个唯一电影种类: {unique_genres}")
    
    # 创建电影-种类矩阵
    genre_matrix = create_genre_vectors(movies_df, unique_genres)
    print(f"电影-种类矩阵形状: {genre_matrix.shape}")
    
    # 生成种类嵌入向量
    genre_to_embedding, genre_embeddings = generate_genre_embeddings(genre_matrix, unique_genres)
    
    # 可视化嵌入向量
    visualize_genre_embeddings(genre_embeddings, unique_genres)
    
    # 打印一些示例嵌入向量
    print("\n示例种类嵌入向量:")
    for genre in unique_genres[:5]:
        print(f"{genre}: {genre_to_embedding[genre][:5]}...")  # 只显示前5个维度

if __name__ == "__main__":
    main()