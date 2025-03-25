import numpy as np


def cholesky_dpp(item_features, rewards, k, theta):
    """
    基于Cholesky分解的k-DPP算法实现
    
    参数:
    - item_features: 物品特征矩阵，每行是一个物品的特征向量
    - rewards: 物品的相关性分数
    - k: 要选择的物品数量
    - theta: 权衡相关性和多样性的参数，范围[0,1]
    
    返回:
    - 选中的k个物品的索引
    """
    n = len(rewards)  # 物品数量

    # 步骤2: 计算相似度矩阵A
    A = np.dot(item_features, item_features.T)
    # 归一化对角线元素为1
    for i in range(n):
        if A[i, i] > 0:
            item_features[i] = item_features[i] / np.sqrt(A[i, i])
    # 重新计算相似度矩阵
    A = np.dot(item_features, item_features.T)
    
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
            # I. 获取矩阵A_S∪{i}的最后一行
            a_i = np.array([A[i, j] for j in S])
            
            # II. 求解线性方程组a_i = L·c_i
            c_i = np.linalg.solve(L, a_i)
            
            # III. 计算d_i^2 = 1 - c_i^T·c_i
            d_i_squared = 1 - np.dot(c_i, c_i)
            
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
    # 示例：生成随机数据进行测试
    n = 100  # 物品数量
    d = 20   # 特征维度
    k = 10   # 要选择的物品数量

    # 生成随机特征向量和奖励值
    np.random.seed(42)
    item_features = np.random.randn(n, d)
    rewards = np.random.rand(n)
    
    # 测试不同theta值
    theta_values = [0.0, 0.2, 0.5, 0.8, 1.0]
    
    print("基于Cholesky分解的k-DPP算法测试:")
    for theta in theta_values:
        selected = cholesky_dpp(item_features.copy(), rewards, k, theta)
        
        # 计算所选物品的平均相似度（多样性的反面）
        similarity_sum = 0
        count = 0
        A = np.dot(item_features, item_features.T)
        for i in range(len(selected)):
            for j in range(i+1, len(selected)):
                similarity_sum += A[selected[i], selected[j]]
                count += 1
        
        avg_similarity = similarity_sum / count if count > 0 else 0
        avg_reward = np.mean([rewards[i] for i in selected])
        
        print(f"\nTheta = {theta}:")
        print(f"  选中的物品索引: {selected}")
        print(f"  平均奖励值: {avg_reward:.4f}")
        print(f"  平均相似度: {avg_similarity:.4f}")
        print(f"  多样性指标: {1 - avg_similarity:.4f}")

if __name__ == "__main__":
    main()