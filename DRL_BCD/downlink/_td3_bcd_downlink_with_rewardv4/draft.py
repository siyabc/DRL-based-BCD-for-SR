import numpy as np
def calculate_C(F, p, sigma):
    """
    计算所有 C_{k \backslash i} 的值

    参数:
    g: 二维数组，表示 g_{j \rightarrow k} 的值
    p: 一维数组，表示 p_j 的值
    sigma: 标量，表示噪声的方差

    返回:
    C: 一维数组，包含所有 C_{k \backslash i} 的值
    """
    num_nodes = len(p)
    C = np.zeros((num_nodes,num_nodes) ) # 初始化 C 数组

    for i in range(num_nodes):
        for k in range(num_nodes):
            if k != i:  # 确保 j != i, k
                numerator = p[k]
                denominator = np.sum(F[k][j] * p[j] for j in range(num_nodes) if j != i and j != k) + sigma[k]
                # C[i][k] = np.log(1 + numerator / denominator)
                C[i][k] = 1 + numerator / denominator

    return C

F = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])  # 连接强度矩阵
p = np.array([0.5, 0.3, 0.2])  # 属性值
sigma = np.array([0.5, 0.3, 0.2])  # 噪声方差

C_values = calculate_C(F, p, sigma)
print(C_values)