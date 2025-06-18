import numpy as np


def power_allocation_algorithm(h, m, alpha, P, max_iter=3000, tol=1e-6):
    """
    多用户功率分配算法实现

    参数:
        h : numpy.ndarray, 信道增益矩阵 (h[i,j] 表示用户j到用户i的信道增益)
        m : numpy.ndarray, 权重系数向量 (m[i] 为用户i的权重)
        alpha : numpy.ndarray, 优先级权重向量 (alpha[i] 为用户i的优先级)
        P : float, 总功率约束
        max_iter : int, 最大迭代次数
        tol : float, 收敛容差

    返回:
        V : numpy.ndarray, 最优功率分配向量
    """
    num_users = len(m)

    # 1. 初始化
    V = np.sqrt(P / (2 * m))  # V_i(0) = sqrt(P / (2m_i))
    lambda_low, lambda_high = 0, 1e6  # 二分法初始范围

    for k in range(max_iter):
        V_prev = V.copy()

        # 2. 计算 u_i^(k+1)
        interference = np.sum((h ** 2) * (V ** 2), axis=1)  # 各用户的干扰项
        u = h.diagonal() * V / (0.05 + interference)  # 避免除以0

        # 3. 计算 w_i^(k+1)
        w = 1 / (1 - u * h.diagonal() * V)

        # 4. 二分法求解 lambda
        # for _ in range(500):  # 二分法迭代
        while abs(lambda_high-lambda_low)>1e-4:
            lambda_mid = (lambda_low + lambda_high) / 2

            # 计算 V_i^(k+1)(lambda)
            numerator = alpha * w * u * h.diagonal()
            denominator = np.sum(alpha * w * (u ** 2) * (h ** 2), axis=0) + lambda_mid * m
            V_new = numerator / denominator

            # 检查功率约束
            total_power = np.sum(m * (V_new ** 2))

            if total_power < P:
                lambda_high = lambda_mid
            else:
                lambda_low = lambda_mid

        # 更新 V
        V = numerator / (denominator )  # 避免除以0

        # 5. 检查收敛
        if np.linalg.norm(V - V_prev) < tol:
            break

    return V


# 示例用法
if __name__ == "__main__":
    # o = np.array([0.09,0.24,0.18,0.11,2.49,0.03,0.99,0.25,7.1,0.96,0.38,0.35,0.05,0.05,0.05,2.91,])
    o = np.array([1.18, 0.05, 0.59, 0.61, 9.67, 0.56, 0.89, 0.99, 3.29, 0.66, 0.45, 0.46, 0.05, 0.05, 0.05, 2.79])
    # label = np.array([0.0,21.453712190650787,49.07719298245613])
    label = np.array([27.804597701149422, 5.502345251826221, 0.0])

    G = o[0:9].reshape(3, 3)
    w = o[9:12].reshape(3)
    sigma = o[12:15].reshape(3,1)
    p_bar = o[15]
    gamma_star = label


    # 参数设置
    num_users = 3
    P = 10.0  # 总功率
    m = np.array([1.0, 1., 1.0])  # 权重系数
    alpha = np.array([1.0, 1.0, 1.0])  # 优先级权重

    # 随机生成信道矩阵 (h[i,j] 是用户j到用户i的增益)
    h = np.abs(np.random.randn(num_users, num_users)) * 0.5 + 1.0

    # 运行算法
    # optimal_V = power_allocation_algorithm(h, m, alpha, P)
    optimal_V = power_allocation_algorithm(G, m, w, p_bar)

    # 验证结果
    print("最优功率分配:", optimal_V**2)
    print("实际总功率:", np.sum(m * (optimal_V ** 2)))
    print("目标总功率:", p_bar)

    w = w.reshape(3,1)
    F = np.dot(np.linalg.inv(np.diag(np.diag(G))), (G - np.diag(np.diag(G))))
    v = np.dot(np.linalg.inv(np.diag(np.diag(G))), sigma)
    p= (optimal_V ** 2).reshape(3,1)
    sinr = (1 / (np.dot(F, p) + v)) * p
    print("sinr:", sinr)

    obj_updated = w.T.dot(np.log(1 + sinr))[0]
    print("obj_updated:", obj_updated)

    obj_star = w.T.dot(np.log(1 + gamma_star))[0]
    sumrate_acc = obj_updated / obj_star
    print("sumrate_acc:", sumrate_acc)

