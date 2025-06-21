import numpy as np
from scipy.linalg import inv, norm
# np.random.seed(0)


def iteration_for_subproblem(B, b):
    z = np.random.rand(len(b))
    tol = 10e-4
    err = 1
    while err>tol:
        z_temp = z
        z = b/(B.T.dot(b/(B.dot(z))))
        err = np.linalg.norm(z_temp-z,1)

    res = B.dot(z)
    til_gamma = np.log(z/res)
    return til_gamma

def bcd_for_wsrm(G,w, sigma,m, p_bar,  y_init):
    F = np.dot(np.linalg.inv(np.diag(np.diag(G))), (G - np.diag(np.diag(G))))
    v = np.dot(np.linalg.inv(np.diag(np.diag(G))), sigma)
    B = F + 1 / p_bar * np.outer(v, m)
    y = y_init
    err = 1
    tol = 1e-3
    obj_before = 0
    step = 0
    # gamma1 = [y_init[0]]
    # gamma2 = [y_init[1]]
    # gamma3 = [y_init[2]]

    # for i in range(13):
    while err > tol:
        # print("step:", step)
        b = w[:, 0] * y
        til_gamma = iteration_for_subproblem(B, b)
        gamma = np.exp(til_gamma)
        # gamma1.append(gamma[0])
        # gamma2.append(gamma[1])
        # gamma3.append(gamma[2])
        y = 1 / (1 /gamma + 1)

        obj_updated = w.T.dot(np.log(1 + gamma))[0]
        err = obj_updated - obj_before
        obj_before = obj_updated
        step += 1
    # print("gamma1:", gamma1)
    # print("gamma2:", gamma2)
    # print("gamma3:", gamma3)
    L = len(w)
    p = inv(np.eye(L) - np.diag(gamma) @ F) @ np.diag(gamma) @ v
    return step, p, gamma


def wsrm_algorithm(H_list, w, m, n, P_bar, max_iter=1000, epsilon=1e-2):
    """
    完整的WSRM算法实现，使用 iteration_for_subproblem 替代步骤2和7.

    参数:
        H_list (list of np.ndarray): 每个用户的信道矩阵 [H_1, ..., H_L], 每个 H_l 形状为 (N_l x N)
        w (np.ndarray): 权重向量 (L x 1), sum(w) = 1
        m (np.ndarray): 功率权重向量 (L x 1)
        n (np.ndarray): 噪声功率向量 (L x 1)
        P_bar (float): 总功率约束
        max_iter (int): 最大迭代次数
        epsilon (float): 收敛阈值

    返回:
        p_opt (np.ndarray): 最优功率分配 (L x 1)
        U_opt (np.ndarray): 发射波束成形矩阵 (N x L)
        V_opt (list of np.ndarray): 接收波束成形向量 [v_1, ..., v_L], 每个 v_l 形状为 (N_l x 1)
    """
    L = len(H_list)  # 用户数
    N = H_list[0].shape[1]  # 发射天线数
    N_l = [H.shape[0] for H in H_list]  # 每个用户的接收天线数

    # 步骤1: 初始化
    p = np.ones(L) * (P_bar / (np.sum(m) * L))  # 初始功率分配
    U = np.random.randn(N, L) + np.random.randn(N, L)  # 随机初始化 U
    U = U / norm(U, axis=0)  # 归一化列向量
    V = [np.random.randn(N_l[l], 1) + np.random.randn(N_l[l], 1) for l in range(L)]
    V = [v / norm(v) for v in V]  # 归一化 v_l
    y = np.random.rand(L) * 1


    for _ in range(max_iter):
        p_pre = p
        U_pre = U
        V_prev = V.copy()
        # print("_:", _)
        # 步骤①: 计算 G 和 F 矩阵
        G = np.zeros((L, L))
        for l in range(L):
            for i in range(L):
                G[l, i] = np.abs(V[l].T @ H_list[l] @ U[:, i:i + 1]) ** 2

        step, p, gamma = bcd_for_wsrm(G, w, n, m, P_bar, y)
        # print("p:",p)

        # 步骤⑤: 更新接收波束成形 V
        for l in range(L):
            interference = sum(p[j] * H_list[l] @ U[:, j:j + 1] @ U[:, j:j + 1].T @ H_list[l].T
                               for j in range(L) if j != l)
            V[l] = inv(interference + n[l] * np.eye(N_l[l])) @ H_list[l] @ U[:, l:l + 1]
            V[l] = V[l] / norm(V[l])

        V_diff = 0.0
        for l in range(L):
            V_diff += norm(V[l] - V_prev[l]) ** 2  # 累加所有用户的Frobenius范数平方
        V_diff = np.sqrt(V_diff)  # 总差异的L2范数

        # 步骤⑥: 重新计算 G 和 F。  最优功率分配: [ 1.69240102+0.j  9.2556927 +0.j 14.6089108 +0.j] 最优功率分配: [ 9.48074892+0.j  3.03579283+0.j 14.73444136+0.j]
        G_prime = np.zeros((L, L))
        for l in range(L):
            for i in range(L):
                G_prime[l, i] = np.abs(U[:, i:i + 1].T @ H_list[i].T @ V[l]) ** 2

        F_prime = np.zeros((L, L))
        for i in range(L):
            for j in range(L):
                if i != j:
                    F_prime[i, j] = G_prime[i, j] / G_prime[i, i]

        step_prime, p_prime, gamma_prime = bcd_for_wsrm(G_prime, w, m, n, P_bar, y)
        # print("p_prime:", p_prime)
        # 步骤⑩: 更新发射波束成形 U
        for l in range(L):
            interference = sum(p_prime[j] * H_list[j].T @ V[j] @ V[j].T @ H_list[j]
                               for j in range(L) if j != l)
            u_l = inv(interference + m[l] * np.eye(N)) @ H_list[l].T @ V[l]
            U[:, l:l + 1] = u_l / norm(u_l)

        # 步骤⑪: 检查收敛条件
        if norm(p_pre - p) < epsilon and norm(U_pre-U) < epsilon and V_diff < epsilon:
            break
        # print("U:", U)
    obj_updated = w.T.dot(np.log(1 + gamma))
    # print("=====obj_updated:", obj_updated)
    return p, U, V,obj_updated

if __name__ == '__main__':
    # 生成随机数据示例
    N = 4  # 发射天线数
    L = 4  # 用户数
    N_l = [2]*L  # 每个用户的接收天线数
    # H_list = [np.random.randn(N_l[l], N) + 1j * np.random.randn(N_l[l], N) for l in range(L)]
    H_list = [np.array([[1.0, 0.3, 0.1, -0.2],
            [0.5, 1.2, -0.3, 0.4]]),
            np.array([[0.8, 0.2, 0.5, 1.0],
                [-0.4, 1.1, 0.3, 0.6]]),
            np.array([[0.7, -0.3, 1.4, 0.1],
                [0.2, 0.9, -0.5, 1.2]])]
    w = (np.ones(L) / L).reshape(L,1)  # 均匀权重
    m = np.ones(L)  # 功率权重
    n = 0.1 * np.ones(L)  # 噪声功率
    P_bar = 10.0  # 总功率约束

    # 运行算法
    p_opt, U_opt, V_opt, obj_updated = wsrm_algorithm(H_list, w, m, n, P_bar)
    print("最优功率分配:", p_opt)
