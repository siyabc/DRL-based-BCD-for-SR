import numpy as np
from scipy.linalg import pinv, norm
from IP_BCD import iteration_for_subproblem,bcd_for_wsrm

def zf_mrc_beamforming(H_list):
    """
    使用 ZF 预编码计算 U，用 MRC 计算 V

    参数:
        H_list (list): 每个用户的信道矩阵列表 [H_1, H_2, ..., H_L],
                      其中 H_l 形状为 (N_l, N)

    返回:
        U (np.ndarray): 发射波束成形矩阵 (N x L), 列已归一化
        V (list): 接收波束成形向量列表 [v_1, v_2, ..., v_L],
                 每个 v_l 形状为 (N_l x 1)
    """
    L = len(H_list)  # 用户数
    N = H_list[0].shape[1]  # 发射天线数

    # 步骤1: 构造组合信道矩阵 H (N x sum(N_l))
    H_combined = np.vstack(H_list)  # 形状 (sum(N_l) x N)

    # 步骤2: 计算 ZF 预编码矩阵 (伪逆)
    H_pseudo_inv = pinv(H_combined)  # 形状 (N x sum(N_l))

    # 步骤3: 提取每个用户的预编码向量并归一化
    U = np.zeros((N, L), dtype=np.complex128)
    for l in range(L):
        # 取 H_pseudo_inv 的对应列 (假设用户顺序与 H_list 一致)
        u_l = H_pseudo_inv[:, l].reshape(-1, 1)
        U[:, l] = u_l.flatten() / norm(u_l)  # 归一化

    # 步骤4: 用 MRC 计算接收波束成形向量 V
    V = []
    for l in range(L):
        H_l = H_list[l]  # 用户 l 的信道矩阵
        u_l = U[:, l].reshape(-1, 1)  # 对应的发射波束成形向量
        v_l = H_l @ u_l  # MRC 计算
        v_l = v_l / norm(v_l)  # 归一化
        V.append(v_l)

    return U, V


# ------------------- 示例使用 -------------------
if __name__ == "__main__":
    # 示例参数
    N = 4  # 发射天线数
    L = 3  # 用户数
    N_l = [2, 2, 2]  # 每个用户的接收天线数

    # 生成确定性信道矩阵 (复数)
    H_list = [np.array([[1.0, 0.3, 0.1, -0.2],
                        [0.5, 1.2, -0.3, 0.4]]),
              np.array([[0.8, 0.2, 0.5, 1.0],
                        [-0.4, 1.1, 0.3, 0.6]]),
              np.array([[0.7, -0.3, 1.4, 0.1],
                        [0.2, 0.9, -0.5, 1.2]])]

    # 计算 ZF 和 MRC 波束成形
    U, V = zf_mrc_beamforming(H_list)
    G = np.zeros((L, L))
    for l in range(L):
        for i in range(L):
            G[l, i] = np.abs(V[l].T @ H_list[l] @ U[:, i:i + 1]) ** 2

    w = (np.ones(L) / L).reshape(L, 1)  # 均匀权重
    m = np.ones(L)  # 功率权重
    n = 0.1 * np.ones(L)  # 噪声功率
    P_bar = 10.0  # 总功率约束
    y = np.random.rand(3) * 1

    step, p, gamma = bcd_for_wsrm(G, w, n, m, P_bar, y)
    print("p:", p)
    obj_updated = w.T.dot(np.log(1 + gamma))
    print("=====obj_updated:", obj_updated)
    # 打印结果
    print("发射波束成形矩阵 U (列已归一化):")
    print(U)
    print("\n接收波束成形向量 V:")
    for l in range(L):
        print(f"v_{l + 1}:", V[l].flatten())