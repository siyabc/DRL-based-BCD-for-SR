from LP_BCD import zf_mrc_beamforming
from IP_BCD import iteration_for_subproblem,bcd_for_wsrm,wsrm_algorithm
import numpy as np
from scipy.linalg import pinv, norm


N = 64  # 发射天线数
L = 3  # 用户数
N_l = [2, 2, 2]  # 每个用户的接收天线数
H_list = [np.random.randn(N_l[l], N) + 1j * np.random.randn(N_l[l], N) for l in range(L)]

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
print("LP-obj:",obj_updated)

p_opt, U_opt, V_opt = wsrm_algorithm(H_list, w, m, n, P_bar)
print("最优功率分配:", p_opt)