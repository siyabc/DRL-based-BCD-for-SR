from LP_BCD import zf_mrc_beamforming
from IP_BCD import iteration_for_subproblem,bcd_for_wsrm,wsrm_algorithm
import numpy as np
from scipy.linalg import pinv, norm
import time


#---
N = 128  # 发射天线数
L = 32  # 用户数
N_l = [2]*L  # 接收天线数
w = (np.ones(L) / L).reshape(L, 1)
m = np.ones(L)
n = 0.1 * np.ones(L)
P_bar = 3*L
num_trials = 10  # 试验次数

acc_list = []
bcd_time_list = []
wsrm_time_list = []
IP_obj_list = []
LP_obj_list = []

for trial in range(num_trials):
    # 生成随机信道
    H_list = [np.random.randn(N_l[l], N) + 1j * np.random.randn(N_l[l], N) for l in range(L)]

    # 计算G矩阵和LP目标值（统计BCD时间）
    U, V = zf_mrc_beamforming(H_list)
    G = np.zeros((L, L))
    for l in range(L):
        for i in range(L):
            G[l, i] = np.abs(V[l].T @ H_list[l] @ U[:, i:i + 1]) ** 2

    start_time = time.time()
    _, p, gamma = bcd_for_wsrm(G, w, n, m, P_bar, np.random.rand(L))
    bcd_time = time.time() - start_time
    bcd_time_list.append(bcd_time)

    LP_obj = w.T.dot(np.log(1 + gamma))[0]
    LP_obj_list.append(LP_obj)

    # 计算IP目标值（统计WSRM时间）
    start_time = time.time()
    _, _, _, IP_obj = wsrm_algorithm(H_list, w, m, n, P_bar)
    wsrm_time = time.time() - start_time
    wsrm_time_list.append(wsrm_time)
    IP_obj_list.append(IP_obj)

    acc = IP_obj / LP_obj
    acc_list.append(acc)

# 计算统计结果
mean_acc = np.mean(acc_list)
std_acc = np.std(acc_list)
mean_bcd_time = np.mean(bcd_time_list)
std_bcd_time = np.std(bcd_time_list)
mean_wsrm_time = np.mean(wsrm_time_list)
std_wsrm_time = np.std(wsrm_time_list)
mean_IP_obj = np.mean(IP_obj_list)
std_IP_obj = np.std(IP_obj_list)
mean_LP_obj = np.mean(LP_obj_list)
std_LP_obj = np.std(LP_obj_list)

# 打印结果
print(f"试验次数: {num_trials}")
print("---------------- 目标值统计 ----------------")
print(f"IP_obj均值: {mean_IP_obj:.4f}, IP_obj方差: {std_IP_obj:.4f}")
print(f"LP_obj均值: {mean_LP_obj:.4f}, LP_obj方差: {std_LP_obj:.4f}")
print(f"acc均值: {mean_acc:.4f}, acc方差: {std_acc:.4f}")
print("\n---------------- 时间统计 ----------------")
print(f"BCD时间均值: {mean_bcd_time:.6f}秒, BCD时间方差: {std_bcd_time:.6f}秒")
print(f"WSRM时间均值: {mean_wsrm_time:.6f}秒, WSRM时间方差: {std_wsrm_time:.6f}秒")