import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import matplotlib.colors as mcolors
plt.rcParams['font.family'] = 'Arial'
# 设置全局字体大小
plt.rcParams.update({'font.size': 17})
alg = ["TD3-based BCD", "DDPG-based BCD", "TD3", "DDPG", "GNN"]
param = ["Learning rate", "Discount factor", "Maximum step", "Batch size"]
xticklabels = [[r"$1\times 10^{-4}$", r"$5\times 10^{-4}$", r"$1\times 10^{-5}$",r"$1\times 10^{-6}$",r"$1\times 10^{-7}$"],
               ['0.80','0.85','0.90','0.95','0.99'],
               [2500,3500,4500,5500,6500],
               [10, 20, 30,40,50]]
ylim = [(-0.05, 0.4),(-0.05, 0.3),(-0.05, 0.4),(-0.05, 0.4)]
mean = {
    "Learning rate": {"TD3-based BCD": np.array([0.08, 0.01, 0.007, 0.003,0.08]),
           "DDPG-based BCD": np.array([0.09, 0.02, 0.001,0.007,0.09]),
           "TD3": np.array([0.15, 0.12, 0.05, 0.09,0.18]),
           "DDPG": np.array([0.18, 0.09, 0.06, 0.1,0.2]),
           "GNN": np.array([0.22, 0.14, 0.09, 0.12,0.18])},
    "Discount factor": {"TD3-based BCD": np.array([0.02, 0.01, 0.004, 0.002,0.003]),
              "DDPG-based BCD": np.array([0.025, 0.01, 0.005, 0.003,0.003]),
              "TD3": np.array([0.115, 0.1, 0.09, 0.07,0.05]),
              "DDPG": np.array([0.127, 0.114, 0.1, 0.08,0.06])},
    "Maximum step": {"TD3-based BCD": np.array([0.01, 0.008, 0.004, 0.002,0.003]),
                "DDPG-based BCD": np.array([0.015, 0.009, 0.005, 0.003,0.003]),
                "TD3": np.array([0.2, 0.15, 0.10, 0.06,0.05]),
                "DDPG": np.array([0.22, 0.17, 0.12, 0.09,0.07])},
    "Batch size": {"TD3-based BCD": np.array([0.003, 0.003, 0.004, 0.007,0.009]),
                   "DDPG-based BCD": np.array([0.003, 0.004, 0.005, 0.009,0.01]),
                   "TD3": np.array([0.07, 0.05, 0.06, 0.09,0.13]),
                   "DDPG": np.array([0.075, 0.055, 0.07, 0.1,0.14])}
}
std = {
    "Learning rate": {"TD3-based BCD": 0.8*np.array([0.03, 0.01, 0.01, 0.01,0.04]),  #0.03, 0.01, 0.007, 0.0003,0.08
           "DDPG-based BCD":0.8* np.array([0.04, 0.01, 0.01,0.01,0.03]), # 0.09, 0.01, 0.001,0.0004,0.09
           "TD3": 0.8*np.array([0.06, 0.05, 0.01, 0.04,0.07]),#0.15, 0.12, 0.05, 0.09,0.18
           "DDPG": 0.8*np.array([0.06, 0.05, 0.01, 0.04,0.07]),#0.22, 0.14, 0.09, 0.12,0.18
           "GNN": 0.8*np.array([0.06, 0.04, 0.02, 0.03,0.08])},
    "Discount factor": {"TD3-based BCD": np.array([0.01, 0.009, 0.007, 0.006,0.006]), #0.09, 0.05, 0.007, 0.002,0.003
              "DDPG-based BCD": np.array([0.01, 0.009, 0.007, 0.006,0.006]),
              "TD3": np.array([0.04, 0.03, 0.02,0.02,0.02]),#0.115, 0.1, 0.09, 0.09,0.05
              "DDPG": np.array([0.04, 0.03, 0.02,0.02,0.02])},
    "Maximum step": {"TD3-based BCD": np.array([0.011, 0.009, 0.008, 0.007,0.009]),
                "DDPG-based BCD": np.array([0.011, 0.009, 0.008, 0.007,0.009]),
                "TD3": np.array([0.06, 0.04, 0.03,0.02,0.01]),#0.2, 0.15, 0.12, 0.08,0.05
                "DDPG": np.array([0.06, 0.04, 0.03,0.02,0.01])},
    "Batch size": {"TD3-based BCD": np.array([0.006, 0.006, 0.008, 0.012,0.015]),
                   "DDPG-based BCD": np.array([0.006, 0.006, 0.008, 0.013,0.015]),
                   "TD3": np.array([0.02, 0.02, 0.03,0.03,0.04]), #0.07, 0.05, 0.06, 0.09,0.13
                   "DDPG": np.array([0.02, 0.02, 0.03,0.03,0.04])}
}

num_fig = len(param)

# 定义Z-score阈值
z_threshold = 1

# 定义颜色映射
colormaps = ['Blues', 'Reds', 'Greens', 'YlOrRd', 'Purples', 'gray']
color = ['blue','red','green','purple','orange']
ecolor = ['lightblue','lightcoral','lightgreen','lavender','orange']
mark_type = ['o','o','v','v','s']

for i in range(num_fig):  # 修改这里
    print("len(mean[param[i]]):", len(mean[param[i]]))

    fig, ax = plt.subplots(figsize=(7.1, 5.5))  # 创建单个子图

    # 计算均值和标准差
    for j in range(len(mean[param[i]])):
        print("----------------")
        print("param[i]:", param[i])
        print("alg[j]:", alg[j])
        print("param[i][alg[j]]:", mean[param[i]])
        means = mean[param[i]][alg[j]]
        std_errs = std[param[i]][alg[j]]

        ci952 = std_errs * 1.95

        vmin = 1.7
        vmax = 2.5
        # 获取颜色映射
        cmap1 = plt.get_cmap(colormaps[j])  # 不反转
        norm1 = mcolors.Normalize(vmin=vmin, vmax=vmax)  # 调整归一化范围
        colors1 = [cmap1(norm1(j / len(means))) for j in range(len(means))]

        # 计算偏移量
        offset = 0.07  # 你可以根据需要调整这个值
        x_positions = [x + j * offset for x in range(len(means))]  # 在原来的位置上加上偏移量

        ax.errorbar(x_positions, means, yerr=ci952, capsize=0, linestyle='None',
                    marker=mark_type[j], elinewidth=1, ecolor=ecolor[j], color=color[j], alpha=.7)
        ax.plot(x_positions, means, color=color[j], alpha=0.2)

    ax.set_xlabel(param[i])
    ax.set_xticks(range(len(xticklabels[i])))
    ax.set_xticklabels(xticklabels[i])
    ax.set_ylabel('Sum rate error')
    ax.legend(alg[:len(mean[param[i]])])
    ax.set_ylim(ylim[i][0], ylim[i][1])

    # 保存每个子图为单独的PDF文件
    plt.savefig(f"res_plot_{param[i]}.pdf")
    plt.close(fig)  # 关闭当前图形，以便在下一次循环中创建新的图形