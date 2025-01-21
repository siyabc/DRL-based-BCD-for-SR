import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import matplotlib.colors as mcolors

# 文件名列表
files = ['reward_rewardv3_sl.csv', 'obj_error_rewardv3_sl.csv', 'step_rewardv3_sl.csv']
plt.rcParams['font.family'] = 'Arial'

# 创建一个图形和子图
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
subfig_colmn = ["Cumulative reward", "Sum rate error", "Convergence Steps"]

# 定义Z-score阈值
z_threshold = 1

# 定义颜色映射
colormaps = ['Blues', 'Reds', 'Greens']
color = ['blue', 'red', 'green']

# 遍历每个文件
for i, file in enumerate(files):
    # 读取CSV文件
    data = pd.read_csv(file).iloc[:, :150]
    data = data.apply(lambda x: np.where(np.abs(stats.zscore(x)) < z_threshold, x, x.mean()), axis=0)

    # 计算均值和标准差
    means = data.mean()
    std_errs = data.sem()

    # 计算95%置信区间
    # ci95 = 1.96 * std_errs
    ci95 = 0.96 * std_errs

    # 绘制带95%置信区间的图
    x = range(len(means))  # x轴为列的索引
    every = 1
    x_selected = x[::every]  # 每隔3个点选择一个点
    means_selected = means[::every]
    ci95_selected = ci95[::every]

    # 获取颜色映射
    cmap = plt.get_cmap(colormaps[i]).reversed()
    norm = mcolors.Normalize(vmin=0.9, vmax=1.3)  # 调整归一化范围，避开两端的浅色
    colors = [cmap(norm(j / len(x_selected))) for j in range(len(x_selected))]

    # 绘制误差条形图
    for j in range(len(x_selected)):
        axs[i].errorbar(x_selected[j], means_selected[j], yerr=ci95_selected[j], fmt='o', capsize=0, linestyle='None',
                        marker='o', color=colors[j], alpha=.5)

    # 连接误差条点
    # axs[i].plot(x_selected, means_selected, color='black', alpha=0.5)  # 连接点之间的线

    # 添加标题和标签
    axs[i].set_xlabel(r'Episodes')
    axs[i].set_ylabel(subfig_colmn[i])

# 调整布局
plt.tight_layout()

# 显示图形
plt.show()
