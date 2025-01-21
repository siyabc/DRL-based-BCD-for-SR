import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import matplotlib.colors as mcolors
# 文件名列表
files = ['res_reward.csv', 'obj_err.csv', 'res_step.csv']
plt.rcParams['font.family'] = 'Arial'
# 创建一个图形和子图
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
subfig_colmn = ["Cumulative reward", "Sum rate error", "Convergence Steps"]
# 定义Z-score阈值
z_threshold = 1
# 定义颜色映射
colormaps = ['Blues', 'Reds', 'Greens']

# 遍历每个文件
for i, file in enumerate(files):
    # 读取CSV文件
    data = pd.read_csv(file).iloc[:, :150]
    data = data.apply(lambda x: np.where(np.abs(stats.zscore(x)) < z_threshold, x, x.mean()), axis=0)
    # 计算均值和标准差
    means = data.mean()
    std_errs = data.sem()
    # 计算95%置信区间
    ci95 = 1.96 * std_errs
    # 绘制带95%置信区间的图
    x = range(len(means))  # x轴为列的索引
    x_selected = x[::3]  # 每隔5个点选择一个点
    means_selected = means[::3]
    ci95_selected = ci95[::3]
    # 获取颜色映射
    cmap = plt.get_cmap(colormaps[i]).reversed()
    norm = mcolors.Normalize(vmin=0, vmax=len(x_selected))
    colors = [cmap(norm(j)) for j in range(len(x_selected))]
    # 绘制误差条形图
    for j in range(len(x_selected)):
        axs[i].errorbar(x_selected[j], means_selected[j], yerr=ci95_selected[j], fmt='o', capsize=0, linestyle='None', marker='o', color=colors[j], alpha=.5)

    # 添加标题和标签
    axs[i].set_xlabel(r'Episodes')
    axs[i].set_ylabel(subfig_colmn[i])
# 调整布局
plt.tight_layout()
# 显示图形
plt.show()
