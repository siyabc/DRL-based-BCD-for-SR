import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import matplotlib.colors as mcolors
# from sympy.printing.pretty.pretty_symbology import line_width

# 文件名列表
# files = ['res_reward.csv','res_reward_ddpg.csv',
#               'obj_err.csv','obj_err_ddpg.csv',
#               'res_step.csv','res_step_ddpg.csv',
#          'reward_rewardv3_td3.csv', 'reward_rewardv3_ddpg.csv',
#          'obj_error_rewardv3_td3.csv', 'obj_error_rewardv3_ddpg.csv',
#          'step_rewardv3_td3.csv', 'step_rewardv3_ddpg.csv'
#          ]
files = [
    'res_reward_v2.csv', 'obj_err_v2.csv', 'res_step_v2.csv',
'res_reward_25.csv', 'obj_err_25.csv', 'res_step_25.csv'
    # 'res_reward_ddpg.csv','obj_err_ddpg.csv','res_step_ddpg.csv'
    # 'reward_rewardv3_td3.csv', 'obj_error_rewardv3_td3.csv','step_rewardv3_td3.csv'
    # 'reward_rewardv3_ddpg.csv','obj_error_rewardv3_ddpg.csv', 'step_rewardv3_ddpg.csv'
]
title = ["TD3-based BCD","TD3-based BCD","TD3-based BCD","DDPG-based BCD","DDPG-based BCD","DDPG-based BCD"]

subfig_colmn = ["Cumulative reward", "Sum rate error", "Convergence Steps",
"Cumulative reward", "Sum rate error", "Convergence Steps",
"Cumulative reward", "Sum rate error", "Convergence Steps",
"Cumulative reward", "Sum rate error", "Convergence Steps"]
plt.rcParams['font.family'] = 'Arial'

# 定义Z-score阈值
z_threshold = 1

# 定义颜色映射
colormaps = ['Blues', 'Reds', 'Greens', 'Blues', 'Reds', 'Greens']
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
    ci95 = 1.96 * std_errs

    # 绘制带95%置信区间的图
    x = range(len(means))  # x轴为列的索引
    every = 1
    x_selected = x[::every]  # 每隔3个点选择一个点
    means_selected = means[::every]
    ci95_selected = ci95[::every]

    # 获取颜色映射
    cmap = plt.get_cmap(colormaps[i]).reversed()
    norm = mcolors.Normalize(vmin=0.15, vmax=1.5)  # 调整归一化范围，避开两端的浅色
    colors = [cmap(norm(j / len(x_selected))) for j in range(len(x_selected))]

    # 创建一个新的图形
    fig, ax = plt.subplots(figsize=(4, 3))

    # 连接误差条点
    ax.plot(x_selected, means_selected, color='gray', alpha=0.9, linewidth=1)  # 连接点之间的线
    # 绘制误差条形图
    for j in range(len(x_selected)):
        ax.errorbar(x_selected[j], means_selected[j], yerr=ci95_selected[j], fmt='o', capsize=0, linestyle='None',
                    marker='o', color=colors[j], alpha=.5, markersize=4)

    # 添加标题和标签
    ax.set_xlabel(r'Episodes')
    ax.set_ylabel(subfig_colmn[i])
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax.set_title(title[i],fontsize=11)
    # 调整布局
    plt.tight_layout()

    # 保存为PDF文件
    pdf_filename = f'plot_{i+1}.pdf'
    plt.savefig(pdf_filename, format='pdf')

    # 关闭图形
    plt.show()
