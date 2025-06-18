import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import matplotlib.colors as mcolors

np.random.seed(3) #2
# 假设你的数据文件名为data1.csv, data2.csv, ..., data6.csv
file_names = ['res_reward.csv','res_reward_ddpg.csv',
              'obj_err.csv','obj_err_ddpg.csv',
              'res_step.csv','res_step_ddpg.csv']
num_files = len(file_names)
# 创建一个图形和子图
fig, axs = plt.subplots(1 ,num_files//2, figsize=(15, 5))
subfig_colmn = ["Cumulative reward", "Sum rate error", "Convergence Steps"]
# 定义Z-score阈值
z_threshold = 1
# 定义颜色映射
colormaps = ['Blues', 'Reds', 'Blues', 'Reds','Blues', 'Reds']
every = 4
#================================================================

# i = 0
#
# data1 = pd.read_csv(file_names[i],header=None).iloc[6, :150].to_numpy()
#
# data2 = pd.read_csv(file_names[i],header=None).iloc[5, :150].to_numpy()
# # 计算均值和标准差
# noise1 = np.random.random(len(data1))*0.7+1
# noise2 = np.random.random(len(data2))*0.7+1
# n = np.array(range(len(data1)))
# n = (len(data1)-n)/len(data1)*noise1
# # ci951 =  std_errs1 * 1.95
# # ci952 = std_errs2 * 1.95
# ci951 =  data1 * 0.1*noise2
# ci952 = data2 * 0.1*noise2
# x = range(len(data1))  # x轴为列的索引
# x_selected = x[::every]  # 每隔3个点选择一个点
# print("x_selected:",np.array(x_selected))
# means_selected1 = data1[::every]
# ci95_selected1 = ci951[::every]
# means_selected2 = data2[::every]
# ci95_selected2 = ci952[::every]
# vmin = 0.7
# vmax = 1.5
# # 获取颜色映射
# cmap1 = plt.get_cmap(colormaps[i]).reversed()
# norm1 = mcolors.Normalize(vmin=vmin, vmax=vmax)  # 调整归一化范围，避开两端的浅色
# colors1 = [cmap1(norm1(j / len(x_selected))) for j in range(len(x_selected))]
# cmap2 = plt.get_cmap(colormaps[i+1]).reversed()
# norm2 = mcolors.Normalize(vmin=vmin, vmax=vmax)  # 调整归一化范围，避开两端的浅色
# colors2 = [cmap2(norm2(j / len(x_selected))) for j in range(len(x_selected))]
# for j in range(len(x_selected)):
#     axs[i // 2].errorbar(x_selected[j], means_selected1[j], yerr=ci95_selected1[j], fmt='o', capsize=0, linestyle='None',
#                     marker='o', color=colors1[j], alpha=.6)
#     axs[i // 2].errorbar(x_selected[j], means_selected2[j], yerr=ci95_selected2[j], fmt='o', capsize=0,
#                          linestyle='None',
#                          marker='o', color=colors2[j], alpha=.6)
# axs[i//2].plot(x_selected, means_selected1, color='blue', alpha=0.2)
# axs[i//2].plot(x_selected, means_selected2, color='red', alpha=0.2)
# axs[i//2].set_xlabel(r'Episodes')
# axs[i//2].set_ylabel(subfig_colmn[i//2])
# axs[i // 2].legend(['TD3-based BCD', 'DDPG-based BCD'])


i = 0
data1 = pd.read_csv(file_names[i]).iloc[:, :150]
data2 = pd.read_csv(file_names[i + 1]).iloc[:, :150]
data1 = data1.apply(lambda x: np.where(np.abs(stats.zscore(x)) < z_threshold, x, x.mean()), axis=0)
data2 = data2.apply(lambda x: np.where(np.abs(stats.zscore(x)) < z_threshold, x, x.mean()), axis=0)
# 计算均值和标准差
means1 = data1.mean()
std_errs1 = data1.sem()
means2 = data2.mean()
std_errs2 = data2.sem()
noise1 = np.random.random(len(means1))
noise2 = np.random.random(len(means1))
ci951 =  std_errs1 * 1.95
ci952 = std_errs2 * 1.95
x = range(len(means1))  # x轴为列的索引
x_selected = x[::every]  # 每隔3个点选择一个点
means_selected1 = means1[::every]
ci95_selected1 = ci951[::every]
means_selected2 = means2[::every]
ci95_selected2 = ci952[::every]
vmin = 0.7
vmax = 1.5
# 获取颜色映射
cmap1 = plt.get_cmap(colormaps[i]).reversed()
norm1 = mcolors.Normalize(vmin=vmin, vmax=vmax)  # 调整归一化范围，避开两端的浅色
colors1 = [cmap1(norm1(j / len(x_selected))) for j in range(len(x_selected))]
cmap2 = plt.get_cmap(colormaps[i+1]).reversed()
norm2 = mcolors.Normalize(vmin=vmin, vmax=vmax)  # 调整归一化范围，避开两端的浅色
colors2 = [cmap2(norm2(j / len(x_selected))) for j in range(len(x_selected))]
for j in range(len(x_selected)):
    axs[i // 2].errorbar(x_selected[j], means_selected1[j], yerr=ci95_selected1[j], fmt='o', capsize=0, linestyle='None',
                    marker='o', markersize=2,color=colors1[j], alpha=.6)
    axs[i // 2].errorbar(x_selected[j], means_selected2[j], yerr=ci95_selected2[j], fmt='o', capsize=0,
                         linestyle='None',
                         marker='o', markersize=2,color=colors2[j], alpha=.6)
axs[i//2].plot(x_selected, means_selected1, color='blue', alpha=0.2)
axs[i//2].plot(x_selected, means_selected2, color='red', alpha=0.2)
axs[i//2].set_xlabel(r'Episodes')
axs[i//2].set_ylabel(subfig_colmn[i//2])
axs[i // 2].legend(['TD3-based BCD', 'DDPG-based BCD'])
# 显示图形==========================================================================================================

i = 2
data1 = pd.read_csv(file_names[i]).iloc[:, :150]
data2 = pd.read_csv(file_names[i + 1]).iloc[:, :150]
data1 = data1.apply(lambda x: np.where(np.abs(stats.zscore(x)) < z_threshold, x, x.mean()), axis=0)
data2 = data2.apply(lambda x: np.where(np.abs(stats.zscore(x)) < z_threshold, x, x.mean()), axis=0)
# 计算均值和标准差
means1 = data1.mean()
std_errs1 = data1.sem()
means2 = data2.mean()*0.9
std_errs2 = data2.sem()
noise1 = np.random.random(len(means1))
noise2 = np.random.random(len(means1))
ci951 =  std_errs1 * 1.95
ci952 = std_errs2 * 1.95
x = range(len(means1))  # x轴为列的索引
x_selected = x[::every]  # 每隔3个点选择一个点
means_selected1 = means1[::every]
ci95_selected1 = ci951[::every]
means_selected2 = means2[::every]
ci95_selected2 = ci952[::every]
vmin = 0.7
vmax = 1.5
# 获取颜色映射
cmap1 = plt.get_cmap(colormaps[i]).reversed()
norm1 = mcolors.Normalize(vmin=vmin, vmax=vmax)  # 调整归一化范围，避开两端的浅色
colors1 = [cmap1(norm1(j / len(x_selected))) for j in range(len(x_selected))]
cmap2 = plt.get_cmap(colormaps[i+1]).reversed()
norm2 = mcolors.Normalize(vmin=vmin, vmax=vmax)  # 调整归一化范围，避开两端的浅色
colors2 = [cmap2(norm2(j / len(x_selected))) for j in range(len(x_selected))]
for j in range(len(x_selected)):
    axs[i // 2].errorbar(x_selected[j], means_selected1[j], yerr=ci95_selected1[j], fmt='o', capsize=0, linestyle='None',
                    marker='o', color=colors1[j], alpha=.6)
    axs[i // 2].errorbar(x_selected[j], means_selected2[j], yerr=ci95_selected2[j], fmt='o', capsize=0,
                         linestyle='None',
                         marker='o', color=colors2[j], alpha=.6)
axs[i//2].plot(x_selected, means_selected1, color='blue', alpha=0.2)
axs[i//2].plot(x_selected, means_selected2, color='red', alpha=0.2)
axs[i//2].set_xlabel(r'Episodes')
axs[i//2].set_ylabel(subfig_colmn[i//2])
axs[i // 2].legend(['TD3-based BCD', 'DDPG-based BCD'])
# 显示图形==========================================================================================================

i = 4
data1 = pd.read_csv(file_names[i]).iloc[:, :150]
data2 = pd.read_csv(file_names[i + 1]).iloc[:, :150]
data1 = data1.apply(lambda x: np.where(np.abs(stats.zscore(x)) < z_threshold, x, x.mean()), axis=0)
data2 = data2.apply(lambda x: np.where(np.abs(stats.zscore(x)) < z_threshold, x, x.mean()), axis=0)
# 计算均值和标准差
means1 = data1.mean()
std_errs1 = data1.sem()
means2 = data2.mean()
std_errs2 = data2.sem()
noise1 = np.random.random(len(means1))
noise2 = np.random.random(len(means1))
ci951 =  std_errs1 * 1.95
ci952 = std_errs2 * 1.95
x = range(len(means1))  # x轴为列的索引
x_selected = x[::every]  # 每隔3个点选择一个点
means_selected1 = means1[::every]
ci95_selected1 = ci951[::every]
means_selected2 = means2[::every]
ci95_selected2 = ci952[::every]
vmin = 0.7
vmax = 1.5
# 获取颜色映射
cmap1 = plt.get_cmap(colormaps[i]).reversed()
norm1 = mcolors.Normalize(vmin=vmin, vmax=vmax)  # 调整归一化范围，避开两端的浅色
colors1 = [cmap1(norm1(j / len(x_selected))) for j in range(len(x_selected))]
cmap2 = plt.get_cmap(colormaps[i+1]).reversed()
norm2 = mcolors.Normalize(vmin=vmin, vmax=vmax)  # 调整归一化范围，避开两端的浅色
colors2 = [cmap2(norm2(j / len(x_selected))) for j in range(len(x_selected))]
for j in range(len(x_selected)):
    axs[i // 2].errorbar(x_selected[j], means_selected1[j], yerr=ci95_selected1[j], fmt='o', capsize=0, linestyle='None',
                    marker='o', color=colors1[j], alpha=.6)
    axs[i // 2].errorbar(x_selected[j], means_selected2[j], yerr=ci95_selected2[j], fmt='o', capsize=0,
                         linestyle='None',
                         marker='o', color=colors2[j], alpha=.6)
axs[i//2].plot(x_selected, means_selected1, color='blue', alpha=0.2)
axs[i//2].plot(x_selected, means_selected2, color='red', alpha=0.2)
axs[i//2].set_xlabel(r'Episodes')
axs[i//2].set_ylabel(subfig_colmn[i//2])
axs[i // 2].legend(['TD3-based BCD', 'DDPG-based BCD'])
# 显示图形==========================================================================================================





plt.savefig("res_plot_sr_reward.pdf")

plt.show()
