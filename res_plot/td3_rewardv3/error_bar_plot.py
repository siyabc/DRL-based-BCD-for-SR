import pandas as pd
import matplotlib.pyplot as plt

# 文件名列表
files = ['reward_rewardv3_td3.csv', 'obj_error_rewardv3_sl.csv', 'step_rewardv3_td3.csv']

# 创建一个图形和子图
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

subfig_colmn = ["Cumulative reward", "Sum rate error", "Convergence Steps"]
# 遍历每个文件
for i, file in enumerate(files):
    # 读取CSV文件
    # data = pd.read_csv(file)
    data = pd.read_csv(file).iloc[:, :150]

    # 计算均值和标准差
    means = data.mean()
    std_devs = data.std()

    # 绘制带误差条的图
    x = range(len(means))  # x轴为列的索引
    axs[i].errorbar(x, means, yerr=std_devs, fmt='o', capsize=5, linestyle='None', marker='o', color='b')

    # 添加标题和标签
    # axs[i].set_title(f'Error Bar Plot for {file}')
    axs[i].set_xlabel(r'Episodes ($\times 10^3$)')
    axs[i].set_ylabel(subfig_colmn[i])
    # axs[i].set_xticks(x)
    # axs[i].set_xticklabels(data.columns)  # 设置x轴标签为列名

# 调整布局
plt.tight_layout()

# 显示图形
plt.show()
