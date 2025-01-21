import pandas as pd
import matplotlib.pyplot as plt

# 文件名列表
files = ['res_reward.csv', 'obj_err.csv', 'res_step.csv']


# 假设你的数据文件名为data1.csv, data2.csv, ..., data6.csv
file_names = ['res_reward.csv','res_reward.csv', 'obj_err.csv','obj_err.csv', 'res_step.csv','res_step.csv']

num_files = len(file_names)

# 创建一个图形和子图
fig, axs = plt.subplots(1, num_files // 2, figsize=(15, 5))

for i in range(0, num_files, 2):
    # 读取两个数据文件
    data1 = pd.read_csv(file_names[i])
    data2 = pd.read_csv(file_names[i + 1])

    # 计算均值和标准差
    mean1 = data1.mean()
    std1 = data1.std()
    mean2 = data2.mean()
    std2 = data2.std()

    # 绘制error bar图
    axs[i // 2].errorbar(range(len(mean1)), mean1, yerr=std1, label=file_names[i], fmt='o')
    axs[i // 2].errorbar(range(len(mean2)), mean2, yerr=std2, label=file_names[i + 1], fmt='o')

    # 设置图例和标题
    axs[i // 2].legend()
    axs[i // 2].set_title(f'Subplot {i // 2 + 1}')

# 显示图形
plt.tight_layout()
plt.show()
