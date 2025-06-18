import matplotlib.pyplot as plt
import numpy as np

# 设置全局字体大小
plt.rcParams.update({'font.size': 15})
ecolor = ['lightblue','lightcoral','lightgreen','lavender','orange']
alg = ["TD3-based BCD", "DDPG-based BCD", "TD3", "DDPG", "GNN"]
plt.rcParams['font.family'] = 'Arial'

colors = ['lightseagreen' , 'lightyellow','lightgreen','lavender', 'orange','lightblue','lightcoral']  # 每个柱子的颜色

alg = ["SCA","BCD", "TD3", "DDPG", "GNN","TD3-based BCD","DDPG-based BCD"]
param = ["L=4", "L=32", "L=64", "L=128"]
xticklabels = alg
mean = {
    "L=4":  np.array([1-0.9217, 1-0.9231, 1-0.9638,1-0.9620, 1-0.93791, 1-0.9910,1-0.9897]),
    "L=32": np.array([1-0.9286, 1-0.92380,1-0.9583, 1-0.9575, 1-0.9485, 1-0.98901, 1-0.9900]),
    "L=64": np.array([1-0.9206, 1-0.9180,1-0.9533, 1-0.9505, 1-0.9485, 1-0.99101, 1-0.9897]),
    "L=128":np.array([1-0.9207,1-0.9230,  1-0.9473,1-0.9433, 1-0.9309, 1-0.9780,1-0.9747])
}
std = {
    "L=4":  np.array([0.02, 0.014, 0.009, 0.012,0.018, 0.008, 0.009]),
    "L=32": 0.1*np.array([0.127, 0.114, 0.1, 0.08,0.06, 0.08, 0.09]),
    "L=64": 0.1*np.array([0.22, 0.2, 0.12, 0.09,0.07, 0.08, 0.09]),
    "L=128":0.1*np.array([0.175, 0.185, 0.07, 0.1,0.14, 0.12, 0.12])
}

num_fig = len(param)
x = np.arange(num_fig)  # x轴的标签

# 在每个子图上绘制柱状图
for i in range(4):
    a = mean[param[i]]
    print(mean[param[i]])
    print(std[param[i]])

    # 创建一个新的图形
    fig, ax = plt.subplots(figsize=(5, 5))

    ax.bar(range(len(mean[param[i]])), mean[param[i]].tolist(), yerr=std[param[i]].tolist(), color=colors, capsize=5, edgecolor='black', alpha=0.7)
    ax.set_ylabel('Sum rate error')

    ax.set_xticks(range(len(xticklabels)))
    ax.set_xticklabels(xticklabels, rotation=20)  # 设置刻度标签并旋转
    ax.set_title(param[i])

    # 调整布局
    plt.tight_layout()

    # 保存为PDF文件
    pdf_filename = f'sr_error_comp_{i+1}.pdf'
    plt.savefig(pdf_filename, format='pdf')

    # 关闭图形
    plt.close(fig)