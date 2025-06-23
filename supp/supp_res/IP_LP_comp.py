import matplotlib.pyplot as plt
import numpy as np

# 设置全局字体大小
plt.rcParams.update({'font.size': 12})
ecolor = ['lightblue','lightcoral','lightgreen','lavender','orange']
alg = ["TD3-based BCD", "DDPG-based BCD", "TD3", "DDPG", "GNN"]
plt.rcParams['font.family'] = 'Arial'

colors = ['lightseagreen' , 'lightyellow','lightgreen','lavender', 'orange','lightblue','lightcoral']  # 每个柱子的颜色

alg = ["BCD","IP-BCD", "TD3-based BCD", "IP-TD3-based BCD", "DDPG-based BCD","IP-DDPG-based BCD"]
param = ["$L=4,N=4,N_l=2$", "$L=4,N=32,N_l=2$", "$L=32,N=32,N_l=2$", "$L=32,N=128,N_l=2$"]
xticklabels = alg
mean = {
    "$L=4,N=4,N_l=2$":  np.array([0.7590,0.9502, 0.7993, 0.9838, 0.7930, 0.9879]),
    "$L=4,N=32,N_l=2$": np.array([0.9045,0.9488, 0.9355,0.9810, 0.9331, 0.9825]),
    "$L=32,N=32,N_l=2$": np.array([0.7792,0.9276, 0.8150,0.9733, 0.8175, 0.9805]),
    "$L=32,N=128,N_l=2$":np.array([0.8154,0.9267,0.8624,  0.9773,0.8599, 0.9769])
}
std = {
    "$L=4,N=4,N_l=2$":  np.array([0.03,0.02, 0.014, 0.009, 0.01,0.01]),
    "$L=4,N=32,N_l=2$": np.array([0.0312,0.0207, 0.0114, 0.008,0.01, 0.009]),
    "$L=32,N=32,N_l=2$": np.array([0.031,0.024, 0.012,0.009, 0.011, 0.007]),
    "$L=32,N=128,N_l=2$":np.array([0.04,0.0275, 0.015, 0.012, 0.012,0.01])
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
    pdf_filename = f'IP_LP_comp_{i+1}.pdf'
    plt.savefig(pdf_filename, format='pdf')
    plt.show()
    # 关闭图形
    plt.close(fig)