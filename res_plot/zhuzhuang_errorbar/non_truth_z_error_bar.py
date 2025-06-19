import matplotlib.pyplot as plt
import numpy as np

# 设置全局字体大小
plt.rcParams.update({'font.size': 12})
ecolor = ['lightblue','lightcoral','lightgreen','lavender','orange']
alg = ["TD3-based BCD", "DDPG-based BCD", "TD3", "DDPG", "GNN"]
plt.rcParams['font.family'] = 'Arial'

colors = ['lightseagreen' , 'lightyellow','lightgreen','lavender', 'orange','lightblue','lightcoral']  # 每个柱子的颜色

alg = ["WMMSE","SCA","BCD", "TD3", "DDPG", "GNN","TD3-based BCD","DDPG-based BCD"]
param = ["L=4", "L=32", "L=64", "L=128"]
xticklabels = alg
mean = {
    "L=4":  np.array([0.9417,0.9237, 0.9531, 0.9638,0.9620, 0.93791, 0.9910,0.9897])/(0.9417),
    "L=32": np.array([0.9487,0.9286, 0.94380,0.9583, 0.9575, 0.9485, 0.98901, 0.9900])/(0.9487),
    "L=64": np.array([0.9507,0.9276, 0.9460,0.9533, 0.9505, 0.9485, 0.99101, 0.9897])/(0.9507),
    "L=128":np.array([0.9387,0.9267,0.9350,  0.9473,0.9433, 0.9309, 0.9780,0.9747])/(0.9387)
}
std = {
    "L=4":  np.array([0.01,0.02, 0.014, 0.009, 0.012,0.018, 0.008, 0.009]),
    "L=32": 0.1*np.array([0.112,0.127, 0.114, 0.1, 0.08,0.06, 0.08, 0.09]),
    "L=64": 0.1*np.array([0.18,0.22, 0.2, 0.12, 0.09,0.07, 0.08, 0.09]),
    "L=128":0.1*np.array([0.2,0.235, 0.215, 0.07, 0.1,0.14, 0.12, 0.12])
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
    ax.set_ylabel('Sum rate')

    ax.set_xticks(range(len(xticklabels)))
    ax.set_xticklabels(xticklabels, rotation=20)  # 设置刻度标签并旋转
    ax.set_title(param[i])
    ax.set_ylim(bottom=0.8)
    # 调整布局
    plt.tight_layout()

    # 保存为PDF文件
    pdf_filename = f'non_truth_sr_error_comp_{i+1}.pdf'
    plt.savefig(pdf_filename, format='pdf')
    plt.show()
    # 关闭图形
    plt.close(fig)