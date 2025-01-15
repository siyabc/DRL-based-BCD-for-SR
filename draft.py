# import matplotlib
# matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt

# 示例数据
x = [1, 2, 3, 4, 5]  # x 轴数据
y = [2, 3, 5, 7, 11]  # y 轴数据

# 创建折线图
plt.plot(x, y, marker='o')  # marker='o' 用于在数据点上添加圆圈标记

# 添加标题和标签


# 显示网格
plt.grid()

# 显示图形
plt.show()
