import numpy as np


def construct_matrix(row_sums, col_sums):
    m = len(row_sums)  # 行数
    n = len(col_sums)  # 列数

    # 检查行和与列和是否相等
    if sum(row_sums) != sum(col_sums):
        raise ValueError("行和与列和不相等，无法构造矩阵。")

    # 初始化矩阵
    matrix = np.zeros((m, n))

    # 使用贪心算法填充矩阵
    for i in range(m):
        for j in range(n):
            # 选择当前行和列的最小值
            value = min(row_sums[i], col_sums[j])
            matrix[i][j] = value

            # 更新行和列的剩余值
            row_sums[i] -= value
            col_sums[j] -= value

    return matrix


# 示例
row_sums = [1.5, 5.34, 0.5]  # 行和
col_sums = [1.1, 3.24, 3]  # 列和

matrix = construct_matrix(row_sums, col_sums)

# 打印构造的矩阵
print("构造的矩阵：")
print(matrix)

# 打印行和和列和
print("行和：", matrix.sum(axis=1))  # 计算行和
print("列和：", matrix.sum(axis=0))  # 计算列和
