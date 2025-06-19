import numpy as np
import time

# 记录开始时间
start_time = time.time()

# 生成128x128随机矩阵
np.random.seed(42)
matrix = np.random.rand(580, 580)

# 计算逆矩阵
try:
    inv_matrix = np.linalg.inv(matrix)
    print("逆矩阵计算成功！")
    print("前5x5子矩阵:\n", inv_matrix[:5, :5])

except np.linalg.LinAlgError:
    print("矩阵是奇异的，无法求逆！")

# 输出总耗时
total_time = time.time() - start_time
print(f"总耗时: {total_time:.4f} 秒")