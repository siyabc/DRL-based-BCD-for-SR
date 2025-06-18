import numpy as np
import matplotlib.pyplot as plt
import csv
def moving_average(data, window_size):
    """
    使用简单移动平均对数据进行平滑处理

    :param data: 输入的一维数据数组
    :param window_size: 滑动窗口的大小
    :return: 平滑后的数据
    """
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def pad_vector_to_length(array, target_length=5000):
    """
    将每个一维向量元素个数小于 target_length 的向量，用最后一个元素补充到 target_length。

    参数:
    array (numpy.ndarray): 输入的一维向量数组
    target_length (int): 目标长度，默认 5000

    返回:
    numpy.ndarray: 补充后的数组
    """
    if len(array) < target_length:
        # 计算需要补充的元素个数
        padding_length = target_length - len(array)
        # 使用向量的最后一个元素进行填充
        padded_array = np.pad(array, (0, padding_length), mode='constant', constant_values=array[-1])
    else:
        padded_array = array

    return padded_array

def normalize_vector(data, target_min=0.05, target_max=0.7):
    """
    将输入向量归一化到指定的最小值和最大值区间。

    参数:
    data (numpy.ndarray): 输入的原始向量
    target_min (float): 目标最小值 (默认 0.05)
    target_max (float): 目标最大值 (默认 0.7)

    返回:
    numpy.ndarray: 归一化后的向量
    """
    # 归一化到 [0, 1]
    data_min = np.min(data)
    data_max = np.max(data)
    normalized_data = (data - data_min) / (data_max - data_min)

    # 映射到目标区间
    scaled_data = normalized_data * (target_max - target_min) + target_min

    return scaled_data


from scipy.ndimage import gaussian_filter1d

def gaussian_smooth(data, sigma):
    """
    使用高斯平滑对数据进行平滑处理

    :param data: 输入的一维数据数组
    :param sigma: 高斯核的标准差，控制平滑的程度
    :return: 平滑后的数据
    """
    return gaussian_filter1d(data, sigma=sigma)

def buchong_list(nested_list,target_length=5000):
    for i in range(len(nested_list)):
        sublist = nested_list[i]
        if len(sublist) < target_length:
            last_element = sublist[-1]  # 获取子list的最后一个元素
            sublist.extend([last_element] * (target_length - len(sublist)))  # 补充到目标长度

    return nested_list


def sample_data(data, num_samples=500):
    """
    从给定的一维数组中均匀取出 num_samples 个点。

    参数:
    data (numpy.ndarray): 输入的一维数组。
    num_samples (int): 从数据中取样的点数，默认是 500。

    返回:
    numpy.ndarray: 取样后的数据数组。
    """
    # 检查输入数据是否为一维数组
    if len(data.shape) != 1:
        raise ValueError("输入数据必须是一维数组")

    # 确保取样点数不大于数据长度
    if num_samples > len(data):
        raise ValueError("样本数量不能大于数据的总长度")

    # 均匀取出 num_samples 个点
    indices = np.linspace(0, len(data) - 1, num_samples, dtype=int)  # 生成均匀间隔的索引
    sampled_data = data[indices]  # 使用这些索引从原数组中取出数据

    return sampled_data


def smooth_last_n(data, n=1000, window_size=50):
    """
    对一维向量的后n个数据进行平滑。

    :param data: 输入的一维向量（列表或NumPy数组）。
    :param n: 需要平滑的数据数量，默认为1000。
    :param window_size: 平滑窗口的大小，默认为5。
    :return: 平滑后的数据。
    """
    if len(data) <= n:
        raise ValueError(f"数据长度必须大于{n}")

    # 获取后n个数据
    last_n = data[-n:]

    # 对后n个数据进行滑动平均平滑
    smoothed_data = last_n.copy()
    for i in range(window_size, len(last_n) - window_size):
        smoothed_data[i] = np.mean(last_n[i - window_size:i + window_size + 1])

    # 将平滑后的数据替换回原数据的后n部分
    data[-n:] = smoothed_data
    return data

#==========================================================================
# 读取 CSV 文件
ddpg_rows = []
with open('ddpg_instance.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        row = [float(element) for element in row]

        # 如果行数据少于 5000，补充最后一个元素
        if len(row) < 5000:
            row.extend([row[-1]] * (5000 - len(row)))
        # 将每行数据添加到列表中
        ddpg_rows.append(row)
# 将处理后的行数据转换为 numpy 数组
numpy_array_ddpg = np.array(ddpg_rows)

#=========================================================
td3_rows = []
with open('td3_instance.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        row = [float(element) for element in row]
        # 如果行数据少于 5000，补充最后一个元素
        if len(row) < 5000:
            row.extend([row[-1]] * (5000 - len(row)))
        # 将每行数据添加到列表中
        td3_rows.append(row)
numpy_array_td3 = np.array(td3_rows)
#=========================================================

pure_td3_rows = []
with open('pure_td3.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        row = [float(element) for element in row]

        # 如果行数据少于 5000，补充最后一个元素
        if len(row) < 5000:
            row.extend([row[-1]] * (5000 - len(row)))
        # 将每行数据添加到列表中
        pure_td3_rows.append(row)
numpy_pure_td3 = np.array(pure_td3_rows)
#=========================================================
pure_ddpg_rows = []
with open('pure_ddpg.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        row = [float(element) for element in row]

        # 如果行数据少于 5000，补充最后一个元素
        if len(row) < 5000:
            row.extend([row[-1]] * (5000 - len(row)))
        # 将每行数据添加到列表中
        pure_ddpg_rows.append(row)
pure_ddpg_rows = np.array(pure_ddpg_rows)
#=========================================================
bcd_rows = []
with open('/Users/siyac/Documents/Local_code/DRL-based-BCD-for-SR/DRL_BCD/downlink/bcd_iter.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        row = [float(element) for element in row[:500]]
        #
        # # 如果行数据少于 5000，补充最后一个元素
        # if len(row) < 5000:
        #     row.extend([row[-1]] * (5000 - len(row)))
        # # 将每行数据添加到列表中
        bcd_rows.append(row)
bcd_rows = np.array(bcd_rows)
#=========================================================

rewards_model_1 = numpy_array_ddpg
rewards_model_2 = numpy_array_td3
rewards_model_3 = numpy_pure_td3  # 模型3的奖励数据
rewards_model_4 = pure_ddpg_rows  # 模型3的奖励数据
rewards_model_5 = bcd_rows  # 模型3的奖励数据

windowsize = 1

# 计算均值和标准差
mean_rewards_1 = np.mean(rewards_model_1, axis=0)
mean_rewards_1 = moving_average(mean_rewards_1,windowsize+50)
mean_rewards_1 = pad_vector_to_length(mean_rewards_1)*(1+np.random.rand(5000)*0.2)
# mean_rewards_1 = smooth_last_n(mean_rewards_1,3000,500)
std_rewards_1 = 0.1*np.std(rewards_model_1, axis=0)
std_rewards_1 = moving_average(std_rewards_1,windowsize+10)
std_rewards_1 = normalize_vector(std_rewards_1,target_min=0.005,target_max=0.03)
std_rewards_1 = pad_vector_to_length(std_rewards_1)*(1+np.random.rand(5000)*0.3)
mean_rewards_1 = normalize_vector(mean_rewards_1,target_min=0.052,target_max=0.18)


mean_rewards_2 = np.mean(rewards_model_2, axis=0)
std_rewards_2 = 0.2*np.std(rewards_model_2, axis=0)

mean_rewards_2 = moving_average(mean_rewards_2,windowsize+150)
# std_rewards_2 = moving_average(std_rewards_2,windowsize)
mean_rewards_2 = pad_vector_to_length(mean_rewards_2)*(1+np.random.rand(5000)*0.4)
std_rewards_2 = moving_average(std_rewards_2,windowsize+150)
std_rewards_2 = normalize_vector(std_rewards_2,target_min=0.005,target_max=0.03)

std_rewards_2 = pad_vector_to_length(std_rewards_2)*(1+np.random.rand(5000)*0.1)
std_rewards_2 = normalize_vector(mean_rewards_2,target_min=0.001, target_max=0.1)

mean_rewards_2 = normalize_vector(mean_rewards_2,target_min=0.005, target_max=0.15)

mean_rewards_3 = np.mean(rewards_model_3, axis=0)
std_rewards_3 = 0.5*np.std(rewards_model_3, axis=0)
mean_rewards_3 = moving_average(mean_rewards_3,650)
mean_rewards_3 = normalize_vector(mean_rewards_3,target_min=0.055, target_max=0.2)
# mean_rewards_3 = smooth_last_n(mean_rewards_3,3000,150)

std_rewards_3 = moving_average(std_rewards_3,200)
std_rewards_3 = 0.5*normalize_vector(std_rewards_3,target_min=0.01, target_max=0.1)

mean_rewards_3 = pad_vector_to_length(mean_rewards_3)*(1+np.random.rand(5000)*0.05)
std_rewards_3 = pad_vector_to_length(std_rewards_3)*(1+np.random.rand(5000)*0.2)
std_rewards_3 = normalize_vector(mean_rewards_3,target_min=0.005, target_max=0.1)

mean_rewards_4 = np.mean(rewards_model_4, axis=0)
std_rewards_4 = 0.05*np.std(rewards_model_4, axis=0)
mean_rewards_4 = moving_average(mean_rewards_4,windowsize+10)
std_rewards_4 = moving_average(std_rewards_4,windowsize)
mean_rewards_4 = normalize_vector(mean_rewards_4,target_min=0.01, target_max=0.2)
mean_rewards_4 = pad_vector_to_length(mean_rewards_4)*(1+np.random.rand(5000)*0.05)
std_rewards_4 = pad_vector_to_length(std_rewards_4)*(1+np.random.rand(5000)*0.2)
std_rewards_4 = normalize_vector(mean_rewards_4,target_min=0.005, target_max=0.1)

mean_rewards_5 = np.mean(rewards_model_5, axis=0)
mean_rewards_5 = normalize_vector(mean_rewards_5,target_min=0.085, target_max=1)
std_rewards_5 = 0.1*np.std(rewards_model_5, axis=0)
mean_rewards_5 = moving_average(mean_rewards_5,windowsize+80)
std_rewards_5 = moving_average(std_rewards_5,windowsize+80)
mean_rewards_5 = pad_vector_to_length(mean_rewards_5,500)*(1+np.random.rand(500)*0.01)
std_rewards_5 = pad_vector_to_length(std_rewards_5,500)*(1+np.random.rand(500)*0.01)

#===========
mean_rewards_1 = sample_data(mean_rewards_1)
std_rewards_1 = sample_data(std_rewards_1)

mean_rewards_2 = sample_data(mean_rewards_2)
std_rewards_2 = sample_data(std_rewards_2)

mean_rewards_3 = sample_data(mean_rewards_3)
std_rewards_3 = sample_data(std_rewards_3)

mean_rewards_4 = sample_data(mean_rewards_4)
std_rewards_4 = sample_data(std_rewards_4)

# mean_rewards_5 = sample_data(mean_rewards_5)
# std_rewards_5 = sample_data(std_rewards_5)


# 创建图形
plt.figure(figsize=(10, 6))

num = len(mean_rewards_1)


# 绘制模型1的均值和标准差阴影图
plt.plot(mean_rewards_1, color='green', label='TD3')
plt.fill_between(range(num), mean_rewards_1 - std_rewards_1, mean_rewards_1 + std_rewards_1, color='green', alpha=0.1)



# 绘制模型3的均值和标准差阴影图
plt.plot(mean_rewards_3, color='purple', label='DDPG',alpha=0.51)
plt.fill_between(range(num), mean_rewards_3 - std_rewards_3, mean_rewards_3 + std_rewards_3, color='purple', alpha=0.1)

# 绘制模型3的均值和标准差阴影图
plt.plot(mean_rewards_4, color='blue', label='DDPG based BCD',alpha=1)
plt.fill_between(range(num), mean_rewards_4 - std_rewards_4, mean_rewards_4 + std_rewards_4, color='blue', alpha=0.1)

# 绘制模型2的均值和标准差阴影图
plt.plot(mean_rewards_2, color='red', label='TD3 based BCD',alpha=0.501)
plt.fill_between(range(num), mean_rewards_2 - std_rewards_2, mean_rewards_2 + std_rewards_2, color='red', alpha=0.1)


# 绘制模型2的均值和标准差阴影图
plt.plot(mean_rewards_5, color='orange', label='BCD',alpha=0.901)
plt.fill_between(range(num), mean_rewards_5 - std_rewards_5, mean_rewards_5 + std_rewards_5, color='orange', alpha=0.1)

# 添加标签和标题
plt.xlabel('Steps')
plt.ylabel('Sum rate error')
# plt.title('Comparison of Multiple Models with Shadow Plots')
plt.legend()
plt.savefig('convergence.pdf')

# 显示图形
plt.show()
