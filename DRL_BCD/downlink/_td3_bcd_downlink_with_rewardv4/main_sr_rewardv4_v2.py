# 对于多个样例来训练
# reward function 选为 “Multi-Agent Deep Reinforcement Learning for
# Dynamic Power Allocation in Wireless Networks”

from td3_sr_rewardv4 import TD3
import matplotlib.pyplot as plt
import numpy as np
import csv
import math
import pandas as pd


def calculate_C(F, p, sigma):
    num_nodes = len(p)
    C = np.zeros((num_nodes,num_nodes) ) # 初始化 C 数组

    for i in range(num_nodes):
        for k in range(num_nodes):
            if k != i:  # 确保 j != i, k
                numerator = p[k]
                denominator = np.sum(F[k][j] * p[j] for j in range(num_nodes) if j != i and j != k) + sigma[k]
                C[i][k] = np.log(1 + numerator / denominator)
    return C

def step(o, a, label):
    G = o[0:9].reshape(3, 3)
    w = o[9:12].reshape(3, 1)
    sigma = o[12:15].reshape(3, 1)
    p_bar = o[15]
    gamma_star = label
    F = np.dot(np.linalg.inv(np.diag(np.diag(G))), (G - np.diag(np.diag(G))))
    v = np.dot(np.linalg.inv(np.diag(np.diag(G))), sigma)
    # print(np.dot(v,np.ones((1, 3))))
    B = F+1/p_bar*np.dot(v,np.ones((1, 3)))
    # y = 1/(1/np.exp(a)+1)
    # b = w[:,0]*y
    b = w[:, 0] * a
    til_gamma = iteration_3u_v2(B, b)
    gamma = np.exp(til_gamma)

    p = np.linalg.inv(np.eye(len(gamma)) - np.diag(gamma) @ F) @ np.diag(gamma) @ v
    C = calculate_C(F, p.reshape(3), v.reshape(3))
    reward = 3*w.T.dot(np.log(1 + gamma))[0]-np.sum(w.T.dot(C.T))

    obj_updated = w.T.dot(np.log(1 + gamma))[0]
    obj_star = w.T.dot(np.log(1 + gamma_star))[0]
    # reward = - (np.abs(obj_updated-obj_star))**2


    obj_err = abs(np.abs(obj_updated - obj_star)) / obj_star

    # reward = - np.linalg.norm(gamma - gamma_star, 1)
    # print("a:", a)
    # print("gamma:", gamma)
    # print("gamma_star:", gamma_star)
    # print("reward:", reward)
    o2 = np.append(o[:-3], gamma)
    d = False
    if reward <= 1e-2:
        d = True
    return o2, reward, d, obj_err


def iteration_3u_v2(B, b):
    z = np.random.rand(3)
    z0 = z[0]
    z1 = z[1]
    z2 = z[2]
    tol = 10e-9
    err = 1
    # print("B:", np.reshape(B, (1,9)))

    while err>tol:
        z0_temp = z0
        z1_temp = z1
        z2_temp = z2
        # print("Z:", z0,z1,z2)

        z0 = b[0]/(b[0]*B[0][0]/(B[0][0]*z0+B[0][1]*z1+B[0][2]*z2) +  b[1]*B[1][0]/(B[1][0]*z0+B[1][1]*z1+B[1][2]*z2) +  b[2]*B[2][0]/(B[2][0]*z0+B[2][1]*z1+B[2][2]*z2))
        z1 = b[1]/(b[0]*B[0][1]/(B[0][0]*z0+B[0][1]*z1+B[0][2]*z2) +  b[1]*B[1][1]/(B[1][0]*z0+B[1][1]*z1+B[1][2]*z2) +  b[2]*B[2][1]/(B[2][0]*z0+B[2][1]*z1+B[2][2]*z2))
        z2 = b[2]/(b[0]*B[0][2]/(B[0][0]*z0+B[0][1]*z1+B[0][2]*z2) +  b[1]*B[1][2]/(B[1][0]*z0+B[1][1]*z1+B[1][2]*z2) +  b[2]*B[2][2]/(B[2][0]*z0+B[2][1]*z1+B[2][2]*z2))

        err = abs(z0_temp - z0)+abs(z1_temp - z1)+abs(z2_temp - z2)
    z = np.array([z0, z1, z2])
    # print("b:", b)
    # print("Z:", z0, z1, z2)
    res = B.dot(z)
    # print(np.log(z[0] / res[0]))
    # print(np.log(z[1] / res[1]))
    # print(np.log(z[2] / res[2]))
    til_gamma = np.array([np.log(z[0] / res[0]),np.log(z[1] / res[1]),np.log(z[2] / res[2])])

    return til_gamma


def main():
    sr_data = list()
    sr_target = list()
    with open("res/data1.csv", "r") as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            sr_data.append(list(map(float, line[:-3])))
            sr_target.append(list(map(float, line[-3:])))

    obs_dim = 19
    act_dim = 3
    td3 = TD3(obs_dim, act_dim)

    # MAX_EPISODE = len(sr_data)
    MAX_EPISODE = 150  # 这里指的是第几个样本
    MAX_STEP = 10000
    update_every = 200  # 100
    batch_size = 10
    start_update = 50  # 10

    all_rewardList = []
    all_err_list = []
    all_stop_step_list = []
    all_obj_err_list = []
    epoch_num = 1

    for epoch in range(epoch_num):
        print("epoch:", epoch)
        rewardList = []
        err_list = []
        stop_step_list = []
        outlier_list = []
        obj_err_list = []

        o_init = np.append(np.array(sr_data[epoch]), np.random.rand(3) * 0.1)
        label = np.array(sr_target[epoch])

        o = o_init
        ep_reward = 0
        stop_step = 0

        for episode in range(MAX_EPISODE):
            # o = env.reset()

            for j in range(MAX_STEP):
                # if episode > 20:
                #     a = td3.get_action(o, td3.act_noise) * 2
                # else:
                #     a = env.action_space.sample()
                # a = td3.get_action(o, td3.act_noise) * 2
                a = td3.get_action(o, 0.001)
                # ======================
                # next state
                o2, r, d, obj_err = step(o, a, label)
                # ======================
                td3.replay_buffer.store(o, a, r, o2, d)
                if episode >= start_update and j % update_every == 0:
                    td3.update(batch_size, update_every)

                o = o2
                ep_reward += r
                stop_step = j
                if d: break
            gamma = o[-3:]
            err = np.linalg.norm(gamma - label)
            print('Episode:', episode, 'gamma:', gamma, 'label:', label, '==========', 'Reward:', ep_reward, 'err:',
                  err, '---', 'obj_err:', obj_err, 'j:', stop_step)
            if math.isnan(ep_reward):
                print("a:", a)
            rewardList.append(ep_reward)
            err_list.append(err)
            obj_err_list.append(obj_err)
            stop_step_list.append(stop_step)
            if episode > 50 and stop_step >= 1000:
                outlier_list.append(episode)

        print("rewardList:", rewardList)
        print("err_list:", err_list)
        print("stop_step_list:", stop_step_list)
        print("outlier_list:", outlier_list)

        plt.figure(figsize=(18, 4))
        all_rewardList = all_rewardList + rewardList
        all_err_list = all_err_list + err_list
        all_obj_err_list = all_obj_err_list + obj_err_list
        all_stop_step_list = all_stop_step_list + stop_step_list

    plt.subplot(1, 4, 1)
    plt.plot(np.arange(len(all_rewardList)), all_rewardList)
    plt.xlabel("Episode", fontsize=10)
    plt.ylabel("Reward", fontsize=10)

    plt.subplot(1, 4, 2)
    plt.plot(np.arange(len(all_err_list)), all_err_list)
    plt.xlabel("Episode", fontsize=10)
    plt.ylabel("SINR error", fontsize=10)

    plt.subplot(1, 4, 3)
    plt.plot(np.arange(len(all_obj_err_list)), all_obj_err_list)
    plt.xlabel("Episode", fontsize=10)
    plt.ylabel("Sum rate error", fontsize=10)

    plt.subplot(1, 4, 4)
    plt.plot(np.arange(len(all_stop_step_list)), all_stop_step_list)
    plt.xlabel("Episode", fontsize=10)
    plt.ylabel("Iteration steps", fontsize=10)

    plt.title("td3_v2")

    plt.savefig("res_v1.pdf")
    plt.show()

    with open('res/res_reward_v2.csv', 'w', newline='') as file:
        mywriter = csv.writer(file, delimiter=',')  # 移到循环外部
        mywriter.writerow(all_rewardList)

    with open('res/res_err_v2.csv', 'w', newline='') as file:
        mywriter = csv.writer(file, delimiter=',')  # 移到循环外部
        mywriter.writerow(all_err_list)

    with open('res/obj_err_v2.csv', 'w', newline='') as file:
        mywriter = csv.writer(file, delimiter=',')  # 移到循环外部
        mywriter.writerow(all_obj_err_list)

    with open('res/res_step_v2.csv', 'w', newline='') as file:
        mywriter = csv.writer(file, delimiter=',')  # 移到循环外部
        mywriter.writerow(all_stop_step_list)


if __name__ == '__main__':
    main()