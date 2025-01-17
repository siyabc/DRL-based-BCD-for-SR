# 对于多个样例来训练
# reward function 选为 sum rate

from td3_sr_rewardv3 import TD3
import matplotlib.pyplot as plt
import numpy as np
import csv
import math
import pandas as pd

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


    obj_updated = w.T.dot(np.log(1 + gamma))[0]
    obj_star = w.T.dot(np.log(1 + gamma_star))[0]
    obj_err = abs(np.abs(obj_updated-obj_star))/obj_star
    # print("===obj_updated:", obj_updated)
    # print("---obj_star:", obj_star)

    old_gamma = o[-3:]
    obj_old = w.T.dot(np.log(1 + old_gamma))[0]
    # print("===old_gamma:", old_gamma)
    # print("---gamma:", gamma,"****gamma_star:", gamma_star)
    # print("===obj_old:", obj_old)
    # print("---obj_updated:", obj_updated)
    # print('-------------------------------')

    reward = obj_updated
    delta_reward = np.abs(obj_updated - obj_old)
    # print("===delta_reward:", delta_reward)

    o2 = np.append(o[:-3], gamma)
    d = False
    if delta_reward <= 1e-2:
        d = True
    d = False

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


if __name__ == '__main__':
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
    MAX_EPISODE = 100 # 这里指的是第几个样本
    MAX_STEP = 8000
    update_every = 100 # 100
    batch_size = 20
    start_update = 30 # 10


    all_rewardList = []
    all_err_list = []
    all_stop_step_list = []
    epoch_num = 1

    for epoch in range(epoch_num):
        print("epoch:", epoch)

        rewardList = []
        err_list = []
        stop_step_list = []
        outlier_list = []
        obj_err_list = []

        for episode in range(MAX_EPISODE):
            # o = env.reset()
            o_init = np.append(np.array(sr_data[episode]), np.random.rand(3)*0.1)
            label = np.array(sr_target[episode])

            o = o_init
            ep_reward = 0
            stop_step = 0
            for j in range(MAX_STEP):
                # if episode > 20:
                #     a = td3.get_action(o, td3.act_noise) * 2
                # else:
                #     a = env.action_space.sample()
                # a = td3.get_action(o, td3.act_noise) * 2
                a = td3.get_action(o, 0.001)
                # print("==a:", a)
                # if sum(a) <=0:
                #     print("==a:", a)
                #     continue
                # o2, r, d, _ = env.step(a)

                # ======================
                # next state
                o2, r, d, obj_err = step(o, a, label)

                # ======================

                td3.replay_buffer.store(o, a, r, o2, d)

                if episode >= start_update and j % update_every == 0:
                    td3.update(batch_size, update_every)
                    # print("j:", j)

                o = o2
                ep_reward += r
                stop_step = j
                if d: break
            gamma = o[-3:]
            err = np.linalg.norm(gamma-label)
            print('Episode:', episode,'gamma:', gamma,'label:', label,'==========', 'Reward:',ep_reward, 'err:', err,'---', 'obj_err:',obj_err, 'j:', stop_step)
            # print('Episode:', episode, '====Reward:',ep_reward, '****err:', err, 'j:', stop_step)

            # print('gamma:', o)
            # print("a:", a)
            if math.isnan (ep_reward):
                print("a:", a)
            rewardList.append(ep_reward)
            err_list.append(err)
            obj_err_list.append(obj_err)
            stop_step_list.append(stop_step)
            if episode > 50 and stop_step>=1000:
                outlier_list.append(episode)

        print("rewardList:", rewardList)
        print("err_list:", err_list)
        print("stop_step_list:", stop_step_list)

        # print("outlier_list:", outlier_list)

        print("outlier_list:", outlier_list)

        plt.figure(figsize=(18, 4))

        plt.subplot(1, 4, 1)
        plt.plot(np.arange(len(rewardList)), rewardList)
        plt.xlabel("Episode", fontsize=10)
        plt.ylabel("Reward", fontsize=10)

        plt.subplot(1, 4, 2)
        plt.plot(np.arange(len(err_list)), err_list)
        plt.xlabel("Episode", fontsize=10)
        plt.ylabel("SINR error", fontsize=10)

        plt.subplot(1, 4, 3)
        plt.plot(np.arange(len(obj_err_list)), obj_err_list)
        plt.xlabel("Episode", fontsize=10)
        plt.ylabel("Sum rate error", fontsize=10)

        plt.subplot(1, 4, 4)
        plt.plot(np.arange(len(stop_step_list)), stop_step_list)
        plt.xlabel("Episode", fontsize=10)
        plt.ylabel("Iteration steps", fontsize=10)

        plt.savefig("res_v1.pdf")
        plt.show()

        all_rewardList.append(rewardList)
        all_err_list.append(err_list)
        all_stop_step_list.append(stop_step_list)

    # data_pd = pd.DataFrame(all_rewardList)
    # data_pd.to_csv('res_reward_v1.csv')
    arr_rewardList = np.array(all_rewardList)
    arr_err_list = np.array(all_err_list)
    arr_stop_step_list = np.array(all_stop_step_list)


    with open('res/res_reward_v3_startupdat10.csv', 'w', newline='') as file:
        for i in range(MAX_EPISODE):
            mywriter = csv.writer(file, delimiter=',')
            a = np.array(arr_rewardList[:,i])
            mywriter.writerow(a)
    with open('res/res_err_v3_startupdat10.csv', 'w', newline='') as file:
        for i in range(MAX_EPISODE):
            mywriter = csv.writer(file, delimiter=',')
            a = np.array(arr_err_list[:, i])
            mywriter.writerow(a)
    with open('res/res_step_v3_startupdat10.csv', 'w', newline='') as file:
        for i in range(MAX_EPISODE):
            mywriter = csv.writer(file, delimiter=',')
            a = np.array(arr_stop_step_list[:, i])
            mywriter.writerow(a)
