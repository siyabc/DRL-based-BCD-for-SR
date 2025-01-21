# 对于多个样例来训练
# reward function 选为 sum rate
# with all features, including previous features (G,v,w...) and new features in rewardv3_v2
# for many instances with each as one eipisode

# sr_reward_v3 for single-instances on sr_iterations.

from td3_sr_rewardv3_v3 import TD3
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

def env_o_init(sr_data_single):
    G = sr_data_single[0:9].reshape(3, 3)
    w = sr_data_single[9:12].reshape(3, 1)
    sigma = sr_data_single[12:15].reshape(3, 1)
    p_bar = sr_data_single[15]
    F = np.dot(np.linalg.inv(np.diag(np.diag(G))), (G - np.diag(np.diag(G))))
    v = np.dot(np.linalg.inv(np.diag(np.diag(G))), sigma)
    # B = F + 1 / p_bar * np.dot(v, np.ones((1, 3)))
    notil_gamma = np.random.rand(3) * 1

    p = np.linalg.inv(np.eye(len(notil_gamma)) - np.diag(notil_gamma) @ F) @ np.diag(notil_gamma) @ v
    Fpv = F.dot(p) + v
    res = np.log(1 + notil_gamma)
    rate = np.diag(w[:, 0]) @ np.log(1 + notil_gamma)

    # o2 = np.append(o[:-3], notil_gamma)
    o_init = np.hstack((sr_data_single, p.reshape(3), Fpv.reshape(3), rate.reshape(3), notil_gamma))
    return o_init

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
    notil_gamma = np.exp(til_gamma)

    p = np.linalg.inv(np.eye(len(notil_gamma)) - np.diag(notil_gamma) @ F) @ np.diag(notil_gamma) @ v
    Fpv = F.dot(p) + v
    res = np.log(1 + notil_gamma)
    rate = np.diag(w[:, 0])@np.log(1 + notil_gamma)

    # o2 = np.append(o[:-3], notil_gamma)
    o2 = np.hstack((o[:16],p.reshape(3), Fpv.reshape(3), rate.reshape(3), notil_gamma))


    obj_updated = w.T.dot(np.log(1 + notil_gamma))[0]
    obj_star = w.T.dot(np.log(1 + gamma_star))[0]
    # reward = - (np.abs(obj_updated-obj_star))**2

    obj_err = abs(np.abs(obj_updated - obj_star)) / obj_star
    # reward = - np.linalg.norm(notil_gamma - gamma_star, 1)
    reward = obj_updated
    old_gamma = o[-3:]
    obj_old = w.T.dot(np.log(1 + old_gamma))[0]

    delta_reward = np.abs(obj_updated - obj_old)

    reward_acc = obj_updated / obj_star

    # print("===delta_reward:", delta_reward)

    # o2 = np.append(o[:-3], notil_gamma)
    d = False
    if delta_reward <= 1e-6:
        d = True
    return o2, reward, d, obj_err, reward_acc

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


def main(epoch_num,MAX_EPISODE=100,MAX_STEP=5000,update_every=50,batch_size=10,start_update=40,
         replay_size=int(1e5), gamma=0.9, pi_lr=1e-6,q_lr=1e-6, policy_delay=5):
    sr_data = list()
    sr_target = list()
    with open("res/data1.csv", "r") as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            sr_data.append(list(map(float, line[:-3])))
            sr_target.append(list(map(float, line[-3:])))

    obs_dim = 28
    nn_dim = 28
    act_dim = 3
    td3 = TD3(obs_dim, act_dim,nn_dim,replay_size=replay_size, gamma=gamma, pi_lr=pi_lr, q_lr=q_lr,
                  act_noise=0.05, target_noise=0.05, noise_clip=0.5, policy_delay=policy_delay)
    # epoch_num = 1

    for epoch in range(epoch_num):
        print("epoch:", epoch)
        filename = f"res/rewardv3_v2_ME{MAX_EPISODE}_SU{start_update}_Ga{gamma}_UE{update_every}_BS{batch_size}_lr{pi_lr}_MS{MAX_STEP}_Epoch{epoch}" + '.pdf'

        rewardList = []
        err_list = []
        stop_step_list = []
        outlier_list = []


        for episode in range(MAX_EPISODE):
            j_converge_list = [MAX_STEP]
            # o = env.reset()
            # o_init = np.append(np.array(sr_data[epoch]), np.random.rand(3) * 1)
            o_init = env_o_init(np.array(sr_data[episode]))
            # sr_data_a = sr_data[epoch]
            # print("o_init:",o_init)
            label = np.array(sr_target[episode])
            o = o_init
            ep_reward = 0
            # stop_step = 0
            obj_err_list = []
            for j in range(MAX_STEP):
                # if episode > 20:
                #     a = td3.get_action(o, td3.act_noise) * 2
                # else:
                #     a = env.action_space.sample()
                # a = td3.get_action(o, td3.act_noise) * 2

                a = td3.get_action(o[:], 0.001)
                # ======================
                # next state
                # o2, r, d, obj_err = step(o, a, label)
                # o2, reward, d, obj_err, reward_acc= step(o, a, label)
                o2, r, d, obj_err, reward_acc = step(o, a, label)
                # ======================
                td3.replay_buffer.store(o, a, r, o2, d)
                if episode >= start_update and j % update_every == 0:
                    td3.update(batch_size, update_every)

                o = o2
                ep_reward += reward_acc
                obj_err_list.append(obj_err)
                # stop_step = j
                if d:
                    j_converge_list.append(j)
            notil_gamma = o[-3:]
            err = np.linalg.norm(notil_gamma - label)
            converge_step = min(j_converge_list)

            print('Episode:', episode, 'notil_gamma:', notil_gamma, 'label:', label, '==========', 'Reward:', ep_reward, 'err:',
                  err, '---', 'obj_err:', obj_err, 'j:', converge_step)
            if math.isnan(ep_reward):
                print("a:", a)


            if episode > 50 and converge_step >= 1000:
                outlier_list.append(episode)

            if episode > 70:
                with open('res/instance_sr_iterations.csv', 'a', newline='') as file:
                    mywriter = csv.writer(file, delimiter=',')  # 移到循环外部
                    mywriter.writerow(obj_err_list)


if __name__ == '__main__':
    epoch_num = 1

    # MAX_EPISODE_list = [150]  # 这里指的是第几个样本
    # MAX_STEP_list = [5000]
    # update_every_list = [100]  # 100
    # batch_size_list = [50]
    # start_update_list = [40]  # 10

    MAX_EPISODE_list = [150]  # 这里指的是第几个样本
    MAX_STEP_list = [10000]
    update_every_list = [200]  # 100
    batch_size_list = [25]
    start_update_list = [40]  # 10

    lr_list = [1e-6]
    gamma_list = [0.99]

    for MAX_EPISODE in MAX_EPISODE_list:
        for start_update in start_update_list:
            for gamma in gamma_list:
                for update_every in update_every_list:
                    for batch_size in batch_size_list:
                        for lr in lr_list:
                            for MAX_STEP in MAX_STEP_list:
                                main(epoch_num=epoch_num,MAX_EPISODE=MAX_EPISODE, MAX_STEP=MAX_STEP,
                                     update_every=update_every, batch_size=batch_size,
                                     start_update=start_update, replay_size=int(1e5), gamma=gamma, pi_lr=lr, q_lr=lr,
                                     policy_delay=2)
