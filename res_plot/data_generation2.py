import numpy as np
import csv
import random
import math


np.random.seed(42)
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
    til_gamma = iteration_for_subproblem(B, b)
    gamma = np.exp(til_gamma)

    obj_updated = w.T.dot(np.log(1 + gamma))[0]
    obj_star = w.T.dot(np.log(1 + gamma_star))[0]
    reward = - (np.abs(obj_updated-obj_star))**2
    # reward = - np.linalg.norm(gamma - gamma_star, 1)
    # print("a:", a)
    # print("gamma:", gamma)
    # print("gamma_star:", gamma_star)
    # print("reward:", reward)
    o2 = np.append(o[:-3], gamma)
    d = False
    if reward >= -1e-3:
        d = True
    return o2, reward, d


def iteration_for_subproblem_v0(B, b):
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

def iteration_for_subproblem(B, b):
    z = np.random.rand(len(b))
    tol = 10e-9
    err = 1
    while err>tol:
        z_temp = z
        z = b/(B.T.dot(b/(B.dot(z))))
        err = np.linalg.norm(z_temp-z,1)

    res = B.dot(z)
    til_gamma = np.log(z/res)
    return til_gamma

def bcd_for_wsrm(G, w, sigma, p_bar, y_init,obj_star):
    F = np.dot(np.linalg.inv(np.diag(np.diag(G))), (G - np.diag(np.diag(G))))
    v = np.dot(np.linalg.inv(np.diag(np.diag(G))), sigma)
    B = F + 1 / p_bar * np.dot(v, np.ones((1, 3)))
    y = y_init
    err = 1
    tol = 1e-3
    obj_before = 0
    step = 0
    sumrate_acc_list = []
    p = np.random.rand(3,1)
    p=p*p_bar/np.sum(p)
    gamma = (1 / (np.dot(F, p) + v)) * p
    obj_updated = w.T.dot(np.log(1 + gamma))[0]

    sumrate_acc = abs(obj_updated - obj_star) / obj_star
    sumrate_acc_list.append(sumrate_acc[0])

    while step<5000:
        # print("step:", step)
        b = w[:, 0] * y
        til_gamma = iteration_for_subproblem(B, b)
        gamma = np.exp(til_gamma)
        # gamma1.append(gamma[0])
        # gamma2.append(gamma[1])
        # gamma3.append(gamma[2])
        y = 1 / (1 /gamma + 1)

        obj_updated = w.T.dot(np.log(1 + gamma))[0]
        err = obj_updated - obj_before
        obj_before = obj_updated

        obj_updated = w.T.dot(np.log(1 + gamma))[0]

        sumrate_acc = abs(obj_updated-obj_star)/obj_star
        sumrate_acc_list.append(sumrate_acc[0])

        step += 1
    # print("gamma1:", gamma1)
    # print("gamma2:", gamma2)
    # print("gamma3:", gamma3)
    return sumrate_acc_list

def brutal_search(G,w,p_bar,sigma):
    diff = 0.01
    max_obj = 10e-8
    F = np.dot(np.linalg.inv(np.diag(np.diag(G))), (G - np.diag(np.diag(G))))
    v = np.dot(np.linalg.inv(np.diag(np.diag(G))), sigma)
    for p_0 in np.arange(0, p_bar[0][0], diff):
        for p_1 in np.arange(0, p_bar[0][0]-p_0, diff):
            for p_2 in np.arange(0, p_bar[0][0]-p_0-p_1, diff):
                p = np.array([[p_0],[p_1],[p_2]])

                sinr = (1 / (np.dot(F, p) + v)) * p # 2*1,m=1
                f_func = np.log(1 + sinr)
                obj = w.T.dot(f_func)[0][0]
                # print("obj:", obj)

                if obj>=max_obj:
                    max_obj = obj
                    # print("max_obj:", max_obj)
                    # print("pinter:", p)
                    p_star = p
    # print("p_star:", p_star)
    sinr_star = (1 / (np.dot(F, p_star) + v)) * p_star
    # gamma_star = np.log(sinr_star)
    return sinr_star


if __name__ == '__main__':
    num = 500
    sigma = np.array([[0.05, 0.05, 0.05]]).T

    with open('/Users/siyac/Documents/Local_code/DRL-based-BCD-for-SR/DRL_BCD/downlink/bcd_iter.csv', 'a', newline='') as file:
        mywriter = csv.writer(file, delimiter=',')

        # 循环 num 次
        for i in range(num):
            print("i:", i)

            # 生成 G_std 矩阵
            G_std = np.round(np.random.rand(3, 3) + np.diag(np.random.rand(3)) * 10, 2)
            print("G_std:", G_std)

            # 计算 G
            G = G_std / 0.05

            # 生成 p_bar
            p_bar = np.round(np.random.rand(1, 1) * 3, 2)
            print("p_bar:", p_bar)

            # 如果 p_bar 小于等于 0.02，跳过本次循环
            if p_bar[0][0] <= 0.02:
                continue

            # 生成 w 向量
            w = np.round(np.random.rand(3, 1), 2)

            # 初始化 y_init
            y_init = np.random.rand(3) * 10

            # 使用 brute force 搜索计算 gamma_star
            gamma_star = brutal_search(G, w, p_bar, sigma)
            obj_star = w.T.dot(np.log(1 + gamma_star))[0]

            # 调用 bcd_for_wsrm 函数获得 sumrate_acc_list
            sumrate_acc_list = bcd_for_wsrm(G, w, sigma, p_bar, y_init, obj_star)

            # 如果 sumrate_acc_list 是一维数组，将其作为一行写入 CSV
            mywriter.writerow(sumrate_acc_list)
