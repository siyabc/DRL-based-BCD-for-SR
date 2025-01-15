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
    til_gamma = iteration_3u_v2(B, b)
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


def sca(G,w, sigma,p_bar,  y_init):
    F = np.dot(np.linalg.inv(np.diag(np.diag(G))), (G - np.diag(np.diag(G))))
    v = np.dot(np.linalg.inv(np.diag(np.diag(G))), sigma)
    B = F + 1 / p_bar * np.dot(v, np.ones((1, 3)))
    y = y_init
    err = 1
    tol = 1e-3
    obj_before = 0
    step = 0
    gamma1 = [y_init[0]]
    gamma2 = [y_init[1]]
    gamma3 = [y_init[2]]

    # for i in range(13):
    while err > tol:
        # print("step:", step)
        b = w[:, 0] * y
        til_gamma = iteration_3u_v2(B, b)
        gamma = np.exp(til_gamma)
        gamma1.append(gamma[0])
        gamma2.append(gamma[1])
        gamma3.append(gamma[2])
        y = 1 / (1 /gamma + 1)

        obj_updated = w.T.dot(np.log(1 + gamma))[0]
        err = obj_updated - obj_before
        obj_before = obj_updated
        step += 1
    # print("gamma1:", gamma1)
    # print("gamma2:", gamma2)
    print("gamma3:", gamma3)
    return step, gamma


def all_data_sca_check():
    sr_data = list()
    sr_target = list()
    all_step = []
    all_gamma_acc = []
    all_sum_rate_acc = []
    with open("../../../DRL_BCD/downlink/_ddpg_bcd_downlink/data1.csv", "r") as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            sr_data.append(list(map(float, line[:-3])))
            sr_target.append(list(map(float, line[-3:])))
    for i in range(len(sr_data)):
        print("i:", i)
        o = np.array(sr_data[i])
        label = np.array(sr_target[i])
        G = o[0:9].reshape(3, 3)
        w = o[9:12].reshape(3, 1)
        sigma = o[12:15].reshape(3, 1)
        p_bar = o[15]
        gamma_star = label
        # y_init = np.array([0.5,0.5,0.5])
        y_init = np.random.rand(3) * 10
        step, gamma = sca(G, w, sigma, p_bar, y_init)
        reward = np.linalg.norm(gamma - label, 1)
        all_step.append(step)
        obj_updated = w.T.dot(np.log(1 + gamma))[0]
        obj_star = w.T.dot(np.log(1 + gamma_star))[0]
        sumrate_acc = min(obj_updated / obj_star, 1)
        all_sum_rate_acc.append(sumrate_acc)

        gamma_acc = (gamma-label)/(label+0.01)
        for gc in gamma_acc:
            if gc >=0:
                acc = 1-min(gc,1)
                all_gamma_acc.append(acc)
            else:
                acc = 1+max(gc,-1)
                all_gamma_acc.append(acc)

        # print("all_gamma_acc:", all_gamma_acc)

        # if reward > 10:
            # print("y_init:", y_init)
            # print("step:", step)
            # print("label:", label)
            # print("gamma:", gamma)
            # print("===============")
    print(max(all_step))
    print(min(all_step))
    print(sum(all_step)/len(all_step))

    print(max(all_sum_rate_acc))
    print(min(all_sum_rate_acc))
    print(sum(all_sum_rate_acc) / len(all_sum_rate_acc))

    print(max(all_gamma_acc))
    print(min(all_gamma_acc))
    print(sum(all_gamma_acc) / len(all_gamma_acc))



def one_data_check():
    # o = np.array([0.09,0.24,0.18,0.11,2.49,0.03,0.99,0.25,7.1,0.96,0.38,0.35,0.05,0.05,0.05,2.91,])
    o = np.array([1.18, 0.05, 0.59, 0.61, 9.67, 0.56, 0.89, 0.99, 3.29, 0.66, 0.45, 0.46, 0.05, 0.05, 0.05, 2.79])
    # label = np.array([0.0,21.453712190650787,49.07719298245613])
    label = np.array([27.804597701149422, 5.502345251826221, 0.0])

    G = o[0:9].reshape(3, 3)
    w = o[9:12].reshape(3, 1)
    sigma = o[12:15].reshape(3, 1)
    p_bar = o[15]
    gamma_star = label
    # y_init = np.array([0.5,0.5,0.5])
    y_init = np.random.rand(3) * 10
    step, gamma = sca(G, w, sigma, p_bar, y_init)
    obj_updated = w.T.dot(np.log(1 + gamma))[0]
    obj_star = w.T.dot(np.log(1 + gamma_star))[0]
    sumrate_acc = obj_updated/obj_star
    print("y_init:", y_init)
    print("step:", step)
    print("label:", label)
    print("gamma:", gamma)
    print("sumrate_acc:", sumrate_acc)


if __name__ == '__main__':
    #3, 122
    # all_data_sca_check()
    one_data_check()
#1,0.9243459280130922,1,1,1,0.7841987150658767,0.9244365828125123,0.9244567961531558,0.7841987150658767

'''
i: 1
y_init: [5.69055847 9.4847671  7.98240829]
step: 13
label: [  0.  128.8   0. ]
gamma: [1.56849245e-01 1.16214024e+02 9.22016204e-05]

i: 14
y_init: [0.70512644 9.38234217 1.02262027]
step: 7
label: [27.8045977   5.50234525  0.        ]
gamma: [1.11863848e-02 5.27679069e+02 2.14263126e-08]


i: 21
y_init: [9.7081906  0.11602395 3.54142111]
step: 4
label: [51.06115108  1.81525424  0.        ]
gamma: [1.17225726e+02 2.75458462e-02 4.49306380e-04]

i: 39
y_init: [6.93678552 0.24933065 4.26878247]
step: 4
label: [  0.  256.5   0. ]
gamma: [1.72733399e+02 2.90922956e-02 3.31111886e-06]

i: 43
y_init: [7.16797255 3.94863682 1.83828856]
step: 9
label: [  0.     0.   145.08]
gamma: [2.40709078e+01 2.13611385e-15 1.08711363e-02]

i: 45
y_init: [3.6112073  6.21185313 7.90196858]
step: 17
label: [163.18   0.     0.  ]
gamma: [1.35846180e-07 5.04065369e-02 5.23656363e+01]

i: 53
y_init: [3.08195011 9.38970264 6.4704976 ]
step: 5
label: [  0.    0.  363.3]
gamma: [4.63679448e-03 3.26815712e+01 2.30597360e+00]

i: 59
y_init: [0.4473254  5.54105962 3.08679828]
step: 10
label: [174.52727273   0.38797814   0.        ]
gamma: [1.33930619e+02 8.06529469e-01 5.54269211e-16]

i: 95
y_init: [7.53981135 0.90513606 7.8199427 ]
step: 8
label: [ 0.         21.45371219 49.07719298]
gamma: [5.21995788e+00 4.43451195e-03 4.05239628e-05]

i: 100
y_init: [7.90707442 7.08508797 0.79779062]
step: 16
label: [0.00000000e+00 2.72068966e+02 2.03571429e-01]
gamma: [4.44759118e-04 2.47744373e+02 4.16728683e-01]

i: 121
y_init: [5.96201964 5.30516244 7.13637685]
step: 11
label: [51.6  0.   0. ]
gamma: [1.07087161e-04 4.43455154e-12 4.13237207e+01]

i: 122
y_init: [7.79908225 3.20393831 8.43492718]
step: 8
label: [  0.         221.72316384   1.1276891 ]
gamma: [3.30492176e-06 2.10643412e+02 1.29238661e+00]


i: 136
y_init: [0.50911845 5.13887435 8.64214935]
step: 12
label: [237.66   0.     0.  ]
gamma: [3.73912280e-04 2.23493628e+02 1.58982473e-07]
'''
