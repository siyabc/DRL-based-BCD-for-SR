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
    res = B.dot(z)
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

def bcd_for_wsrm(G,w, sigma,p_bar,  y_init):
    F = np.dot(np.linalg.inv(np.diag(np.diag(G))), (G - np.diag(np.diag(G))))
    v = np.dot(np.linalg.inv(np.diag(np.diag(G))), sigma)
    B = F + 1 / p_bar * np.dot(v, np.ones((1, 3)))
    y = y_init
    err = 1
    tol = 1e-3
    obj_before = 0
    step = 0
    # gamma1 = [y_init[0]]
    # gamma2 = [y_init[1]]
    # gamma3 = [y_init[2]]

    # for i in range(13):
    while err > tol:
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
        step += 1
    # print("gamma1:", gamma1)
    # print("gamma2:", gamma2)
    # print("gamma3:", gamma3)
    return step, gamma


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
    y_init = np.random.rand(3) * 1
    step, gamma = bcd_for_wsrm(G, w, sigma, p_bar, y_init)
    obj_updated = w.T.dot(np.log(1 + gamma))[0]
    obj_star = w.T.dot(np.log(1 + gamma_star))[0]
    sumrate_acc = obj_updated/obj_star
    print("y_init:", y_init)
    print("step:", step)
    print("label:", label)
    print("gamma:", gamma)
    print("obj_star:", obj_star)

    print("sumrate_acc:", sumrate_acc)



if __name__ == '__main__':
    one_data_check()

