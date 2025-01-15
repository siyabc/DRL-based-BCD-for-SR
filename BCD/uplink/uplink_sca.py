import numpy as np


def iter_from_lambda_to_z(B, _lambda):
    tol = 10e-7
    err = 1
    z = np.random.rand(2, 1)

    while err > tol:
        z_temp = z
        a = (np.dot(B, z))
        z = _lambda / (np.dot(B.T, _lambda / (np.dot(B, z))))
        err = np.linalg.norm(z_temp - z)
    return z

def norm_lambda_to_w(lambda_1,lambda_2, w):
    lambda_1_t = lambda_1
    lambda_2_t = lambda_2
    lambda_1 = lambda_1_t / (lambda_1_t + lambda_2_t) * w
    lambda_2 = lambda_2_t / (lambda_1_t + lambda_2_t) * w
    return lambda_1,lambda_2


G = np.array([[2.0,0.5],[0.3,2.4]])
sigma = np.array([[0.05],[0.06]])
p_bar = np.array([[2.0],[2.5]])
e1 = np.array([[1],[0]])
# e1 = np.ones((2,1))
e2 = np.array([[0],[1]])
w = np.array([[0.2],[0.3]])
# w = np.random.rand(2, 1)

F = np.dot(np.linalg.inv(np.diag(np.diag(G))), (G - np.diag(np.diag(G))))
v = np.dot(np.linalg.inv(np.diag(np.diag(G))), sigma)
B1 = F + 1 / p_bar[0][0] * np.dot(v, e1.T)
B2 = F + 1 / p_bar[1][0] * np.dot(v, e2.T)
# print(B2)

# lambda_1 = np.array([[1],[2]])
# lambda_2 = np.array([[1],[2]])
lambda_1 = np.random.rand(2, 1)
lambda_2 = np.random.rand(2, 1)

lambda_1,lambda_2 = norm_lambda_to_w(lambda_1,lambda_2, w)
# lambda_1 = lambda_1/sum(lambda_1)
# lambda_2 = lambda_2/sum(lambda_2)

z_1 = iter_from_lambda_to_z(B1, lambda_1)
z_2 = iter_from_lambda_to_z(B2, lambda_2)
print(z_1)
print(z_2)




