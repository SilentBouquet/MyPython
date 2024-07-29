import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import norm, solve


def FK(x):
    y1 = x[0, 0] - 0.7 * np.sin(x[0, 0]) - 0.2 * np.cos(x[1, 0])
    y2 = x[1, 0] - 0.7 * np.cos(x[0, 0]) + 0.2 * np.sin(x[1, 0])
    arr = np.array([y1, y2])
    fk = np.transpose(arr)
    fk.shape = (2, 1)
    return fk


def JFK(x):
    JF = np.array([[1 - 0.7 * np.cos(x[0, 0]), 0.2 * np.sin(x[1, 0])],
                   [0.7 * np.sin(x[0, 0]), 1 + 0.2 * np.cos(x[1, 0])]])
    return JF


def lmm(x0):
    maxk = 100
    rho = 0.55
    sigma = 0.4
    muk = norm(FK(x0))
    k = 0
    epsilon = 1e-06
    n = len(x0)
    Err = []
    tk = []
    Err.append(0.5 * norm(FK(x0)) ** 2)
    tk.append(k)
    while k < maxk:
        fk = FK(x0)
        jfk = JFK(x0)
        gk = jfk.T @ fk
        dk = solve(-(jfk.T @ jfk + muk * np.eye(n)), gk)
        if norm(gk) < epsilon:
            break
        m = 0
        mk = 0
        while m < 20:
            newf = 0.5 * norm(FK(x0 + rho ** m * dk)) ** 2
            oldf = 0.5 * norm(FK(x0)) ** 2 + sigma * rho ** m * gk.T @ dk
            if newf < oldf:
                mk = m
                break
            m = m + 1
        x0 = x0 + rho ** mk * dk
        muk = norm(FK(x0))
        k += 1
        Err.append(0.5 * muk ** 2)
        tk.append(k)
    x = x0
    val = 0.5 * muk ** 2
    return x, val, k, Err, tk


x0 = np.transpose(np.array([-5, -5]))
x0.shape = (2, 1)
x, val, k, Err, tk = lmm(x0)
print(x, val)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'cm'
plt.xlabel('迭代次数 k')
plt.ylabel('目标函数值')
plt.plot(tk, Err, 'b-')
plt.show()