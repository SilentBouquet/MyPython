import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm, solve
from 阻尼牛顿法 import dampnm


def func(x):
    f = 100 * (x[0, 0] ** 2 - x[1, 0]) ** 2 + (x[0, 0] - 1) ** 2
    return f


def gfunc(x):
    arr = np.array([400 * x[0, 0] * (x[0, 0] ** 2 - x[1, 0]) + 2 * (x[0, 0] - 1), -200 * (x[0, 0] ** 2 - x[1, 0])])
    g = np.transpose(arr)
    g.shape = (2, 1)
    return g


def Hess(x):
    n = len(x)
    He = np.zeros((n, n))
    He = np.array([[1200 * x[0, 0] ** 2 - 400 * x[1, 0] + 2, -400 * x[0, 0]], [-400 * x[0, 0], 200]])
    return He


def revisenm(x0):
    n = len(x0)
    maxk = 150
    rho = 0.55
    sigma = 0.4
    tau = 0.0
    k = 0
    Err = []
    tk = []
    epsilon = 1e-05
    while k < maxk:
        gk = gfunc(x0)
        muk = norm(gk) ** (1 + tau)
        Gk = Hess(x0)
        Ak = Gk + muk * np.eye(n)
        dk = solve(-Ak, gk)
        if norm(gk) < epsilon:
            break
        m = 0
        mk = 0
        while m < 20:
            if func(x0 + rho ** m * dk) < func(x0) + sigma * rho ** m * np.transpose(gk) @ dk:
                mk = m
                break
            m = m + 1
        x0 = x0 + rho ** mk * dk
        err = func(x0)
        Err.append(err)
        tk.append(k)
        k += 1
    x = x0
    val = func(x)
    return x, val, k, Err, tk


x0 = np.transpose(np.array([0, 0]))
x0.shape = (2, 1)
x1, val1, k1, Err1, tk1 = dampnm(x0)
x2, val2, k2, Err2, tk2 = revisenm(x0)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'cm'
plt.yscale('log')
plt.xlabel('迭代次数 k')
plt.ylabel('目标函数值')
plt.plot(tk1, Err1, 'r--^', label="阻尼牛顿法")
plt.plot(tk2, Err2, 'b-*', label="修正牛顿法")
plt.legend()
plt.show()