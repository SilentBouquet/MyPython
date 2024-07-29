import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm, inv
from 对称秩1算法 import sr1
from BFGS算法 import bfgs
from DFP算法 import dfp


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


def broyden(x0):
    maxk = 1e4
    rho = 0.55
    sigma = 0.4
    epsilon = 1e-05
    phi = 0.5
    k = 0
    Err = []
    tk = []
    Hk = inv(Hess(x0))
    while k < maxk:
        gk = gfunc(x0)
        if norm(gk) < epsilon:
            break
        dk = -Hk @ gk
        m = 0
        mk = 0
        while m < 20:
            if func(x0 + rho ** m * dk) < func(x0) + sigma * rho ** m * np.transpose(gk) @ dk:
                mk = m
                break
            m = m + 1
        x = x0 + rho ** mk * dk
        val = func(x0)
        Err.append(val)
        tk.append(k)
        sk = x - x0
        yk = gfunc(x) - gk
        Hy = Hk @ yk
        sy = np.transpose(sk) @ yk
        yHy = np.transpose(yk) @ Hk @ yk
        if sy < 0.2 * yHy:
            theta = 0.8 * yHy / (yHy - sy)
            sk = theta * sk + (1 - theta) * Hy
            sy = 0.2 * yHy
        vk = np.sqrt(np.abs(yHy)) * (sk / sy - Hy / yHy)
        Hk = (Hk - (Hy @ np.transpose(Hy)) / yHy + (sk @ np.transpose(sk)) / sy + phi * vk @ np.transpose(vk))
        k = k + 1
        x0 = x
    x = x0
    val = func(x)
    return x, val, k, Err, tk


x0 = np.transpose(np.array([5, 5]))
x0.shape = (2, 1)
x1, val1, k1, Err1, tk1 = sr1(x0)
x2, val2, k2, Err2, tk2 = bfgs(x0)
x3, val3, k3, Err3, tk3 = dfp(x0)
x4, val4, k4, Err4, tk4 = broyden(x0)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'cm'
plt.yscale('log')
plt.xlabel('迭代次数 k')
plt.ylabel('目标函数值')
plt.plot(tk1, Err1, 'b-', label="SR1")
plt.plot(tk2, Err2, 'r--', label="BFGS")
plt.plot(tk3, Err3, 'y-^', label="DFP")
plt.plot(tk4, Err4, 'g-*', label="Broyden")
plt.legend()
plt.show()