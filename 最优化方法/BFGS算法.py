import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm, solve
from 对称秩1算法 import sr1


def func(x):
    f = 100 * (x[0, 0] ** 2 - x[1, 0]) ** 2 + (x[0, 0] - 1) ** 2
    return f


def gfunc(x):
    arr = np.array([400*x[0, 0]*(x[0, 0]**2-x[1, 0])+2*(x[0, 0]-1), -200*(x[0, 0]**2-x[1, 0])])
    g = np.transpose(arr)
    g.shape = (2, 1)
    return g


def bfgs(x0):
    maxk = 500
    rho = 0.55
    sigma = 0.4
    epsilon = 1e-05
    k = 0
    Err = []
    tk = []
    n = len(x0)
    Bk = np.eye(n)
    while k < maxk:
        gk = gfunc(x0)
        if norm(gk) < epsilon:
            break
        dk = solve(-Bk, gk)
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
        if np.transpose(yk) @ sk > 0:
            Bk = (Bk - (Bk @ sk @ np.transpose(sk) @ Bk) / (np.transpose(sk) @ Bk @ sk)
                  + (yk @ np.transpose(yk)) / (np.transpose(yk) @ sk))
        k = k + 1
        x0 = x
    x = x0
    val = func(x)
    return x, val, k, Err, tk


x0 = np.transpose(np.array([10, 10]))
x0.shape = (2, 1)
x1, val1, k1, Err1, tk1 = bfgs(x0)
x2, val2, k2, Err2, tk2 = sr1(x0)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'cm'
plt.yscale('log')
plt.xlabel('迭代次数 k')
plt.ylabel('目标函数值')
plt.plot(tk1, Err1, 'b-', label="BFGS")
plt.plot(tk2, Err2, 'r--', label="SR1")
plt.legend()
plt.show()