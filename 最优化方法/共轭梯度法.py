import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm


def func(x):
    f = 100 * (x[0, 0] ** 2 - x[1, 0]) ** 2 + (x[0, 0] - 1) ** 2
    return f


def gfunc(x):
    arr = np.array([400*x[0, 0]*(x[0, 0]**2-x[1, 0])+2*(x[0, 0]-1), -200*(x[0, 0]**2-x[1, 0])])
    g = np.transpose(arr)
    g.shape = (2, 1)
    return g


def frcg(x0):
    maxk = 5000
    rho = 0.6
    sigma = 0.4
    k = 0
    Err = []
    tk = []
    epsilon = 0.0001
    n = len(x0)
    g0 = gfunc(x0)
    d0 = 0
    while k < maxk:
        g = gfunc(x0)
        itern = k - (n + 1) * int(np.floor(k / (n + 1)))
        itern = itern + 1
        if itern == 1:
            d = -g
        else:
            beta = (np.transpose(g) @ g) / (np.transpose(g0) @ g0)
            d = -g + beta * d0
            gd = np.transpose(g) @ d
            if gd >= 0.0:
                d = -g
        if norm(g) < epsilon:
            break
        m = 0
        mk = 0
        while m < 20:
            if func(x0 + rho ** m * d) < func(x0) + sigma * rho ** m * np.transpose(g) @ d:
                mk = m
                break
            m = m + 1
        x0 = x0 + rho ** mk * d
        val = func(x0)
        Err.append(val)
        tk.append(k)
        g0 = g
        d0 = d
        k = k + 1
    x = x0
    val = func(x)
    return x, val, k, Err, tk


x0 = np.transpose(np.array([0, 0]))
x0.shape = (2, 1)
x, val, k, Err, tk = frcg(x0)
print(x, val, k)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'cm'
plt.yscale('log')
plt.xlabel('迭代次数 k')
plt.ylabel('目标函数值')
plt.plot(tk, Err, 'b-.')
plt.show()