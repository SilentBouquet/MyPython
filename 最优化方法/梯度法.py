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


def grad(x0):
    maxk = 5000
    rho = 0.5
    sigma = 0.4
    k = 0
    epsilon = 1e-05
    Err = []
    tk = []
    while k < maxk:
        g = gfunc(x0)
        d = -g
        if norm(d) < epsilon:
            break
        m = 0
        mk = 0
        while m < 20:
            if func(x0 + rho ** m * d) < func(x0) + sigma * rho ** m * np.transpose(g) @ d:
                mk = m
                break
            m = m + 1
        x0 = x0 + rho ** mk * d
        err = func(x0)
        Err.append(err)
        tk.append(k)
        k = k + 1
    x = x0
    val = func(x0)
    return x, val, k, Err, tk


x0 = np.transpose(np.array([0, 0]))
x0.shape = (2, 1)
x, val, k, Err, tk = grad(x0)
print(x, val, k)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'cm'
plt.yscale('log')
plt.xlabel('迭代次数 k')
plt.ylabel('目标函数值')
plt.plot(tk, Err, 'b-.')
plt.show()