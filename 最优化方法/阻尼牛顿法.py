import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm, solve


def func(x):
    f = 100 * (x[0, 0] ** 2 - x[1, 0]) ** 2 + (x[0, 0] - 1) ** 2
    return f


def gfunc(x):
    arr = np.array([400*x[0, 0]*(x[0, 0]**2-x[1, 0])+2*(x[0, 0]-1), -200*(x[0, 0]**2-x[1, 0])])
    g = np.transpose(arr)
    g.shape = (2, 1)
    return g


def Hess(x):
    n = len(x)
    He = np.zeros((n, n))
    He = np.array([[1200 * x[0, 0] ** 2 - 400 * x[1, 0] + 2, -400 * x[0, 0]], [-400 * x[0, 0], 200]])
    return He


def dampnm(x0):
    maxk = 100
    rho = 0.55
    sigma = 0.4
    k = 0
    Err = []
    tk = []
    epsilon = 1e-5
    while k < maxk:
        gk = gfunc(x0)
        Gk = Hess(x0)
        dk = solve(-Gk, gk)
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


x0 = np.transpose(np.array([20, 20]))
x0.shape = (2, 1)
x, val, k, Err, tk = dampnm(x0)
print(x, val, k)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'cm'
plt.yscale('log')
plt.xlabel('迭代次数 k')
plt.ylabel('目标函数值')
plt.plot(tk, Err, 'b-.')
plt.show()