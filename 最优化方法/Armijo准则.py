import numpy as np


def func(x):
    f = 100 * (x[0, 0] ** 2 - x[1, 0]) ** 2 + (x[0, 0] - 1) ** 2
    return f


def gfunc(x):
    gf = np.asmatrix([400*x[0, 0]*(x[0, 0]**2-x[1, 0])+2*(x[0, 0]-1), -200*(x[0, 0]**2-x[1, 0])]).T
    return gf


def armijo(xk, dk):
    beta = 0.5
    sigma = 0.2
    m = 0
    mk = 0
    mmax = 20
    while m <= mmax:
        if func(xk + beta ** m * dk) <= (func(xk) + sigma * beta ** m * gfunc(xk).T * dk):
            mk = m
            break
        m = m + 1
    alpha = beta ** mk
    newxk = xk + alpha*dk
    fk = func(xk)
    newfk = func(newxk)
    return mk, alpha, newxk, fk, newfk


xk = np.asmatrix([-1, 1]).T
dk = np.asmatrix([1, -2]).T
result = armijo(xk, dk)
print(result)
