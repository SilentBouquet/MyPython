import numpy as np
from numpy.linalg import norm, solve


def f1(x):
    f = (x[0, 0] - 2.0) ** 2 + (x[1, 0] - 1.0) ** 2
    return np.mat([f])


def h1(x):
    he = x[0, 0] - 2.0 * x[1, 0] + 1.0
    return np.mat([he])


def g1(x):
    g = -0.25 * x[0, 0] ** 2 - x[1, 0] ** 2 + 1
    return np.mat([g])


def df1(x):
    g = np.mat([2.0 * (x[0, 0] - 2.0), 2.0 * (x[1, 0] - 1.0)]).T
    return g


def dh1():
    dhe = np.mat([1.0, -2.0]).T
    return dhe


def dg1(x):
    dgi = np.mat([-0.5 * x[0, 0], -2.0 * x[1, 0]]).T
    return dgi


def func(x, mu, lambda_, sigma):
    f = (f1(x) - mu * h1(x) + sigma/2 * (h1(x)) ** 2 +
         1 / (2 * sigma)) * ((min(0, sigma * g1(x) - lambda_)) ** 2 - lambda_ ** 2)
    return f


def gfunc(x, mu, lambda_, sigma):
    g = df1(x) - dh1() * mu + dh1() * h1(x) * sigma + dg1(x) * min((sigma * g1(x) - lambda_), 0)
    g = np.transpose(g)
    g.shape = (2, 1)
    return g


def bfgs(x0, mu, lambda_, sigma):
    maxk = 500
    rho = 0.55
    sigma2 = 0.4
    epsilon = 1e-05
    k = 0
    n = len(x0)
    Bk = np.eye(n)
    while k < maxk:
        gk = gfunc(x0, mu, lambda_, sigma)
        if norm(gk) < epsilon:
            break
        dk = solve(-Bk, gk)
        m = 0
        mk = 0
        while m < 20:
            if (func(x0 + rho ** m * dk, mu, lambda_, sigma) < func(x0, mu, lambda_, sigma)
                    + sigma2 * rho ** m * np.transpose(gk) @ dk):
                mk = m
                break
            m = m + 1
        x = x0 + rho ** mk * dk
        sk = x - x0
        yk = gfunc(x, mu, lambda_, sigma) - gk
        if np.transpose(yk) @ sk > 0:
            Bk = (Bk - (Bk @ sk @ np.transpose(sk) @ Bk) / (np.transpose(sk) @ Bk @ sk)
                  + (yk @ np.transpose(yk)) / (np.transpose(yk) @ sk))
        k = k + 1
        x0 = x
    x = x0
    val = func(x, mu, lambda_, sigma)
    return x, val


def multphr(x0):
    maxk = 500
    sigma = 2.0
    eta = 2.0
    theta = 0.8
    k = 0
    epsilon = 1e-05
    x = x0
    he = h1(x)
    gi = g1(x)
    l1 = len(he)
    m = len(gi)
    mu = 0.1 * np.mat(np.ones((l1, 1)))
    lambda_ = 0.1 * np.mat(np.ones((m, 1)))
    btak = 10
    btaold = 10
    while btak > epsilon and k < maxk:
        x, v = bfgs(x0, mu, lambda_, sigma)
        he = h1(x)
        gi = g1(x)
        btak = 0.0
        for i in range(l1):
            btak = btak + he[i] ** 2
        for i2 in range(m):
            temp = min(gi[i2], lambda_[i2]/sigma)
            btak = btak + temp ** 2
        btak = np.sqrt(btak)
        if btak > epsilon:
            if k >= 2 and btak > theta * btaold:
                sigma = eta * sigma
            for i3 in range(l1):
                mu[i3] = mu[i3] - sigma * he[i3]
            for i4 in range(m):
                lambda_[i4] = max(0.0, lambda_[i4] - sigma * gi[i4])
        k = k + 1
        btaold = btak
    f = f1(x)
    return x, mu, lambda_, f, k, btak


x0 = np.mat([3, 3]).T
x, mu, lambda_, f, k, btak = multphr(x0)
print(x, mu, lambda_, f, k, btak)