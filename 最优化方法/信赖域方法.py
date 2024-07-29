import numpy as np
import matplotlib.pyplot as plt


def func(x):
    f = 100 * (x[0, 0] ** 2 - x[1, 0]) ** 2 + (x[0, 0] - 1) ** 2
    return f


def gfunc(x):
    arr = np.array([400 * x[0, 0] * (x[0, 0] ** 2 - x[1, 0]) + 2 * (x[0, 0] - 1), -200 * (x[0, 0] ** 2 - x[1, 0])])
    g = np.transpose(arr)
    g.shape = (2, 1)
    return g


def Hess(x):
    He = np.array([[1200 * x[0, 0] ** 2 - 400 * x[1, 0] + 2, -400 * x[0, 0]], [-400 * x[0, 0], 200]])
    return He


def mk_function(x, p):
    p = np.array(p)
    fk = func(x)
    gk = gfunc(x)
    Bk = Hess(x)
    mk = fk + np.dot(gk.T, p) + 0.5 * np.dot(np.dot(p.T, Bk), p)
    return mk


def Dogleg_Method(x, delta):
    g = gfunc(x)
    B = Hess(x)
    g = g.astype(float)
    B = B.astype(float)
    inv_B = np.linalg.inv(B)
    PB = np.dot(-inv_B, g)
    PU = -(np.dot(g.T, g) / (np.dot(g.T, B).dot(g))) * g
    PB_U = PB - PU
    PB_norm = np.linalg.norm(PB)
    PU_norm = np.linalg.norm(PU)
    PB_U_norm = np.linalg.norm(PB_U)
    if PB_norm <= delta:
        tao = 2
    elif PU_norm >= delta:
        tao = delta / PU_norm
    else:
        factor = np.dot(PU.T, PB_U) * np.dot(PU.T, PB_U)
        tao = -2 * np.dot(PU.T, PB_U) + 2 * np.sqrt(
            factor - PB_U_norm * PB_U_norm * (PU_norm * PU_norm - delta * delta))
        tao = tao / (2 * PB_U_norm * PB_U_norm) + 1
    s_k = 0
    if 0 <= tao <= 1:
        s_k = tao * PU
    elif 1 < tao <= 2:
        s_k = PU + (tao - 1) * (PB - PU)
    return s_k


def TrustRegion(x, delta_max):
    delta = delta_max
    k = 0
    epsilon = 1e-9
    maxk = 1000
    Err = []
    tk = []
    Err.append(func(x))
    tk.append(k)
    while True:
        g_norm = np.linalg.norm(gfunc(x))
        if g_norm < epsilon:
            break
        if k > maxk:
            break
        sk = Dogleg_Method(x, delta)
        x_new = x + sk
        fun_k = func(x)
        fun_new = func(x_new)
        r = (fun_k - fun_new) / (mk_function(x, [[0], [0]]) - mk_function(x, sk))
        if r < 0.25:
            delta = delta / 4
        elif r > 0.75 and np.linalg.norm(sk) == delta:
            delta = np.min((2 * delta, delta_max))
        else:
            pass
        if r <= 0:
            pass
        else:
            x = x_new
            k = k + 1
            Err.append(func(x))
            tk.append(k)
    return x, k, Err, tk


x = np.transpose(np.array([10, 10]))
x.shape = (2, 1)
delta_max = 20
X, k, Err, tk = TrustRegion(x, delta_max)
print(X)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'cm'
plt.yscale('log')
plt.xlabel('迭代次数 k')
plt.ylabel('目标函数值')
plt.plot(tk, Err, 'b-.')
plt.show()