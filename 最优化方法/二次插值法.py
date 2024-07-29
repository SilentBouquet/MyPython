from math import sin
from matplotlib import pyplot as plt


def phi(x):
    return x ** 2 - sin(x)


def qmin(a, b, delta, epsilon):
    s0 = a
    maxj = 20
    maxk = 30
    big = 1e6
    err = 1
    k = 1
    tk = []
    Err = []
    S = [s0]
    cond = 0
    h = 1
    ds = 0.00001
    while k < maxk and err > epsilon and cond != 5:
        f1 = (phi(s0 + ds) - phi(s0 - ds)) / (2 * ds)
        if f1 > 0:
            h = -abs(h)
        s1 = s0 + h
        s2 = s0 + 2 * h
        phi0 = phi(s0)
        phi1 = phi(s1)
        phi2 = phi(s2)
        cond = 0
        j = 0
        err = s2 - s0
        Err.append(abs(err))
        tk.append(k)
        while j < maxj and abs(h) > delta and cond == 0:
            if phi0 <= phi1:
                s2 = s1
                phi2 = phi1
                h = 0.5 * h
                s1 = s0 + h
                phi1 = phi(s1)
            elif phi2 < phi1:
                s1 = s2
                phi1 = phi2
                h = 2 * h
                s2 = s0 + 2 * h
                phi2 = phi(s2)
            else:
                cond = -1
            j = j + 1
            if abs(h) > big or abs(s0) > big:
                cond = 5
        else:
            d = 2 * (2 * phi1 - phi0 - phi2)
            if d < 0:
                barh = h * (4 * phi1 - 3 * phi0 - phi2) / d
            else:
                barh = h / 3
                cond = 4
            bars = s0 + barh
            barphi = phi(bars)
            h = abs(h)
            h0 = abs(barh)
            h1 = abs(barh - h)
            h2 = abs(barh - 2 * h)
            if h0 < h:
                h = h0
            if h1 < h:
                h = h1
            if h2 < h:
                h = h2
            if h == 0:
                h = barh
            if h < delta:
                cond = 1
            if abs(h) > big or abs(bars) > big:
                cond = 5
            err = abs(phi1-barphi)
            s0 = bars
            k = k + 1
            S.append(s0)
        if cond == 2 and h < delta:
            cond = 3
    s = s0
    phis = phi(s)
    ds = h
    dphi = err
    return s, phis, k, ds, dphi, S, Err, tk


s, phis, k, ds, dphi, S, Err, tk = qmin(0, 1, 1e-4, 1e-6)
print(s, phis, k, ds, dphi)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'cm'
plt.xticks(range(1, k))
plt.xlabel('迭代次数 k')
plt.ylabel('误差 err')
plt.plot(tk, Err, 'b-.')
plt.show()