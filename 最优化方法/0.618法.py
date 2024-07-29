import math
import matplotlib.pyplot as plt


def phi(x):
    return x ** 2 - math.sin(x)


def golds(a, b, delta, epsilon):
    t = (math.sqrt(5) - 1) / 2
    tk = []
    err = []
    h = b - a
    phia = phi(a)
    phib = phi(b)
    p = a+(1-t)*h
    q = a+t*h
    phip = phi(p)
    phiq = phi(q)
    k = 1
    G = [[a, p, q, b]]
    while abs(phib - phia) > epsilon or h > delta:
        if phip < phiq:
            b = q
            phib = phiq
            q = p
            phiq = phip
            h = b - a
            p = a+(1-t)*h
            phip = phi(p)
        else:
            a = p
            phia = phi(a)
            p = q
            phip = phiq
            h = b - a
            q = a+t*h
            phiq = phi(q)
        err.append(abs(phib - phia))
        tk.append(k)
        k += 1
        G.append([a, p, q, b])
    ds = abs(b-a)
    dphi = abs(phib-phia)
    if phip <= phiq:
        s = p
        phis = phip
    else:
        s = q
        phis = phiq
    E = [ds, dphi]
    return s, phis, k, G, E, err, tk


s, phis, k, G, E, err, tk = golds(0, 1, 1e-4, 1e-5)
print(s, k, E)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'cm'
plt.yscale('log')
plt.xticks(range(1, k+1, 1))
plt.xlabel('迭代次数 k')
plt.ylabel('误差 err')
plt.plot(tk, err, 'b-.')
plt.show()