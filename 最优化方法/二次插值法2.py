from math import sin
from matplotlib import pyplot as plt


def phi(x):
    return x ** 2 - sin(x)


x1 = 0
x3 = 1
x2 = (x1 + x3) / 2
delta = 1e-4
epsilon = 1e-6
Err = []
X = []
k = -1
K = []
while True:
    k = k + 1
    f1 = phi(x1)
    f2 = phi(x2)
    f3 = phi(x3)
    a1 = 2 * (f1 * (x2 - x3) + f2 * (x2 - x1) + f3 * (x1 - x2))
    a2 = (f1 * (x2 ** 2 - x3 ** 2) + f2 * (x2 ** 2 - x1 ** 2) + f3 * (x1 ** 2 - x2 ** 2))
    xx = a2 / a1
    K.append(k)
    X.append(xx)
    Err.append(abs(f2-phi(xx)))
    if abs(x2-xx) < delta and abs(f2-phi(xx)) < epsilon:
        x = xx
        fx = phi(xx)
        break
    else:
        if (xx - x1) * (xx - x2) < 0:
            if phi(xx) <= f2:
                x3 = x2
                x2 = xx
            else:
                x1 = xx
        else:
            if phi(xx) <= f2:
                x1 = x2
                x2 = xx
            else:
                x3 = xx

print(f'最优解为{x}，此时函数值为{fx}')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'cm'
plt.xlim(1, k)
plt.xticks(range(1, k+1))
plt.xlabel('迭代次数 k')
plt.ylabel('误差 err')
plt.plot(K, Err, 'b-.')
plt.show()