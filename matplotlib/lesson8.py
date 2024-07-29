import matplotlib.pyplot as plt
import numpy as np


def f(x, y):
    return (1-x/2+x**5+y**3)*np.exp(-x**2-y**2)


n = 256
m = np.linspace(-3, 3, n)
n = np.linspace(-3, 3, n)
X, Y = np.meshgrid(m, n)

plt.contourf(X, Y, f(X, Y), 7, alpha=0.75, cmap='rainbow')

C = plt.contour(X, Y, f(X, Y), 7, colors='black')
plt.clabel(C, inline=True, fontsize=10)

plt.xticks(())
plt.yticks(())
plt.show()