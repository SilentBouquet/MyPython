import matplotlib.pyplot as plt
import numpy as np

n = 12
X = np.arange(1, n+1)
Y1 = (1 - X/float(n) * np.random.uniform(0.5, 1.0, n))
Y2 = (1 - X/float(n) * np.random.uniform(0.5, 1.0, n))

plt.bar(X, +Y1, facecolor='#9999ff', edgecolor='white')
plt.bar(X, -Y2, facecolor='#ff9999', edgecolor='white')

for x, y in zip(X, Y1):
    plt.text(x, y + 0.05, '%.2f' % y, ha='center', va='bottom')

for x, y in zip(X, Y2):
    plt.text(x, -y - 0.05, '-%.2f' % y, ha='center', va='top')

plt.xlim(0, n+1)
plt.xticks(())
plt.ylim(-1.25, 1.25)
plt.yticks(())

plt.show()