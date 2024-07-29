import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-3, 3, 50)
y1 = 2*x + 1
y2 = x**2
y3 = np.sin(x)
plt.figure(num=1)
plt.plot(x, y1)
plt.figure(num=2)
plt.plot(x, y2)
plt.plot(x, y3, color='red', linewidth=1.5, linestyle='--')
plt.show()