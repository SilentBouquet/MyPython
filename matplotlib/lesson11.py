import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-3, 3, 100)
y1 = np.sin(np.exp(x))
y2 = np.cos(np.exp(x))
y3 = np.exp(np.sin(x))
y4 = np.exp(np.sin(x))
plt.figure()
ax_1 = plt.subplot2grid((3, 3), (0, 0), colspan=3, rowspan=1)
ax_1.plot([1, 2], [1, 2])
ax_1.set_title('ax_1')
ax_2 = plt.subplot2grid((3, 3), (1, 0), colspan=2, rowspan=1)
ax_2.plot(x, y1)
ax_3 = plt.subplot2grid((3, 3), (1, 2), rowspan=2)
ax_3.plot(x, y2)
ax_4 = plt.subplot2grid((3, 3), (2, 0))
ax_4.plot(x, y3)
ax_5 = plt.subplot2grid((3, 3), (2, 1))
ax_5.plot(x, y4)

plt.show()