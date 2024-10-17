import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-3, 3, 50)
y = 0.1*x

plt.figure(num=1)
plt.plot(x, y, linewidth=10, alpha=0.7)
plt.ylim(-2, 2)

ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.spines['bottom'].set_position(('数据分析实训', 0))
ax.spines['left'].set_position(('数据分析实训', 0))

for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontsize(12)
    label.set_bbox(dict(facecolor='pink', edgecolor='none', alpha=0.1))

plt.show()