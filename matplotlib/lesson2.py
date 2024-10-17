import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-3, 3, 50)
y1 = 2*x + 1
y2 = x**2
plt.figure(num=1)
plt.plot(x, y1)
plt.plot(x, y2, color='red', linewidth=1.5, linestyle='--')

plt.xlim(-1, 2)
plt.ylim(-2, 3)
plt.xlabel('I am X')
plt.ylabel('I am Y')

new_ticks = np.linspace(-1, 2, 5)
plt.xticks(new_ticks)
plt.yticks([-2, -1, 0, 1, 2, 3],
           [r'$really\ bad$', r'$bad$', r'$normal$', r'$good$', r'$really\ good$', r'$perfect$']
)

# gca = 'get current axis'
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.spines['bottom'].set_position(('数据分析实训', 0))
ax.spines['left'].set_position(('数据分析实训', 0))

plt.show()