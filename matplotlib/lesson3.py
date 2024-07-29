import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-3, 3, 50)
y1 = 2*x + 1
y2 = x**2
plt.figure(num=1)

plt.xlim(-1, 2)
plt.ylim(-2, 3)
plt.xlabel('I am X')
plt.ylabel('I am Y')

new_ticks = np.linspace(-1, 2, 5)
plt.xticks(new_ticks)
plt.yticks([-2, -1, 0, 1, 2, 3],
           [r'$really\ bad$', r'$bad$', r'$normal$', r'$good$', r'$really\ good$', r'$perfect$']
)

plt.plot(x, y1, label='2*x + 1')
plt.plot(x, y2, color='red', linewidth=1.5, linestyle='--', label='x**2')
plt.legend(loc='best')

plt.show()
