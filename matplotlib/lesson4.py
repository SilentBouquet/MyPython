import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-3, 3, 50)
y = 2 * x + 1

plt.figure(num=1)
plt.plot(x, y)

ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.spines['bottom'].set_position(('data', 0))
ax.spines['left'].set_position(('data', 0))

x0 = 1
y0 = 2 * x0 + 1
plt.scatter(x0, y0, s=50, color='blue')
plt.plot([x0, x0], [y0, 0], 'k--')
plt.annotate(r'$2x+1=3$', xy=(x0, y0), xycoords='data', xytext=(+30, -30), textcoords='offset points',
             fontsize=14, arrowprops=dict(arrowstyle='->', connectionstyle='arc3, rad=0.2'))

plt.text(-3.7, 3, r'$This\ is\ the\ some\ text.\ \mu\ \sigma_i\ \alpha_t$',
         fontdict={'size': 14, 'color': 'r'})

plt.show()
