import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.patheffects import withStroke
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

# 自定义颜色
royal_blue = [0, 20/256, 82/256]

np.random.seed(1)
# 生成数据
X = np.linspace(0.5, 3.5, 100)
Y1 = 3 + np.cos(X)
Y2 = 1 + np.cos(1 + X / 0.75) / 2
Y3 = np.random.uniform(Y1, Y2, size=len(X))

# 创建并配置图形和轴
fig = plt.figure(figsize=(8, 8))
ax = fig.add_axes((0.2, 0.17, 0.68, 0.7), aspect=1)

# 设置主要和次要刻度定位器
# x轴的主要刻度间隔
ax.xaxis.set_major_locator(MultipleLocator(1.000))
# x轴的次要刻度间隔
ax.xaxis.set_minor_locator(AutoMinorLocator(4))
# y轴的主要刻度间隔
ax.yaxis.set_major_locator(MultipleLocator(1.000))
# y轴的次要刻度间隔
ax.yaxis.set_minor_locator(AutoMinorLocator(4))
# 设置次要刻度的格式
ax.xaxis.set_minor_formatter("{x:.2f}")

