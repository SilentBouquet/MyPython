import matplotlib.pyplot as plt
import numpy as np


def entropy(p):
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def gini(p):
    return p * (1 - p) + (1 - p) * p


def error(p):
    return 1 - np.max([p, 1 - p])


if __name__ == '__main__':
    plt.rcParams['font.sans-serif'] = ['KaiTi']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['mathtext.fontset'] = 'cm'
    x = np.arange(0, 1, 0.01)
    ent = [entropy(p) if p != 0 else 0 for p in x]
    sc_ent = [e * 0.5 if e else 0 for e in ent]
    err = [error(i) for i in x]
    fig = plt.figure(figsize=(8, 6))
    # 对fig创建默认的坐标区（一行一列一个坐标区）
    ax = fig.add_subplot(111)
    for i, lab, ls, c in zip([ent, sc_ent, gini(x), err],
                              [r'$Entropy$', r'$Entropy (Scaled)$',
                               r'$Gini\ Impurity$', r'$Misclassification Error$'],
                              ['-', '-', '--', '-.'],
                              ['black', 'lightgray', 'red', 'green', 'cyan']):
        line = ax.plot(x, i, label=lab, linestyle=ls, lw=2, color=c)
    # bbox_to_anchor指定图例的位置，ncol设置图例分为n列展示
    # fancybox控制是否在构成图例背景的周围启用圆边，shadow控制是否在图例后面画一个阴影
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1),
              ncol=4, fancybox=True, shadow=False)
    ax.axhline(y=0.5, linewidth=1, color='k', linestyle='--')
    ax.axhline(y=1, linewidth=1, color='k', linestyle='--')
    plt.ylim(0, 1.2)
    plt.xlabel(r'$p(i=1)$')
    plt.ylabel(r'$Impurity\ index$')
    plt.title("0和1之间不同类成员概率的不同杂质指标")
    plt.tight_layout()
    plt.show()