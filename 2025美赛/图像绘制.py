import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_function1(func, x_range, title, xlabel, ylabel, A):
    # 生成x值
    x = np.linspace(x_range[0], x_range[1], x_range[2])

    # 设置Seaborn风格
    sns.set(style="whitegrid")
    # 定义颜色方案
    colors = sns.color_palette('viridis', len(A))

    plt.figure(figsize=(12, 6))
    plt.rcParams['font.sans-serif'] = ['Lucida Sans Unicode']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['mathtext.fontset'] = 'cm'
    for i, a in enumerate(A):
        y = func(x, A=a)
        # 绘制图像
        plt.plot(x, y, color=colors[i], linewidth=2, label=f'A={a}')
    plt.title(title, fontsize=17)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()


def plot_function2(func, x_range, title, xlabel, ylabel, r):
    # 生成x值
    x = np.linspace(x_range[0], x_range[1], x_range[2])

    # 设置Seaborn风格
    sns.set(style="whitegrid")
    # 定义颜色方案
    colors = sns.color_palette('muted', len(r))

    plt.figure(figsize=(12, 6))
    plt.rcParams['font.sans-serif'] = ['Lucida Sans Unicode']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['mathtext.fontset'] = 'cm'
    for i, a in enumerate(r):
        y = func(x, r=a)
        # 绘制图像
        plt.plot(x, y, color=colors[i], linewidth=2, label=f'r={a}')
    plt.title(title, fontsize=17)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()


def plot_function3(func, x_range, title, xlabel, ylabel, sigma):
    # 生成x值
    x = np.linspace(x_range[0], x_range[1], x_range[2])

    # 设置Seaborn风格
    sns.set(style="whitegrid")
    # 定义颜色方案
    colors = sns.color_palette('muted', len(sigma))

    plt.figure(figsize=(12, 6))
    plt.rcParams['font.sans-serif'] = ['Lucida Sans Unicode']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['mathtext.fontset'] = 'cm'
    for i, a in enumerate(sigma):
        y = func(x, sigma=a)
        # 绘制图像
        plt.plot(x, y, color=colors[i], linewidth=2, label=f'sigma={a}')
    plt.title(title, fontsize=17)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()


def plot_function4(func, x_range, title, xlabel, ylabel, alpha):
    # 生成x值
    x = np.linspace(x_range[0], x_range[1], x_range[2])

    # 设置Seaborn风格
    sns.set(style="whitegrid")
    # 定义颜色方案
    colors = sns.color_palette('muted', len(alpha))

    plt.figure(figsize=(12, 6))
    plt.rcParams['font.sans-serif'] = ['Lucida Sans Unicode']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['mathtext.fontset'] = 'cm'
    for i, a in enumerate(alpha):
        y = func(x, alpha=a)
        # 绘制图像
        plt.plot(x, y, color=colors[i], linewidth=2, label=f'alpha={a}')
    plt.title(title, fontsize=17)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()


def my_function(x, A=3, r=0.15, sigma=-0.05, alpha=0.5, P=1.5):
    # 定义权重系数
    a1, a2, a3, a4 = 0.6, 0.2, 0.1, 0.1
    w1, w2, w3 = 0.5, 0.2, 0.3
    n1, n2 = 0.5, 0.5
    N_max = 3
    N = 1
    U_min = 0.5
    Y_min = 0.5

    return (a1 + a3 * w1) * (P + r * (N - 2 * x) - alpha) * 2 * x \
        + (a2 - a3 * w3 + a4 * n1) * A * 2 * x ** sigma - (a3 * w2 + a4 * n2) * 2 * x


plot_function1(my_function, x_range=(0, 10, 400), title=r'$Objective\ Function\ Image$',
               xlabel=r'$Number\ of\ passengers\ (10,000\ people)$',
               ylabel=r'$Total\ Social\ Welfare$', A=[1, 3, 5, 7, 10])
plot_function2(my_function, x_range=(0, 10, 400), title=r'$Objective\ Function\ Image$',
               xlabel=r'$Number\ of\ passengers\ (10,000\ people)$',
               ylabel=r'$Total\ Social\ Welfare$', r=[0.1, 0.2, 0.3, 0.4, 0.5])
plot_function3(my_function, x_range=(0, 3, 200), title=r'$Objective\ Function\ Image$',
               xlabel=r'$Number\ of\ passengers\ (10,000\ people)$',
               ylabel=r'$Total\ Social\ Welfare$', sigma=[-0.1, -0.3, -0.5, -0.7, -0.9])
plot_function4(my_function, x_range=(0, 5, 400), title=r'$Objective\ Function\ Image$',
               xlabel=r'$Number\ of\ passengers\ (10,000\ people)$',
               ylabel=r'$Total\ Social\ Welfare$', alpha=[0.25, 0.5, 0.75, 1, 1.5])