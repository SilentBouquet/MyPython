import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, ticker

matplotlib.use("Qt5Agg")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'cm'


def conjugate_gradient(A, b, x0, tol=1e-10, max_iter=None):
    n = len(b)
    if max_iter is None:
        max_iter = n

    x = x0.copy()
    r = b - np.dot(A, x)
    p = r.copy()
    errors = [np.linalg.norm(r)]

    for i in range(max_iter):
        Ap = np.dot(A, p)
        alpha = np.dot(r.T, r) / np.dot(p.T, Ap)
        x += alpha * p
        r_prev = r.copy()
        r -= alpha * Ap
        errors.append(np.linalg.norm(r))

        beta = np.dot(r.T, r) / np.dot(r_prev.T, r_prev)
        p = r + beta * p

        if errors[-1] < tol:
            break

    return x, errors


def solve_and_plot(matrix_file, x_ast):
    A = pd.read_excel(matrix_file, header=None).values

    # 右端项 b = A * x_ast
    b = np.dot(A, x_ast)

    # 初始向量 x0 = 0
    n = A.shape[0]
    x0 = np.zeros(n)

    # 使用共轭梯度法求解
    x, errors = conjugate_gradient(A, b, x0)

    # 绘制误差收敛曲线
    plt.figure(figsize=(10, 6))
    plt.plot(errors, 'b-', marker='o', markersize=3)
    plt.xlabel('迭代次数')
    plt.ylabel('误差')
    plt.title(f'共轭梯度法误差收敛曲线 ({A.shape[0]}阶矩阵)')
    plt.grid(True)
    plt.yscale('log', base=10)  # 使用对数坐标

    ax1 = plt.gca()  # 获取当前图像的坐标轴

    # 更改坐标轴字体，避免出现指数为负的情况
    tick_font = font_manager.FontProperties(family='DejaVu Sans', size=9.0)
    for labelx in ax1.get_xticklabels():
        labelx.set_fontproperties(tick_font)  # 设置 x轴刻度字体
    for labely in ax1.get_yticklabels():
        labely.set_fontproperties(tick_font)  # 设置 y轴刻度字体
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))  # x轴刻度设置为整数
    plt.tight_layout()
    plt.show()

    return x


x_ast_100 = np.ones(100)
x_ast_400 = np.ones(400)
x_ast_1000 = np.ones(1000)

# 定义矩阵的加载路径
array_path_100 = r"D:\QQ\QQdownloads\第1个系数矩阵(100阶)(1).xlsx"
array_path_400 = r"D:\QQ\QQdownloads\第2个系数矩阵(400阶)(1).xlsx"
array_path_1000 = r"D:\QQ\QQdownloads\第3个系数矩阵(1000阶)(1).xlsx"

# 解决三个矩阵问题
print("处理100阶矩阵...")
solve_and_plot(array_path_100, x_ast_100)

print("\n处理400阶矩阵...")
solve_and_plot(array_path_400, x_ast_400)

print("\n处理1000阶矩阵...")
solve_and_plot(array_path_1000, x_ast_1000)


def steepest_descent(A, b, x0, tol=1e-10, max_iter=None):
    n = len(b)
    if max_iter is None:
        max_iter = n

    x = x0.copy()
    r = b - np.dot(A, x)
    errors = [np.linalg.norm(r)]

    for i in range(max_iter):
        alpha = np.dot(r.T, r) / np.dot(r.T, np.dot(A, r))
        x += alpha * r
        r_prev = r.copy()
        r = b - np.dot(A, x)
        errors.append(np.linalg.norm(r))

        if errors[-1] < tol:
            break

    return x, errors


def compare_methods(matrix_file, x_ast):
    A = pd.read_excel(matrix_file, header=None).values

    # 右端项 b = A * x_ast
    b = np.dot(A, x_ast)

    # 初始向量 x0 = 0
    n = A.shape[0]
    x0 = np.zeros(n)

    # 使用共轭梯度法求解
    _, errors_cg = conjugate_gradient(A, b, x0)

    # 使用最速下降法求解
    _, errors_sd = steepest_descent(A, b, x0)

    # 绘制误差收敛曲线
    plt.figure(figsize=(10, 6))
    plt.plot(errors_cg, 'b-', marker='o', markersize=2, label='共轭梯度法')
    plt.plot(errors_sd, 'r--', marker='x', markersize=2, label='最速下降法')
    plt.xlabel('迭代次数')
    plt.ylabel('误差')
    plt.title(f'共轭梯度法与最速下降法收敛速度比较 ({A.shape[0]}阶矩阵)')
    plt.grid(True)
    plt.yscale('log', base=10)  # 使用对数坐标
    ax1 = plt.gca()
    # 更改坐标轴字体，避免出现指数为负的情况
    tick_font = font_manager.FontProperties(family='DejaVu Sans', size=9.0)
    for labelx in ax1.get_xticklabels():
        labelx.set_fontproperties(tick_font)  # 设置 x轴刻度字体
    for labely in ax1.get_yticklabels():
        labely.set_fontproperties(tick_font)  # 设置 y轴刻度字体
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))  # x轴刻度设置为整数
    plt.tight_layout()
    plt.legend()
    plt.show()


# 比较方法
print("\n比较共轭梯度法和最速下降法...")
compare_methods(array_path_100, x_ast_100)
compare_methods(array_path_400, x_ast_400)
compare_methods(array_path_1000, x_ast_1000)