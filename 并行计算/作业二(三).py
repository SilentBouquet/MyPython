import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 系数矩阵A
A = np.array([
    [10.84, 1.95, 2.62, 2.42, 2.25, 2.35, 2.21, 2.31, 2.28, 2.33],
    [1.95, 11.43, 2.75, 2.54, 2.36, 2.47, 2.32, 2.43, 2.40, 2.45],
    [2.62, 2.75, 11.97, 2.73, 2.53, 2.65, 2.49, 2.61, 2.57, 2.63],
    [2.42, 2.54, 2.73, 11.70, 2.34, 2.45, 2.30, 2.41, 2.38, 2.43],
    [2.25, 2.36, 2.53, 2.34, 11.41, 2.28, 2.14, 2.24, 2.21, 2.26],
    [2.35, 2.47, 2.65, 2.45, 2.28, 11.59, 2.24, 2.34, 2.31, 2.36],
    [2.21, 2.32, 2.49, 2.30, 2.14, 2.24, 11.37, 2.20, 2.17, 2.22],
    [2.31, 2.43, 2.61, 2.41, 2.24, 2.34, 2.20, 11.49, 2.27, 2.32],
    [2.28, 2.40, 2.57, 2.38, 2.21, 2.31, 2.17, 2.27, 11.45, 2.29],
    [2.33, 2.45, 2.63, 2.43, 2.26, 2.36, 2.22, 2.32, 2.29, 11.52]
])

# 问题参数
x_star = np.arange(1, 11, dtype=float)  # 精确解
b = A @ x_star  # 右端项
x0 = np.zeros(10)  # 初始向量
max_iter = 10  # 迭代次数
tol = 1e-8  # 精度阈值


def steepest_descent(A, b, x0, max_iter, x_star, tol):
    x = x0.copy()
    for _ in range(max_iter):
        r = b - A @ x
        alpha = (r @ r) / (r @ A @ r)
        x += alpha * r
        if np.linalg.norm(x - x_star) < tol:
            break
    final_err = np.linalg.norm(x - x_star)
    return x, final_err


def conjugate_gradient(A, b, x0, max_iter, x_star, tol):
    x = x0.copy()
    r = b - A @ x
    p = r.copy()
    for _ in range(max_iter):
        alpha = (r @ r) / (p @ A @ p)
        x += alpha * p
        r_new = r - alpha * A @ p
        beta = (r_new @ r_new) / (r @ r) if (r @ r) != 0 else 0
        p = r_new + beta * p
        r = r_new
        if np.linalg.norm(x - x_star) < tol:
            break
    final_err = np.linalg.norm(x - x_star)
    return x, final_err


# 执行计算
x_sd, err_sd = steepest_descent(A, b, x0, max_iter, x_star, tol)
x_cg, err_cg = conjugate_gradient(A, b, x0, max_iter, x_star, tol)

# 打印最终结果
print("最速下降法：")
print(f"最终解向量：{x_sd.round(6)}")
print(f"最终误差：{err_sd:.6e}\n")

print("共轭梯度法：")
print(f"最终解向量：{x_cg.round(6)}")
print(f"最终误差：{err_cg:.6e}")


# 绘制误差曲线
def get_errors(method, A, b, x0, max_iter, x_star):
    errors = []
    x = x0.copy()
    for _ in range(max_iter):
        if method == 'sd':
            r = b - A @ x
            alpha = (r @ r) / (r @ A @ r)
            x += alpha * r
        elif method == 'cg':
            if _ == 0:
                r = b - A @ x
                p = r.copy()
            alpha = (r @ r) / (p @ A @ p)
            x += alpha * p
            r_new = r - alpha * A @ p
            beta = (r_new @ r_new) / (r @ r) if (r @ r) != 0 else 0
            p = r_new + beta * p
            r = r_new
        errors.append(np.linalg.norm(x - x_star))
    return errors


err_sd_plot = get_errors('sd', A, b, x0, max_iter, x_star)
err_cg_plot = get_errors('cg', A, b, x0, max_iter, x_star)

plt.figure(figsize=(10, 6))
plt.semilogy(range(1, max_iter + 1), err_sd_plot, label='最速下降法', marker='o', markersize=4)
plt.semilogy(range(1, max_iter + 1), err_cg_plot, label='共轭梯度法', marker='s', markersize=4)
plt.xlabel('迭代次数')
plt.ylabel('误差范数（2-范数）')
plt.title('误差下降曲线')
plt.legend()
plt.grid(ls='--', alpha=0.7)
plt.show()
