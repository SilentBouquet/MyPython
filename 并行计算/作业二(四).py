import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 系数矩阵A（10x10）
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
x_star = np.arange(1, 11, dtype=float)  # 精确解 [1,2,...,10]^T
b = A @ x_star  # 右端项 b = A x*
x0 = np.zeros_like(x_star)  # 初始向量（全0）
max_iter = 100  # 最大迭代次数
tol = 1e-8  # 收敛精度
omega = 1.1  # 单一松弛因子


def sor_method(A, b, x0, omega, max_iter, x_star, tol):
    """SOR迭代法：返回最终解、最终误差、误差列表"""
    n = len(b)
    D = np.diag(np.diag(A))  # 对角矩阵
    L = -np.tril(A, -1)  # 严格下三角部分的负矩阵
    U = -np.triu(A, 1)  # 严格上三角部分的负矩阵
    M = (1 / omega) * D - L  # SOR左侧矩阵
    N = ((1 - omega) / omega) * D + U  # SOR右侧矩阵

    x = x0.copy()
    errors = []
    for _ in range(max_iter):
        x_new = np.linalg.solve(M, N @ x + b)  # 迭代更新
        error = np.linalg.norm(x_new - x_star)  # 误差计算
        errors.append(error)
        if error < tol:  # 满足精度则提前停止
            break
        x = x_new.copy()
    return x, error, errors  # 最终解、最终误差、误差序列


# 执行SOR迭代
x_final, final_err, errors = sor_method(A, b, x0, omega, max_iter, x_star, tol)

# 打印最终结果和误差
print(f"SOR迭代法（松弛因子ω={omega}）：")
print(f"  最终解向量：{x_final.round(6)}")  # 保留6位小数
print(f"  最终误差：{final_err:.6e}")  # 科学计数法表示误差

# 绘制误差下降曲线
plt.figure(figsize=(10, 6))
plt.semilogy(range(1, len(errors) + 1), errors, label=f'ω={omega}', marker='.', markersize=4)
plt.xlabel('迭代次数')
plt.ylabel('误差范数（2-范数）')
plt.title(f'SOR迭代法（ω={omega}）的误差下降曲线')
plt.legend()
plt.grid(ls='--', alpha=0.7)
plt.show()