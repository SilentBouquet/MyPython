import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 定义线性方程组 A x = b
A = np.array([[3, 2, 1],
              [-1, 1, 2],
              [2, -1, 4]], dtype=float)
b = np.array([0, 4, 1], dtype=float)
x_true = np.array([-1, 1, 1], dtype=float)  # 精确解

# 迭代参数设置
x0 = np.zeros_like(b)  # 初始解
max_iter = 100  # 最大迭代次数
tol = 1e-8  # 收敛容差
omega = 0.9  # SOR松弛因子


def jacobi_method(A, b, x0, max_iter, tol, x_true):
    """Jacobi迭代法"""
    n = len(b)
    D = np.diag(np.diag(A))
    L_plus_U = D - A
    D_inv = np.diag(1 / np.diag(D))

    x = x0.copy()
    errors = []

    for _ in range(max_iter):
        x_new = D_inv @ (b - L_plus_U @ x)
        error = np.linalg.norm(x_new - x_true)
        errors.append(error)

        if error < tol:
            break
        x = x_new.copy()

    return x, errors


def gauss_seidel_method(A, b, x0, max_iter, tol, x_true):
    """Gauss-Seidel迭代法"""
    n = len(b)
    D = np.diag(np.diag(A))
    L = -np.tril(A, -1)
    U = -np.triu(A, 1)

    x = x0.copy()
    errors = []

    for _ in range(max_iter):
        x_new = np.linalg.solve(D - L, U @ x + b)
        error = np.linalg.norm(x_new - x_true)
        errors.append(error)

        if error < tol:
            break
        x = x_new.copy()

    return x, errors


def sor_method(A, b, x0, omega, max_iter, tol, x_true):
    """SOR迭代法"""
    n = len(b)
    D = np.diag(np.diag(A))
    L = -np.tril(A, -1)
    U = -np.triu(A, 1)

    M = (1 / omega) * D - L
    N = (1 / omega - 1) * D + U

    x = x0.copy()
    errors = []

    for _ in range(max_iter):
        x_new = np.linalg.solve(M, N @ x + b)
        error = np.linalg.norm(x_new - x_true)
        errors.append(error)

        if error < tol:
            break
        x = x_new.copy()

    return x, errors


# 执行迭代计算
x_jacobi, err_jacobi = jacobi_method(A, b, x0, max_iter, tol, x_true)
x_gs, err_gs = gauss_seidel_method(A, b, x0, max_iter, tol, x_true)
x_sor, err_sor = sor_method(A, b, x0, omega, max_iter, tol, x_true)

# 输出结果
print(f"Jacobi方法最终解: {x_jacobi.round(6)}，误差: {np.linalg.norm(x_jacobi - x_true):.2e}")
print(f"Gauss-Seidel方法最终解: {x_gs.round(6)}，误差: {np.linalg.norm(x_gs - x_true):.2e}")
print(f"SOR方法(ω={omega})最终解: {x_sor.round(6)}，误差: {np.linalg.norm(x_sor - x_true):.2e}")

# 绘制收敛曲线
plt.figure(figsize=(10, 6))
plt.semilogy(err_jacobi, label='Jacobi方法')
plt.semilogy(err_gs, label='Gauss-Seidel方法')
plt.semilogy(err_sor, label=f'SOR方法 (ω={omega})')

plt.xlabel('迭代次数')
plt.ylabel('误差范数')
plt.title('三种迭代方法的收敛曲线')
plt.legend()
plt.grid(True, which="both", linestyle='--')
plt.show()
