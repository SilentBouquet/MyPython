import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lu, solve_banded
from scipy.sparse.linalg import cg
import time

# 参数设置
a = 2
epsilon = 2
n = 201
h = 1.0 / (n - 1)
x = np.linspace(0, 1, n)


# 离散化方程并构造矩阵和右侧向量
def setup_cd_eq(n, a, epsilon):
    h = 1.0 / (n - 1)
    alpha = epsilon / h ** 2
    beta = a / h

    # 主对角线元素
    diag = np.ones(n - 2) * (2 * alpha + beta)
    # 上下对角线元素
    off_diag = np.ones(n - 3) * (-alpha)

    # 构造三对角矩阵（使用带状矩阵格式）
    A = np.diag(diag) + np.diag(off_diag, -1) + np.diag(off_diag, 1)

    # 右侧向量
    b = np.ones(n - 2) * a

    # 处理边界条件
    A[0, 0] = alpha + beta
    A[-1, -1] = alpha + beta

    return A, b


# LU分解直接法
def solve_cd_lu(A, b):
    start = time.perf_counter()  # 使用更高精度的计时器
    solution = np.linalg.solve(A, b)
    end = time.perf_counter()
    return solution, end - start


# 带状LU分解直接法
def solve_cd_banded_lu(A, b):
    start = time.perf_counter()
    # 提取对角线
    ab = np.zeros((3, len(b)))
    ab[1, :] = np.diag(A)  # 主对角线
    ab[0, 1:] = np.diag(A, -1)  # 下对角线
    ab[2, :-1] = np.diag(A, 1)  # 上对角线

    solution = solve_banded((1, 1), ab, b)
    end = time.perf_counter()
    return solution, end - start


# SOR迭代法
def solve_cd_sor(A, b, omega=1.25, max_iter=1e5, tol=1e-6):
    start = time.perf_counter()
    n = len(b)
    x = np.zeros(n)
    residual = np.linalg.norm(b)

    for iteration in range(int(max_iter)):
        x_prev = x.copy()

        for i in range(n):
            x[i] = (1 - omega) * x[i] + omega / A[i, i] * (
                        b[i] - np.dot(A[i, :i], x[:i]) - np.dot(A[i, i + 1:], x_prev[i + 1:]))

        residual = np.linalg.norm(b - np.dot(A, x))
        if residual < tol:
            break

    end = time.perf_counter()
    return x, end - start, iteration + 1


# 共轭梯度法
def solve_cd_cg(A, b, max_iter=1e5, tol=1e-6):
    start = time.perf_counter()
    x0 = np.zeros_like(b)
    x, info = cg(A, b, x0=x0, atol=tol, maxiter=int(max_iter))
    end = time.perf_counter()
    return x, end - start, info


# 计算解析解
def analytical_solution(x, a, epsilon):
    return ((1 - a) / (1 - np.exp(-1 / epsilon)) * (1 - np.exp(-x / epsilon)) + a * x)


# 主程序
A, b = setup_cd_eq(n, a, epsilon)

# LU分解求解
lu_sol, lu_time = solve_cd_lu(A, b)
print(f"LU分解计算时间: {lu_time:.6f}秒")

# 带状LU分解求解
try:
    banded_lu_sol, banded_lu_time = solve_cd_banded_lu(A, b)
    print(f"带状LU分解计算时间: {banded_lu_time:.6f}秒")
except np.linalg.LinAlgError:
    print("带状LU分解失败：矩阵奇异")

# SOR迭代求解
sor_sol, sor_time, sor_iter = solve_cd_sor(A, b)
print(f"SOR迭代计算时间: {sor_time:.6f}秒, 迭代次数: {sor_iter}")

# 共轭梯度法求解
cg_sol, cg_time, cg_info = solve_cd_cg(A, b)
print(f"共轭梯度法计算时间: {cg_time:.6f}秒, 退出代码: {cg_info}")

# 计算解析解
x_all = np.linspace(0, 1, n)
y_exact = analytical_solution(x_all, a, epsilon)

# 绘制结果
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'cm'
plt.figure(figsize=(10, 6))
plt.plot(x_all[1:-1], lu_sol, label='LU分解')
plt.plot(x_all[1:-1], sor_sol, label='SOR迭代')
plt.plot(x_all[1:-1], cg_sol, label='共轭梯度法')
plt.plot(x_all, y_exact, 'r--', label='解析解')
plt.xlabel('x')
plt.ylabel('y')
plt.title('1维对流扩散方程数值解')
plt.legend()
plt.grid()
plt.show()