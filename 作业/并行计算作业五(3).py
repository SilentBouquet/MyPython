import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.linalg import lu_factor, lu_solve
import time
import matplotlib.pyplot as plt

n = 51
h = 1.0 / n
m = n - 1
total_nodes = m * m

A = sp.lil_matrix((total_nodes, total_nodes))
b = np.zeros(total_nodes)

for j in range(1, m + 1):
    for i in range(1, m + 1):
        # 当前节点坐标和全局索引
        x = i * h
        y = j * h
        k = (i - 1) + (j - 1) * m

        # 中心系数
        A[k, k] = 1 + (h ** 2) / 4

        # 右端项初始化为 (h²/4)*f(x,y)
        b[k] = (h ** 2 / 4) * np.sin(x * y)

        # 处理四个方向的邻居
        # 左邻居 (i-1, j)
        if i > 1:
            A[k, k - 1] = -0.25
        else:
            g_left = 0 + y ** 2  # g(0,y)=0²+y²
            b[k] += 0.25 * g_left

        # 右邻居 (i+1, j)
        if i < m:
            A[k, k + 1] = -0.25
        else:
            g_right = 1 + y ** 2  # g(1,y)=1+y²
            b[k] += 0.25 * g_right

        # 下邻居 (i, j-1)
        if j > 1:
            A[k, k - m] = -0.25
        else:
            g_bottom = x ** 2 + 0  # g(x,0)=x²+0²
            b[k] += 0.25 * g_bottom

        # 上邻居 (i, j+1)
        if j < m:
            A[k, k + m] = -0.25
        else:
            g_top = x ** 2 + 1  # g(x,1)=x²+1²
            b[k] += 0.25 * g_top

A_csr = A.tocsr()

A_dense = A_csr.toarray()
start_lu = time.time()
lu, piv = lu_factor(A_dense)
x_lu = lu_solve((lu, piv), b)
time_lu = time.time() - start_lu


def sor(A, b, omega=1.8, maxiter=int(1e5), tol=1e-6):
    x = np.zeros_like(b)
    diag = A.diagonal()
    A_nodiag = A.copy()
    A_nodiag.setdiag(0)
    A_nodiag = A_nodiag.tocsr()
    b_norm = np.linalg.norm(b)

    for it in range(maxiter):
        x_new = x.copy()
        for i in range(A.shape[0]):
            row = A_nodiag[i].indices
            data = A_nodiag[i].data
            sum_neighbors = data @ x_new[row]
            x_new[i] = (1 - omega) * x[i] + omega * (b[i] - sum_neighbors) / diag[i]

        residual = np.linalg.norm(A @ x_new - b)
        if residual < tol * b_norm:
            return x_new, it + 1, residual
        x = x_new.copy()
    return x, maxiter, residual


start_sor = time.time()
x_sor, sor_iters, sor_res = sor(A_csr, b)
time_sor = time.time() - start_sor

start_cg = time.time()
x_cg, info = spla.cg(A_csr, b, atol=1e-6, maxiter=int(1e5))
time_cg = time.time() - start_cg
residual_cg = np.linalg.norm(A_csr @ x_cg - b)

print(f"LU分解时间: {time_lu:.4f}s")
print(f"SOR迭代时间: {time_sor:.4f}s | 迭代次数: {sor_iters} | 残差: {sor_res:.2e}")
print(f"CG迭代时间: {time_cg:.4f}s | 残差: {residual_cg:.2e}")


def reshape_solution(u):
    """将解向量转换为二维网格"""
    u_grid = np.zeros((n + 1, n + 1))
    # 填充边界条件
    for i in range(n + 1):
        for j in range(n + 1):
            if i == 0 or i == n or j == 0 or j == n:
                u_grid[i, j] = (i * h) ** 2 + (j * h) ** 2  # g(x,y)=x²+y²
    # 填充内部解
    for j in range(1, m + 1):
        for i in range(1, m + 1):
            k = (i - 1) + (j - 1) * m
            u_grid[i, j] = u[k]
    return u_grid


u_lu_grid = reshape_solution(x_lu)
u_sor_grid = reshape_solution(x_sor)
u_cg_grid = reshape_solution(x_cg)

# 创建网格坐标
x_coords = np.linspace(0, 1, n + 1)
y_coords = np.linspace(0, 1, n + 1)
X, Y = np.meshgrid(x_coords, y_coords)

# 绘制LU分解结果
fig = plt.figure(figsize=(18, 6))
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'cm'

ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(X, Y, u_lu_grid.T, cmap='viridis')
ax1.set_title("LU分解解")

ax2 = fig.add_subplot(132, projection='3d')
ax2.plot_surface(X, Y, u_sor_grid.T, cmap='viridis')
ax2.set_title("SOR迭代解")

ax3 = fig.add_subplot(133, projection='3d')
ax3.plot_surface(X, Y, u_cg_grid.T, cmap='viridis')
ax3.set_title("共轭梯度解")

plt.tight_layout()
plt.show()