import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse.linalg import norm as spnorm
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
        k = (i - 1) + (j - 1) * m
        A[k, k] = -4.0 / h ** 2
        if i > 1: A[k, k - 1] = 1.0 / h ** 2  # 左邻点
        if i < m: A[k, k + 1] = 1.0 / h ** 2  # 右邻点
        if j > 1: A[k, k - m] = 1.0 / h ** 2  # 下邻点
        if j < m: A[k, k + m] = 1.0 / h ** 2  # 上邻点
        if j == m: b[k] = -1.0 / h ** 2  # 顶边界条件

A_csr = A.tocsr()

A_dense = A_csr.toarray()
start_lu = time.time()
lu, piv = lu_factor(A_dense)
x_lu = lu_solve((lu, piv), b)
time_lu = time.time() - start_lu
res_lu = np.linalg.norm(A_dense @ x_lu - b)


def sor(A, b, omega=1.9, maxiter=1000, tol=1e-6):
    x = np.zeros_like(b)
    diag = A.diagonal()
    A_nodiag = A.copy()
    A_nodiag.setdiag(0)
    A_nodiag = A_nodiag.tocsr()
    residuals = []
    for it in range(maxiter):
        x_new = x.copy()
        for i in range(A.shape[0]):
            row = A_nodiag[i].indices
            sum_neighbors = A_nodiag[i].data @ x_new[row]
            x_new[i] = (1 - omega) * x[i] + omega * (b[i] - sum_neighbors) / diag[i]
        res = np.linalg.norm(A @ x_new - b)
        residuals.append(res)
        if res < tol:
            return x_new, it + 1, res, residuals
        x = x_new
    return x, maxiter, res, residuals


start_sor = time.time()
x_sor, sor_iters, sor_res, res_hist = sor(A_csr, b, omega=1.9)
time_sor = time.time() - start_sor

start_cg = time.time()
x_cg, info = spla.cg(A_csr, b, atol=1e-6, maxiter=1000)
time_cg = time.time() - start_cg
res_cg = np.linalg.norm(A_csr @ x_cg - b)

print(f"LU时间: {time_lu:.3f}s | 残差: {res_lu:.1e}")
print(f"SOR时间: {time_sor:.3f}s | 迭代: {sor_iters} | 最终残差: {sor_res:.1e}")
print(f"CG时间: {time_cg:.3f}s | 残差: {res_cg:.1e}")


def plot_solution(u, title):
    u_full = np.zeros((n + 1, n + 1))
    u_full[:, -1] = 1  # 顶边界u=1
    for j in range(m):
        for i in range(m):
            k = i + j * m
            u_full[i + 1, j + 1] = u[k]

    X, Y = np.meshgrid(np.linspace(0, 1, n + 1), np.linspace(0, 1, n + 1))

    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(X, Y, u_full.T, cmap='viridis')
    ax1.set_title(f'{title} 3D视图')

    ax2 = fig.add_subplot(122)
    cont = ax2.contourf(X, Y, u_full.T, levels=20, cmap='viridis')
    plt.colorbar(cont)
    ax2.set_title(f'{title} 等高线')
    plt.tight_layout()


# 绘制三种解法的结果
plot_solution(x_lu, "LU分解解")
plot_solution(x_sor, "SOR迭代解")
plot_solution(x_cg, "共轭梯度解")

# 绘制残差下降曲线
plt.figure()
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'cm'
plt.semilogy(res_hist, 'r-', label='SOR残差')
plt.axhline(1e-6, color='k', linestyle='--', label='收敛阈值')
plt.xlabel("迭代次数")
plt.ylabel("残差范数")
plt.title("SOR残差下降历史 (ω=1.9)")
plt.legend()
plt.grid()
plt.show()