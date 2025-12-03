import numpy as np
from scipy.linalg import qr, inv, norm
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib import rcParams

plt.rcParams['font.sans-serif'] = ['SimHei']  # 正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号


# 案例1.32：实用幂法计算主特征值和特征向量
def case1_32():
    A = np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])
    u = np.array([1.0, 1.0, 1.0])
    lambda_true = 2 + np.sqrt(2)
    x_true = np.array([-np.sqrt(2) / 2, 1.0, -np.sqrt(2) / 2])

    max_iter, tol = 100, 1e-6
    lambda_est = 0.0
    for k in range(max_iter):
        v = A @ u
        max_v = np.max(np.abs(v))
        u_new = v / max_v
        if np.abs(max_v - lambda_est) < tol:
            break
        lambda_est, u = max_v, u_new

    x_err = norm(u - x_true / np.max(np.abs(x_true))) / norm(x_true)
    print("案例1.32：")
    print(f"  主特征值估计：{lambda_est:.6f}")
    print(f"  真实主特征值：{lambda_true:.6f}")
    print(f"  特征值误差：{np.abs(lambda_est - lambda_true):.6e}")
    print(f"  特征向量误差：{x_err:.6e}")


# 案例1.33：瑞利商加速的实用幂法和反幂法
def case1_33():
    A = np.array([[3, 1], [1, 3]])
    u = np.array([0.0, 1.0])
    lambda1_true, lambda2_true = 4.0, 2.0
    x1_true = np.array([1.0, 1.0])

    # 瑞利商加速幂法
    max_iter, tol = 20, 1e-6
    lambda_est = 0.0
    for k in range(max_iter):
        v = A @ u
        u_new = v / np.max(np.abs(v))
        rayleigh = (u.T @ v) / (u.T @ u)
        if np.abs(rayleigh - lambda_est) < tol:
            break
        lambda_est, u = rayleigh, u_new

    x_err = norm(u - x1_true / np.max(np.abs(x1_true))) / norm(x1_true)
    print("案例1.33：")
    print("  瑞利商加速结果：")
    print(f"    主特征值估计：{lambda_est:.6f}")
    print(f"    真实主特征值：{lambda1_true:.6f}")
    print(f"    特征值误差：{np.abs(lambda_est - lambda1_true):.6e}")
    print(f"    特征向量误差：{x_err:.6e}")

    # 反幂法（修正sigma值）
    sigma = 1.99  # 更接近真实值的移位
    A_shifted = A - sigma * np.eye(2)
    A_inv = inv(A_shifted)
    u_inv = np.array([0.0, 1.0])
    lambda_inv_est = 0.0
    for k in range(max_iter):
        v_inv = A_inv @ u_inv
        max_v_inv = np.max(np.abs(v_inv))
        u_inv = v_inv / max_v_inv
        lambda_inv_est_new = 1.0 / max_v_inv + sigma
        if np.abs(lambda_inv_est_new - lambda_inv_est) < tol:
            break
        lambda_inv_est = lambda_inv_est_new

    print("  反幂法结果：")
    print(f"    最小特征值估计：{lambda_inv_est:.6f}")
    print(f"    真实最小特征值：{lambda2_true:.6f}")
    print(f"    特征值误差：{np.abs(lambda_inv_est - lambda2_true):.6e}")


# 案例1.34：QR算法计算所有特征值
def case1_34():
    A = np.array([[2.9766, 0.3945, 0.4198, 1.1159],
                  [0.3945, 2.7328, -0.3097, 0.1129],
                  [0.4198, -0.3097, 2.5675, 0.6079],
                  [1.1159, 0.1129, 0.6079, 1.7231]])
    true_eigs = np.array([4.0, 3.0, 2.0, 1.0])

    max_iter, tol = 50, 1e-6
    A_copy = A.copy()
    for k in range(max_iter):
        Q, R = qr(A_copy)
        A_new = R @ Q
        if np.max(np.abs(A_new - A_copy)) < tol:
            break
        A_copy = A_new

    eigs_est = np.sort(np.diag(A_copy))
    true_sorted = np.sort(true_eigs)
    err = norm(eigs_est - true_sorted)
    print("案例1.34：")
    print(f"  估计特征值：{[round(e, 6) for e in eigs_est]}")
    print(f"  真实特征值：{true_sorted.tolist()}")
    print(f"  整体误差：{err:.6e}")


# 案例1.35：机械振动模型数值求解
def case1_35():
    M = np.diag([2, 1, 1, 3])
    B = np.diag([0.4, 0.4, 0.4, 0.4])
    K = np.diag([2, 2, 2, 1]) + np.diag([-1, -1, -1], 1) + np.diag([-1, -1, -1], -1)

    def ode(t, y):
        x, v = y[:4], y[4:]
        return np.concatenate([v, inv(M) @ (-B @ v + K @ x)])

    y0 = np.array([-0.25, 0.0, 0.0, 0.25, -1.0, 0.0, 0.0, 1.0])
    t_span = (0, 10)
    sol = solve_ivp(ode, t_span, y0, method='RK45', t_eval=[10])

    x_final = sol.y[:4, -1]
    v_final = sol.y[4:, -1]
    print("案例1.35：")
    print(f"  t=10 时位移：{[round(x, 6) for x in x_final]}")
    print(f"  t=10 时速率：{[round(v, 6) for v in v_final]}")


# 案例1.36：QR算法计算矩阵特征值并展示在二维平面
def construct_stokes_matrix(v, q):
    h = 1 / (q + 1)
    size_B = 2 * q
    B = (4 * v / h ** 2) * np.eye(size_B) - (v / h ** 2) * (np.eye(size_B, k=1) + np.eye(size_B, k=-1))
    E = np.eye(size_B)
    return np.block([[B, E.T], [-E, np.zeros((size_B, size_B))]])


def case1_36():
    vs, qs = [1, 0.01], [4, 8]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()  # 转为一维数组便于索引

    for idx, (v, q) in enumerate([(v, q) for v in vs for q in qs]):
        A = construct_stokes_matrix(v, q)
        max_iter, tol = 50, 1e-6
        A_copy = A.copy()
        for k in range(max_iter):
            Q, R = qr(A_copy)
            A_new = R @ Q
            if np.max(np.abs(A_new - A_copy)) < tol:
                break
            A_copy = A_new

        eigs_est = np.diag(A_copy)
        real_parts = np.real(eigs_est)
        imag_parts = np.imag(eigs_est)

        # 绘制特征值分布
        ax = axes[idx]
        ax.scatter(real_parts, imag_parts, c='b', alpha=0.7, s=50)
        ax.set_title(f'v={v}, q={q} 特征值分布')
        ax.set_xlabel('实部')
        ax.set_ylabel('虚部')
        ax.grid(True, linestyle='--', alpha=0.7)

        # 输出数值结果
        print(f"案例1.36(v={v}, q={q})：")
        print(f"  特征值实部范围：[{np.min(real_parts):.4f}, {np.max(real_parts):.4f}]")
        print(f"  虚部最大绝对值：{np.max(np.abs(imag_parts)):.4e}")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    case1_32()
    print()
    case1_33()
    print()
    case1_34()
    print()
    case1_35()
    print()
    case1_36()