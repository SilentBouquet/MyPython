import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


# 案例 1.32: 实用幂法计算主特征值
def case_1_32():
    # 定义矩阵
    A = np.array([[2, -1, 0],
                  [-1, 2, -1],
                  [0, -1, 2]], dtype=float)

    # 初始化向量
    v = np.array([1, 1, 1], dtype=float)
    v = v / np.linalg.norm(v)

    # 幂法迭代
    tolerance = 1e-6
    max_iterations = 1000
    lambda_old = 0

    for _ in range(max_iterations):
        # 计算 Av
        w = np.dot(A, v)
        # 计算 Rayleigh 商 (特征值估计)
        lambda_new = np.dot(v, w)
        # 更新特征向量
        v = w / np.linalg.norm(w)

        # 检查收敛性
        if abs(lambda_new - lambda_old) < tolerance:
            break
        lambda_old = lambda_new

    print("案例1.32结果:")
    print("主特征值 λ1 ≈", lambda_new)
    print("主特征向量 x1 ≈", v)
    print()


# 案例 1.33: 实用幂法和瑞利商加速的实用幂法计算主特征值
def case_1_33():
    # 定义矩阵
    A = np.array([[3, 1],
                  [1, 3]], dtype=float)

    # 初始化向量
    v = np.array([0, 1], dtype=float)
    v = v / np.linalg.norm(v)

    # 幂法迭代
    tolerance = 1e-6
    max_iterations = 1000
    lambda_old = 0

    for _ in range(max_iterations):
        # 计算 Av
        w = np.dot(A, v)
        # 计算 Rayleigh 商 (特征值估计)
        lambda_new = np.dot(v, w)
        # 更新特征向量
        v = w / np.linalg.norm(w)

        # 检查收敛性
        if abs(lambda_new - lambda_old) < tolerance:
            break
        lambda_old = lambda_new

    print("案例1.33结果:")
    print("主特征值 λ1 ≈", lambda_new)
    print("主特征向量 x1 ≈", v)
    print()


# 案例 1.34: QR 算法计算所有特征值
def case_1_34():
    # 定义矩阵
    A = np.array([[2.9766, 0.3945, 0.4198, 1.1159],
                  [0.3945, 2.7328, -0.3097, 0.1129],
                  [0.4198, -0.3097, 2.5675, 0.6079],
                  [1.1159, 0.1129, 0.6079, 1.7231]], dtype=float)

    # 拷贝矩阵以避免修改原始矩阵
    H = A.copy()

    # 定义 QR 分解函数 (使用改进的方法)
    def qr_decomposition(H):
        m, n = H.shape
        Q = np.eye(m)
        R = H.copy()

        for i in range(n):
            # 计算 Householder 变换
            x = R[i:, i]
            e1 = np.zeros_like(x)
            e1[0] = 1
            # 选择反射向量
            v = np.zeros_like(x)
            v[0] = 1.0
            v = v + x / np.linalg.norm(x)
            v = v / np.linalg.norm(v)
            # 应用 Householder 变换
            H = np.eye(m)
            H[i:, i:] -= 2 * np.outer(v, v)
            Q = np.dot(Q, H)
            R = np.dot(H, R)
        return Q, R

    # 定义 QR 算法函数
    def qr_algorithm(H, tol=1e-6, max_iter=1000):
        m, n = H.shape
        eigenvalues = []
        for _ in range(max_iter):
            Q, R = qr_decomposition(H)
            H = np.dot(R, Q)
            # 检查是否收敛（子对角线元素是否足够小）
            converged = True
            for i in range(1, m):
                if abs(H[i, i-1]) > tol:
                    converged = False
                    break
            if converged:
                break
        # 提取特征值（对角线元素）
        eigenvalues = np.diag(H)
        return eigenvalues

    # 计算特征值
    eigenvalues = qr_algorithm(H)
    print("案例1.34结果:")
    print("矩阵的所有特征值依次为：")
    for i, val in enumerate(sorted(eigenvalues, reverse=True), start=1):
        print(f"λ{i} = {val}")
    print()


# 案例 1.35: 机械振动模型求解
def case_1_35():
    # 定义参数
    m = [2, 1, 1, 3]  # 质量
    b = [0.4, 0.4, 0.4, 0.4]  # 阻尼系数
    k = [1, 1, 1, 1]  # 弹簧常数
    n = 4  # 质量块数量

    # 初始条件
    x0 = [-1 / 4, 0, 0, 1 / 4]  # 初始位移
    v0 = [-1, 0, 0, 1]  # 初始速度
    y0 = x0 + v0  # 初始状态向量

    # 定义状态方程
    def model(y, t, m, b, k, n):
        x = y[:n]  # 位移
        v = y[n:]  # 速度
        dxdt = v.copy()  # 位移变化率
        dvdt = np.zeros(n)  # 速度变化率

        for i in range(n):
            if i == 0:
                dvdt[i] = (-b[i] * v[i] - k[i] * x[i] + k[i + 1] * (x[i + 1] - x[i])) / m[i]
            elif i == n - 1:
                dvdt[i] = (-b[i] * v[i] - k[i] * x[i] + k[i - 1] * (x[i - 1] - x[i])) / m[i]
            else:
                dvdt[i] = (-b[i] * v[i] - k[i] * x[i] + k[i - 1] * (x[i - 1] - x[i]) + k[i + 1] * (x[i + 1] - x[i])) / \
                          m[i]

        return np.concatenate((dxdt, dvdt))

    # 定义时间范围
    t = np.linspace(0, 10, 1000)

    # 求解微分方程
    sol = odeint(model, y0, t, args=(m, b, k, n))

    # 绘制位移随时间变化
    plt.figure(figsize=(10, 6))
    for i in range(n):
        plt.plot(t, sol[:, i], label=f'位移 x{i + 1}(t)')
    plt.title('各弹簧的位移随时间的变化')
    plt.xlabel('时间 t')
    plt.ylabel('位移 x')
    plt.legend()
    plt.grid()
    plt.show()

    # 绘制速度随时间变化
    plt.figure(figsize=(10, 6))
    for i in range(n):
        plt.plot(t, sol[:, n + i], label=f'速度 v{i + 1}(t)')
    plt.title('各弹簧的速率随时间的变化')
    plt.xlabel('时间 t')
    plt.ylabel('速度 v')
    plt.legend()
    plt.grid()
    plt.show()


# 执行所有案例
if __name__ == "__main__":
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['mathtext.fontset'] = 'cm'
    case_1_32()
    case_1_33()
    case_1_34()
    case_1_35()