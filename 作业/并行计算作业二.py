import numpy as np
import sympy as sp


# 案例1.16的雅可比和SOR迭代法求解
def case1_16():
    A = np.array([[4, 0, -2],
                 [0, 3, 0],
                 [-2, 0, 3]], dtype=float)
    b = np.array([-6, 3, 5], dtype=float)
    x0 = np.zeros(3)
    tol = 1e-3

    # 雅可比迭代法
    D = np.diag(np.diag(A))
    L = -np.tril(A, -1)
    U = -np.triu(A, 1)
    J = np.linalg.inv(D) @ (L + U)
    D_inv_b = np.linalg.solve(D, b)

    x = x0.copy()
    iteration_jacobi = 0
    while np.linalg.norm(b - A @ x) > tol:
        x = J @ x + D_inv_b
        iteration_jacobi += 1

    print(f"案例1.16雅可比迭代法收敛，迭代次数：{iteration_jacobi}")
    print(f"解为：{x}\n")

    # SOR方法
    def sor(A, b, omega, x0, tol, max_iter=1000):
        x = x0.copy()
        n = len(b)
        residual_norm = np.linalg.norm(b - A @ x)
        iteration = 0
        while residual_norm > tol and iteration < max_iter:
            for i in range(n):
                sigma = 0.0
                for j in range(i):
                    sigma += A[i, j] * x[j]
                for j in range(i + 1, n):
                    sigma += A[i, j] * x[j]
                x[i] = (1 - omega) * x[i] + omega * (b[i] - sigma) / A[i, i]
            residual_norm = np.linalg.norm(b - A @ x)
            iteration += 1
        return x, iteration

    omegas = [1.0, 1.2, 1.5]
    for omega in omegas:
        x_sor, iter_sor = sor(A, b, omega, x0, tol)
        print(f"案例1.16 SOR方法(ω={omega})收敛，迭代次数：{iter_sor}")
        print(f"解为：{x_sor}\n")


# 案例1.17的雅可比和SOR迭代法求解
def case1_17():
    t_value = 1.0  # 选择一个具体的t值进行测试
    A = np.array([[3, 7, 1],
                 [0, 4, t_value + 1],
                 [0, -t_value + 1, -1]], dtype=float)
    b = np.array([1, 1, 0], dtype=float)
    x0 = np.zeros(3)
    tol = 1e-3

    # 雅可比迭代法
    D = np.diag(np.diag(A))
    L = -np.tril(A, -1)
    U = -np.triu(A, 1)
    J = np.linalg.inv(D) @ (L + U)
    D_inv_b = np.linalg.solve(D, b)

    x = x0.copy()
    iteration_jacobi = 0
    while np.linalg.norm(b - A @ x) > tol:
        x = J @ x + D_inv_b
        iteration_jacobi += 1

    print(f"案例1.17雅可比迭代法收敛，迭代次数：{iteration_jacobi}")
    print(f"解为：{x}\n")

    # SOR方法
    def sor(A, b, omega, x0, tol, max_iter=1000):
        x = x0.copy()
        n = len(b)
        residual_norm = np.linalg.norm(b - A @ x)
        iteration = 0
        while residual_norm > tol and iteration < max_iter:
            for i in range(n):
                sigma = 0.0
                for j in range(i):
                    sigma += A[i, j] * x[j]
                for j in range(i + 1, n):
                    sigma += A[i, j] * x[j]
                x[i] = (1 - omega) * x[i] + omega * (b[i] - sigma) / A[i, i]
            residual_norm = np.linalg.norm(b - A @ x)
            iteration += 1
        return x, iteration

    omegas = [1.0, 1.2, 1.5]
    for omega in omegas:
        x_sor, iter_sor = sor(A, b, omega, x0, tol)
        print(f"案例1.17 SOR方法(ω={omega})收敛，迭代次数：{iter_sor}")
        print(f"解为：{x_sor}\n")

    # 收敛条件分析
    t = sp.symbols('t', real=True)
    J = sp.Matrix([
        [0, 7/3, 1/3],
        [0, 0, (t + 1)/4],
        [0, t - 1, 0]
    ])

    eigenvalues = J.eigenvals()
    max_abs_eigenvalue = max([sp.Abs(eig) for eig in eigenvalues.keys()])
    convergence_condition = sp.solve(max_abs_eigenvalue < 1, t)
    print(f"案例1.17雅可比迭代法收敛条件：{convergence_condition}")


# 案例1.20的雅可比和SOR迭代法求解
def case1_20():
    t_value = 2.0  # 选择一个具体的t值进行测试
    A = np.array([[t_value, 1, 1],
                 [1/t_value, t_value, 0],
                 [1/t_value, 0, t_value]], dtype=float)
    b = np.array([0, 1, 2], dtype=float)
    x0 = np.zeros(3)
    tol = 1e-3

    # 雅可比迭代法
    D = np.diag(np.diag(A))
    L = -np.tril(A, -1)
    U = -np.triu(A, 1)
    J = np.linalg.inv(D) @ (L + U)
    D_inv_b = np.linalg.solve(D, b)

    x = x0.copy()
    iteration_jacobi = 0
    while np.linalg.norm(b - A @ x) > tol:
        x = J @ x + D_inv_b
        iteration_jacobi += 1

    print(f"案例1.20雅可比迭代法收敛，迭代次数：{iteration_jacobi}")
    print(f"解为：{x}\n")

    # SOR方法
    def sor(A, b, omega, x0, tol, max_iter=1000):
        x = x0.copy()
        n = len(b)
        residual_norm = np.linalg.norm(b - A @ x)
        iteration = 0
        while residual_norm > tol and iteration < max_iter:
            for i in range(n):
                sigma = 0.0
                for j in range(i):
                    sigma += A[i, j] * x[j]
                for j in range(i + 1, n):
                    sigma += A[i, j] * x[j]
                x[i] = (1 - omega) * x[i] + omega * (b[i] - sigma) / A[i, i]
            residual_norm = np.linalg.norm(b - A @ x)
            iteration += 1
        return x, iteration

    omegas = [1.0, 1.2, 1.5]
    for omega in omegas:
        x_sor, iter_sor = sor(A, b, omega, x0, tol)
        print(f"案例1.20 SOR方法(ω={omega})收敛，迭代次数：{iter_sor}")
        print(f"解为：{x_sor}\n")

    # 收敛条件分析
    t = sp.symbols('t', real=True)
    M_gs = sp.Matrix([
        [0, 1/t, 1/t],
        [0, 0, 0],
        [0, 0, 0]
    ])

    eigenvalues_gs = M_gs.eigenvals()
    max_abs_eigenvalue_gs = max([sp.Abs(eig) for eig in eigenvalues_gs.keys()], default=0)
    convergence_condition_gs = sp.solve(max_abs_eigenvalue_gs < 1, t)
    print(f"案例1.20 GS迭代法收敛条件：当 t > 0 时，GS迭代法收敛。")


def main():
    print("案例1.16的求解结果：")
    case1_16()
    print("案例1.17的求解结果：")
    case1_17()
    print("案例1.20的求解结果：")
    case1_20()


if __name__ == "__main__":
    main()