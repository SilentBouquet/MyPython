import numpy as np


# 向前替换求解Ly = b
def forward_substitution(L, b):
    n = len(b)
    y = np.zeros_like(b, dtype=np.float64)
    for i in range(n):
        y[i] = b[i]
        for j in range(i):
            y[i] -= L[i, j] * y[j]
        y[i] /= L[i, i]
    return y


# 向后替换求解Ux = y
def backward_substitution(U, y):
    n = len(y)
    x = np.zeros_like(y, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        x[i] = y[i]
        for j in range(i + 1, n):
            x[i] -= U[i, j] * x[j]
        x[i] /= U[i, i]
    return x


# 基本LU分解 (Doolittle)
def lu_decomposition(A):
    n = A.shape[0]
    L = np.eye(n, dtype=np.float64)
    U = np.zeros_like(A, dtype=np.float64)

    for i in range(n):
        # 计算U的第i行
        for j in range(i, n):
            U[i, j] = A[i, j]
            for k in range(i):
                U[i, j] -= L[i, k] * U[k, j]

        # 计算L的第i列
        for j in range(i + 1, n):
            L[j, i] = A[j, i]
            for k in range(i):
                L[j, i] -= L[j, k] * U[k, i]
            L[j, i] /= U[i, i]

    return L, U


# 行选主元LU分解
def lu_decomposition_row_pivoting(A):
    n = A.shape[0]
    L = np.eye(n, dtype=np.float64)
    U = np.zeros_like(A, dtype=np.float64)
    P = np.eye(n, dtype=np.float64)  # 行置换矩阵
    A_copy = A.copy().astype(np.float64)

    for i in range(n):
        # 选主元
        max_row = np.argmax(np.abs(A_copy[i:, i])) + i
        if max_row != i:
            A_copy[[i, max_row]] = A_copy[[max_row, i]]
            P[[i, max_row]] = P[[max_row, i]]
            if i > 0:
                L[[i, max_row], :i] = L[[max_row, i], :i]

        # 计算U行和L列
        for j in range(i, n):
            U[i, j] = A_copy[i, j]
            for k in range(i):
                U[i, j] -= L[i, k] * U[k, j]

        for j in range(i + 1, n):
            L[j, i] = A_copy[j, i]
            for k in range(i):
                L[j, i] -= L[j, k] * U[k, i]
            L[j, i] /= U[i, i]

    return L, U, P


# 列选主元LU分解
def lu_decomposition_column_pivoting(A):
    n = A.shape[0]
    L = np.eye(n, dtype=np.float64)
    U = np.zeros_like(A, dtype=np.float64)
    P = np.eye(n, dtype=np.float64)  # 列置换矩阵
    A_copy = A.copy().astype(np.float64)

    for i in range(n):
        # 选主元
        max_col = np.argmax(np.abs(A_copy[i, i:])) + i
        if max_col != i:
            A_copy[:, [i, max_col]] = A_copy[:, [max_col, i]]
            P[:, [i, max_col]] = P[:, [max_col, i]]

        # 计算U行和L列
        for j in range(i, n):
            U[i, j] = A_copy[i, j]
            for k in range(i):
                U[i, j] -= L[i, k] * U[k, j]

        for j in range(i + 1, n):
            L[j, i] = A_copy[j, i]
            for k in range(i):
                L[j, i] -= L[j, k] * U[k, i]
            L[j, i] /= U[i, i]

    return L, U, P


# Cholesky分解（对称正定矩阵）
def cholesky_decomposition(A):
    n = A.shape[0]
    L = np.zeros_like(A, dtype=np.float64)

    for i in range(n):
        # 对角线元素
        sum_val = A[i, i]
        for k in range(i):
            sum_val -= L[i, k] ** 2
        L[i, i] = np.sqrt(sum_val)

        # 下三角元素
        for j in range(i + 1, n):
            sum_val = A[j, i]
            for k in range(i):
                sum_val -= L[j, k] * L[i, k]
            L[j, i] = sum_val / L[i, i]

    return L


# 求解函数
def solve_with_lu(A, b):
    L, U = lu_decomposition(A)
    y = forward_substitution(L, b)
    x = backward_substitution(U, y)
    return x


def solve_with_lu_row_pivoting(A, b):
    L, U, P = lu_decomposition_row_pivoting(A)
    y = forward_substitution(L, P @ b)
    x = backward_substitution(U, y)
    return x


def solve_with_lu_column_pivoting(A, b):
    L, U, P = lu_decomposition_column_pivoting(A)
    y = forward_substitution(L, b)
    x_permuted = backward_substitution(U, y)
    return P @ x_permuted


def solve_with_cholesky(A, b):
    L = cholesky_decomposition(A)
    y = forward_substitution(L, b)
    x = backward_substitution(L.T, y)
    return x


# 验证解
def verify(A, x, b, case):
    error = np.linalg.norm(A @ x - b)
    print(f"Case {case}: 解 = {x.round(6)}, 误差 = {error:.6e}")


# 案例
def case1_6():
    A = np.array([[1, 2, 3, -4], [-3, -4, -12, 13], [2, 10, 0, -3], [4, 14, 9, -13]], dtype=np.float64)
    b = np.array([-2, 5, 10, 7], dtype=np.float64)
    x = solve_with_lu(A, b)
    verify(A, x, b, "1.6")


def case1_7():
    A = np.array([[6, 2, 1, -1], [2, 4, 1, 0], [1, 1, 4, -1], [-1, 0, -1, 3]], dtype=np.float64)
    b = np.array([6, -1, 5, -5], dtype=np.float64)
    x = solve_with_lu(A, b)
    verify(A, x, b, "1.7")


def case1_8():
    A = np.array([[2, 1, 1, 0], [4, 3, 3, 1], [8, 7, 9, 5], [6, 7, 9, 8]], dtype=np.float64)
    b = np.array([4, 11, 29, 30], dtype=np.float64)
    x = solve_with_lu_row_pivoting(A, b)
    verify(A, x, b, "1.8")


def case1_9():
    A = np.array([[1, 2, 3], [3, 1, 5], [2, 5, 2]], dtype=np.float64)
    b = np.array([14, 20, 18], dtype=np.float64)
    x = solve_with_lu_row_pivoting(A, b)
    verify(A, x, b, "1.9")


def case1_10():
    A = np.array([[0.3e-15, 59.14, 3, 1], [5.291, -6.13, -1, 2], [11.2, 9, 5, 2], [1, 2, 1, 1]], dtype=np.float64)
    b = np.array([59.17, 46.78, 1, 2], dtype=np.float64)

    x1 = solve_with_lu(A, b)
    print(f"Case 1.10 (不选主元): 解 = {x1.round(6)}, 误差 = {np.linalg.norm(A @ x1 - b):.6e}")

    x2 = solve_with_lu_column_pivoting(A, b)
    print(f"Case 1.10 (列选主元): 解 = {x2.round(6)}, 误差 = {np.linalg.norm(A @ x2 - b):.6e}")


def case1_11():
    A = np.array([[4, 2, 8, 0], [2, 10, 10, 9], [8, 10, 21, 6], [0, 9, 6, 34]], dtype=np.float64)
    b = np.array([14, 31, 45, 49], dtype=np.float64)
    x = solve_with_cholesky(A, b)
    verify(A, x, b, "1.11")


# 主函数
if __name__ == "__main__":
    case1_6()
    case1_7()
    case1_8()
    case1_9()
    case1_10()
    case1_11()