import numpy as np
from scipy.linalg import lu, lu_factor, lu_solve, cholesky

# 案例 1.6
A16 = np.array([[1, 2, 3, -4],
                [-3, -4, -12, 13],
                [2, 10, 0, -3],
                [4, 14, 9, -13]], dtype=float)
b16 = np.array([-2, 5, 10, 7])

# 案例 1.7
A17 = np.array([[6, 2, 1, -1],
                [2, 4, 1, 0],
                [1, 1, 4, -1],
                [-1, 0, -1, 3]], dtype=float)
b17 = np.array([6, -1, 5, -5])

# 案例 1.8
A18 = np.array([[2, 1, 1, 0],
                [4, 3, 3, 1],
                [8, 7, 9, 5],
                [6, 7, 9, 8]], dtype=float)
b18 = np.array([4, 11, 29, 30])

# 案例 1.9
A19 = np.array([[1, 2, 3],
                [3, 1, 5],
                [2, 5, 2]], dtype=float)
b19 = np.array([14, 20, 18])

# 案例 1.10
A110 = np.array([[0.3e-15, 59.14, 3, 1],
                 [5.291, -6.13, -1, 2],
                 [11.2, 9, 5, 2],
                 [1, 2, 1, 1]], dtype=float)
b110 = np.array([59.17, 46.78, 1, 2])

# 案例 1.11
A111 = np.array([[4, 2, 8, 0],
                 [2, 10, 10, 9],
                 [8, 10, 21, 6],
                 [0, 9, 6, 34]], dtype=float)
b111 = np.array([14, 31, 45, 49])

# LU 分解求解案例 1.6
P16, L16, U16 = lu(A16)
y16 = np.linalg.solve(L16, P16 @ b16)
x16 = np.linalg.solve(U16, y16)
print("案例 1.6 的解为：")
print(x16)
print()

# LU 分解求解案例 1.7
P17, L17, U17 = lu(A17)
y17 = np.linalg.solve(L17, P17 @ b17)
x17 = np.linalg.solve(U17, y17)
print("案例 1.7 的解为：")
print(x17)
print()

# 选主元 LU 分解求解案例 1.8
from scipy.linalg import lu_factor, lu_solve
lu18, piv18 = lu_factor(A18)
x18 = lu_solve((lu18, piv18), b18)
print("案例 1.8 的解为：")
print(x18)
print()

# 列选主元 LU 分解求解案例 1.9
lu19, piv19 = lu_factor(A19)
x19 = lu_solve((lu19, piv19), b19)
print("案例 1.9 的解为：")
print(x19)
print()

# 不选主元和列选主元 LU 分解求解案例 1.10
# 不选主元 LU 分解
P110, L110, U110 = lu(A110)
y110 = np.linalg.solve(L110, P110 @ b110)
x110_no_piv = np.linalg.solve(U110, y110)
print("案例 1.10 不选主元的解为：")
print(x110_no_piv)
print()

# 列选主元 LU 分解
lu110, piv110 = lu_factor(A110)
x110_piv = lu_solve((lu110, piv110), b110)
print("案例 1.10 列选主元的解为：")
print(x110_piv)
print()

# Cholesky 分解求解案例 1.11
L111 = cholesky(A111, lower=True)
y111 = np.linalg.solve(L111, b111)
x111 = np.linalg.solve(L111.T, y111)
print("案例 1.11 的解为：")
print(x111)
print()