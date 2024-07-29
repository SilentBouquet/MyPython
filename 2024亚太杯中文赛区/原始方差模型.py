import cvxpy as cp
import numpy as np

# 定义预期收益和协方差矩阵
mu = np.array([0.1, 0.2, 0.15, 0.05, 0.1])
sigma = np.array([
    [0.005, 0.002, 0.001, 0.000, 0.000],
    [0.002, 0.010, 0.003, 0.000, 0.000],
    [0.001, 0.003, 0.015, 0.000, 0.000],
    [0.000, 0.000, 0.000, 0.020, 0.001],
    [0.000, 0.000, 0.000, 0.001, 0.025]
])

# 定义变量
x = cp.Variable(5)

# 定义目标函数和约束条件
objective = cp.Minimize(cp.quad_form(x, sigma))
constraints = [cp.sum(x) == 1, mu @ x == 0.12, x >= 0]

# 求解问题
prob = cp.Problem(objective, constraints)
prob.solve()

# 输出结果
su = 0
lst = x.value
for i in range(0, len(lst)):
    weight = lst[i] * mu[i]
    su += weight
print("最优投资比例：", x.value)
print("最小化的风险：", prob.value)