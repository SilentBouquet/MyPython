import numpy as np
from scipy.optimize import minimize


class MultiVariableOptimizer:
    def __init__(self, objective_func, bounds, constraints, initial_guess):
        """
        参数:
        objective_func: 目标函数，需要最小化。
        bounds: 变量的边界，格式为 [(min1, max1), (min2, max2), ...]。
        constraints: 约束条件，格式为 [{'type': 'ineq', 'fun': constraint_func1}, {'type': 'eq', 'fun': constraint_func2}, ...]。
        initial_guess: 初始猜测值，格式为 [value1, value2, ...]。
        """
        self.objective_func = objective_func
        self.bounds = bounds
        self.constraints = constraints
        self.initial_guess = initial_guess

    def optimize(self):
        result = minimize(self.objective_func, self.initial_guess, bounds=self.bounds, constraints=self.constraints)
        return result


if __name__ == "__main__":
    a1, a2, a3, a4 = 0.6, 0.2, 0.1, 0.1
    w1, w2, w3 = 0.5, 0.2, 0.3
    n1, n2 = 0.5, 0.5
    A = 3
    r = 0.15
    sigma = -0.05
    alpha = 0.5
    P = 1.5
    N_max = 3
    U_min = 0.5
    Y_min = 0.5

    # 定义目标函数
    def objective_func(x):
        return -((a1 + a3 * w1) * (P + r * (2 - 2 * x) - alpha) * 2 * x
                 + (a2 - a3 * w3 + a4 * n1) * A * 2 * x ** sigma - (a3 * w2 + a4 * n2) * 2 * x)

    # 定义约束条件
    def constraint1(x):
        return w1 * (P + r * (P - 2 * x) - alpha) * 2 * x \
            - w2 * 2 * x + w3 * A * 2 * x ** sigma - U_min


    def constraint2(x):
        return P + r * (P - 2 * x) - alpha


    def constraint3(x):
        return A * 2 * x ** sigma - Y_min


    # 变量边界
    bounds = [(0, N_max)]

    # 约束条件列表
    constraints = [{'type': 'ineq', 'fun': constraint1},
                   {'type': 'ineq', 'fun': constraint2}]

    # 初始猜测值
    initial_guess = 1

    # 创建优化器实例
    optimizer = MultiVariableOptimizer(objective_func, bounds, constraints, initial_guess)

    # 执行优化
    result = optimizer.optimize()

    # 输出结果
    print("优化结果：", result)
    print("最优值：", result.fun)
    print("最优解：", result.x)