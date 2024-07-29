import neal
import numpy as np
from pyqubo import Binary

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
x1, x2, x3, x4, x5 = Binary('x1'), Binary('x2'), Binary('x3'), Binary('x4'), Binary('x5')
x = [x1, x2, x3, x4, x5]
x_ = ['x1', 'x2', 'x3', 'x4', 'x5']
max_sigma = 0
sum = 0
P1 = 10
P2 = 5
s = 0
for k in range(0, len(mu)):
    s += mu[k] * x[k]
H1 = P1 * (x1 + x2 + x3 + x4 + x5 - 3) ** 2
H2 = P2 * (0.12 - s) ** 2

H = 0.5 * sum - s + P1 * (x1 + x2 + x3 + x4 + x5 - 3) ** 2 + P2 * (0.12 - s) ** 2
model = H.compile()


def weight(sigma, mu, x):
    arr = 0
    for i in range(0, len(sigma)):
        arr += mu[i] * x[i]
        for j in range(0, len(sigma)):
            arr -= 0.5 * sigma[i][j] * x[i] * x[j]
    return arr


solvers = []
w_arr = []
QUBO = []
Dic = []
for i in range(0, 1000):
    solve = []
    su = 0
    qubo, offset = model.to_qubo()
    sampler = neal.SimulatedAnnealingSampler()
    raw_solution = sampler.sample_qubo(qubo)
    dic = raw_solution.first.sample
    for j in range(0, len(x)):
        answer = dic[x_[j]]
        su += answer
        solve.append(answer)
    if su > 3:
        continue
    w_ = weight(sigma, mu, solve)
    w_arr.append(w_)
    solvers.append(solve)
    QUBO.append(qubo)
    Dic.append(dic)
maxw = w_arr[0]
for i in range(0, len(w_arr)):
    if w_arr[i] > maxw:
        maxw = w_arr[i]
location = w_arr.index(maxw)
group = solvers[location]
d = Dic[location]
print(maxw)
print(group)
print(d)
Q = QUBO[location]
print(Q)

num = len(x_)
qubo_matrix = np.zeros((num, num))
for i in range(0, num):
    for j in range(0, num):
        m = x_[i]
        n = x_[j]
        if (m, n) in Q:
            qubo_matrix[i, j] = Q[(m, n)]
        else:
            qubo_matrix[i, j] = 0
print(qubo_matrix)