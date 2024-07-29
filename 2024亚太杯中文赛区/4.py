import neal
import numpy as np
from pyqubo import Binary

D1 = {
    1: "上海",
    2: "西安",
    3: "昆明",
    4: "深圳",
    5: "天津",
    6: "郑州",
}

W12 = [[0, 28500, 54000, 36500, 28500, 18500],
       [28500, 0, 38500, 40500, 19000, 8500],
       [54000, 38500, 0, 17500, 61500, 47000],
       [36500, 40500, 17500, 0, 50500, 37500],
       [28500, 19000, 61500, 50500, 0, 10500],
       [18500, 8500, 47000, 37500, 10500, 0]]

W5 = [[0, 20000, 37000, 26000, 18000, 12000],
      [20000, 0, 28000, 30000, 10000, 4000],
      [37000, 28000, 0, 11000, 47000, 32000],
      [26000, 30000, 11000, 0, 38000, 27000],
      [18000, 10000, 47000, 38000, 0, 6000],
      [12000, 4000, 32000, 27000, 6000, 0]]

w11 = W12[0][2]
w12 = W12[0][3]
w13 = W12[0][4]
w21 = W12[1][2]
w22 = W12[1][3]
w23 = W12[1][4]
w31 = W12[5][2]
w32 = W12[5][3]
w33 = W12[5][4]

w11_ = W5[0][2]
w12_ = W5[0][3]
w13_ = W5[0][4]
w21_ = W5[1][2]
w22_ = W5[1][3]
w23_ = W5[1][4]
w31_ = W5[5][2]
w32_ = W5[5][3]
w33_ = W5[5][4]

w = [w11, w12, w13, w21, w22, w23, w31, w32, w33,
     w11, w12, w13, w21, w22, w23, w31, w32, w33,
     w11, w12, w13, w21, w22, w23, w31, w32, w33,
     w11, w12, w13, w21, w22, w23, w31, w32, w33,
     w11_, w12_, w13_, w21_, w22_, w23_, w31_, w32_, w33_]

# 定义二进制的自变量
# a类表示上海到其他城市是否选择12t的运输方式，a_则是第二次是否选择，a1则是上海到昆明
# b类表示西安到其他城市是否选择12t的运输方式，b_则是第二次是否选择，b1则是西安到昆明
# c类表示郑州到其他城市是否选择12t的运输方式，c_则是第二次是否选择，c1则是郑州到昆明
# d类表示上海到其他城市是否选择5t的运输方式，d1则是上海到昆明
# e类表示西安到其他城市是否选择5t的运输方式，e1则是西安到昆明
# f类表示郑州到其他城市是否选择5t的运输方式，f1则是郑州到昆明
a1, a2, a3, b1, b2, b3, c1, c2, c3 = (Binary('a1'), Binary('a2'), Binary('a3'), Binary('b1'),
                                      Binary('b2'), Binary('b3'), Binary('c1'), Binary('c2'), Binary('c3'))
a1_, a2_, a3_, b1_, b2_, b3_, c1_, c2_, c3_ = (Binary('a1_'), Binary('a2_'), Binary('a3_'), Binary('b1_'),
                                               Binary('b2_'), Binary('b3_'), Binary('c1_'), Binary('c2_'),
                                               Binary('c3_'))
a1_2, a2_2, a3_2, b1_2, b2_2, b3_2, c1_2, c2_2, c3_2 = (Binary('a1_2'), Binary('a2_2'), Binary('a3_2'), Binary('b1_2'),
                                                        Binary('b2_2'), Binary('b3_2'), Binary('c1_2'), Binary('c2_2'),
                                                        Binary('c3_2'))
a1_3, a2_3, a3_3, b1_3, b2_3, b3_3, c1_3, c2_3, c3_3 = (Binary('a1_3'), Binary('a2_3'), Binary('a3_3'), Binary('b1_3'),
                                                        Binary('b2_3'), Binary('b3_3'), Binary('c1_3'), Binary('c2_3'),
                                                        Binary('c3_3'))
d1, d2, d3, e1, e2, e3, f1, f2, f3 = (Binary('d1'), Binary('d2'), Binary('d3'), Binary('e1'),
                                      Binary('e2'), Binary('e3'), Binary('f1'), Binary('f2'), Binary('f3'))
x = [a1, a2, a3, b1, b2, b3, c1, c2, c3, a1_, a2_, a3_, b1_, b2_, b3_, c1_,
     c2_, c3_, a1_2, a2_2, a3_2, b1_2, b2_2, b3_2, c1_2,
     c2_2, c3_2, a1_3, a2_3, a3_3, b1_3, b2_3, b3_3, c1_3,
     c2_3, c3_3, d1, d2, d3, e1, e2, e3, f1, f2, f3]
x_ = ['a1', 'a2', 'a3', 'b1', 'b2', 'b3', 'c1', 'c2', 'c3', 'a1_', 'a2_', 'a3_', 'b1_', 'b2_', 'b3_', 'c1_',
      'c2_', 'c3_', 'a1_2', 'a2_2', 'a3_2', 'b1_2', 'b2_2', 'b3_2', 'c1_2',
      'c2_2', 'c3_2', 'a1_3', 'a2_3', 'a3_3', 'b1_3', 'b2_3', 'b3_3', 'c1_3',
      'c2_3', 'c3_3', 'd1', 'd2', 'd3', 'e1', 'e2', 'e3', 'f1', 'f2', 'f3']
# 定义罚函数
P1 = max(w11, w12, w13, w11_, w12_, w13_)
P2 = max(w21, w22, w23, w21_, w22_, w23_)
P3 = max(w31, w32, w33, w31_, w32_, w33_)
P4 = max(w11, w21, w31, w11_, w21_, w31_)
P5 = max(w12, w22, w32, w12_, w22_, w32_)
P6 = max(w13, w23, w33, w13_, w23_, w33_)
H1 = P1 * (33 - 12 * (a1 + a2 + a3 + a1_ + a2_ + a3_ + a1_2 + a2_2 + a3_2 + a1_3 + a2_3 + a3_3) - 5 * (
            d1 + d2 + d3)) ** 2
H2 = P2 * (44 - 12 * (b1 + b2 + b3 + b1_ + b2_ + b3_ + b1_2 + b2_2 + b3_2 + b1_3 + b2_3 + b3_3) - 5 * (
            e1 + e2 + e3)) ** 2
H3 = P3 * (30 - 12 * (c1 + c2 + c3 + c1_ + c2_ + c3_ + c1_2 + c2_2 + c3_2 + c1_3 + c2_3 + c3_3) - 5 * (
            f1 + f2 + f3)) ** 2
H4 = P4 * (46 - 12 * (a1 + a1_ + a1_2 + a1_3 + b1 + b1_ + b1_2 + b1_3 + c1 + c1_ + c1_2 + c1_3) - 5 * (
            d1 + e1 + f1)) ** 2
H5 = P5 * (35 - 12 * (a2 + a2_ + a2_2 + a2_3 + b2 + b2_ + b2_2 + b2_3 + c2 + c2_ + c2_2 + c2_3) - 5 * (
            d2 + e2 + c2)) ** 2
H6 = P6 * (26 - 12 * (a3 + a3_ + a3_2 + a3_3 + b3 + b3_ + b3_2 + b3_3 + c3 + c3_ + c3_2 + c3_3) - 5 * (
            d3 + e3 + c3)) ** 2

H = [H1, H2, H3, H4, H5, H6]

# 定义哈密顿算符W
W = 0 * a1
for i in range(0, len(w)):
    s = w[i] * x[i] ** 2
    W += s
for j in range(0, len(H)):
    W += H[j]
model = W.compile()


def weight(w, x):
    arr = 0
    for i in range(0, len(x)):
        arr += w[i] * x[i]
    return arr


# 无视offset就行了，QUBO用来做模拟退火的输入
solvers = []
w_arr = []
QUBO = []
Dic = []
for i in range(0, 1000):
    solve = []
    qubo, offset = model.to_qubo()
    sampler = neal.SimulatedAnnealingSampler()
    raw_solution = sampler.sample_qubo(qubo)
    dic = raw_solution.first.sample
    for j in range(0, len(x)):
        solve.append(dic[x_[j]])
    w_ = weight(w, solve)
    w_arr.append(w_)
    solvers.append(solve)
    QUBO.append(qubo)
    Dic.append(dic)
minw = w_arr[0]
for i in range(0, len(w_arr)):
    if w_arr[i] < minw:
        minw = w_arr[i]
location = w_arr.index(minw)
group = solvers[location]
d = Dic[location]
print(minw)
print(group)
print(d)
Q = QUBO[location]

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