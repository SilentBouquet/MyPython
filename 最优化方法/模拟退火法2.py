import numpy as np


def C(s):
    return 1 / (s * np.sin(s) + 12)


# 初始化
# 设定初始温度
# 函数原型：numpy.random.uniform(low,high,size)
# 从一个均匀分布[low,high)中随机采样，注意定义域是左闭右开，即包含low，不包含high.
t0 = np.var(np.random.uniform(0, 12.55, 100))

# 设定初始解
s0 = np.random.uniform(0, 12.55, 1)

# 设定迭代次数
iters = 3000

# 设定终止条件，连续ct个新解都没有接受（ct/cn大于某个值）时终止算法
ct = 200
ct_array = []

# 保存历史最好的状态，默认取上边界值
best = 12.55

for t in range(1, iters + 1):
    # 在s0附近，产生新解，但又能包含定义内的所有值
    s1 = np.random.normal(s0, 2, 1)
    while s1 < 0 or s1 > 12.55:
        s1 = np.random.normal(s0, 2, 1)
    # 计算能量增量
    delta_t = C(s1) - C(s0)
    if delta_t < 0:
        s0 = s1
        ct_array.append(1)
    else:
        p = np.exp(-delta_t / t0)
        if np.random.uniform(0, 1) < p:
            s0 = s1
            ct_array.append(1)
        else:
            ct_array.append(0)

    best = s0 if C(s0) < C(best) else best

    # 更新温度
    t0 = t0 / np.log(1 + t)

    # 检查终止条件
    if len(ct_array) > ct and np.sum(ct_array[-ct:]) == 0:
        print("迭代 ", t, " 次，连续 ", ct, " 次没有接受新解，算法终止！")
        break

# 状态最终停留位置
print(s0)
# 最佳状态，即对应最优解的状态
print(best)