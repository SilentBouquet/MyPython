import torch
from torch import tensor

# 1. Scalar：标量，通常就是一个数值
x = tensor(42.)
print(x.item(), x.dim())

# 2. Vector：向量，通常指特征
v = tensor([1, 2, 3])
print(v, v.dim(), v.size())

# 3. Matrix：一般计算的都是矩阵，通常是多维的
M = tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
print(M, M.dim(), M.size())