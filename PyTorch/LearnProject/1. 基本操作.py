import torch
import numpy as np

# 查看torch的版本
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.current_device())
print(torch.cuda.device_count())

# 创建一个矩阵
x = torch.empty(5, 3)
print(x)
# 创建一个随机矩阵
y = torch.rand(5, 3)
print(y)
# 创建一个全零矩阵
z = torch.zeros(5, 3, dtype=torch.long)
print(z)
# 直接传入数据
a = torch.tensor([1, 2, 3], dtype=torch.long)
print(a)
# 创建一个全一矩阵
b = x.new_ones(5, 3, dtype=torch.double)
print(b)
# 创建和b格式一样的随机矩阵
c = torch.rand_like(b, dtype=torch.float)
print(c)
# 打印矩阵大小
print(c.size())

# 基本计算方法
# 加法
x = torch.rand(5, 3, dtype=torch.double)
y = torch.rand(5, 3, dtype=torch.double)
print(x, '\n', y)
print(x + y)
print(torch.add(x, y))
# 索引
print(x[:, 1])
# 改变矩阵维度
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)   # -1代表自动计算的意思
print(x.size(), y.size(), z.size())

# 与Numpy的协同操作
a = torch.ones(3, 3)
b = a.numpy()
print(a, '\n', b, '\n', type(b))

a = np.ones((4, 4))
b = torch.from_numpy(a)
print(a, '\n', b, '\n', type(b))