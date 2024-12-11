import torch

# 数据类型转换
a = torch.tensor([1, 2, 3])
a_new = a.to(torch.int64)
print(a_new, a_new.dtype)

# 张量转置
t = torch.rand(3, 5)
t_tr = torch.transpose(t, 0, 1)
print(t.shape, t_tr.shape)

# 张量重塑
t = torch.zeros(30)
t_reshape = torch.reshape(t, (5, 6))
print(t.shape, t_reshape.shape)

# 删除不必要的维度
t = torch.zeros(1, 2, 1, 4, 1)
t_sqz = torch.squeeze(t, 2)
print(t.shape, t_sqz.shape)
print(t_sqz)

# 随机数种子
torch.manual_seed(1)
t1 = 2 * torch.rand(5, 2) - 1
t2 = torch.normal(0, 1, size=(5, 2))
# 元素乘积
t3 = torch.multiply(t1, t2)
print(t3)

# 均值
t4 = torch.mean(t3)
print(t4)

# 矩阵乘积
t5 = torch.matmul(t1, torch.transpose(t2, 0, 1))
print(t5)
t6 = torch.matmul(torch.transpose(t1, 0, 1), t2)
print(t6)

# 计算范数
norm_t1 = torch.linalg.norm(t1, ord=2, dim=1)
print(norm_t1)

# 拆分张量
# 提供拆分数量
t = torch.rand(6)
t_split = torch.chunk(t, 3)
print([item.numpy() for item in t_split])
# 提供拆分后的大小
t = torch.rand(5)
t_split = torch.split(t, [3, 2])
print([item.numpy() for item in t_split])

# 连接张量
A = torch.ones(3)
B = torch.zeros(3)
C = torch.cat([A, B], dim=0)
print(C)

# 张量的堆叠
S = torch.stack([A, B], dim=0)
print(S)