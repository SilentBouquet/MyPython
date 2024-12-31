import torch
import torch.nn as nn

# 定义BCE损失
loss_func = nn.BCELoss()    # 二元交叉熵损失函数，适用于二分类问题
loss = loss_func(torch.tensor([0.9]), torch.tensor([1.0]))
print("BCE Loss: ", loss)

#  L2 正则化的系数，通常是一个小的正数，它决定了正则化项对总损失的影响程度
l2_lamda = 0.001

# 定义卷积层
# kernel_size为卷积核的大小
conv_layer = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=5)
# torch.norm(p, p=2) 计算张量的 L2 范数，平方后，它等价于 (p ** 2).sum()
l2_penalty_conv = l2_lamda * sum([torch.norm(p, p=2) ** 2 for p in conv_layer.parameters()])
loss_with_conv_penalty = loss + l2_penalty_conv
print("Conv Loss with L2 Penalty:", loss_with_conv_penalty)

# 定义全连接层
linear_layer = nn.Linear(10, 16)
# 计算全连接层的 L2 正则化项
l2_penalty_linear = l2_lamda * sum([torch.norm(p, p=2) ** 2 for p in linear_layer.parameters()])
# 计算带正则化的总损失
loss_with_linear_penalty = loss + l2_penalty_linear
print("Linear Loss with L2 Penalty:", loss_with_linear_penalty)