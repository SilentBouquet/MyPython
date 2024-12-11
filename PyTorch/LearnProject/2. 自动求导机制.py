import torch

x = torch.randn(3, 4, requires_grad=True)
y = torch.randn(3, 4, requires_grad=True)
z = x + y
t = z.sum()
print(t)
print(t.backward())
print(y.grad)
print(x.requires_grad, y.requires_grad, z.requires_grad)

# 反向传播计算
x = torch.randn(1)
b = torch.randn(1, requires_grad=True)
w = torch.randn(1, requires_grad=True)
y = w * x
z = y + b
print(x.requires_grad, b.requires_grad, w.requires_grad, y.requires_grad, z.requires_grad, z.requires_grad)
print(x.is_leaf, w.is_leaf, b.is_leaf, y.is_leaf, z.is_leaf)
z.backward(retain_graph=True)
print(w.grad)
print(b.grad)