import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

X_train = np.arange(10, dtype=np.float32).reshape(10, 1)
y_train = np.array([1.0, 1.3, 3.1, 2.0, 5.0, 6.3, 6.6, 7.4, 8.0, 9.0], dtype=np.float32)

X_train_norm = (X_train - np.mean(X_train)) / np.std(X_train)
X_train_norm = torch.from_numpy(X_train_norm)
y_train = torch.from_numpy(y_train).float()
# TensorDataset用于将输入数据和目标数据（标签）组合在一起
train_ds = TensorDataset(X_train_norm, y_train)
batch_size = 1
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

torch.manual_seed(1)
weight = torch.randn(1)
# 这里将weight的requires_grad属性设置为True。这意味着在反向传播过程中，PyTorch将会计算weight的梯度
weight.requires_grad_()
bias = torch.zeros(1, requires_grad=True)

loss = 0
learning_rate = 0.001
num_epochs = 200
log_epochs = 10
loss_func = nn.MSELoss(reduction='mean')
input_size = 1
output_size = 1
model = nn.Linear(input_size, output_size)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs+1):
    for X_batch, y_batch in train_loader:
        pred = model(X_batch)[:, 0]
        loss = loss_func(pred, y_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    if epoch % log_epochs == 0:
        print(f'Epoch {epoch}:  Loss {loss.item():.4f}')
print('Final Parameters:', model.weight.item(), model.bias.item())

X_test = np.linspace(0, 9, num=100, dtype=np.float32).reshape(-1, 1)
X_test_norm = (X_test - np.mean(X_train)) / np.std(X_train)
X_test_norm = torch.from_numpy(X_test_norm)
# detach()将y_pred张量从当前计算图中分离出来。这意味着y_pred将不再需要梯度，也不会在反向传播中被计算梯度
y_pred = model(X_test_norm).detach().numpy()

fig = plt.figure(figsize=(13, 6))
plt.plot(X_train_norm, y_train, 'o', markersize=10)
plt.plot(X_test_norm, y_pred, '--', lw=3)
plt.legend(['Training example', 'Linear regression'], fontsize=15)
plt.xlabel('x', size=15)
plt.ylabel('y', size=15)
# tick_params是用于控制刻度线和刻度标签的外观的函数
# axis='both'表示同时设置x轴和y轴的刻度参数
# which='major'指定了要设置的刻度类型。'major'表示主要刻度，即主刻度线和标签
plt.tick_params(axis='both', which='major', labelsize=15)
plt.show()