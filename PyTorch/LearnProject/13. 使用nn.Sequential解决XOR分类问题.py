import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

model = nn.Sequential(
    nn.Linear(4, 16),
    nn.ReLU(),
    nn.Linear(16, 32),
    nn.ReLU()
)

# 初始化第一个全连接层
nn.init.xavier_uniform_(model[0].weight)
# 对第二个全连接层权重进行L1正则化
l1_weight = 0.01
l1_penalty = l1_weight * model[2].weight.abs().sum()

torch.manual_seed(0)
np.random.seed(0)
x = np.random.uniform(-1, 1, (200, 2))
y = np.ones(len(x))
y[x[:, 0] * x[:, 1] < 0] = 0
n_train = 100
x_train = torch.tensor(x[:n_train, :], dtype=torch.float32)
y_train = torch.tensor(y[:n_train], dtype=torch.float32)
x_valid = torch.tensor(x[n_train:, :], dtype=torch.float32)
y_valid = torch.tensor(y[n_train:], dtype=torch.float32)

fig = plt.figure(figsize=(6, 6))
plt.plot(x[y == 0, 0], x[y == 0, 1], 'o', alpha=0.75, markersize=10)
plt.plot(x[y == 1, 0], x[y == 1, 1], '<', alpha=0.75, markersize=10)
plt.xlabel(r'$x_1$', size=12)
plt.ylabel(r'$x_2$', size=12)
plt.show()