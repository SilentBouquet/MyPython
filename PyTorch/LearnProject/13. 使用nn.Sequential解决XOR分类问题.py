import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

# 生成数据并拆分为训练数据和验证数据
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

# 绘制散点图
fig = plt.figure(figsize=(6, 6))
plt.plot(x[y == 0, 0], x[y == 0, 1], 'o', alpha=0.75, markersize=10)
plt.plot(x[y == 1, 0], x[y == 1, 1], '<', alpha=0.75, markersize=10)
plt.xlabel(r'$x_1$', size=12)
plt.ylabel(r'$x_2$', size=12)
plt.show()

# 定义模型
model = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 4),
    nn.ReLU(),
    nn.Linear(4, 1),
    nn.Sigmoid()
)
loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.015)

# 创建数据加载器
train_data = TensorDataset(x_train, y_train)
batch_size = 2
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# 训练模型
num_epochs = 200


def train(model, num_epochs, train_dl, x_valid, y_valid):
    loss_hist_train = [0] * num_epochs
    accuracy_hist_train = [0] * num_epochs
    loss_hist_valid = [0] * num_epochs
    accuracy_hist_valid = [0] * num_epochs
    for epoch in range(num_epochs):
        for x_batch, y_batch in train_dl:
            pred = model(x_batch)[:, 0]
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_hist_train[epoch] += loss.item()
            is_correct = np.array((pred >= 0.5).float() == y_batch, dtype=float)
            accuracy_hist_train[epoch] += is_correct.mean()
        loss_hist_train[epoch] /= n_train / batch_size
        accuracy_hist_train[epoch] /= n_train / batch_size
        pred = model(x_valid)[:, 0]
        loss = loss_fn(pred, y_valid)
        loss_hist_valid[epoch] = loss.item()
        is_correct = np.array((pred >= 0.5).float() == y_valid, dtype=float)
        accuracy_hist_valid[epoch] += is_correct.mean()
    return loss_hist_train, accuracy_hist_train, loss_hist_valid, accuracy_hist_valid


history = train(model, num_epochs, train_loader, x_valid, y_valid)

# 绘制学习过程
fig2 = plt.figure(figsize=(12, 6))
ax = fig2.add_subplot(121)
plt.plot(history[0], lw=4)
plt.plot(history[2], lw=4)
plt.legend([r'$Train\ Loss$', r'$Validation\ Loss$'], fontsize=12, loc='best')
ax.set_xlabel(r'$Epochs$', size=12)
ax = fig2.add_subplot(122)
plt.plot(history[1], lw=4)
plt.plot(history[3], lw=4)
plt.legend([r'$Train\ Acc$', r'$Validation\  Acc$'], fontsize=12, loc='best')
ax.set_xlabel(r'$Epochs$', size=12)
plt.show()