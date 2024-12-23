import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from mlxtend.plotting import plot_decision_regions

n_train = 100
batch_size = 2
num_epochs = 200
# 生成数据并拆分为训练数据和验证数据
torch.manual_seed(0)
np.random.seed(0)
x = np.random.uniform(-1, 1, (200, 2))
y = np.ones(len(x))
y[x[:, 0] * x[:, 1] < 0] = 0
x_train = torch.tensor(x[:n_train, :], dtype=torch.float32)
y_train = torch.tensor(y[:n_train], dtype=torch.float32)
x_valid = torch.tensor(x[n_train:, :], dtype=torch.float32)
y_valid = torch.tensor(y[n_train:], dtype=torch.float32)
# 创建数据加载器
train_data = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)


class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        l1 = nn.Linear(2, 4)
        a1 = nn.ReLU()
        l2 = nn.Linear(4, 4)
        a2 = nn.ReLU()
        l3 = nn.Linear(4, 1)
        a3 = nn.Sigmoid()
        L = [l1, a1, l2, a2, l3, a3]
        self.module_list = nn.ModuleList(L)

    def forward(self, x):
        for module in self.module_list:
            x = module(x)
        return x

    def predict(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        pred = self.forward(x)[:, 0]
        return (pred >= 0.5).float()


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


model = MyModule()
loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.015)
history = train(model, num_epochs, train_loader, x_valid, y_valid)

# 绘制学习过程和决策区域
fig = plt.figure(figsize=(16, 6))
ax = fig.add_subplot(1, 3, 1)
plt.plot(history[0], lw=4)
plt.plot(history[2], lw=4)
plt.legend([r'$Train\ Loss$', r'$Validation\ Loss$'], fontsize=12, loc='best')
ax.set_xlabel(r'$Epochs$', size=12)
ax = fig.add_subplot(1, 3, 2)
plt.plot(history[1], lw=4)
plt.plot(history[3], lw=4)
plt.legend([r'$Train\ Acc$', r'$Validation\  Acc$'], fontsize=12, loc='best')
ax.set_xlabel(r'$Epochs$', size=12)
ax = fig.add_subplot(1, 3, 3)
plot_decision_regions(X=x_valid.numpy(),
                      y=y_valid.numpy().astype(np.int32), clf=model)
ax.set_xlabel(r'$x_1$', size=14)
ax.xaxis.set_label_coords(1, -0.025)
ax.set_ylabel(r'$x_2$', size=14)
ax.yaxis.set_label_coords(-0.025, 1)
plt.tight_layout()
plt.show()