import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

# 加载鸢尾花数据集
iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1. / 3, random_state=1)

# 数据特征归一化
X_train_norm = (X_train - np.mean(X_train)) / np.std(X_train)
X_train_norm = torch.from_numpy(X_train_norm).float()
y_train = torch.from_numpy(y_train)
train_ds = TensorDataset(X_train_norm, y_train)

torch.manual_seed(1)
batch_size = 2
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = nn.Sigmoid()(x)
        x = self.layer2(x)
        return x


input_size = X_train_norm.shape[1]
hidden_size = 16
output_size = 3
model = Model(input_size, hidden_size, output_size)
learning_rate = 0.001
# 交叉熵损失函数
loss_func = nn.CrossEntropyLoss()
# Adam优化器
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 对模型进行训练
num_epochs = 100
loss_hist = [0] * num_epochs
accuracy_hist = [0] * num_epochs
for epoch in range(num_epochs):
    for x_batch, y_batch in train_loader:
        pred = model(x_batch)
        loss = loss_func(pred, y_batch.long())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # 累加当前批次的损失，并乘以批次的大小（y_batch.size(0)），以便在每个epoch结束后计算平均损失
        loss_hist[epoch] += loss.item() * y_batch.size(0)
        # 计算预测是否正确，torch.argmax(pred, dim=1)得到预测的类别索引，然后与真实标签y_batch进行比较
        is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
        # 累加当前批次的正确预测数量，并乘以批次的大小，以便在每个epoch结束后计算平均准确率
        accuracy_hist[epoch] += is_correct.sum()
    # 计算平均损失和平均准确率
    loss_hist[epoch] /= len(train_loader.dataset)
    accuracy_hist[epoch] /= len(train_loader.dataset)

# 学习过程可视化
fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(1, 2, 1)
ax.plot(loss_hist, lw=3)
ax.set_title('Training Loss', size=14)
ax.set_xlabel('Epoch', size=14)
ax.tick_params(axis='both', which='major', labelsize=12)
ax = fig.add_subplot(1, 2, 2)
ax.plot(accuracy_hist, lw=3)
ax.set_title('Training Accuracy', size=14)
ax.set_xlabel('Epoch', size=14)
ax.tick_params(axis='both', which='major', labelsize=12)
plt.show()

# 在测试数据集上评估训练好的模型
X_test_norm = (X_test - np.mean(X_train)) / np.std(X_train)
X_test_norm = torch.from_numpy(X_test_norm).float()
y_test = torch.from_numpy(y_test)
pred_test = model(X_test_norm)
correct = (torch.argmax(pred_test, dim=1) == y_test).float()
accuracy = correct.mean()
print(f'Test Accuracy: {accuracy}')

# 保存训练好的模型
path = '../My_Model/iris_classifier.pt'
torch.save(model, path)

# 重新加载模型
model_new = torch.load(path)
model_new.eval()
pred_test = model_new(X_test_norm)
correct = (torch.argmax(pred_test, dim=1) == y_test).float()
accuracy = correct.mean()
print(f'Test Accuracy: {accuracy}')