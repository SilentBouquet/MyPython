import torch
import torchvision
import numpy as np
import torch.nn as nn
from torchvision import transforms
from matplotlib import pyplot as plt
from torch.utils.data import Subset, DataLoader

# 加载MNIST数据集并创建训练集和测试集
image = '../'
# 定义数据预处理方法，将图像数据转换为PyTorch张量，每个像素值会被归一化到 [0, 1] 范围内
transform = transforms.Compose([transforms.ToTensor()])
mnist_dataset = torchvision.datasets.MNIST(
    root=image, train=True, download=True, transform=transform
)
# 选取前10000个样本作为验证集
mnist_valid_dataset = Subset(mnist_dataset, torch.arange(10000))
# 选取剩余样本作为训练集
mnist_train_dataset = Subset(mnist_dataset, torch.arange(10000, len(mnist_dataset)))
mnist_test_dataset = torchvision.datasets.MNIST(
    root=image, train=False, download=False, transform=transform
)

# 为训练集和验证集创建数据加载器
batch_size = 64
torch.manual_seed(0)
train_dataloader = DataLoader(mnist_train_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(mnist_valid_dataset, batch_size=batch_size, shuffle=False)

# 构建卷积神经网络
model = nn.Sequential()
model.add_module('conv1', nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2))
model.add_module('relu1', nn.ReLU())
model.add_module('pool1', nn.MaxPool2d(kernel_size=2))
model.add_module('conv2', nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2))
model.add_module('relu2', nn.ReLU())
model.add_module('pool2', nn.MaxPool2d(kernel_size=2))
model.add_module('flatten', nn.Flatten())
model.add_module('fc1', nn.Linear(in_features=3136, out_features=1024))
model.add_module('relu3', nn.ReLU())
model.add_module('dropout', nn.Dropout(0.5))
model.add_module('fc2', nn.Linear(in_features=1024, out_features=10))

# 创建损失函数和优化器
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# 定义模型的训练函数
def train(model, num_epochs, train_dataloader, valid_dataloader):
    loss_hist_train = [0] * num_epochs
    accuracy_hist_train = [0] * num_epochs
    loss_hist_valid = [0] * num_epochs
    accuracy_hist_valid = [0] * num_epochs
    for epoch in range(num_epochs):
        # 设置模型为训练模式，启用Dropout
        model.train(True)
        for x_batch, y_batch in train_dataloader:
            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_hist_train[epoch] += loss.item() * y_batch.size(0)
            is_correct = (
                    torch.argmax(pred, dim=1) == y_batch
            ).float()
            accuracy_hist_train[epoch] += is_correct.sum()
        loss_hist_train[epoch] /= len(train_dataloader.dataset)
        accuracy_hist_train[epoch] /= len(train_dataloader.dataset)

        # 设置模型为评估模式，禁用Dropout
        model.eval()
        # 禁用梯度计算，减少内存消耗
        with torch.no_grad():
            for x_batch, y_batch in valid_dataloader:
                pred = model(x_batch)
                loss = loss_fn(pred, y_batch)
                loss_hist_valid[epoch] += loss.item() * y_batch.size(0)
                is_correct = (
                        torch.argmax(pred, dim=1) == y_batch
                ).float()
                accuracy_hist_valid[epoch] += is_correct.sum()
        loss_hist_valid[epoch] /= len(valid_dataloader.dataset)
        accuracy_hist_valid[epoch] /= len(valid_dataloader.dataset)

        print(f'Epoch {epoch + 1}   accuracy：{accuracy_hist_train[epoch]:.4f}'
              f'  val_accuracy：{accuracy_hist_valid[epoch]:.4f}')
    return loss_hist_train, loss_hist_valid, accuracy_hist_train, accuracy_hist_valid


torch.manual_seed(0)
num_epochs = 20
hist = train(model, num_epochs, train_dataloader, valid_dataloader)

# 可视化学习曲线
plt.rcParams['font.sans-serif'] = ['Lucida Sans Unicode']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'cm'
x_arr = np.arange(len(hist[0])) + 1
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(121)
ax.plot(x_arr, hist[0], '-o', label=r'$Train\ Loss$')
ax.plot(x_arr, hist[1], '--<', label=r'$Validation\ Loss$')
ax.legend(loc='best', fontsize=13)
ax.set_xlabel('$Epoch$', fontsize=13)
ax.set_ylabel('$Loss$', fontsize=13)
ax = fig.add_subplot(122)
ax.plot(x_arr, hist[2], '-o', label=r'$Train\ Acc$')
ax.plot(x_arr, hist[3], '--<', label=r'$Validation\ Acc$')
ax.legend(loc='best', fontsize=13)
ax.set_xlabel('$Epoch$', fontsize=13)
ax.set_ylabel('$Accuracy$', fontsize=13)
plt.show()

# 在测试集上评估训练好的模型
# unsqueeze(1): 在第 1 维度上增加一个维度
# 将数据的形状从 [num_samples, 28, 28] 转换为 [num_samples, 1, 28, 28]
# / 255.: 将像素值归一化到 [0, 1] 范围
pred = model(mnist_test_dataset.data.unsqueeze(1) / 255.)
is_correct = (
    torch.argmax(pred, dim=1) == mnist_test_dataset.targets
).float()
print(f'Test Accuracy：{is_correct.mean():.4f}')

# 展示部分预测标签
fig = plt.figure(figsize=(12, 6))
for i in range(12):
    ax = fig.add_subplot(2, 6, i + 1)
    ax.set_xticks([])
    ax.set_yticks([])
    img = mnist_test_dataset[i][0][0, :, :]
    pred = model(img.unsqueeze(0).unsqueeze(1))
    y_pred = torch.argmax(pred)
    # cmap='gray_r': 使用黑白反转的颜色映射（黑色为背景，白色为前景）
    ax.imshow(img, cmap='gray_r')
    # 0.9, 0.1: 指定文本的位置，相对坐标（左下角为 (0, 0)，右上角为 (1, 1)）。
    # horizontalalignment='center': 文本水平居中。
    # verticalalignment='center': 文本垂直居中。
    # transform=ax.transAxes: 使用子图的坐标系。
    ax.text(0.9, 0.1, str(y_pred.item()),
            size=15, color='blue',
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes)
plt.show()