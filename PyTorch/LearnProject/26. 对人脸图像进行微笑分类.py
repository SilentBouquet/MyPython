import torch
import torchvision
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader, Subset

# 从attributes列表中提取微笑标签
get_smile = lambda attr: attr[31]
# 定义生成转换图像的函数
transform_train = transforms.Compose([
    transforms.RandomCrop([178, 178]),
    transforms.RandomHorizontalFlip(),
    transforms.Resize([64, 64]),
    transforms.ToTensor()
])
transform = transforms.Compose([
    transforms.CenterCrop([178, 178]),
    transforms.Resize([64, 64]),
    transforms.ToTensor()
])

# 调用transform_train函数处理训练数据集
image_path = '../'
celeba_train_dataset = torchvision.datasets.CelebA(
    image_path, split='train', download=False, target_type='attr',
    transform=transform_train, target_transform=get_smile
)
torch.manual_seed(1)
data_loader = DataLoader(celeba_train_dataset, batch_size=2)
fig = plt.figure(figsize=(14, 6))
num_epochs = 5
for j in range(num_epochs):
    img_batch, label_batch = next(iter(data_loader))
    img = img_batch[0]
    ax = fig.add_subplot(2, 5, j + 1, xticks=[], yticks=[])
    ax.set_title(f'Epoch {j}', size=13)
    ax.imshow(img.permute(1, 2, 0))

    img2 = img_batch[1]
    ax2 = fig.add_subplot(2, 5, j + 6, xticks=[], yticks=[])
    ax2.imshow(img2.permute(1, 2, 0))
plt.show()

# 使用transform函数处理验证集和测试机
celeba_valid_dataset = torchvision.datasets.CelebA(
    image_path, split='valid', download=False, target_type='attr',
    transform=transform, target_transform=get_smile
)
celeba_test_dataset = torchvision.datasets.CelebA(
    image_path, split='test', download=False, target_type='attr',
    transform=transform, target_transform=get_smile
)

# 分割数据集
celeba_train_dataset = Subset(celeba_train_dataset, torch.arange(16000))
celeba_test_dataset = Subset(celeba_test_dataset, torch.arange(1000))

# 创建数据加载器
batch_size = 32
torch.manual_seed(1)
train_dl = DataLoader(celeba_train_dataset, batch_size=batch_size, shuffle=True)
test_dl = DataLoader(celeba_test_dataset, batch_size=batch_size, shuffle=False)
valid_dl = DataLoader(celeba_valid_dataset, batch_size=batch_size, shuffle=False)

# 训练卷积神经网络微笑分类器
model = nn.Sequential()
model.add_module('conv1', nn.Conv2d(3, 32, kernel_size=3, padding=1))
model.add_module('relu1', nn.ReLU())
model.add_module('pool1', nn.MaxPool2d(2))
model.add_module('dropout1', nn.Dropout(0.5))
model.add_module('conv2', nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1))
model.add_module('relu2', nn.ReLU())
model.add_module('pool2', nn.MaxPool2d(2))
model.add_module('dropout2', nn.Dropout(0.5))
model.add_module('conv3', nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1))
model.add_module('relu3', nn.ReLU())
model.add_module('pool3', nn.MaxPool2d(2))
model.add_module('conv4', nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1))
model.add_module('relu4', nn.ReLU())
model.add_module('pool4', nn.AvgPool2d(8))
model.add_module('flatten', nn.Flatten())
model.add_module('fc', nn.Linear(in_features=256, out_features=1))
model.add_module('sigmoid', nn.Sigmoid())

# 创建损失函数和优化器
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# 定义训练函数
def train(model, num_epochs, train_dl, valid_dl):
    loss_hist_train = [0] * num_epochs
    loss_hist_valid = [0] * num_epochs
    accuracy_hist_train = [0] * num_epochs
    accuracy_hist_valid = [0] * num_epochs
    for epoch in range(num_epochs):
        model.train()
        for x_batch, y_batch in train_dl:
            pred = model(x_batch)[:, 0]
            loss = loss_fn(pred, y_batch.float())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_hist_train[epoch] += loss.item() * y_batch.size()[0]
            is_correct = np.array((pred >= 0.5).float() == y_batch)
            accuracy_hist_train[epoch] += is_correct.sum()
        loss_hist_train[epoch] /= len(train_dl.dataset)
        accuracy_hist_train[epoch] /= len(train_dl.dataset)

        model.eval()
        with torch.no_grad():
            for x_batch, y_batch in valid_dl:
                pred = model(x_batch)[:, 0]
                loss = loss_fn(pred, y_batch.float())
                loss_hist_valid[epoch] += loss.item() * y_batch.size()[0]
                is_correct = np.array((pred >= 0.5).float() == y_batch)
                accuracy_hist_valid[epoch] += is_correct.sum()
        loss_hist_valid[epoch] /= len(valid_dl.dataset)
        accuracy_hist_valid[epoch] /= len(valid_dl.dataset)

        print(f'Epoch {epoch + 1}   Accuracy：{accuracy_hist_train[epoch]:.4f}'
              f'  Val_Accuracy：{accuracy_hist_valid[epoch]:.4f}')

    return loss_hist_train, loss_hist_valid, accuracy_hist_train, accuracy_hist_valid


# 训练模型
torch.manual_seed(1)
num_epochs = 30
hist = train(model, num_epochs, train_dl, valid_dl)

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

# 在测试集上评估模型性能
accuracy_test = 0
model.eval()
with torch.no_grad():
    for x_batch, y_batch in test_dl:
        pred = model(x_batch)[:, 0]
        is_correct = np.array((pred >= 0.5).float() == y_batch)
        accuracy_test += is_correct.sum()
accuracy_test /= len(test_dl.dataset)
print(f'Test Accuracy：{accuracy_test:.4f}')

# 可视化部分测试样本的预测概率和真实标签
# permute 是一个用于交换张量（tensor）维度顺序的方法
# PyTorch 通常使用 (channels, height, width) 的格式来存储图像数据
# 而 Matplotlib 的 imshow 方法需要 (height, width, channels) 的格式
pred = model(x_batch)[:, 0] * 100
fig = plt.figure(figsize=(15, 8))
for j in range(0, 8):
    ax = fig.add_subplot(2, 4, j + 1, xticks=[], yticks=[])
    ax.imshow(x_batch[j].permute(1, 2, 0))
    if y_batch[j] == 1:
        label = 'Smile'
    else:
        label = 'Not Smile'
    ax.text(0.5, -0.15, f'GT: {label:s}\nPr (Smile) = {pred[j]:.0f}%',
            size=13, transform=ax.transAxes, ha='center', va='center')
plt.show()