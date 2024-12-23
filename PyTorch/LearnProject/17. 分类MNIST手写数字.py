import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms

# 加载并处理数据集
image_path = '../'
transform = transforms.Compose([transforms.ToTensor()])
mnist_train_dataset = torchvision.datasets.MNIST(root=image_path, train=True, transform=transform, download=False)
mnist_test_dataset = torchvision.datasets.MNIST(root=image_path, train=False, transform=transform, download=False)
batch_size = 64
torch.manual_seed(0)
train_loader = torch.utils.data.DataLoader(dataset=mnist_train_dataset, batch_size=batch_size, shuffle=True)

# 构建神经网络模型
hidden_units = [32, 16]
image_size = mnist_train_dataset[0][0].shape
input_size = image_size[0] * image_size[1] * image_size[2]
all_layers = [nn.Flatten()]
for hidden_unit in hidden_units:
    layer = nn.Linear(in_features=input_size, out_features=hidden_unit)
    all_layers.append(layer)
    all_layers.append(nn.ReLU())
    input_size = hidden_unit
all_layers.append(nn.Linear(in_features=hidden_units[-1], out_features=10))
model = nn.Sequential(*all_layers)

# 训练模型
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 20
for epoch in range(num_epochs + 1):
    accuracy_hist_train = 0
    for x_batch, y_batch in train_loader:
        pred = model(x_batch)
        loss = loss_func(pred, y_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        is_correct = (torch.argmax(pred, dim=-1) == y_batch).float()
        accuracy_hist_train += is_correct.sum()
    accuracy_hist_train /= len(train_loader.dataset)
    print(f'Epoch {epoch}, Accuracy: {accuracy_hist_train:.4f}')

# 评估模型在测试集上的性能
pred = model(mnist_test_dataset.data / 255.)
is_correct = (torch.argmax(pred, dim=1) == mnist_test_dataset.targets).float()
print(f'Test Accuracy: {is_correct.mean():.4f}')