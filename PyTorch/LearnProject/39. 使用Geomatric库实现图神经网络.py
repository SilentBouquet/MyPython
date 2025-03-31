import torch
import matplotlib
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from torch_geometric.nn import NNConv, global_add_pool

dataset = QM9('F:/Deep Learning Datasets/QM9')
print(len(dataset))


# 定义图卷积
class ExampleNet(nn.Module):
    def __init__(self, num_node_features, num_edge_features):
        super().__init__()
        conv1_net = nn.Sequential(
            nn.Linear(num_edge_features, 32),
            nn.ReLU(),
            nn.Linear(32, 32 * num_node_features),
        )
        conv2_net = nn.Sequential(
            nn.Linear(num_edge_features, 32),
            nn.ReLU(),
            nn.Linear(32, 32 * 16),
        )
        self.conv1 = NNConv(num_node_features, 32, conv1_net)
        self.conv2 = NNConv(32, 16, conv2_net)
        self.fc1 = nn.Linear(16, 32)
        self.out = nn.Linear(32, 1)

    def forward(self, data):
        batch, x, edge_index, edge_attr = data.batch, data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1(x))
        output = self.out(x)
        return output


# 将数据集拆分为训练集、验证集和测试集
train_set, valid_set, test_set = random_split(dataset, [110000, 10831, 10000])
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=32, shuffle=True)

# 初始化并训练图神经网络
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
qm9_node_features, qm9_edge_features = 11, 4
net = ExampleNet(qm9_node_features, qm9_edge_features)
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
epochs = 4
target_idx = 1
net.to(device)
for total_epochs in range(epochs):
    epoch_loss = 0
    total_graphs = 0
    net.train()
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        output = net(batch)
        loss = F.mse_loss(output, batch.y[:, target_idx].unsqueeze(1))
        loss.backward()
        epoch_loss += loss.item()
        total_graphs += batch.num_graphs
        optimizer.step()
    train_avg_loss = epoch_loss / total_graphs
    val_loss = 0
    total_graphs = 0
    net.eval()
    for batch in valid_loader:
        batch = batch.to(device)
        output = net(batch)
        loss = F.mse_loss(output, batch.y[:, target_idx].unsqueeze(1))
        val_loss += loss.item()
        total_graphs += batch.num_graphs
    val_avg_loss = val_loss / total_graphs
    print(f'Epochs: {total_epochs} | '
          f'epoch avg loss: {train_avg_loss:.4f} | '
          f'val avg loss: {val_avg_loss:.4f}')

# 使用模型预测测试数据标签
net.eval()
predictions = []
real = []
for batch in test_loader:
    output = net(batch.to(device))
    predictions.append(output.detach().cpu().numpy())
    real.append(batch.y[:, target_idx].detach().cpu().numpy())
real = np.concatenate(real)
predictions = np.concatenate(predictions)

matplotlib.use('Qt5Agg')
plt.scatter(real[:500], predictions[:500], s=10, c=predictions[:500], cmap='viridis')
plt.xlabel(r'$Isotropic/ polarizability$')
plt.ylabel(r'$Predicted/ isotropic/ polarizability$')
plt.show()