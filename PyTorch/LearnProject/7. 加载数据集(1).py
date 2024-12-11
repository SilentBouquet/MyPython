# 使用已有张量创建PyTorch DataLoader
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

t = torch.arange(6, dtype=torch.float32)
data_loader = DataLoader(t)
for data in data_loader:
    print(data)

data_loader = DataLoader(t, batch_size=3, drop_last=False)
for i, data in enumerate(data_loader):
    print(f'batch {i}', data)


# 将两个张量组合成一个联合数据集
# 自定义Dataset类
class JointDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# torch.manual_seed(0)
x = torch.rand([4, 3], dtype=torch.float32)
y = torch.arange(4, dtype=torch.float32)
# 或者使用现成的TensorDataset
# joint_dataset = TensorDataset(x, y)
joint_dataset = JointDataset(x, y)
for example in joint_dataset:
    print('x: ', example[0], '      y: ', example[1])

# 创建一个乱序的数据加载器
data_loader = DataLoader(dataset=joint_dataset, batch_size=2, shuffle=True)
for i, batch in enumerate(data_loader):
    print(f'batch {i}: ', 'x: ', batch[0], '\n               y: ', batch[1])