import torch
import sklearn
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import sklearn.model_selection as ms
from torch.nn.functional import one_hot
from torch.utils.data import TensorDataset, DataLoader

# 数据加载和预处理
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower',
                'Weight', 'Acceleration', 'Model Year', 'Origin']
# na_values='?'：将问号?视为缺失值
# comment='\t'：将制表符\t视为注释符号，注释掉的数据不会被读取
# skipinitialspace=True：跳过字段前的空格
df = pd.read_csv(url, names=column_names, na_values='?',
                 comment='\t', sep=' ', skipinitialspace=True)
# 移除df中包含缺失值的行
df = df.dropna()
# 重置df的索引，并使用drop=True参数丢弃旧的索引
df = df.reset_index(drop=True)
df_train, df_test = ms.train_test_split(df, train_size=0.8, random_state=0)
# 对训练集df_train进行描述性统计，并使用transpose()方法将结果转置，以便于后续按列名访问统计数据
train_stats = df_train.describe().transpose()

# 归一化连续特征
numeric_column_names = ['Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration']
df_train_norm, df_test_norm = df_train.copy(), df_test.copy()
for col in numeric_column_names:
    mean = train_stats.loc[col, 'mean']
    std = train_stats.loc[col, 'std']
    df_train_norm.loc[:, col] = (df_train_norm.loc[:, col] - mean) / std
    df_test_norm.loc[:, col] = (df_test_norm.loc[:, col] - mean) / std

# 对数据进行分组
boundaries = torch.tensor([73, 76, 79])
v = torch.tensor(df_train_norm['Model Year'].values)
# torch.bucketize根据boundaries对v进行分桶
# right=True参数表示桶的边界是右开的，即每个区间包括右边界但不包括左边界
df_train_norm['Model Year Bucketed'] = torch.bucketize(v, boundaries, right=True)
v = torch.tensor(df_test_norm['Model Year'].values)
df_test_norm['Model Year Bucketed'] = torch.bucketize(v, boundaries, right=True)
numeric_column_names.append('Model Year Bucketed')

# 对类别特征进行独热编码
total_origin = len(set(df_train_norm['Origin']))
origin_encoded = one_hot(torch.from_numpy(df_train_norm['Origin'].values) % total_origin)
x_train_numeric = torch.tensor(df_train_norm[numeric_column_names].values)
# 使用torch.cat将数值特征张量和独热编码张量沿着列（dim=1）合并
x_train = torch.cat([x_train_numeric, origin_encoded], dim=1).float()
origin_encoded = one_hot(torch.from_numpy(df_test_norm['Origin'].values) % total_origin)
x_test_numeric = torch.tensor(df_test_norm[numeric_column_names].values)
x_test = torch.cat([x_test_numeric, origin_encoded], dim=1).float()
y_train = torch.tensor(df_train_norm['MPG'].values).float()
y_test = torch.tensor(df_test_norm['MPG'].values).float()

# 创建数据加载器
train_ds = TensorDataset(x_train, y_train)
batch_size = 8
torch.manual_seed(0)
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

# 训练DNN回归模型
hidden_units = [8, 4]
input_size = x_train.shape[1]
all_layers = []
for hidden_unit in hidden_units:
    layer = nn.Linear(input_size, hidden_unit)
    all_layers.append(layer)
    all_layers.append(nn.ReLU())
    input_size = hidden_unit
all_layers.append(nn.Linear(hidden_units[-1], 1))
model = nn.Sequential(*all_layers)
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

num_epochs = 200
log_epoch = 20
for epoch in range(num_epochs + 1):
    loss_hist_train = 0
    for x_batch, y_batch in train_dl:
        pred = model(x_batch)[:, 0]
        loss = loss_fn(pred, y_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_hist_train += loss.item()
    if epoch % log_epoch == 0:
        print(f'Epoch {epoch}/{num_epochs}, Loss: {loss_hist_train / len(train_dl)}')

with torch.no_grad():
    pred = model(x_test.float())[:, 0]
    loss = loss_fn(pred, y_test)
    print(f'Test MSE: {loss.item():.4f}')
    print(f'Test MAE: {nn.L1Loss()(pred, y_test).item():.4f}')