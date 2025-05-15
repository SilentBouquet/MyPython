import matplotlib
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


# 添加缺失值处理函数
def handle_missing_values(df):
    # 对每一列进行前向填充，如果有NaN值则用前一个有效值填充
    df.ffill(inplace=True)
    # 如果开头仍有NaN值，则用后向填充
    df.bfill(inplace=True)
    return df


class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length, pred_length):
        self.load_data = data[:, 0].reshape(-1, 1)
        self.weather_time_data = data[:, 1:]
        self.seq_length = seq_length
        self.pred_length = pred_length

    def __len__(self):
        return len(self.load_data) - self.seq_length - self.pred_length + 1

    def __getitem__(self, index):
        x_load = self.load_data[index:index + self.seq_length]
        x_weather_time = self.weather_time_data[index:index + self.seq_length]
        x_combined = np.concatenate([x_load, x_weather_time], axis=1)
        y = self.load_data[index + self.seq_length:index + self.seq_length + self.pred_length]
        y = y.reshape(-1)
        return x_combined, y


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


# 读取处理好的数据
processed_data_path = 'processed_data.xlsx'
processed_data = pd.read_excel(processed_data_path)

# 数据归一化
scaler_features = MinMaxScaler(feature_range=(0, 1))
scaler_labels = MinMaxScaler(feature_range=(0, 1))

# 提取特征和标签
features = processed_data[['最高温度℃', '最低温度℃', '平均温度℃', '相对湿度(平均)', '降雨量（mm）']]
labels = processed_data[['Load']]

# 处理缺失值
features = handle_missing_values(features)
labels = handle_missing_values(labels)

# 归一化特征和标签
scaled_features = scaler_features.fit_transform(features)
scaled_labels = scaler_labels.fit_transform(labels)

# 合并特征和标签为一个数组
combined_data = np.concatenate([scaled_labels, scaled_features], axis=1)

# 划分训练集和验证集
train_ratio = 0.8
train_size = int(len(combined_data) * train_ratio)
train_data = combined_data[:train_size]
val_data = combined_data[train_size:]

# 创建数据集和数据加载器
seq_length = 7 * 24  # 使用过去7天的数据（每天24个时间步）
pred_length = 24     # 预测未来24小时的负荷

train_dataset = TimeSeriesDataset(train_data, seq_length, pred_length)
val_dataset = TimeSeriesDataset(val_data, seq_length, pred_length)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 定义模型参数
input_size = features.shape[1] + 1  # 特征数量（天气特征5个 + 负荷1个）
hidden_size = 64
num_layers = 2
output_size = pred_length  # 预测未来24小时的负荷

# 创建模型实例
model = LSTMModel(input_size, hidden_size, num_layers, output_size)

# 设置训练设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# 定义早停类
class EarlyStopping:
    def __init__(self, patience=10, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


early_stopping = EarlyStopping(patience=10, delta=0.0001)

# 训练模型
num_epochs = 200
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    train_loss = 0.0
    for inputs, targets in train_loader:
        inputs = torch.tensor(inputs, dtype=torch.float32).to(device)
        targets = torch.tensor(targets, dtype=torch.float32).to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    # 验证阶段
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = torch.tensor(inputs, dtype=torch.float32).to(device)
            targets = torch.tensor(targets, dtype=torch.float32).to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

    # 早停检查
    early_stopping(val_loss)
    if early_stopping.early_stop:
        print(f"Early stopping triggered at epoch {epoch + 1}")
        break

    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# 保存模型
torch.save(model.state_dict(), 'best_model.pt')

# 绘制训练和验证损失曲线
matplotlib.use("Qt5Agg")
plt.figure(figsize=(15, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()