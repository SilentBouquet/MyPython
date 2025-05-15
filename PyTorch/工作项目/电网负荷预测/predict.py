import matplotlib
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


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


# 加载处理好的数据
processed_data_path = 'processed_data.xlsx'

processed_data = pd.read_excel(processed_data_path)

# 数据归一化
scaler_features = joblib.load('scaler_features.gz')
scaler_labels = joblib.load('scaler_labels.gz')

# 提取特征和标签
features = processed_data[['最高温度℃', '最低温度℃', '平均温度℃', '相对湿度(平均)', '降雨量（mm）']]
labels = processed_data[['Load']]

# 归一化特征和标签
scaled_features = scaler_features.transform(features)
scaled_labels = scaler_labels.transform(labels)

# 合并特征和标签为一个数组
combined_data = np.concatenate([scaled_labels, scaled_features], axis=1)

# 定义序列长度和预测长度
seq_length = 7 * 24  # 使用过去7天的数据（每天24个时间步）
pred_length = 24     # 预测未来24小时的负荷

# 创建数据集和数据加载器
test_dataset = TimeSeriesDataset(combined_data[-seq_length - pred_length:], seq_length, pred_length)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 加载训练好的模型
model = LSTMModel(input_size=features.shape[1] + 1, hidden_size=64, num_layers=2, output_size=pred_length)
model.load_state_dict(torch.load('best_model.pt', weights_only=True))  # 添加 weights_only=True
model.eval()

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 进行预测
predictions = []
with torch.no_grad():
    for inputs, _ in test_loader:
        inputs = inputs.to(device).float()
        outputs = model(inputs)
        predictions.append(outputs.cpu().numpy())

# 反归一化预测结果
predictions = np.concatenate(predictions, axis=0)
predicted_loads = scaler_labels.inverse_transform(predictions.reshape(-1, 1))

# 获取真实值
true_loads = labels.values[-pred_length:]

# 绘制预测结果与真实值对比图
matplotlib.use("Qt5Agg")
plt.figure(figsize=(15, 6))
plt.plot(true_loads, label='True Load')
plt.plot(predicted_loads, label='Predicted Load')
plt.xlabel('Time Step')
plt.ylabel('Load')
plt.title('True vs Predicted Load')
plt.legend()
plt.show()

# 计算误差
errors = np.abs(predicted_loads - true_loads)

# 绘制预测值与真实值的误差热力图
plt.figure(figsize=(20, 6))  # 增加宽度以减少拥挤
sns.heatmap(errors.reshape(1, -1), annot=True, fmt=".2f", cmap='viridis', cbar_kws={'label': 'Error'},
            annot_kws={'rotation': 90, 'fontsize': 8})  # 旋转注释并减小字体大小
plt.title('Prediction Error Heatmap')
plt.xlabel('Time Step')
plt.ylabel('Prediction Error')
plt.xticks(ticks=np.arange(len(errors)) + 0.5, labels=np.arange(len(errors)), rotation=45, fontsize=8)  # 旋转刻度标签并减小字体大小
plt.show()

# 绘制预测值与真实值的对比热力图
heatmap_data = pd.DataFrame({
    'Time Step': np.arange(len(true_loads)),
    'True Load': true_loads.flatten(),
    'Predicted Load': predicted_loads.flatten()
})

plt.figure(figsize=(20, 6))  # 增加宽度以减少拥挤
sns.heatmap(heatmap_data.set_index('Time Step')[['True Load', 'Predicted Load']], annot=True, fmt=".2f", cmap='viridis', cbar_kws={'label': 'Load'})
plt.title('True vs Predicted Load Heatmap')
plt.xlabel('Load')
plt.ylabel('Time Step')
plt.yticks(fontsize=8)  # 减小刻度标签字体大小
plt.show()

# 绘制预测误差的时间分布热力图
plt.figure(figsize=(20, 6))  # 增加宽度以减少拥挤
sns.heatmap(errors.reshape(1, -1), annot=True, fmt=".2f", cmap='viridis', cbar_kws={'label': 'Error'},
            annot_kws={'rotation': 90, 'fontsize': 8})  # 旋转注释并减小字体大小
plt.title('Error Distribution Over Time Heatmap')
plt.xlabel('Time Step')
plt.ylabel('Error')
plt.xticks(ticks=np.arange(len(errors)) + 0.5, labels=np.arange(len(errors)), rotation=45, fontsize=8)  # 旋转刻度标签并减小字体大小
plt.show()