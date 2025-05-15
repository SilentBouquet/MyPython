import math
import torch
import matplotlib
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
from imblearn.combine import SMOTEENN
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score
import matplotlib.pyplot as plt  # 添加 Matplotlib 用于绘图

# 加载信用卡诈骗数据
file_path = 'creditcard.csv'
data = pd.read_csv(file_path)

# 特征和标签
X = data.drop(columns=['Class'])  # 特征
y = data['Class']  # 标签

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2, random_state=42, stratify=y)

# 过采样与欠采样的混合方法
smoteen = SMOTEENN(random_state=42)
X_resampled, y_resampled = smoteen.fit_resample(X_train, y_train)

# 标准化特征
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_resampled)
X_test_scaled = scaler.transform(X_test)


class TransactionDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = np.array(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        feature = self.X[idx].astype(np.float32)
        label = self.y[idx].astype(np.float32)
        feature = feature.reshape(1, -1)  # 添加 sequence_length 维度
        return torch.tensor(feature, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


# 创建数据集
test_dataset = TransactionDataset(X_test_scaled, y_test)
train_dataset = TransactionDataset(X_train_scaled, y_resampled)

# 创建数据加载器
batch_size = 64
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=64):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # 添加 batch 维度
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, hidden_dim)
        pe = self.pe[:, :x.size(1), :]  # 截取位置编码到与输入张量的 sequence_length 相同
        x = x + pe
        return x


class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads, num_layers, dropout=0.1):
        super(TransformerModel, self).__init__()
        # 嵌入层
        self.embedding = nn.Linear(input_dim, hidden_dim)
        # 位置编码
        self.pos_encoder = PositionalEncoding(hidden_dim)
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_dim)
        x = self.embedding(x)  # (batch_size, sequence_length, hidden_dim)
        x = self.pos_encoder(x)  # 添加位置编码
        x = self.transformer(x)  # (batch_size, sequence_length, hidden_dim)
        x = x.mean(dim=1)  # 对 sequence_length 维度取平均，得到 (batch_size, hidden_dim)
        x = self.fc(x)  # (batch_size, output_dim)
        return x


# 使用焦点损失
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return torch.mean(focal_loss)


# 模型参数
input_dim = X_train.shape[1]
print(input_dim)
hidden_dim = 256  # 隐藏层大小
output_dim = 1
num_heads = 2  # 注意力头数
num_layers = 2  # 编码器层数
dropout = 0.2  # dropout 率

model = TransformerModel(input_dim, hidden_dim, output_dim, num_heads, num_layers, dropout)

# 使用加权损失函数
pos_weight = torch.tensor([len(y_train) / y_train.sum() - 1])
criterion = FocalLoss(alpha=0.25, gamma=2)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 使用学习率调度器
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# 训练模型
num_epochs = 20  # 训练轮数
early_stopping_patience = 5  # 早停法参数
best_val_loss = float('inf')
patience_counter = 0

# 用于记录训练和验证损失
train_losses = []
val_losses = []

accumulation_steps = 4  # 每4个小批次累积一次梯度
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for batch_idx, (sequences, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs.squeeze(), labels)
        loss = loss / accumulation_steps  # 平均损失
        loss.backward()
        # 每4个小批次更新一次参数
        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        train_loss += loss.item()
    scheduler.step()

    train_loss /= len(train_loader)
    train_losses.append(train_loss)  # 记录训练损失
    print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}')

    # 验证阶段
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for sequences, labels in test_loader:
            outputs = model(sequences)
            loss = criterion(outputs.squeeze(), labels)
            val_loss += loss.item()
            # 动态调整阈值
            preds = torch.sigmoid(outputs).squeeze()
            threshold = 0.5  # 可以根据验证集调整
            val_correct += (preds > threshold).eq(labels).sum().item()
            val_total += labels.size(0)

    val_loss /= len(test_loader)
    val_losses.append(val_loss)  # 记录验证损失
    val_accuracy = val_correct / val_total
    print(f'Epoch {epoch + 1}/{num_epochs}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

    # 早停法
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1
        if patience_counter >= early_stopping_patience:
            print(f'Early stopping after {early_stopping_patience} epochs without improvement.')
            break


# 绘制学习曲线
def plot_learning_curve(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Learning Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


matplotlib.use('Qt5Agg')
plot_learning_curve(train_losses, val_losses)


# 评估模型
model.load_state_dict(torch.load('best_model.pth', weights_only=True))
model.eval()
y_true = []
y_pred = []

with torch.no_grad():
    for sequences, labels in test_loader:
        outputs = model(sequences)
        preds = torch.sigmoid(outputs).squeeze()  # 获取预测概率
        preds_label = (preds > 0.5).float()  # 将概率转换为二进制标签
        y_true.extend(labels.numpy())
        y_pred.extend(preds_label.numpy())

# 确保 y_true 和 y_pred 是 numpy 数组
y_true = np.array(y_true)
y_pred = np.array(y_pred)

print(classification_report(y_true, y_pred, zero_division=0))
# 计算F1-score
f1 = f1_score(y_true, y_pred)
# 计算PR曲线
precision, recall, _ = precision_recall_curve(y_true, y_pred)
# 计算ROC-AUC
roc_auc = roc_auc_score(y_true, y_pred)
print(f'AUC: {roc_auc}')
print(f'F1: {f1}')
print(f'PR: {precision}')