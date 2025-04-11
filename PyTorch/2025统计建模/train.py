import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
from imblearn.combine import SMOTEENN
from sklearn.metrics import classification_report
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score


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


# 定义数据集类
class TransactionDataset(Dataset):
    def __init__(self, X, y, sequence_length=10):
        self.X = X
        self.y = y
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.X) - self.sequence_length

    def __getitem__(self, idx):
        sequence = self.X.iloc[idx:idx + self.sequence_length].values.astype(np.float32)
        label = self.y.iloc[idx + self.sequence_length].astype(np.float32)
        return torch.tensor(sequence), torch.tensor(label)


# 创建数据集
sequence_length = 10
test_dataset = TransactionDataset(X_test, y_test, sequence_length)
train_dataset = TransactionDataset(X_resampled, y_resampled, sequence_length)

# 创建数据加载器
batch_size = 64
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads, num_layers, dropout=0.1):
        super(TransformerModel, self).__init__()
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,  # 输入特征维度
            nhead=num_heads,  # 注意力头数
            dim_feedforward=hidden_dim,  # 前馈网络隐藏层维度
            dropout=dropout,  # dropout率
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,  # 编码器层
            num_layers=num_layers  # 编码器层数
        )

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # 第一个全连接层
            nn.LeakyReLU(),  # 激活函数
            nn.Dropout(dropout),  # dropout
            nn.Linear(hidden_dim, output_dim)  # 第二个全连接层
        )

    def forward(self, x):
        # x shape: (sequence_length, batch_size, input_dim)
        x = self.transformer(x)  # 输出形状: (sequence_length, batch_size, input_dim)
        x = x[-1, :, :]  # 取最后一个时间步的输出，形状: (batch_size, input_dim)
        x = self.fc(x)  # 输出形状: (batch_size, output_dim)
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
num_epochs = 10  # 训练轮数
early_stopping_patience = 3  # 早停法参数
best_val_loss = float('inf')
patience_counter = 0

accumulation_steps = 4  # 每4个小批次累积一次梯度
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for batch_idx, (sequences, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        sequences = sequences.permute(1, 0, 2)
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
    print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}')

    # 验证阶段
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences = sequences.permute(1, 0, 2)
            outputs = model(sequences)
            loss = criterion(outputs.squeeze(), labels)
            val_loss += loss.item()

            # 动态调整阈值
            preds = torch.sigmoid(outputs).squeeze()
            threshold = 0.5  # 可以根据验证集调整
            val_correct += (preds > threshold).eq(labels).sum().item()
            val_total += labels.size(0)

    val_loss /= len(test_loader)
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

# 评估模型
model.load_state_dict(torch.load('best_model.pth'))
model.eval()
y_true = []
y_pred = []

with torch.no_grad():
    for sequences, labels in test_loader:
        sequences = sequences.permute(1, 0, 2)
        outputs = model(sequences)
        preds = torch.sigmoid(outputs).squeeze() > 0.5
        y_true.extend(labels.numpy())
        y_pred.extend(preds.numpy())

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