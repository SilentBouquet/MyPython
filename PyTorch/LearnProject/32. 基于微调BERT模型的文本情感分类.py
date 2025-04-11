import gzip
import time
import shutil
import matplotlib
import requests
import torch
import pandas as pd
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
import matplotlib.pyplot as plt  # 添加可视化所需的库

# 设置模型参数
torch.backends.cudnn.deterministic = True
torch.manual_seed(123)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 3

# 加载并处理数据
url = "https://github.com/rasbt/machine-learning-book/raw/main/ch08/movie_data.csv.gz"
file_name = "F:/Deep Learning Datasets/movie_data.csv.gz"

with open(file_name, "wb") as f:
    r = requests.get(url)
    f.write(r.content)

with gzip.open(file_name, "rb") as f:
    with open("F:/Deep Learning Datasets/movie_data/movie_data.csv", 'wb') as f_out:
        shutil.copyfileobj(f, f_out)

df = pd.read_csv("F:/Deep Learning Datasets/movie_data/movie_data.csv")
print(df.head(3))

# 拆分数据集
train_texts = df.iloc[:35000]['review'].values
train_labels = df.iloc[:35000]['sentiment'].values
val_texts = df.iloc[35000:40000]['review'].values
val_labels = df.iloc[35000:40000]['sentiment'].values
test_texts = df.iloc[40000:]['review'].values
test_labels = df.iloc[40000:]['sentiment'].values

# 数据集分词
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
train_encodings = tokenizer(list(train_texts), padding=True, truncation=True)
val_encodings = tokenizer(list(val_texts), padding=True, truncation=True)
test_encodings = tokenizer(list(test_texts), padding=True, truncation=True)


# 自定义数据集类并创建数据加载器
class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = IMDbDataset(train_encodings, train_labels)
val_dataset = IMDbDataset(val_encodings, val_labels)
test_dataset = IMDbDataset(test_encodings, test_labels)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=True)

# 加载和微调预训练的BERT模型
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
model.to(device)
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)


# 计算准确率函数
def compute_accuracy(model, data_loader, device):
    with torch.no_grad():
        correct_pred, num_examples = 0, 0
        for batch_idx, batch in enumerate(data_loader):
            inputs_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=inputs_ids, attention_mask=attention_mask)
            logits = outputs['logits']
            predicted_labels = torch.argmax(logits, 1)
            num_examples += labels.size(0)
            correct_pred += (predicted_labels == labels).sum()
    return correct_pred.float() / num_examples * 100


# 记录训练和验证损失
train_losses = []
val_losses = []

# 训练模型
start_time = time.time()
for epoch in range(num_epochs):
    model.train()
    epoch_train_loss = 0.0  # 记录每个 epoch 的训练损失
    for batch_idx, batch in enumerate(train_loader):
        inputs_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids=inputs_ids, attention_mask=attention_mask, labels=labels)
        loss, logits = outputs['loss'], outputs['logits']
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()  # 累加训练损失
        if not batch_idx % 250:
            print(
                f'Epoch: {epoch + 1:04d}/{num_epochs:04d} | Batch {batch_idx:04d}/{len(train_loader):04d} | Loss: {loss:.4f}')

    # 计算平均训练损失
    avg_train_loss = epoch_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # 验证阶段
    model.eval()
    epoch_val_loss = 0.0  # 记录每个 epoch 的验证损失
    with torch.set_grad_enabled(False):
        for batch in val_loader:
            inputs_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=inputs_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs['loss']
            epoch_val_loss += loss.item()  # 累加验证损失

    # 计算平均验证损失
    avg_val_loss = epoch_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    # 打印训练和验证的准确率
    print(
        f'Epoch {epoch + 1} | Training Accuracy: {compute_accuracy(model, train_loader, device):.2f}%'
        f' | Val Accuracy: {compute_accuracy(model, val_loader, device):.2f}%')
    print(f'Training Loss: {avg_train_loss:.4f} | Validation Loss: {avg_val_loss:.4f}')
    print(f'Time elapsed: {(time.time() - start_time) / 60:.2f} min')

print(f'Total Training Time: {(time.time() - start_time) / 60:.2f} min')
print(f'Test Accuracy: {compute_accuracy(model, test_loader, device):.2f}%')

# 绘制学习曲线
matplotlib.use('Qt5Agg')
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Learning Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# 保存模型
model_path = "../Runs/"
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")