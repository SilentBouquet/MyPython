import os
import re
import torch
from torch import nn
from collections import Counter
from torch.utils.data import Dataset, DataLoader, random_split


# 定义数据读取函数
def read_imdb_data(data_type="train"):
    """
    读取IMDB数据集的正负样本文件，并返回文本列表和标签列表。
    :param data_type: "train" 或 "test"
    :return: 文本列表和标签列表
    """
    extracted_dir = r"F:\Deep Learning Datasets\aclImdb"
    data_dir = os.path.join(extracted_dir, data_type)
    texts = []
    labels = []

    # 遍历正负样本文件夹
    for label_type in ["pos", "neg"]:
        dir_name = os.path.join(data_dir, label_type)
        for fname in os.listdir(dir_name):
            if fname.endswith(".txt"):
                with open(os.path.join(dir_name, fname), encoding='utf-8') as f:
                    texts.append(f.read())
                    labels.append(1 if label_type == "pos" else 0)

    return texts, labels


# 自定义Dataset类
class IMDBDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        return text, label


# 加载训练集和测试集
train_texts, train_labels = read_imdb_data("train")
test_texts, test_labels = read_imdb_data("test")

# 创建Dataset对象
train_dataset = IMDBDataset(train_texts, train_labels)
test_dataset = IMDBDataset(test_texts, test_labels)

# 打印数据集大小
print(f"Training set size: {len(train_dataset)}")
print(f"Test set size: {len(test_dataset)}")
print()

# 划分训练集和验证集
torch.manual_seed(0)
train_dataset, val_dataset = random_split(train_dataset, [20000, 5000])


# 文本预处理函数
def tokenizer(text):
    # 去除 HTML 标签
    text = re.sub('<[^>]*>', '', text)
    # 提取表情符号
    emoticons = re.findall(
        '[:;=]-?[()DP]', text.lower()
    )
    # 规范化文本并重新加入表情符号
    text = re.sub(r'\W+', ' ', text.lower() +
                  ' '.join(emoticons).replace('-', ''))
    # 分词
    tokenized = text.split()
    return tokenized


# 使用原始的文本预处理函数处理数据集
token_counts = Counter()
for text, label in train_dataset:
    tokens = tokenizer(text)
    token_counts.update(tokens)

print('Vocab-size:', len(token_counts))
print()


# 自定义词汇表构建函数
def build_vocab(counter):
    # 添加特殊标记
    special_tokens = ['<pad>', '<unk>']
    # 按词频排序
    sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    # 构建词汇表
    tokens = special_tokens + [token for token, _ in sorted_by_freq_tuples]
    # 创建词到索引的映射
    vocab = {token: idx for idx, token in enumerate(tokens)}
    # 设置默认索引为<unk>的索引
    default_index = vocab['<unk>']
    return vocab, default_index


# 构建词汇表
vocab, default_index = build_vocab(token_counts)
# 测试词汇表
test_tokens = ['this', 'is', 'an', 'example']
test_indices = [vocab.get(token, default_index) for token in test_tokens]
print(test_indices)
print()

# 定义文本处理管道，将文本转换为词汇表中的索引序列
text_pipeline = lambda x: [vocab.get(token, default_index) for token in tokenizer(x)]
# 定义标签处理管道，将标签转换为二进制值（1.0 表示正面，0.0 表示负面）
label_pipeline = lambda x: 1. if x == 'pos' else 0.


# 定义批处理函数，用于将一个批次的数据整理成模型可以接受的格式
def collate_batch(batch):
    # 初始化用于存储标签、处理后的文本和文本长度的列表
    label_list, text_list, lengths = [], [], []

    # 遍历批次中的每个样本
    for text, label in batch:
        # 处理标签并添加到标签列表
        label_list.append(label_pipeline(label))
        # 处理文本：将文本转换为词汇索引列表，并转换为 PyTorch 张量
        processed_text = torch.tensor(text_pipeline(text), dtype=torch.int64)
        text_list.append(processed_text)  # 添加处理后的文本到文本列表
        # 记录文本长度
        lengths.append(processed_text.size(0))

    # 将标签列表转换为 PyTorch 张量
    label_list = torch.tensor(label_list)
    # 将长度列表转换为 PyTorch 张量
    lengths = torch.tensor(lengths)
    # 对文本列表进行填充，使其长度相同，便于组合成一个二维张量
    padded_text_list = nn.utils.rnn.pad_sequence(text_list, batch_first=True)
    # 返回填充后的文本张量、标签张量和长度张量
    return padded_text_list, label_list, lengths


# 创建数据加载器，指定数据集、批次大小、是否打乱数据以及自定义的批处理函数
dataloader = DataLoader(
    train_dataset,  # 训练数据集
    batch_size=4,  # 每个批次的大小
    shuffle=False,  # 不打乱数据顺序
    collate_fn=collate_batch  # 使用自定义的批处理函数
)

# 测试第一个批数据
text_batch, label_batch, lengths = next(iter(dataloader))
print('text_batch：', text_batch)
print('label_batch：', label_batch)
print('lengths：', lengths)
print('text_batch.shape：', text_batch.shape)
print()

# 创建数据加载器
batch_size = 32
train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

# 特征嵌入
# num_embeddings：模型的输入的类别数
# embedding_dim：嵌入向量的长度
# padding_idx：填充词元的索引
embedding = nn.Embedding(num_embeddings=10, embedding_dim=3, padding_idx=0)
# 测试包含两个例子的批数据
# LongTensor 函数用于创建一个数据类型为64位整数（int64）的张量
text_encoded_input = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 0]])
print(embedding(text_encoded_input))
print()


# 创建双层循环神经网络模型
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, hidden = self.rnn(x)
        out = hidden[-1, :, :]
        out = self.fc(out)
        return out


model = RNN(input_size=64, hidden_size=32)
print(model)
print(model(torch.randn(5, 3, 64)))
print()


# 构建情感分析循环神经网络模型
class LSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, rnn_hidden_size, fc_hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.LSTM(embed_dim, rnn_hidden_size, batch_first=True)
        self.fc1 = nn.Linear(rnn_hidden_size, fc_hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc_hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text, lengths):
        out = self.embedding(text)
        # 使用pack_padded_sequence函数对填充后的序列进行打包
        # 以便在RNN处理时能够忽略填充部分，提高计算效率并减少内存占用
        out = nn.utils.rnn.pack_padded_sequence(
            out, lengths.cpu().numpy(), batch_first=True, enforce_sorted=False
        )
        out, (hidden, cell) = self.rnn(out)
        out = hidden[-1, :, :]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out


vocab_size = len(vocab)
embed_dim = 20
rnn_hidden_size = 64
fc_hidden_size = 64
torch.manual_seed(0)
model = LSTM(vocab_size, embed_dim, rnn_hidden_size, fc_hidden_size)
print(model)

# 创建损失函数和优化器
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# 定义训练函数
def train(dataloader):
    # 将模型设置为训练模式
    model.train()
    total_loss, total_acc = 0, 0
    for text_batch, label_batch, lengths in dataloader:
        # 清空优化器的梯度缓存，以免梯度累加
        optimizer.zero_grad()
        pred = model(text_batch, lengths)[:, 0]
        loss = loss_fn(pred, label_batch)
        loss.backward()
        optimizer.step()
        total_acc += ((pred >= 0.5).float() == label_batch).float().sum().item()
        total_loss += loss.item() * label_batch.size(0)
    return total_acc / len(dataloader.dataset), total_loss / len(dataloader.dataset)


# 衡量模型在给定数据集上的性能
def evaluate(dataloader):
    model.eval()
    total_loss, total_acc = 0, 0
    with torch.no_grad():
        for text_batch, label_batch, lengths in dataloader:
            pred = model(text_batch, lengths)[:, 0]
            loss = loss_fn(pred, label_batch)
            total_acc += ((pred >= 0.5).float() == label_batch).float().sum().item()
            total_loss += loss.item() * label_batch.size(0)
        return total_acc / len(dataloader.dataset), total_loss / len(dataloader.dataset)


# 训练模型
num_epochs = 3
torch.manual_seed(0)
for epoch in range(num_epochs):
    acc_train, loss_train = train(train_dl)
    acc_val, loss_val = evaluate(val_dl)
    print(f'Epoch: {epoch:02d} accuracy: {acc_train:.4f}'
          f'   val_accuracy: {acc_val:.4f}')

# 使用测试数据对模型进行评估
acc_test, loss_test = evaluate(test_dl)
print(f'Test accuracy: {acc_test:.4f}')


# 双向循环神经网络
class RNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, rnn_hidden_size, fc_hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # bidirectional设置为True时允许输入序列从两个方向通过循环层
        self.rnn = nn.LSTM(embed_dim, rnn_hidden_size, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(rnn_hidden_size * 2, fc_hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc_hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text, lengths):
        out = self.embedding(text)
        out = nn.utils.rnn.pack_padded_sequence(
            out, lengths.cpu().numpy(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, cell) = self.rnn(out)
        # 在使用双向循环神经网络（如双向LSTM）时，LSTM会有两个隐藏状态：一个来自正向处理，一个来自反向处理
        # hidden[-2, :, :]：表示获取正向LSTM最后一个时间步的隐藏状态
        # hidden[-1, :, :]：表示获取反向LSTM最后一个时间步的隐藏状态
        # torch.cat 用于将输入的张量沿着指定的维度连接在一起
        out = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out


torch.manual_seed(0)
model = RNN(vocab_size, embed_dim, rnn_hidden_size, fc_hidden_size)
print(model)