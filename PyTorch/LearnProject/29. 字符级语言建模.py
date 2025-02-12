import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
from torch.utils.data import Dataset, DataLoader

# 数据加载与预处理
with open(r"F:\Deep Learning Datasets\文本数据集\博尔赫斯小说集.txt", 'r', encoding='utf-8') as f:
    text = f.read().replace('\n', ' ')
char_set = set(text)
print('Total Length：', len(text))
print('Unique Characters：', len(char_set))

# 构建字典，将字符映射成整数，并通过Numpy数组索引实现反向映射
chars_sorted = sorted(char_set)
char_to_int = {ch: i for i, ch in enumerate(chars_sorted)}
char_array = np.array(chars_sorted)
text_encoded = np.array([char_to_int[ch] for ch in text], dtype=np.int32)
print('Text encoded shape', text_encoded.shape)
# print(text[:20], '== Encoding ==>', text_encoded[:20])
# print(text_encoded[21:30], '== Reverse ==>', ''.join(char_array[text_encoded[21:30]]))

# 将编码文本分割成文本块
seq_length = 40
chunk_size = seq_length + 1
text_chunks = [text_encoded[i:i+chunk_size] for i in range(len(text_encoded) - chunk_size + 1)]


class TextDataset(Dataset):
    def __init__(self, text_chunks):
        self.text_chunks = text_chunks

    def __len__(self):
        return len(self.text_chunks)

    def __getitem__(self, idx):
        text_chunk = self.text_chunks[idx]
        return text_chunk[:-1].long(), text_chunk[1:].long()


seq_dataset = TextDataset(torch.tensor(np.array(text_chunks)))
# 查看转换后数据集中的一些序列样本
for i, (seq, target) in enumerate(seq_dataset):
    print('  Input (x)：', repr(' '.join(char_array[seq])))
    print('  Target (y)：', repr(' '.join(char_array[target])))
    print()
    if i == 1:
        break

# 将数据集拆分为小批量数据集
batch_size = 64
torch.manual_seed(0)
seq_dl = DataLoader(seq_dataset, batch_size=batch_size, shuffle=True, drop_last=True)


# 构建循环神经网络
class RNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, rnn_hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn = nn.LSTM(embed_dim, rnn_hidden_size, batch_first=True)
        self.fc = nn.Linear(rnn_hidden_size, vocab_size)

    def forward(self, x, hidden, cell):
        # unsqueeze(1)在第二个维度上增加一个维度，以便与LSTM的输入形状匹配
        out = self.embedding(x).unsqueeze(1)
        out, (hidden, cell) = self.rnn(out, (hidden, cell))
        out = self.fc(out).reshape(out.size(0), -1)
        return out, hidden, cell

    def init_hidden(self, batch_size):
        hidden = torch.zeros(1, batch_size, self.rnn_hidden_size)
        cell = torch.zeros(1, batch_size, self.rnn_hidden_size)
        return hidden, cell


# 初始化模型参数
vocab_size = len(char_array)
embed_dim = 256
rnn_hidden_size = 512
torch.manual_seed(0)
model = RNN(vocab_size, embed_dim, rnn_hidden_size)
print(model)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

'''
# 训练模型
num_epochs = 10000
torch.manual_seed(0)
for epoch in range(num_epochs):
    hidden, cell = model.init_hidden(batch_size)
    seq_batch, target_batch = next(iter(seq_dl))
    optimizer.zero_grad()
    loss = 0
    # 每次循环处理序列中的一个字符位置
    for c in range(seq_length):
        pred, hidden, cell = model(seq_batch[:, c], hidden, cell)
        loss += loss_fn(pred, target_batch[:, c])
    loss.backward()
    optimizer.step()
    loss = loss.item() / seq_length
    if epoch % 500 == 0:
        print(f'Epoch  {epoch}  loss：{loss:.4f}')


# 保存模型
torch.save(model.state_dict(), '../My_Model/Borges.pth')
print("模型已保存")
'''

# 加载模型
model.load_state_dict(torch.load('../My_Model/Borges.pth'))
print("模型已加载")


# 文本生成函数
# scale_factor 为生成字符概率分布的缩放因子
def sample(model, starting_str, len_generated_text=1000, scale_factor=1.0):
    encoded_input = torch.tensor([char_to_int[s] for s in starting_str])
    # -1表示自动计算序列的长度
    encoded_input = torch.reshape(encoded_input, (1, -1))
    generated_str = starting_str
    # 将模型设置为评估模式，关闭dropout等训练时特有的操作
    model.eval()
    hidden, cell = model.init_hidden(1)
    # 将起始字符串的每个字符依次输入到模型中，更新模型的隐藏状态和细胞状态
    # 从而使模型能够基于起始字符串的上下文开始生成新文本
    for c in range(len(starting_str) - 1):
        # view(1) 用于重新塑形为一个二维张量
        _, hidden, cell = model(encoded_input[:, c].view(1), hidden, cell)
    # 作用：获取起始字符串的最后一个字符的编码。
    # 目的：在生成新文本时，从这个最后一个字符开始继续生成
    last_char = encoded_input[:, -1]
    for i in range(len_generated_text):
        logits, hidden, cell = model(last_char.view(1), hidden, cell)
        # 去除logits张量中维度为1的维度，使其形状变为(vocab_size,)
        logits = logits.squeeze(0)
        scaled_logits = logits * scale_factor
        # 创建分类分布对象
        m = Categorical(logits=scaled_logits)
        # 从分布m中采样得到一个字符的索引，作为下一个字符
        last_char = m.sample()
        generated_str += str(char_array[last_char])
        if len(generated_str) % 80 == 0:
            generated_str += '\n'
    return generated_str


# 生成新文本
torch.manual_seed(0)
print(sample(model, starting_str='小径分岔的花园是一个', scale_factor=3))