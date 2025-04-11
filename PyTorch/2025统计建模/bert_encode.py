import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from transformers import BertTokenizer, BertModel

# 加载数据
file_path = 'processed_transactions.xlsx'
data = pd.read_excel(file_path)

# 特征和标签
X = data.drop(columns=['是否欺诈'])  # 特征
y = data['是否欺诈']  # 标签

# 确保所有特征都是数值型（除了文本描述）
for col in X.columns:
    if col != '交易描述' and X[col].dtype == 'object':
        X[col] = X[col].astype(str).astype('category').cat.codes


# 使用 BERT 对文本进行编码并输出一个数字
class BertTextEncoderWithOutput(nn.Module):
    def __init__(self, model_name='bert-base-uncased', max_length=32, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(BertTextEncoderWithOutput, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name)
        self.bert.to(device)
        self.bert.eval()  # 设置为评估模式
        self.max_length = max_length
        self.device = device
        # 添加一个全连接层，将 BERT 的输出映射到一个数字
        self.fc = nn.Linear(self.bert.config.hidden_size, 1).to(device)

    def encode(self, texts, batch_size=32):
        all_outputs = []
        # 分批处理文本
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            inputs = self.tokenizer(
                batch_texts.tolist(),
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            # 获取 BERT 的输出
            with torch.no_grad():
                outputs = self.bert(**inputs)
            # 提取 [CLS] 向量
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            # 通过全连接层得到一个数字
            batch_outputs = self.fc(cls_embeddings).squeeze()
            # 使用 detach() 分离梯度，然后转换为 NumPy 数组
            all_outputs.append(batch_outputs.detach().cpu().numpy())
            # 清理缓存
            torch.cuda.empty_cache()
        # 合并所有批次的输出
        all_outputs = np.concatenate(all_outputs, axis=0)
        return all_outputs


# 初始化 BERT 编码器
bert_encoder = BertTextEncoderWithOutput()

# 编码文本描述并输出一个整数
text_descriptions = X['交易描述'].values
encoded_values = bert_encoder.encode(text_descriptions, batch_size=32)

# 将编码后的整数替换到原始的“交易描述”列中
data['交易描述'] = encoded_values

# 指定需要归一化的列
columns_to_normalize = ['年龄', '交易金额', '账户余额', '交易描述']

# 使用 Min-Max 归一化
scaler = MinMaxScaler(feature_range=(0, 1))
data[columns_to_normalize] = scaler.fit_transform(data[columns_to_normalize])

# 保存编码后的数据
data.to_excel('encoded_transactions.xlsx', index=False)

print("数据编码完成并保存到 encoded_transactions.xlsx")