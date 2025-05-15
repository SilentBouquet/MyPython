import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer

# 字符词元化
text = "Tokenizing text is a core task of NLP."
tokenized_text = list(text)
print(tokenized_text)

token2idx = {ch: idx for idx, ch in enumerate(sorted(set(tokenized_text)))}
print(token2idx)

input_ids = [token2idx[token] for token in tokenized_text]
print(input_ids)

# 独热编码
input_ids = torch.tensor(input_ids)
one_hot_encodings = F.one_hot(input_ids, num_classes=len(token2idx))
print(one_hot_encodings.shape)
print(f'Token: {tokenized_text[0]}')
print(f'Tensor index: {input_ids[0]}')
print(f'One-hot encodings: {one_hot_encodings[0]}')

# 单词词元化
tokenized_text = text.split()
print(tokenized_text)

# 子词词元化
model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
encoded = tokenizer(text)
print(encoded)

tokens = tokenizer.convert_ids_to_tokens(encoded.input_ids)
print(tokens)
print(tokenizer.convert_tokens_to_string(tokens))

print("词表大小：", tokenizer.vocab_size)
print("模型的最大上下文大小：", tokenizer.model_max_length)
print("模型期望的字段名称：", tokenizer.model_input_names)


# 对整个数据集进行词元化
def tokenize(batch):
    # truncation=True：将样本截断为模型的最大上下文大小
    return tokenizer(batch["text"], padding=True, truncation=True)


emotions = load_dataset('emotion')
print(tokenize(emotions["train"][:2]))

emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)
print(emotions_encoded["train"].column_names)