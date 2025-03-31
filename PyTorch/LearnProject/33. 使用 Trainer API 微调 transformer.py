import gzip
import shutil
import time
import torch
import pandas as pd
import numpy as np
import wandb
from evaluate import load
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
from transformers import Trainer, TrainingArguments

# 登录 W&B 并配置 API 密钥
wandb.login(key="2e5d5b34a134df5f26ce18b50c1972c899725ed7")

# 设置模型参数
torch.backends.cudnn.deterministic = True
torch.manual_seed(123)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 2

# 加载并处理数据
url = "https://github.com/rasbt/machine-learning-book/raw/main/ch08/movie_data.csv.gz"
file_name = "F:/Deep Learning Datasets/movie_data.csv.gz"

'''
with open(file_name, "wb") as f:
    r = requests.get(url)
    f.write(r.content)
'''

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
metric = load("accuracy", trust_remote_code=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


training_args = TrainingArguments(
    output_dir='../Runs/Fine-tuning-BERT',
    num_train_epochs=num_epochs,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    logging_dir=None,
    logging_steps=1e6,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    optimizers=(optimizer, None)
)

start_time = time.time()
trainer.train()

print(f'Total Training time: {(time.time() - start_time) / 60:.2f} min')
print(trainer.evaluate())