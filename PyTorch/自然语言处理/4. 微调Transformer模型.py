import torch
import matplotlib
import numpy as np
import pandas as pd
from datasets import load_dataset
from huggingface_hub import login
from matplotlib import pyplot as plt
from torch.nn.functional import cross_entropy
from transformers import AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments, AutoTokenizer, pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

num_labels = 6
model_ckpt = "distilbert-base-uncased"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModelForSequenceClassification.from_pretrained(
    model_ckpt, num_labels=num_labels, output_hidden_states=True
).to(device)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions[0].argmax(axis=-1)
    f1 = f1_score(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}


# 对整个数据集进行词元化
def tokenize(batch):
    # truncation=True：将样本截断为模型的最大上下文大小
    return tokenizer(batch["text"], padding=True, truncation=True)


def extract_hidden_states(batch):
    inputs = {k: v.to(device) for k, v in batch.items()
              if k in tokenizer.model_input_names}
    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden_state = outputs.hidden_states[-1]  # 获取最后一层的隐藏状态
        return {"hidden_state": np.asarray(last_hidden_state[:, 0].cpu())}


# 查看混淆矩阵
def plot_confusion_matrix(y_preds, y_true, labels):
    # 计算归一化的混淆矩阵
    cm = confusion_matrix(y_true, y_preds, normalize='true')
    fig, ax = plt.subplots(figsize=(8, 8))
    # 可视化混淆矩阵
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)
    plt.title("Normalized confusion matrix")
    plt.tight_layout()
    plt.show()


emotions = load_dataset('emotion')
emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)
emotions_encoded.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
emotions_hidden = emotions_encoded.map(extract_hidden_states, batched=True)

# 创建特征矩阵
X_train = np.asarray(emotions_hidden['train']['hidden_state'])
y_train = np.asarray(emotions_hidden['train']['label'])
X_val = np.asarray(emotions_hidden['validation']['hidden_state'])
y_val = np.asarray(emotions_hidden['validation']['label'])
labels = emotions["train"].features["label"].names

batch_size = 64
logging_steps = len(emotions_encoded["train"]) // batch_size
model_name = f"../My_Model/{model_ckpt}"
training_args = TrainingArguments(
    output_dir=model_name,
    num_train_epochs=2,
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    # 权重衰减（L2 正则化）的系数，用于防止过拟合
    weight_decay=0.01,
    # 评估策略设置为每个 epoch 结束时进行一次评估
    evaluation_strategy="epoch",
    disable_tqdm=False,
    logging_steps=logging_steps,
    push_to_hub=False,
    # 日志级别设置为 "error"，只记录错误级别的日志
    log_level="error"
)

trainer = Trainer(
    model=model, args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=emotions_encoded["train"],
    eval_dataset=emotions_encoded["validation"],
    tokenizer=tokenizer
)

trainer.train()

# 查看训练指标
preds_output = trainer.predict(emotions_encoded["validation"])
print(preds_output.metrics)
# 查看混淆矩阵
y_preds = np.argmax(preds_output.predictions[0], axis=1)
matplotlib.use("Qt5Agg")
plot_confusion_matrix(y_preds, y_val, labels)


# 返回损失以及预测标注
def forward_pass_with_label(batch):
    inputs = {k: v.to(device) for k, v in batch.items()
              if k in tokenizer.model_input_names}

    with torch.no_grad():
        outputs = model(**inputs)
        pred_labrl = torch.argmax(outputs.logits, dim=-1)
        loss = cross_entropy(outputs.logits, batch["label"].to(device), reduction='none')

    return {"loss": loss.cpu().numpy(), "predicted_label": pred_labrl.cpu().numpy()}


emotions_encoded.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
emotions_encoded["validation"] = emotions_encoded["validation"].map(
    forward_pass_with_label, batched=True, batch_size=16
)
emotions_encoded.set_format("pandas")
cols = ["text", "label", "predicted_label", "loss"]
df_test = emotions_encoded["validation"][:][cols]


def label_int2str(row):
    return emotions['train'].features['label'].int2str(row)


df_test["label"] = df_test["label"].apply(label_int2str)
df_test["predicted_label"] = df_test["predicted_label"].apply(label_int2str)

# 查看损失最高的数据样本
print(df_test.sort_values(by="loss", ascending=False).head(10))
# 查看损失最小的数据样本呢
print(df_test.sort_values(by="loss", ascending=True).head(10))

# 存储与共享模型
trainer.push_to_hub(commit_message="Training completed!")

# 使用微调模型对新推文进行预测
model_id = f"SilentBouquet/{model_ckpt}"
classifier = pipeline("text-classification", model=model_id)

tweet = "I saw a movie today and it was really good."
preds = classifier(tweet, return_all_scores=True)
pred_df = pd.DataFrame(preds[0])
plt.bar(labels, 100 * pred_df["score"], color='C0')
plt.title(f'{tweet}')
plt.ylabel("Class probability (%)")
plt.tight_layout()
plt.show()