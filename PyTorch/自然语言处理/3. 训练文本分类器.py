import torch
import matplotlib
import numpy as np
import pandas as pd
from umap.umap_ import UMAP
from datasets import load_dataset
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from transformers import AutoModel, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

model_ckpt = "distilbert-base-uncased"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModel.from_pretrained(model_ckpt).to(device)

text = "This is a text."
inputs = tokenizer(text, return_tensors="pt")
print(f"Input tensor shape: {inputs['input_ids'].size()}")
print(inputs)

# 提取最终的隐藏状态
inputs = {k: v.to(device) for k, v in inputs.items()}
print(inputs)
with torch.no_grad():
    outputs = model(**inputs)
print(outputs)
print(outputs.last_hidden_state.size())
print(outputs.last_hidden_state[:, 0].size())


def extract_hidden_states(batch):
    inputs = {k: v.to(device) for k, v in batch.items()
              if k in tokenizer.model_input_names}
    with torch.no_grad():
        last_hidden_state = model(**inputs).last_hidden_state
        return {"hidden_state": np.asarray(last_hidden_state[:, 0].cpu())}


# 对整个数据集进行词元化
def tokenize(batch):
    # truncation=True：将样本截断为模型的最大上下文大小
    return tokenizer(batch["text"], padding=True, truncation=True)


emotions = load_dataset('emotion')
emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)
emotions_encoded.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
emotions_hidden = emotions_encoded.map(extract_hidden_states, batched=True)
print(emotions_hidden['train'].column_names)

# 创建特征矩阵
X_train = np.asarray(emotions_hidden['train']['hidden_state'])
y_train = np.asarray(emotions_hidden['train']['label'])
X_val = np.asarray(emotions_hidden['validation']['hidden_state'])
y_val = np.asarray(emotions_hidden['validation']['label'])
print("X_train shape:", X_train.shape)
print("X_val shape:", X_val.shape)

# 可视化训练集
X_scaled = MinMaxScaler().fit_transform(X_train)
# 将高维数据降维到二维，以便于可视化
mapper = UMAP(n_components=2).fit(X_scaled)
# mapper.embedding_：UMAP 对象的属性，包含降维后的坐标
df_emb = pd.DataFrame(mapper.embedding_, columns=['X', 'Y'])
df_emb["label"] = y_train
print(df_emb.head())

# 绘制点密度图
matplotlib.use("Qt5Agg")
fig, axes = plt.subplots(2, 3, figsize=(10, 6))
# 将二维子图数组展平为一维数组，方便后续循环处理
axes = axes.flatten()
cmaps = ["Greys", "Blues", "Oranges", "Reds", "Purples", "Greens"]
labels = emotions["train"].features["label"].names

for i, (label, color) in enumerate(zip(labels, cmaps)):
    # query 方法是 Pandas 中用于筛选数据的一个方法，它允许你使用字符串形式的表达式来过滤 DataFrame 中的行
    df_emb_sub = df_emb.query(f"label == {i}")
    # hexbin 用于绘制二维点密度图
    # gridsize=20：将数据点划分为 20x20 的六边形单元格
    # linewidths=(0,)：设置六边形边框的宽度为 0，以避免显示边框
    axes[i].hexbin(df_emb_sub["X"], df_emb_sub["Y"], gridsize=20, cmap=color, linewidths=(0,))
    axes[i].set_title(label)
    axes[i].set_xticks([])
    axes[i].set_yticks([])
plt.tight_layout()
plt.show()

# 模型训练
lr_clf = LogisticRegression(max_iter=3000)
lr_clf.fit(X_train, y_train)
print(lr_clf.score(X_val, y_val))


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


y_pred = lr_clf.predict(X_val)
plot_confusion_matrix(y_pred, y_val, labels)