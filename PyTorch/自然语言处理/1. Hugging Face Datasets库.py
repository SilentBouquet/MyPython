import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset
from huggingface_hub import list_datasets

'''
all_datasets = list(list_datasets())
print(f'一共有{len(all_datasets)}个可从Hub正确加载的数据集')
print(f'前十个数据集分别为：{all_datasets[:10]}')
'''

emotions = load_dataset('emotion')
print(emotions)

train_ds = emotions['train']
print(train_ds)
print(len(train_ds))
print(train_ds[0])
print(train_ds.column_names)
print(train_ds.features)
print(train_ds[:5])
print(train_ds["text"][:5])

# 从Datasets到DataFrame
emotions.set_format('pandas')
df = emotions['train'][:]
print(df.head())


def label_int2str(row):
    return emotions['train'].features['label'].int2str(row)


# apply 方法用于对数据进行自定义处理。它支持对 Series 或 DataFrame 的每个元素、行或列应用函数
df["label_name"] = df["label"].apply(label_int2str)
print(df.head())

# 查看类分布
matplotlib.use("Qt5Agg")
# value_counts() 用于统计 Series 中每个唯一值的出现次数
# plot.barh() 用于绘制水平条形图
df["label_name"].value_counts(ascending=True).plot.barh()
plt.title(r"$Frequency\ of\ Classes$")
plt.show()

# 计算每条推文的单词数，并将结果存储到一个新的列中
df["Words Per Tweet"] = df["text"].str.split().apply(len)
# showfliers=False：不显示异常值（离群点）
df.boxplot(column="Words Per Tweet", by="label_name", grid=False,
           showfliers=False, color="black")
plt.suptitle("")
plt.xlabel("")
plt.show()

# 重置数据集的输出格式
emotions.reset_format()