import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# 加载数据
df = pd.read_csv("C:/Users/21336/Documents/WXWork/1688858313404405/Cache/File/2025-01/Percentage of households by "
                 "income brackets_ other cities.csv")
df = df[['Entity properties name', 'Variable observation value']]

# 将数据整理成适合绘图的格式
columns = ['Area Name', '$10,000 or Less', '$10,000-$14,999', '$20,000-$24,999', '$40,000-$44,999',
           '$50,000-$59,999', '$60,000-$74,999', '$75,000-$99,999', '$100,000-$124,999',
           '$125,000-$149,999', '$150,000-$199,999', '$200,000 or More']
Data = []

# 遍历原始数据，整理成新的格式
for name, group in df.groupby('Entity properties name'):
    data = [name]
    val = group['Variable observation value']
    for i in range(len(val)):
        data.append(val.iloc[i])
    Data.append(data)

# 创建整理后的DataFrame
df_cleaned = pd.DataFrame(Data, columns=columns)

# 绘制分组条形图
fig, ax = plt.subplots(figsize=(16, 6))
plt.rcParams['font.sans-serif'] = ['Lucida Sans Unicode']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'cm'

# 设置条形图的宽度
bar_width = 0.09
index = range(len(df_cleaned))

# 设置Seaborn风格
sns.set(style="whitegrid")
# 定义颜色方案
colors = sns.color_palette('viridis', len(columns[1:]))

# 绘制每个收入区间的条形图
for i, col in enumerate(columns[1:]):
    print(i, col)
    ax.bar([x + bar_width * i for x in index], df_cleaned[col], bar_width, label=col, color=colors[i])

# 添加图例
ax.legend(title=r'$Income\ Range$', bbox_to_anchor=(1.05, 0, 0.2, 1), loc='center left',
          ncol=1, prop={'size': 10, 'style': 'italic'})

# 设置X轴标签
ax.set_xticks([x + bar_width * (len(columns) - 2) / 2 for x in index])
ax.set_xticklabels(df_cleaned['Area Name'], rotation=45, ha='right', fontsize=8)

# 添加标题和轴标签
ax.set_title(r'$Household\ Income\ Distribution\ By\ Region$', fontsize=14)
ax.set_xlabel(r'$Area\ Name$', fontsize=12)
ax.set_ylabel(r'$Number\ of\ Households$', fontsize=12)

# 添加网格线
ax.yaxis.grid(True, linestyle='--', linewidth=0.5, color='gray')

# 显示图表
plt.tight_layout()
plt.show()