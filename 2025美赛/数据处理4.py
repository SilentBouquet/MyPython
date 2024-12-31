import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取Excel文件
file_path = "C:/Users/21336/Documents/WXWork/1688858313404405/Cache/File/2025-01/西安虚构经济数据.xlsx"
df = pd.read_excel(file_path)

# 数据清洗：删除重复行和说明行
df = df.drop_duplicates()
df = df[df['年份'].apply(lambda x: isinstance(x, int))]

# 转换年份为整数
df['年份'] = df['年份'].astype(int)

# 绘制关系图
fig, ax1 = plt.subplots(figsize=(14, 8))

# 绘制价格指数
sns.lineplot(data=df, x='年份', y='价格指数（CPI）', marker='o', label='价格指数（CPI）', ax=ax1, color='tab:blue')
ax1.set_xlabel('年份')
ax1.set_ylabel('价格指数（CPI）', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')

# 创建第二个y轴
ax2 = ax1.twinx()

# 绘制GDP和常驻劳动人口
sns.lineplot(data=df, x='年份', y='GDP（亿元）', marker='o', label='GDP（亿元）', ax=ax2, color='tab:orange')
sns.lineplot(data=df, x='年份', y='常驻劳动力人口数（万人）', marker='o', label='常驻劳动力人口数（万人）', ax=ax2, color='tab:green')
ax2.set_ylabel('GDP（亿元）/常驻劳动力人口数（万人）', color='tab:orange')
ax2.tick_params(axis='y', labelcolor='tab:orange')

# 添加图例
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper left', fontsize=12, markerscale=1.5)

# 添加标题
plt.title('价格指数、GDP和常驻劳动人口的关系')

# 显示图表
plt.show()