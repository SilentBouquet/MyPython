import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_excel("C:/Users/21336/Documents/WXWork/1688858313404405/Cache/File/2025-01/朱诺经济数据.xlsx", sheet_name="饼图")

# 提取需要的列
income_structure = data[['Unnamed: 11', 'Unnamed: 10']].copy()

# 确保 Count 列是整数类型
income_structure['Unnamed: 10'] = income_structure['Unnamed: 10'].astype(int)

# 定义颜色方案
cmap = plt.get_cmap('Pastel1')
colors = cmap(income_structure['Unnamed: 10'] / income_structure['Unnamed: 10'].max())

# 绘制饼图
plt.figure(figsize=(12, 10))
plt.rcParams['font.sans-serif'] = ['Lucida Sans Unicode']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'cm'
wedges, texts, autotexts = plt.pie(
    income_structure['Unnamed: 10'],
    labels=income_structure['Unnamed: 11'],
    autopct='%1.1f%%',
    startangle=140,
    colors=colors,
    shadow=True,
    textprops={'fontsize': 10, 'color': 'black'},
    wedgeprops={'edgecolor': 'white', 'linewidth': 0.5}
)

# 调整百分比标签的字体大小和颜色
for autotext in autotexts:
    autotext.set_size(10)
    autotext.set_color('black')

# 添加标题
plt.title('Income Structure in Juneau City and Borough (2022)', fontsize=16, fontweight='bold')
plt.show()

data = pd.read_excel("C:/Users/21336/Documents/WXWork/1688858313404405/Cache/File/2025-01/朱诺经济数据.xlsx", sheet_name="朱诺经济指标")

# 提取需要的列
economic_data = data[['year', '人口数量', '贫穷人口', '劳动力人口数']].copy()

# 确保年份列是整数类型
economic_data['year'] = economic_data['year'].astype(int)

# 绘制折线图
plt.figure(figsize=(12, 6))
plt.plot(economic_data['year'], economic_data['人口数量'], label='$Population$', marker='o', color='orangered', linestyle='-', linewidth=1, markersize=8)
plt.plot(economic_data['year'], economic_data['贫穷人口'], label=r'$Poor\ People$', marker='s', color='forestgreen', linestyle='--', linewidth=1, markersize=8)
plt.plot(economic_data['year'], economic_data['劳动力人口数'], label=r'$Labor\ force\ Population$', marker='^', color='dodgerblue', linestyle='-.', linewidth=1, markersize=8)
plt.title(r'$Population,\ poverty\ and\ labor\ force\ changes\ over\ time$', fontsize=16, fontweight='bold')
plt.legend(fontsize=12, loc='upper left')
plt.xlabel('$Year$', fontsize=12)
plt.ylabel('$Population$', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(economic_data['year'], rotation=45)
plt.ylim(0, 45000)
plt.yticks(range(0, int(economic_data[['人口数量', '贫穷人口', '劳动力人口数']].max().max() * 1.1), 5000))
plt.tight_layout()
plt.show()