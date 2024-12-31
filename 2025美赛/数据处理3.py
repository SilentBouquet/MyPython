import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取Excel文件
file_path = "C:/Users/21336/Documents/WXWork/1688858313404405/Cache/File/2025-01/西安虚构经济数据.xlsx"
df = pd.read_excel(file_path)

df.columns = ['Year', 'Price Index (CPI)', 'Number of resident labor force (10,000 people)',
              'Historical and cultural protection funds', 'Number of historical and cultural sites (locations)',
              'Crowding level (people/square kilometer)', 'Tourism and cultural investment funds',
              'GDP (100 million yuan)']

# 数据清洗：删除重复行和说明行
df = df.drop_duplicates()
df = df[df['Year'].apply(lambda x: isinstance(x, int))]

# 转换年份为整数
df['Year'] = df['Year'].astype(int)

# 绘制对比图
plt.figure(figsize=(12, 6))

# 将数据转换为长格式
df_melted = df.melt(id_vars=['Year'], value_vars=['Historical and cultural protection funds',
                                                  'Tourism and cultural investment funds'], var_name='资金类型',
                    value_name='资金数（亿元）')

# 绘制柱状图
plt.rcParams['font.sans-serif'] = ['Lucida Sans Unicode']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'cm'
sns.barplot(data=df_melted, x='Year', y='资金数（亿元）', hue='资金类型', palette='plasma_r', alpha=0.4)
plt.title(r'$Comparison\ chart\ of\ different\ investment\ funds$', fontsize=16)
plt.xlabel('$Year$', fontsize=14)
plt.ylabel(r'$Amount\ of\ funds\ (100\ million\ yuan)$', fontsize=14)
plt.legend()
plt.show()