import pandas as pd

# 从字典创建DataFrame
data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'Score': [90, 85, 88]
}
df = pd.DataFrame(data)
print(df)

# 从列表创建DataFrame
data = [['Alice', 25, 90], ['Bob', 30, 85], ['Charlie', 35, 88]]
df = pd.DataFrame(data, columns=['Name', 'Age', 'Score'])
print('\n', df)

# 从外部数据源读取，可以包含路径
# df = pd.read_csv('data.csv')
# print(df)

# 查看DataFrame
print('\n', df.head())    # 查看头部几行数据
print('\n', df.tail())      # 查看尾部几行数据
print('\n', df.shape)     # 查看数据的维度
print('\n', df.columns)      # 查看列名
print('\n', df.index)     # 查看索引
print('\n', df.describe())      # 查看数据摘要统计信息
print('\n', df.dtypes)      # 查看数据类型

# 数据访问
print('\n', df['Name'])     # 访问列
print('\n', df.iloc[0])     # 使用整数位置进行行访问
print('\n', df.loc[0])      # 使用标签进行行访问
print('\n', df.at[0, 'Name'])      # 使用标签访问特定行列的值
print('\n', df.iat[0, 0])     # 使用整数位置访问特定行列的值

# 数据操作
df['New_Column'] = [1, 2, 3]     # 添加列
df.drop('New_Column', axis=1, inplace=True)     # 删除列
df.at[0, 'Age'] = 26     # 修改数据
selected_data = df[df['Age'] >= 30]     # 根据条件选择数据
print('\n', selected_data)
df.sort_values(by='Score', ascending=False, inplace=True)      # 对数据进行排序
print('\n', df)

# 数据分析
df.drop('Name', axis=1, inplace=True)
print('\n', df.mean())     # 计算平均值
grouped = df.groupby(['Age'])
print('\n', grouped.mean())     # 分组统计
print('\n', df['Age'].sum())      # 总和
print('\n', df['Age'].max())      # 最大值
print('\n', df['Age'].min())      # 最小值
print('\n', df['Age'].median())     # 中位数