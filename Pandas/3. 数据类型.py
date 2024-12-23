import pandas as pd

data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'Score': [90, 85, 88]
}
df = pd.DataFrame(data)

# 数据类型分类
# 整数类型
s = pd.Series([1, 2, 3, 4, 5])
print(s.dtype)
# 浮点类型
s = pd.Series([1.0, 2.5, 3.7, 4.2, 5.9])
print(s.dtype)
# 字符串类型
s = pd.Series(['apple', 'banana', 'grape', 'kiwi'])
print(s.dtype)
# 布尔类型
s = pd.Series([True, False, True, False])
print(s.dtype)
# 日期时间类型
s = pd.Series(['2024-03-10', '2024-03-11', '2024-03-12'])
s = pd.to_datetime(s)
print(s.dtype, '\n', s)
# 分类类型
s = pd.Series(['male', 'female', 'female', 'male'])
s = s.astype('category')
print(s.dtype, '\n', s)

# 分类类型
data = ['A', 'B', 'C', 'A', 'B', 'C']
s = pd.Series(data)
cat_series = pd.Categorical(s)
# 查看分类的唯一值列表
print('\n', cat_series.categories)
# 查看每个元素在分类中的位置索引
print('\n', cat_series.codes)
# 指定分类变量的顺序
cat_series = pd.Categorical(s, categories=['A', 'B', 'C'], ordered=True)
print('\n', cat_series)
# 使用分类类型数据
cat_series = cat_series.sort_values()
print('\n', cat_series)