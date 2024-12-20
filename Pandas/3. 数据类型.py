import pandas as pd

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
