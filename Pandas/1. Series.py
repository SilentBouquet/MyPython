import pandas as pd
import numpy as np

# 创建Series
data = [1, 2, 3, 4, 5]
series = pd.Series(data)
print(series)
data_np = np.array(data)
series_np = pd.Series(data)
print(series_np)

# 默认索引
series_default_index = pd.Series(data)
# 自定义索引
custom_index = ['a', 'b', 'c', 'd', 'e']
series_custom_index = pd.Series(data, index=custom_index)
print(series_custom_index)

# 数据访问
print(series[0])
print(series_custom_index['b'])

# 数据操作
print(series * 2)
print(np.sqrt(series))
print(series > 3)

# 缺失值处理
data_with_nan = [1, 2, np.nan, 4, 5, np.nan]
series_with_nan = pd.Series(data_with_nan)
mask_nan = series_with_nan.isnull()     # 检测缺失值
series_filled = series_with_nan.fillna(0)      # 填充缺失值
print(series_filled)

# 名称和属性
series_with_name = pd.Series(data, name='MySeries')
index = series.index
dtype = series.dtype
print(series_with_name)
print(index, dtype)