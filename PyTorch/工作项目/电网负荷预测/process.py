import pandas as pd
import numpy as np

# 读取数据
file_path = 'data.xlsx'
xls = pd.ExcelFile(file_path)

# 提取第一个地区的电力负荷数据和天气数据
area1_load = pd.read_excel(xls, sheet_name='Area1_Load')
area1_weather = pd.read_excel(xls, sheet_name='Area1_Weather')

# 重命名天气数据的日期列
area1_weather.rename(columns={area1_weather.columns[0]: 'YMD'}, inplace=True)

# 确保天气数据的列名正确
area1_weather.columns = ['YMD', '最高温度℃', '最低温度℃', '平均温度℃', '相对湿度(平均)', '降雨量（mm）']

# 将YMD列转换为datetime格式
area1_load['YMD'] = pd.to_datetime(area1_load['YMD'].astype(str), format='%Y%m%d')
area1_weather['YMD'] = pd.to_datetime(area1_weather['YMD'].astype(str), format='%Y%m%d')

# 将电力负荷数据从15分钟间隔转换为每小时间隔
# 提取每小时的四个15分钟数据点并计算平均值
hourly_load = []
for _, row in area1_load.iterrows():
    date = row['YMD'].strftime('%Y-%m-%d')  # 将日期格式化为字符串
    for hour in range(24):
        # 提取该小时的四个15分钟数据点
        values = [
            row.get(f'T{hour:02d}00', np.nan),
            row.get(f'T{hour:02d}15', np.nan),
            row.get(f'T{hour:02d}30', np.nan),
            row.get(f'T{hour:02d}45', np.nan)
        ]
        # 计算平均值，如果所有值都是NaN，则保留为NaN
        if all(pd.isna(value) for value in values):
            avg_value = np.nan
        else:
            avg_value = np.nanmean(values)
        hourly_load.append({
            'YMD': date,
            'Load': avg_value
        })

hourly_load_df = pd.DataFrame(hourly_load)


# 异常值检测和处理
def detect_and_replace_outliers(df, column):
    # 使用Z-Score方法检测异常值
    df['Z-Score'] = (df[column] - df[column].mean()) / df[column].std()
    # 使用均值替换异常值
    # np.where是一个NumPy函数，用于根据条件从两个数组中选择元素
    df[column] = np.where(np.abs(df['Z-Score']) > 3, df[column].mean(), df[column])
    # inplace=True：操作会直接在原始数据框或序列上进行修改
    df.drop(columns=['Z-Score'], inplace=True)
    return df


# 对电力负荷数据进行异常值处理
hourly_load_df = detect_and_replace_outliers(hourly_load_df, 'Load')

# 对天气数据进行异常值处理
for column in ['最高温度℃', '最低温度℃', '平均温度℃', '相对湿度(平均)', '降雨量（mm）']:
    area1_weather = detect_and_replace_outliers(area1_weather, column)

# 按指定日期范围提取数据
filtered_load = hourly_load_df[(hourly_load_df['YMD'] >= '2012-01-01') & (hourly_load_df['YMD'] <= '2015-01-10')].copy()
filtered_weather = area1_weather[(area1_weather['YMD'] >= '2012-01-01') & (area1_weather['YMD'] <= '2015-01-10')].copy()

# 将天气数据的 YMD 列转换为字符串格式
filtered_weather['YMD'] = filtered_weather['YMD'].dt.strftime('%Y-%m-%d')

# 合并数据
merged_data = pd.merge(filtered_load, filtered_weather, on='YMD', how='left')

# 提取测试数据（2015-01-11到2015-01-17）
test_weather = area1_weather[(area1_weather['YMD'] >= '2015-01-11') & (area1_weather['YMD'] <= '2015-01-17')].copy()

# 将测试天气数据的 YMD 列转换为字符串格式
test_weather['YMD'] = test_weather['YMD'].dt.strftime('%Y-%m-%d')

# 保存处理好的数据到Excel文件
merged_data.to_excel('processed_data.xlsx', index=False)
test_weather.to_excel('processed_test_data.xlsx', index=False)

print("数据预处理完成，处理好的数据已保存到processed_data.xlsx和processed_test_data.xlsx。")