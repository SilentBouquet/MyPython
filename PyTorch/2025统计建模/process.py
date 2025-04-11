import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 定义路径
path = r"C:\Users\21336\Documents\WXWork\1688858313404405\Cache\File\2025-03\Bank_Transaction_Fraud_Detection.xlsx"

# 定义要处理的列
drop_features = ['顾客ID', '交易ID', '商户ID', '交易货币', '客户联系方式', '客户邮箱', '交易时间戳', '顾客姓名', '交易日期', '交易时间']
categorical_features = ['性别', '州', '城市', '银行分行', '账户类型', '交易类型', '商户类别', '交易设备', '交易地点', '设备类型']

# 加载数据
data = pd.read_excel(path, engine='openpyxl')  # 使用 openpyxl 加速读取

# 按时间排序
# 将“交易日期”和“交易时间”合并成一个时间戳
data['交易时间'] = data['交易时间'].astype(str)  # 将时间转换为字符串
data['交易时间戳'] = pd.to_datetime(data['交易日期'] + ' ' + data['交易时间'], format='%d-%m-%Y %H:%M:%S')

# 按时间戳排序
data = data.sort_values(by='交易时间戳')

# 删除不需要的列
data = data.drop(columns=drop_features)

# 初始化归一化和类别编码器
scaler = StandardScaler()
label_encoders = {}

# 对分类型特征进行类别编码
for col in categorical_features:
    label_encoders[col] = LabelEncoder()
    data[col] = label_encoders[col].fit_transform(data[col])

# 保存处理好的数据到新的文件
output_file = 'processed_transactions.xlsx'
data.to_excel(output_file, index=False, engine='openpyxl')  # 使用 openpyxl 加速保存