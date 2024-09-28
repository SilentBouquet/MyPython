from datetime import datetime
from datetime import date

t1 = datetime(2024, 9, 28, 15, 47, 12)
t2 = datetime(2024, 10, 1, 6, 00, 00)
print((t2 - t1).total_seconds())

# 格式化一个时间
t = datetime.now()
print(t.strftime("%Y年%m月%d日 %H小时%M分钟%S秒"))

# 让用户输入两个时间，计算时间差
s1 = input("请输入第一个时间(yyyy-mm-dd HH:MM:SS)：")
s2 = input("请输入第二个时间(yyyy-mm-dd HH:MM:SS)：")

# 把字符串转化成时间
t1 = datetime.strptime(s1, "%Y-%m-%d %H:%M:%S")
t2 = datetime.strptime(s2, "%Y-%m-%d %H:%M:%S")
print(t2 - t1)