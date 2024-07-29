import re

# findall：匹配字符串中所有符合正则的内容
lst = re.findall(r'\d+', '我的电话号码是：10086，我女朋友的电话号码是：10010')
print(lst)

# finditer：匹配字符串中所有的内容（返回一个迭代器），拿数据需要.group()
it = re.finditer(r'\d+', '我的电话号码是：10086，我女朋友的电话号码是：10010')
for i in it:
    print(i.group())

# search：找到一个结果就返回，返回的结果是match对象，拿数据需要.group()
s = re.search(r'\d+', '我的电话号码是：10086，我女朋友的电话号码是：10010')
print(s.group())

# match是从头开始匹配
m = re.match(r'\d+', '10086，我女朋友的电话号码是：10010')
print(m.group())

# 预加载正则表达式
obj = re.compile(r'\d+')
ret = obj.finditer('我的电话号码是：10086，我女朋友的电话号码是：10010')
for i in ret:
    print(i.group())