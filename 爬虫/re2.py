import re

s = """
<div class='jay'><span id='1'>中国联通</span></div>
<div class='journey'><span id='2'>移动</span></div>
<div class='jack'><span id='3'>电信</span></div>
"""

# (?P<分组名字>正则) 可以单独从正则匹配的内容中进一步提取内容
obj = re.compile(r"<div class='(?P<class>.*?)'><span id='(?P<id>\d+)'>(?P<name>.*?)</span></div>", re.S)
# re.S：让.能匹配换行符
ret = obj.finditer(s)
for i in ret:
    print(i.group('class'))
    print(i.group('id'))
    print(i.group('name'))