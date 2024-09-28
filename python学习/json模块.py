# json是前后端交互的枢纽，相当于编程界的普通话
# 1. 把python中的字典或者列表，转化成json字符串
# 2. 把前端返回的json字符串转化成python中的字典
import json

dic = {"id": 1, "name": "樊芮冉", "usertype": 0}
# json处理中文要加ensure_ascii=False
s = json.dumps(dic, ensure_ascii=False)
print(s, type(s))

s = '{"id": 1, "name": "樊芮冉", "usertype": 0}'
dic2 = json.loads(s)
print(dic2, type(dic2))
print(dic2["name"])

# dumps：把对象转换成json
# loads：把json转化成对象

# 前端的json和python中字典的区别：数据类型的写法不一样
dic3 = {"id": 1, "islogin": True, "hasgirl": None}
print(json.dumps(dic3))

json.dump(dic, open("../操作文件/data.txt", "w", encoding="utf-8"), ensure_ascii=False)
d = json.load(open("../操作文件/data.txt", "r", encoding="utf-8"))
print(d, type(d))