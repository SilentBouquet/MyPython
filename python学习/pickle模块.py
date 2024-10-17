import pickle

lst = ["周星驰", "成龙", "周润发", "黄渤"]
bs = pickle.dumps(lst)
print(bs)

lst2 = pickle.loads(bs)
print(lst2)

'''
dic = {"name": "admin", "password": 123456}
f = open("数据分析实训.txt", mode="w", encoding="utf-8")
f.write(str(dic))

f = open("数据分析实训.txt", mode="r", encoding="utf-8")
s = f.read()
print(s, type(s))
d = eval(s)         # eval对安全性有影响
print(d, type(d))
f.close()
'''

# 存储数据到文件最合理的方案就是用pickle
dic = {"name": "admin", "password": 123456}
pickle.dump(dic, open("数据分析实训.txt", mode="wb"))

# 读取序列化之后的文件
dic = pickle.load(open("数据分析实训.txt", mode="rb"))
print(dic)

'''
1. dumps        把数据转换成字节
2. loads          把字节转化成数据
3. dump          把对象序列化成字节后写入文件
4. load            把文件中的字节反序列化成数据
'''