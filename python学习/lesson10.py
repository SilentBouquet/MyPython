# 迭代器
a = '你叫什么名字啊'
it1 = iter(a)
print(it1)
for i in range(1,len(a)+1):
    print(next(it1))

print()
it2 = a.__iter__()
print(it2)
for i in range(1,len(a)+1):
    print(it2.__next__())