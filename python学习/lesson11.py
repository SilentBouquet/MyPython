# 模拟for循环工作原理
s = '你叫什么名字啊'
it = s.__iter__()
while 1:
    try:
        data = it.__next__()
        print(data)
    except StopIteration:
        break
print('>>>')