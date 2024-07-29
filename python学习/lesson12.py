# 生成器函数
def order():
    lst = []
    for i in range(1,10001):
        lst.append(f"衣服{i}")
        if len(lst) == 50:
            yield lst
            lst = []


gen = order()
print(gen.__next__())
print(gen.__next__())