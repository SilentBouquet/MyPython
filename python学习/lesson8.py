# 闭包
def func():
    a = 10

    def inner():
        nonlocal a
        a = a + 1
        return a
    return inner


ret = func()
r1 = ret()
print(r1)
r2 = ret()
print(r2)