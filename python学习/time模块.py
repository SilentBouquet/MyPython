import time


def test1():
    i = 5
    while i != 0:
        print("抓取百度的信息")
        # 可以控制程序执行的效率
        time.sleep(1)
        i -= 1


def test2():
    start = time.time()        # 时间戳，数字类型的时间
    for i in range(10000):
        print(i)
    end = time.time()
    print(end - start)


test1()
test2()