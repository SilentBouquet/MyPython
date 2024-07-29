# 函数的嵌套
def func1():
    print(123)

    def func2():
        print(456)

        def func3():
            print(789)
        print(1)
        func3()
        print(2)
    print(3)
    func2()
    print(4)


func1()
print()


def func():
    def inner():
        print(123)
    print(inner)
    return inner    # 此时把一个函数当成一个变量进行返回


b = func()      # b是func的内部函数inner
print(b)
b()