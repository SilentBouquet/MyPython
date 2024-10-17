try:
    # print(1/0)
    # open("不存在的文件.txt", mode="r", encoding="utf-8").read()
    lst = []
    lst.__iter__().__next__()
except ZeroDivisionError as z:
    print("除数为0")
except FileNotFoundError as f:
    print("文件不存在")
except TypeError as t:
    print("类型错误")
except Exception as e:
    print("系统错误")
finally:
    print("程序结束")


# 程序是可以自己抛出异常的
def func(a, b):  # 计算两个int类型的数字之和
    if type(a) is int and type(b) is int:
        return a + b
    else:
        raise TypeError


a = input("请输入第一个数字：")
b = input("请输入第二个数字：")
func(a, b)