# 反射：提供了一种更加灵活的方式让你去实现对象中的操作成员（以字符串的形式）
# python中提供了4个内置函数来支持反射：
# 1. getattr：去对象中获取成员
# v1 = getattr(对象, "成员名称")
# v2 = getattr(对象, "成员名称", 不存在时的默认值)
# 2. setattr：去对象中设置成员
# setattr(对象, "成员名称", 值)
# 3. hasattr：对象中是否包含成员
# v1 = hasattr(对象, "成员名称")
# 4. delattr：删除对象中的成员
# delattr(对象, "成员名称")
class Account(object):
    def login(self):
        print("这是login方法")
        pass

    def register(self):
        print("这是register方法")
        pass

    def index(self):
        print("这是index方法")
        pass


def run():
    name = input("请输入要执行的方法名称：")
    method = getattr(Account(), name, None)
    if not method:
        print("输入错误！")
        return
    method()


run()
# 在python中一切皆为对象，所以，基于反射也可以对类、模块中的成员进行操作