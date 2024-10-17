# 方法
# 1. 绑定方法：默认有一个self参数，由对象进行调用（此时self就等于调用方法的这个对象）
# 2. 类方法：默认有一个cls参数，用类或对象都可以调用（此时cls就等于调用方法的这个类）
# 3. 静态方法：无默认参数，用类和对象都可以调用
# 在python中比较灵活，方法都可以通过对象和类进行调用
class Foo(object):
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def f1(self):
        print('绑定方法', self.name)

    @classmethod
    def f2(cls):
        print("类方法", cls)

    @staticmethod
    def f3():
        print("静态方法")


# 绑定方法
obj = Foo("樊芮冉", 20)
obj.f1()
Foo.f1(obj)
# 类方法
Foo.f2()
obj.f2()
# 静态方法
Foo.f3()
obj.f3()