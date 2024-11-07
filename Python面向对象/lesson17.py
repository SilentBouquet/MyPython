# 内置函数补充
# 1. callable：是否可在后面加括号执行
class Foo(object):
    def __call__(self, *args, **kwargs):
        pass


obj = Foo()
print(callable(obj))


# 2. super：按照mro继承关系向上找成员
# 应用场景：假设有一个类，它原来已经实现了某些功能，但我们想在他的基础上再扩展点功能，重新写一遍比较麻烦，此时可以用super
class Bar(object):
    def message(self, num):
        print("Bar.massage", num)


class Base(object):
    def message(self, num):
        print("Base.message", num)
        super().message(1000)


class Foo2(Base, Bar):
    pass


obj2 = Foo2()
obj2.message(1)

# 3. type：获取一个对象的类型
s = "樊芮冉"
result = type(s)
print(result == str)


# 4. isinstance：判断对象是否是某个类或其子类的实例
class Top(object):
    pass


class Base2(Top):
    pass


class Foo3(Base2):
    pass


obj3 = Foo3()
print(isinstance(obj3, Foo3))
print(isinstance(obj3, Base2))
print(isinstance(obj3, Top))


# 5. issubclass：判断类是否是某个类的子孙类
print(isinstance(obj3, Base2))
print(isinstance(obj3, Top))