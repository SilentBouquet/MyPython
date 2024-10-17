# 特殊成员
# 在python的类中存在一些特殊的方法，这些方法都是__方法__格式，这些方式在内部均有特殊的含义
# 1. __init__：初始化方法
class Foo(object):
    def __init__(self, name):
        self.name = name


obj = Foo("樊芮冉")


# 2. __new__：构造方法
class Foo2(object):
    def __init__(self, name):
        print("第二步：初始化对象，在空对象中创建数据")
        self.name = name

    def __new__(cls, *args, **kwargs):
        print("第一步：先创建空对象并返回")
        return object.__new__(cls)


obj2 = Foo2("樊芮冉")


# 3. __call__
class Foo3(object):
    def __call__(self, *args, **kwargs):
        print("执行call方法")


obj3 = Foo3()
obj3()


# 4. __str__
class Foo4(object):
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __str__(self):
        return "{} {}岁".format(self.name, self.age)


obj4 = Foo4("樊芮冉", 19)
print(str(obj4))
print(obj4)     # 两个是一样的


# 5. __dict__
class Foo5(object):
    def __init__(self, name, age):
        self.name = name
        self.age = age


obj5 = Foo5("樊芮冉", 19)
print(obj5.__dict__)


# 6. __getitem__、__setitem__、__delitem__
class Foo6(object):
    def __setitem__(self, key, value):
        print(key, value)

    def __getitem__(self, key):
        return key

    def __delitem__(self, key):
        print(key)


obj6 = Foo6()
obj6["xxx"] = 123
print(obj6['yyy'])
del obj6['zzz']


# 7. __enter__、__exit__
# with 对象 as f      在内部会执行__enter__方法
# 当with缩进中的代码执行完毕，会自动执行__exit__方法
class Foo7(object):
    def __enter__(self):
        print("进来了")
        return 666

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


with Foo7() as f:
    print(f)


# 8. __add__等
class Foo8(object):
    def __init__(self, name):
        self.name = name

    def __add__(self, other):
        return "{}-{}".format(self.name, other.name)


v1 = Foo8("yy")
v2 = Foo8("frr")
# 对象+值时内部会去执行对象的__add__方法，并将+后面的值当参数传递过去
v3 = v1 + v2
print(v3)