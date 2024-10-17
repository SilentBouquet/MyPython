# 成员修饰符
# Python中成员的修饰符指的是：共有、私有
# 1. 共有：在任何地方都可以调用这个成员
# 2. 私有：只有在类的内部才可以调用成员（以两个下划线开头，则表示该成员私有）
# 父类中的私有成员，子类无法继承
class Foo(object):
    def __init__(self, name, age):
        self.__name = name
        self.age = age

    def get_data(self):
        return self.__name

    def get_age(self):
        return self.age


obj = Foo("樊芮冉", 20)
print(obj.age)
v1 = obj.get_age()
print(v1)
# print(obj.__name)是错误的
v2 = obj.get_data()
print(v2)