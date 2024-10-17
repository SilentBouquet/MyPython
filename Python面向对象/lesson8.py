# 属性
# 属性其实是由绑定方法和特殊装饰器组合创造出来的，让我们以后在调用方法时可以不用加括号
class Foo:
    def __init__(self, name):
        self.name = name

    def f1(self):
        return 123

    @property
    def f2(self):
        return 456


obj = Foo("樊芮冉")
v1 = obj.f1()
print(v1)
v2 = obj.f2
print(v2)