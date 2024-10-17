# 关于属性的编写有两种方式
# 方式一：基于装饰器
class C(object):
    @property
    def x(self):
        return 123

    @x.setter
    def x(self, value):
        pass

    @x.deleter
    def x(self):
        pass


obj = C()
print(obj.x)
obj.x = 123
del obj.x


# 方式二：基于定义变量
class D(object):
    def getx(self):
        print("getter")
        return 1

    def setx(self, value):
        print("setter")

    def delx(self):
        print("deleter")

    x = property(getx, setx, delx, "I am the 'x' property.")


obj2 = D()
print(obj2.x)
obj2.x = 123
del obj2.x