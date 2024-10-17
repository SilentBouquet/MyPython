# 面向对象编程的三大特性：封装、继承、多态
# 一、封装
# 1. 将同一类方法封装到了一个类中
# 2. 将数据封装到对象中
# 二、继承
# 在面向对象中，子类可以继承父类中的方法和类变量
class Base:
    def f1(self):
        self.f2()
        print("Base.f1")

    def f2(self):
        print("Base.f2")


class Foo(Base):
    def f2(self):
        print("Foo.f2")


# 优先在自己的类中找，自己没有才去父类
obj = Foo()
obj.f1()