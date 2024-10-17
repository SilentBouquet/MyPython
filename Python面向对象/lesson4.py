# python支持多继承：先继承左边的，再继承右边的
class Base:
    def f1(self):
        print("Base.f1")


class Base2:
    def f1(self):
        print("Base2.f1")


class Son(Base, Base2):
    def run(self):
        print("before")
        self.f1()
        print("after")


s = Son()
s.run()