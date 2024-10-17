# 变量
# 1. 实例变量：属于对象，每个对象中各自维护自己的数据
# 2. 类变量：属于类，可以被所有对象共享，一般用于给对象提供公共数据（类似于全局变量）
class Person(object):
    country = "中国"
    
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def show(self):
        message = "{}-{}-{}".format(self.country, self.name, self.age)
        print(message)


p1 = Person("樊芮冉", 20)
p1.show()