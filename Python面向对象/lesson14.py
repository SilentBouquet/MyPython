# __iter__ ：迭代器
# 迭代器类型的定义：
# 1. 当类中定义了__iter__和__next__两个方法
# 2. __iter__方法需要返回对象本身，即self
# 3. __next__方法：返回下一个数据，如果没有了，则需要抛出一个StopIteration的异常

# 创建迭代器类型
class IT(object):
    def __init__(self):
        self.counter = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.counter += 1
        if self.counter == 3:
            raise StopIteration
        return self.counter


obj = IT()
print(next(obj))
print(next(obj))

obj2 = IT()
for i in obj2:
    print(i)


# 生成器：生成器也是一个特殊的迭代器
# 创建生成器函数
def func():
    yield 3
    yield 4


# 创建生成器对象（内部是根据生成器类generator的对象），生成器类的内部也声明了__iter__、__next__方法
obj3 = func()
print(next(obj3))
print(next(obj3))
obj4 = func()
for i in obj4:
    print(i)


# 可迭代对象
# 如果一个类中有__iter__方法且返回一个迭代器对象，则我们称以这个类创建的对象为可迭代对象
class Foo(object):
    def __iter__(self):
        return IT()


obj5 = Foo()
# 可迭代对象是可以使用for来进行循环的
# 在循环的内部其实是先执行__iter__方法，获取其迭代器对象
# 然后再在内部执行这个迭代器对象的__next__方法，逐步取值
for i in obj5:
    print(i)