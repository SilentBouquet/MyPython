# 基于可迭代对象和迭代器实现自定义range
class IterRange(object):
    def __init__(self, num):
        self.num = num
        self.counter = -1

    def __iter__(self):
        return self

    def __next__(self):
        self.counter += 1
        if self.counter == self.num:
            raise StopIteration
        return self.counter


class Xrange(object):
    def __init__(self, max_num):
        self.max_num = max_num

    def __iter__(self):
        return IterRange(self.max_num)


obj = Xrange(100)
obj_iter = iter(obj)
while True:
    try:
        element = next(obj_iter)
    except StopIteration:
        print("循环结束！")
        break
    print(element)


from collections.abc import Iterable, Iterator

v1 = list([11, 22, 33, 44, 55, 66])
# 判断是否是可迭代对象
# 判断依据：是否只有__iter__方法且返回迭代器对象
print(isinstance(v1, Iterable))
# 判断是否是迭代器
# 判断依据：是否有__iter__和__next__方法
print(isinstance(v1, Iterator))

v2 = v1.__iter__()
print(isinstance(v2, Iterable))
print(isinstance(v2, Iterator))