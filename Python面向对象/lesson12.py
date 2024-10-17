class Student(object):
    def __init__(self, name, age, class_object):
        self.name = name
        self.age = age
        self.class_object = class_object

    def message(self):
        data = "我是一名{}班学生，我叫{}，我今年{}岁".format(self.class_object.title, self.name, self.age)
        print(data)


class Classes(object):
    def __init__(self, title, school_object):
        self.title = title
        self.school_object = school_object


class School(object):
    def __init__(self, name):
        self.name = name


s1 = School("北京校区")
s2 = School("上海校区")

c1 = Classes("Python全栈", s1)
c2 = Classes("Linux云计算", s2)

user_object_list = [
    Student("樊芮冉", 19, c1),
    Student("晏勇", 20, c1),
    Student("灰灰", 20, c2)
]

for obj in user_object_list:
    print(obj.name, obj.age, obj.class_object.title, obj.class_object.school_object.name)