# 面向对象中的嵌套
# 学生类
class Student(object):
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def message(self):
        data = "我是一名学生，我叫{}，我今年{}岁".format(self.name, self.age)
        print(data)


s1 = Student("樊芮冉", 20)
s2 = Student("晏勇", 20)
s3 = Student("灰灰", 20)


# 班级类
class Classes(object):
    def __init__(self, title):
        self.title = title
        self.student_list = []

    def add_student(self, stu_object):
        self.student_list.append(stu_object)

    def add_students(self, stu_object_list):
        for stu in stu_object_list:
            self.add_student(stu)

    def show_members(self):
        for item in self.student_list:
            print(item)


c = Classes("三年二班")
c.add_student(s1)
c.add_students([s2, s3])
print(c.title)
for item in c.student_list:
    item.message()