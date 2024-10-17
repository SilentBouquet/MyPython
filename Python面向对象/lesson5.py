# 多态：多种形态
# 在Java或其他编程语言中的多态是基于：
# 接口或抽象类和抽象方法来实现，让数据可以以多种形态存在
# python则不一样，由于python对数据类型没有任何限制，所以他天生支持多态
class Email(object):
    def send(self):
        print("发邮件")


class Message(object):
    def send(self):
        print("发短信")


def func(arg):
    v1 = arg.send()


v1 = Email()
func(v1)
v2 = Message()
func(v2)