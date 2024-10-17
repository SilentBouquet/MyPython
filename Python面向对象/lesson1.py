# 面向对象的思想：将一些数据封装到对象中，在执行方法时，再去对象中获取
# 函数式的思想：函数内容需要的数据均通过参数的形式传递
# self，是一个python内部会提供的参数，本质上就是调用当前方法的那个对象
# 对象，基于类实例化出来的一块内存，默认里面没有数据，经过类的__init__方法，可以在内存中初始化一些数据
# 在编写面向对象相关代码时，最常见的成员有：
# 1. 实例变量：属于对象，只能通过对象调用
# 2. 绑定方法：属于类，通过对象调用或通过类调用
class Message:
    def __init__(self, content):
        # 实例变量
        self.data = content

    # 绑定方法
    def send_email(self, email):
        data = "给{}发邮件，内容是：{}".format(email, self.data)
        print(data)

    def send_wechat(self, email):
        data = "给{}发微信，内容是：{}".format(email, self.data)
        print(data)


msg_object = Message("注册成功！")
msg_object.send_email("2133657519@qq.com")
msg_object.send_wechat("小樊宝贝")