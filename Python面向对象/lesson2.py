# 应用实例
class UserInfo:
    def __init__(self, name, password):
        self.name = name
        self.password = password


def run():
    user_object_list = []
    # 用户注册
    while True:
        username = input("用户名：")
        if username.upper() == 'Q':
            break
        password = input("密码：")

        user_object = UserInfo(username, password)
        user_object_list.append(user_object)

    for obj in user_object_list:
        print(obj.name, obj.password)


if __name__ == '__main__':
    run()