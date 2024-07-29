# 装饰器
login_flag = False


def login_verify(fn):
    def inner(*args, **kwargs):
        global login_flag
        if not login_flag:
            while True:
                username = input('请输入用户名：')
                password = input('请输入密码：')
                if username == 'admit' and password == '123456':
                    print('登录成功')
                    login_flag = True
                    break
                else:
                    print('登录失败，用户名或密码错误')
        ret = fn(*args, **kwargs)
        return ret
    return inner


@login_verify
def add():
    print('添加员工信息')


@login_verify
def delete():
    print('删除员工信息')


@login_verify
def update():
    print('修改员工信息')


@login_verify
def search():
    print('查询员工信息')


add()
delete()
update()
search()