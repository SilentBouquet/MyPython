from utils import db
from flask import Blueprint, render_template, request, redirect, session

# 蓝图对象
ac = Blueprint('account', __name__)


@ac.route('/login32', methods=['GET', 'POST'])
def login32():
    if request.method == 'GET':
        return render_template("login.html")
    userType = request.form['userType']
    phone = request.form['phone']
    password = request.form['password']

    # 连接MySQL并查询用户和密码是否正确
    user_dict = db.fetch_one("select * from user where role = %s and mobile = %s and password = %s",
                   (userType, phone, password))

    if user_dict:
        session['user'] = {
            "role": userType,
            "real_name": user_dict['real_name'],
            'id': user_dict['id']
        }
        return redirect('/order/list')
    return render_template("login.html", error="UserName or Password is Incorrect")


@ac.route('/users')
def users():
    return "用户列表"