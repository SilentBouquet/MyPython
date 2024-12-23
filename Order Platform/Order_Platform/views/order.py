from utils import db
from flask import Blueprint, session, render_template, request, redirect
from utils import cache
# 蓝图对象
od = Blueprint('order', __name__)


@od.route('/order/list')
def order_list():
    # 读取cookies解密获取用户信息
    user = session.get('user')
    role = user['role']
    if role == '2':
        data_list = db.fetch_all('select * from `order` left join user on `order`.user_id = user.id', [])
    else:
        data_list = db.fetch_all('select * from `order` left join user on `order`.user_id = user.id '
                                 'where user_id = %s', [user['id'], ])
    return render_template("order_list.html", data_list=data_list, user=user)


@od.route('/order/create', methods=['GET', 'POST'])
def creat_list():
    if request.method == 'GET':
        user = session.get('user')
        return render_template("order_create.html", user=user)
    else:
        url = request.form.get('url')
        count = request.form.get('count')
        # 写入数据库
        user = session.get('user')
        params = [url, count, user['id'], ]
        order_id = db.insert("insert into `order`(url, count, user_id, status) values (%s, %s, %s, '待处理')", params)
        # 写入Redis队列
        cache.push_queue(order_id)
        return redirect('/order/list')


@od.route('/order/delete')
def delete_list():
    return "删除订单"