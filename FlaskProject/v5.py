import hashlib
import pymysql
from flask import Flask, request, jsonify
from dbutils.pooled_db import PooledDB

app = Flask(__name__)

# 连接池
Pool = PooledDB(
    creator=pymysql,    # 使用连接数据库的模块
    mincached=2,    # 初始化时，连接池中至少创建的空闲连接，0表示不创建
    maxcached=3,    # 连接池中最多闲置的连接，0和None表示不限制
    maxconnections=10,      # 连接池允许的最大连接数，0和None表示不限制连接数
    blocking=True,      # 连接池中如果没有可用连接后，是否阻塞等待。True，等待；False，不等待然后报错
    setsession=[],      # 开始会话前执行的命令列表。如：["set datestyle to ...", "set time zone ..."]
    ping=0,      # 检查与数据库的连接是否正常

    host='127.0.0.1', port=3306, user='root', passwd='yy040806', db='mydatabase', charset='utf8'
)


def fetch_one(sql, params):
    conn = Pool.connection()
    cursor = conn.cursor()
    cursor.execute(sql, params)
    result = cursor.fetchone()
    cursor.close()
    conn.close()    # 此处的close不再是关闭连接，而是将次连接交还给连接池
    return result


@app.route('/bili', methods=['POST'])
def bili():  # put application's code here
    # 请求的URL中需要带    /bili?token=0d9603c7-0bfb-46b5-b1f6-d3a7a953bbfa
    # 请求的数据格式要求：{"ordered_string": "......"}

    # 1. token是否为空
    token = request.args.get('token')
    if not token:
        return jsonify({'status': False, 'data': "认证失败！"})

    # 2. token是否合法，连接MySQL执行命令
    result = fetch_one('select * from user where token=%s', [token, ])
    if not result:
        return jsonify({'status': False, 'data': "认证失败！"})

    ordered_string = request.json.get('ordered_string')
    if not ordered_string:
        return jsonify({'status': False, 'data': "参数错误！"})

    # 调用核心算法，生成sign签名
    encrypted_string = ordered_string + "8273hjsadg891287934us"
    obj = hashlib.md5(encrypted_string.encode('utf-8'))
    sign = obj.hexdigest()
    return jsonify({'status': True, 'data': sign})


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)