import hashlib
import pymysql
from flask import Flask, request, jsonify

app = Flask(__name__)


def fetch_one(sql, params):
    conn = pymysql.connect(host='127.0.0.1', port=3306, user='root', passwd='yy040806', db='mydatabase', charset='utf8')
    cursor = conn.cursor()
    cursor.execute(sql, params)
    result = cursor.fetchone()
    cursor.close()
    conn.close()
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