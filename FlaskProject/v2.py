import hashlib
from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route('/bili', methods=['POST'])
def bili():  # put application's code here
    # 请求的数据格式要求：{"ordered_string": "......"}
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