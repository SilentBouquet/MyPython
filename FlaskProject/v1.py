from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route('/index', methods=['GET', 'POST'])
def index():  # put application's code here
    name = request.args.get("name")
    pwd = request.args.get("pwd")
    print(name, pwd)

    xx = request.form.get("xx")
    yy = request.form.get("yy")
    print(xx, yy)

    if name == "frr" and pwd == "123456":
        # 调用核心算法，生成sign签名
        return jsonify({'status': '登陆成功！', 'data': 'Hello World!'})
    else:
        return jsonify({'status': '登陆失败！', 'data': 'Not Hello!'})


@app.route('/home', methods=['GET', 'POST'])
def home():  # put application's code here
    return 'How are you?'


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)