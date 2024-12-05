import uuid
import redis
from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route('/task', methods=['POST'])
def task():  # put application's code here
    # 请求的URL中需要带    /bili?token=0d9603c7-0bfb-46b5-b1f6-d3a7a953bbfa
    # 请求的数据格式要求：{"ordered_string": "......"}

    ordered_string = request.json.get('ordered_string')
    if not ordered_string:
        return jsonify({'status': False, 'data': "参数错误！"})
    # 生成任务ID
    tid = str(uuid.uuid4())
    # 1. 把任务放到Redis队列中
    info_dict = {'tid': tid, 'data': ordered_string}
    conn = redis.Redis(host='localhost', port=6379, db=0)
    # 2. 给用户返回任务ID
    return jsonify({'status': True, 'data': tid, 'message': '正在处理中...预计一分钟完成'})


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)