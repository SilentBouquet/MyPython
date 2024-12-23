import json
import uuid
import redis
from flask import Flask, request, jsonify

app = Flask(__name__)

Redis_Pool = redis.ConnectionPool(host='192.168.233.128', port=6379, encoding='utf-8', max_connections=100)


@app.route('/task', methods=['POST'])
def task():
    # 请求的URL中需要带    /bili?token=0d9603c7-0bfb-46b5-b1f6-d3a7a953bbfa
    # 请求的数据格式要求：{"ordered_string": "......"}
    ordered_string = request.json.get('ordered_string')
    if not ordered_string:
        return jsonify({'status': False, 'data': "参数错误！"})
    # 生成任务ID
    tid = str(uuid.uuid4())
    # 1. 把任务放到Redis队列中
    task_dict = {'tid': tid, 'data': ordered_string}

    conn = redis.Redis(connection_pool=Redis_Pool)
    conn.lpush("spider_task_list", json.dumps(task_dict))
    # 2. 给用户返回任务ID
    return jsonify({'status': True, 'data': tid, 'message': '正在处理中...预计一分钟完成'})


@app.route('/result', methods=['GET'])
def result():
    tid = request.args.get("tid")
    if not tid:
        return jsonify({'status': False, 'data': "参数错误！"})
    # 设置decode_responses=True自动将所有返回的字节串解码为字符串
    conn = redis.Redis(connection_pool=Redis_Pool, decode_responses=True)
    sign = conn.hget("spider_result_dict", tid)
    # 即时删除结果
    conn.hdel("spider_result_dict", tid)
    if not sign:
        return jsonify({'status': True, 'data': "", "message": "未完成，请继续等待"})
    return jsonify({'status': True, 'data': sign})


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)