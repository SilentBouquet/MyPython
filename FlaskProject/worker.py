# 去队列中获取任务，执行并写入到结果队列
import hashlib
import redis
import json

Redis_Pool = redis.ConnectionPool(host='192.168.233.128', port=6379, encoding='utf-8', max_connections=100)


def get_data():
    conn = redis.Redis(connection_pool=Redis_Pool)
    data = conn.brpop(["spider_task_list"], timeout=5)
    if not data:
        return None
    return json.loads(data[1].decode("utf-8"))


def set_result(tid, value):
    conn = redis.Redis(connection_pool=Redis_Pool)
    conn.hset("spider_result_dict", tid, value)


def run():
    while True:
        # 1. 从Redis中获取任务
        task_dict = get_data()
        if not task_dict:
            continue
        print(task_dict)
        # 2. 执行耗时操作
        ordered_string = task_dict["data"]
        encrypted_string = ordered_string + "8273hjsadg891287934us"
        obj = hashlib.md5(encrypted_string.encode('utf-8'))
        sign = obj.hexdigest()
        # 3. 写入到结果队列
        set_result(task_dict["tid"], sign)


if __name__ == '__main__':
    run()