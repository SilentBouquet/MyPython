import redis

Redis_Connection = {
    "host": '192.168.233.128',
    "port": 6379,
    "encoding": 'utf-8',
    "password": None,
    "db": 0
}
conn = redis.Redis(**Redis_Connection, decode_responses=True)
conn.lpush("test_task_list", "123")
# conn.lpush("test_task_list", "456")
print("加入成功")

data = conn.brpop(["test_task_list"], timeout=5)
print(data)

sign = conn.hget("spider_result_dict", tid)
print(sign)