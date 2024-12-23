import redis

# 创建redis连接池
Redis_Pool = redis.ConnectionPool(host='192.168.233.128', port=6379, encoding='utf-8', max_connections=100)


def push_queue(value):
    conn = redis.Redis(connection_pool=Redis_Pool)
    conn.lpush("Order_Platform_Task_Queue", value)