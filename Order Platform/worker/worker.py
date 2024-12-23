import time
import redis
from pymysql import cursors
from utils import db

# 创建redis连接池
Redis_Pool = redis.ConnectionPool(host='192.168.233.128', port=6379, encoding='utf-8', max_connections=100)


def pop_queue():
    conn = redis.Redis(connection_pool=Redis_Pool)
    data = conn.brpop(["Order_Platform_Task_Queue"], timeout=5)
    if not data:
        return
    return data[1].decode('utf-8')


def update_order(order_id, status):
    conn = db.Pool.connection()
    cursor = conn.cursor(cursor=cursors.DictCursor)
    cursor.execute("update `order` set status=%s where id=%s", (status, order_id))
    conn.commit()
    cursor.close()
    conn.close()


def db_queue_init():
    # 1. 去数据库获取待执行的订单ID
    db_list = db.fetch_all("select id from `order` where status in ('待处理', '正在执行')", [])
    db_id_list = {item['id'] for item in db_list}
    # 2. redis中获取队列中所有的ID
    conn = redis.Redis(connection_pool=Redis_Pool)
    total_count = conn.llen("Order_Platform_Task_Queue")
    cache_list = conn.lrange("Order_Platform_Task_Queue", 0, total_count)
    cache_int_list = {int(item.decode('utf-8')) for item in cache_list}
    # 3. 找到数据库中有且redis队列中没有的所有订单ID，重新放到redis队列中
    need_push = db_id_list - cache_int_list
    if need_push:
        conn.lpush("Order_Platform_Task_Queue", *need_push)


def run():
    # 1. 初始化数据库未在队列里的订单
    db_queue_init()
    while True:
        # 2. 去队列中获取订单
        order_id = pop_queue()
        if not order_id:
            continue
        print(order_id)
        # 3. 检查订单是否存在
        order_dict = db.fetch_one("select * from `order` where id=%s", order_id)
        if not order_dict:
            continue
        # 4. 更新订单状态
        update_order(order_id, '正在执行')
        # 5. 执行订单
        print("执行订单任务：", order_dict)
        time.sleep(5)
        # 6. 执行完成
        update_order(order_id, '已处理')


if __name__ == '__main__':
    run()