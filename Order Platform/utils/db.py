import pymysql
from pymysql import cursors
from dbutils.pooled_db import PooledDB

# 连接池
Pool = PooledDB(
    creator=pymysql,    # 使用连接数据库的模块
    mincached=2,    # 初始化时，连接池中至少创建的空闲连接，0表示不创建
    maxcached=3,    # 连接池中最多闲置的连接，0和None表示不限制
    maxconnections=10,      # 连接池允许的最大连接数，0和None表示不限制连接数
    blocking=True,      # 连接池中如果没有可用连接后，是否阻塞等待。True，等待；False，不等待然后报错
    setsession=[],      # 开始会话前执行的命令列表。如：["set datestyle to ...", "set time zone ..."]
    ping=0,      # 检查与数据库的连接是否正常

    host='127.0.0.1', port=3306, user='root', passwd='yy040806', db='order_platform', charset='utf8'
)


def fetch_one(sql, params):
    conn = Pool.connection()
    cursor = conn.cursor(cursor=cursors.DictCursor)
    cursor.execute(sql, params)
    result = cursor.fetchone()
    cursor.close()
    conn.close()  # 此处的close不再是关闭连接，而是将次连接交还给连接池
    return result


def fetch_all(sql, params):
    conn = Pool.connection()
    cursor = conn.cursor(cursor=cursors.DictCursor)
    cursor.execute(sql, params)
    result = cursor.fetchall()
    cursor.close()
    conn.close()  # 此处的close不再是关闭连接，而是将次连接交还给连接池
    return result


def insert(sql, params):
    conn = Pool.connection()
    cursor = conn.cursor(cursor=cursors.DictCursor)
    cursor.execute(sql, params)
    conn.commit()
    cursor.close()
    conn.close()  # 此处的close不再是关闭连接，而是将次连接交还给连接池
    return  cursor.lastrowid