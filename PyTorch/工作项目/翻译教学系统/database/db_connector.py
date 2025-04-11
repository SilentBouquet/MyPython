import mysql.connector
import time
from config import CONFIG


class DatabaseConnector:
    def __init__(self):
        self.config = CONFIG['database']
        self.connection = None
        self.max_retries = 3
        self.retry_delay = 1
        # 初始化时就建立连接
        self.connect()

    def connect(self):
        """连接到MySQL数据库，包含重试机制"""
        retries = 0
        while retries < self.max_retries:
            try:
                if self.connection is None:
                    self.connection = mysql.connector.connect(
                        host=self.config['host'],
                        user=self.config['user'],
                        password=self.config['password'],
                        database=self.config['db'],
                        connect_timeout=10
                    )
                return self.connection
            except mysql.connector.Error as e:
                retries += 1
                print(f"连接数据库失败 (尝试 {retries}/{self.max_retries}): {e}")
                if retries < self.max_retries:
                    time.sleep(self.retry_delay)
                else:
                    raise
        return None

    def is_connection_valid(self):
        """检查数据库连接是否有效"""
        try:
            if self.connection:
                # 使用简单的查询来测试连接
                cursor = self.connection.cursor()
                cursor.execute("SELECT 1")
                cursor.fetchone()
                cursor.close()
                return True
            return False
        except mysql.connector.Error:
            return False

    def get_connection(self):
        """获取有效的数据库连接"""
        try:
            if not self.is_connection_valid():
                self.connection = None  # 重置无效连接
                self.connect()
            return self.connection
        except mysql.connector.Error as e:
            print(f"获取数据库连接失败: {e}")
            raise

    def close(self):
        """关闭数据库连接"""
        if self.connection:
            try:
                self.connection.close()
            except mysql.connector.Error:
                pass
            finally:
                self.connection = None

    def commit(self):
        """提交事务"""
        if self.connection:
            try:
                self.connection.commit()
            except mysql.connector.Error as e:
                print(f"提交事务失败: {e}")
                raise

    def rollback(self):
        """回滚事务"""
        if self.connection:
            try:
                self.connection.rollback()
            except mysql.connector.Error as e:
                print(f"回滚事务失败: {e}")
                raise

    def get_cursor(self):
        """获取数据库游标"""
        # 检查连接是否有效，无效则重新连接
        if not self.is_connection_valid():
            self.connect()
        return self.connection.cursor()

    def execute(self, query, params=None):
        """执行SQL查询"""
        cursor = self.get_cursor()
        try:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            return cursor
        except mysql.connector.Error as e:
            print(f"Error executing query: {e}")
            raise

    def executemany(self, query, params_list):
        """批量执行SQL查询"""
        cursor = self.get_cursor()
        try:
            cursor.executemany(query, params_list)
            return cursor
        except mysql.connector.Error as e:
            print(f"Error executing query: {e}")
            raise