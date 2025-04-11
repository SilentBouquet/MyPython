import mysql.connector
from database.db_connector import DatabaseConnector
import datetime
import os
import json


class UserModel:
    def __init__(self):
        # 获取数据库连接器的单例实例
        self.db = DatabaseConnector()
        # 确保data文件夹存在
        os.makedirs('data', exist_ok=True)
        # 确保sentences.txt文件存在
        if not os.path.exists('data/sentences.txt'):
            with open('data/sentences.txt', 'w', encoding='utf-8') as f:
                f.write('')
        # 确保knowledge_base.json文件存在
        if not os.path.exists('data/knowledge_base.json'):
            with open('data/knowledge_base.json', 'w', encoding='utf-8') as f:
                json.dump({
                    'grammar_points': [],
                    'phrases': [],
                    'collocations': []
                }, f, ensure_ascii=False, indent=2)

    def create_tables(self):
        """创建必要的数据库表"""
        # 重用已有连接而不是创建新连接
        conn = self.db.get_connection()
        cursor = conn.cursor()

        # 创建用户表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTO_INCREMENT,
            username VARCHAR(255) NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE KEY unique_username (username)
        )
        ''')

        # 创建翻译记录表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS translation_records (
            id INTEGER PRIMARY KEY AUTO_INCREMENT,
            user_id INTEGER,
            source_text TEXT,
            target_translation TEXT,
            user_translation TEXT,
            accuracy REAL,
            translation_type TEXT,
            created_at TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        ''')

        # 创建知识点表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS knowledge_points (
            id VARCHAR(20) PRIMARY KEY,
            type VARCHAR(20) NOT NULL,
            category VARCHAR(20) NOT NULL,
            text TEXT NOT NULL,
            explanation TEXT NOT NULL,
            example TEXT,
            difficulty INTEGER DEFAULT 2,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
        )
        ''')

        conn.commit()

    def add_user(self, username, password):
        """添加新用户"""
        conn = None
        cursor = None
        try:
            conn = self.db.get_connection()
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO users (username, password) VALUES (%s, %s)',
                (username, password)
            )
            conn.commit()
            return True
        except mysql.connector.IntegrityError:
            if conn:
                conn.rollback()
            return False
        except mysql.connector.Error as e:
            print(f"添加用户出错: {e}")
            if conn:
                conn.rollback()
            return False
        finally:
            if cursor:
                cursor.close()

    def get_user_by_username(self, username):
        """通过用户名获取用户"""
        cursor = None
        try:
            conn = self.db.get_connection()
            cursor = conn.cursor(dictionary=True)
            cursor.execute('SELECT * FROM users WHERE username = %s', (username,))
            user = cursor.fetchone()
            return user
        except mysql.connector.Error as e:
            print(f"数据库查询出错: {e}")
            return None
        finally:
            if cursor:
                cursor.close()

    def authenticate_user(self, username, password):
        """验证用户"""
        try:
            user = self.get_user_by_username(username)
            if user and password == user['password']:
                return user
            return None
        except Exception as e:
            print(f"用户认证出错: {e}")
            return None

    def add_translation_record(self, user_id, source_text, target_translation, user_translation,
                               accuracy, translation_type, **kwargs):
        """添加用户翻译记录 (简化版)"""
        # 使用 get_connection() 获取已有连接
        conn = self.db.get_connection()
        cursor = conn.cursor()

        now = datetime.datetime.now()

        try:
            cursor.execute('''
                INSERT INTO translation_records 
                (user_id, source_text, target_translation, user_translation, accuracy, translation_type, created_at) 
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            ''', (user_id, source_text, target_translation, user_translation, accuracy, translation_type, now))
            conn.commit()
            return True
        except Exception as e:
            print(f"添加翻译记录时出错: {e}")
            self.db.rollback()  # 使用 db 的方法
            return False

    def get_user_progress(self, user_id):
        """获取用户学习进度"""
        conn = self.db.connect()
        cursor = conn.cursor(dictionary=True)

        # 获取翻译记录
        query = '''
        SELECT accuracy, created_at 
        FROM translation_records 
        WHERE user_id = %s 
        ORDER BY created_at
        '''
        cursor.execute(query, (user_id,))
        records = cursor.fetchall()

        # 准备准确率历史数据
        accuracy_history = [record['accuracy'] for record in records if record['accuracy'] is not None]

        # 获取统计数据
        total_exercises = len(records)
        avg_accuracy = sum(accuracy_history) / len(accuracy_history) if accuracy_history else 0

        # 获取连续登录天数
        consecutive_days = self.get_consecutive_days(user_id)

        # 获取成就
        achievements = self.get_achievements(total_exercises, consecutive_days, avg_accuracy)

        return {
            'total_exercises': total_exercises,
            'avg_accuracy': round(avg_accuracy, 2),
            'consecutive_days': consecutive_days,
            'accuracy_history': accuracy_history,
            'achievements': achievements
        }

    def get_consecutive_days(self, user_id):
        """获取用户连续登录天数 (简化版)"""
        conn = self.db.get_connection()
        cursor = conn.cursor(dictionary=True)
        try:
            query = '''
            SELECT COUNT(DISTINCT DATE(created_at)) as days 
            FROM translation_records 
            WHERE user_id = %s 
            AND DATE(created_at) BETWEEN DATE_SUB(CURDATE(), INTERVAL 7 DAY) AND CURDATE()
            '''
            cursor.execute(query, (user_id,))
            result = cursor.fetchone()
            return result['days'] if result and result['days'] else 0
        except Exception as e:
            print(f"获取连续天数时出错: {e}")
            return 0

    def get_achievements(self, total_exercises, consecutive_days, avg_accuracy):
        """获取用户成就"""
        return [
            {
                'name': '翻译新手',
                'description': '完成10次翻译练习',
                'achieved': total_exercises >= 10,
                'progress': min(int(total_exercises / 10 * 100), 100),
                'icon': 'star-fill'
            },
            {
                'name': '精准翻译者',
                'description': '翻译准确率超过90%',
                'achieved': avg_accuracy >= 0.9,  # 需要更复杂的逻辑来判断
                'progress': 0,
                'icon': 'award'
            },
            {
                'name': '翻译大师',
                'description': '完成100次翻译练习',
                'achieved': total_exercises >= 100,
                'progress': min(int(total_exercises / 100 * 100), 100),
                'icon': 'journal-check'
            },
            {
                'name': '坚持不懈',
                'description': '连续30天进行练习',
                'achieved': consecutive_days >= 30,
                'progress': min(int(consecutive_days / 30 * 100), 100),
                'icon': 'calendar-check'
            }
        ]

    def add_knowledge_point(self, knowledge_id, knowledge_type, category, text, explanation, example, difficulty):
        """添加知识点到数据库"""
        conn = self.db.get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute('''
                INSERT INTO knowledge_points 
                (id, type, category, text, explanation, example, difficulty) 
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                type = VALUES(type),
                category = VALUES(category),
                text = VALUES(text),
                explanation = VALUES(explanation),
                example = VALUES(example),
                difficulty = VALUES(difficulty)
            ''', (knowledge_id, knowledge_type, category, text, explanation, example, difficulty))
            conn.commit()
            return True
        except Exception as e:
            print(f"添加知识点时出错: {e}")
            self.db.rollback()
            return False

    def delete_knowledge_point(self, knowledge_id):
        """从数据库删除知识点"""
        conn = self.db.get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute('DELETE FROM knowledge_points WHERE id = %s', (knowledge_id,))
            conn.commit()
            return cursor.rowcount > 0  # 返回是否有行被删除
        except Exception as e:
            print(f"删除知识点时出错: {e}")
            self.db.rollback()
            return False

    def get_knowledge_points(self, category=None):
        """获取知识点列表"""
        conn = self.db.get_connection()
        cursor = conn.cursor(dictionary=True)

        try:
            if category:
                cursor.execute('SELECT * FROM knowledge_points WHERE category = %s', (category,))
            else:
                cursor.execute('SELECT * FROM knowledge_points')

            results = cursor.fetchall()

            # 按类别组织结果
            organized = {
                'grammar_points': [],
                'phrases': [],
                'collocations': []
            }

            for item in results:
                category = item['category']
                if category in organized:
                    organized[category].append(item)

            return organized
        except Exception as e:
            print(f"获取知识点时出错: {e}")
            return {
                'grammar_points': [],
                'phrases': [],
                'collocations': []
            }