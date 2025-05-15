import pymysql
import os
import sys
import json
from werkzeug.security import generate_password_hash
import datetime

# 这里填入你的数据库密码
PASSWORD = 'yy040806'


def initialize_database():
    try:
        # 创建数据库连接
        conn = pymysql.connect(
            host='localhost',
            user='root',
            password=PASSWORD
        )
        cursor = conn.cursor()

        # 创建数据库
        cursor.execute("CREATE DATABASE IF NOT EXISTS remote_sensing_db")

        # 切换到新创建的数据库
        cursor.execute("USE remote_sensing_db")

        # 读取SQL脚本
        with open('database_schema.sql', 'r', encoding='utf-8') as f:
            sql_script = f.read()

        # 分割并执行SQL语句，忽略表已存在的错误
        for statement in sql_script.split(';'):
            if statement.strip():
                try:
                    cursor.execute(statement)
                except pymysql.err.InternalError as e:
                    # 忽略"表已存在"的错误，但报告其他错误
                    if "already exists" not in str(e):
                        print(f"执行SQL语句出错: {e}")
                        print(f"出错的SQL: {statement}")

        # 直接跳到后续步骤，确保通知和权限表存在
        try:
            # 检查notifications表是否存在
            cursor.execute("SHOW TABLES LIKE 'notifications'")
            if not cursor.fetchone():
                cursor.execute("""
                CREATE TABLE notifications (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT NOT NULL,
                    title VARCHAR(100) NOT NULL,
                    content TEXT NOT NULL,
                    type VARCHAR(50) NOT NULL,
                    is_read BOOLEAN DEFAULT FALSE,
                    created_at DATETIME NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                )
                """)
                print("创建notifications表")
        except Exception as e:
            print(f"创建notifications表出错: {e}")

        try:
            # 检查permission_requests表是否存在
            cursor.execute("SHOW TABLES LIKE 'permission_requests'")
            if not cursor.fetchone():
                cursor.execute("""
                CREATE TABLE permission_requests (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT NOT NULL,
                    request_type VARCHAR(50) NOT NULL,
                    request_reason TEXT NOT NULL,
                    additional_info JSON,
                    status VARCHAR(20) DEFAULT 'pending',
                    created_at DATETIME NOT NULL,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                )
                """)
                print("创建permission_requests表")
        except Exception as e:
            print(f"创建permission_requests表出错: {e}")

        # 检查users表中是否存在新字段，不存在则添加
        try:
            cursor.execute("DESCRIBE users")
            columns = [row[0] for row in cursor.fetchall()]

            if 'name' not in columns:
                cursor.execute("ALTER TABLE users ADD COLUMN name VARCHAR(100)")
                print("添加users.name字段")

            if 'phone' not in columns:
                cursor.execute("ALTER TABLE users ADD COLUMN phone VARCHAR(20)")
                print("添加users.phone字段")

            if 'bio' not in columns:
                cursor.execute("ALTER TABLE users ADD COLUMN bio TEXT")
                print("添加users.bio字段")

            if 'is_admin' not in columns:
                cursor.execute("ALTER TABLE users ADD COLUMN is_admin BOOLEAN DEFAULT FALSE")
                print("添加users.is_admin字段")
        except Exception as e:
            print(f"更新users表结构出错: {e}")

        # 创建模型公开申请表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_publish_requests (
            id INT AUTO_INCREMENT PRIMARY KEY,
            model_id INT NOT NULL,
            user_id INT NOT NULL,
            reason TEXT,
            status ENUM('pending', 'approved', 'rejected') NOT NULL DEFAULT 'pending',
            admin_comment TEXT,
            created_at DATETIME NOT NULL,
            updated_at DATETIME,
            FOREIGN KEY (model_id) REFERENCES models(id) ON DELETE CASCADE,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
        ''')

        # 创建数据库用户并授权
        try:
            cursor.execute("CREATE USER IF NOT EXISTS 'remote_sensing_user'@'localhost' IDENTIFIED BY 'your_password'")
            cursor.execute("GRANT ALL PRIVILEGES ON remote_sensing_db.* TO 'remote_sensing_user'@'localhost'")
            cursor.execute("FLUSH PRIVILEGES")
        except Exception as e:
            print(f"创建数据库用户出错: {e}")

        conn.commit()
        conn.close()

        print("数据库初始化成功！")
        return True
    except Exception as e:
        print(f"数据库初始化失败: {e}")
        return False


def add_default_models():
    try:
        # 创建数据库连接
        conn = pymysql.connect(
            host='localhost',
            user='root',
            password=PASSWORD,
            database='remote_sensing_db',
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )
        cursor = conn.cursor()

        # 清空现有系统模型
        cursor.execute("DELETE FROM models WHERE is_system = 1")

        # 获取当前脚本的绝对路径
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # 确保模型目录存在
        model_dir = os.path.join(current_dir, 'models', 'system')
        os.makedirs(model_dir, exist_ok=True)

        # 扫描模型目录中已存在的模型文件
        model_files = []
        for file in os.listdir(model_dir):
            if file.endswith(('.pt', '.pth', '.h5')):
                model_files.append(file)

        # 只加载实际存在的模型文件
        if model_files:
            print(f"发现{len(model_files)}个模型文件")
            example_image_path = os.path.join('static', 'images', 'default_input.jpg')
            result_example_image_path = os.path.join('static', 'images', 'default_result.jpg')
            for file in model_files:
                model_path = os.path.join(model_dir, file)

                # 尝试从文件名中提取模型类型
                model_type = 'general'
                if 'landcover' in file.lower():
                    model_type = 'landcover'
                    model_name = '土地覆盖分类模型'
                    description = '系统默认的遥感图像分类模型，支持十种地物类型识别'
                elif 'building' in file.lower():
                    model_type = 'building'
                    model_name = '建筑物提取模型'
                    description = '专为城市规划设计，能够精确提取不同类型建筑物轮廓'
                elif 'water' in file.lower():
                    model_type = 'water'
                    model_name = '水体监测模型'
                    description = '结合多时相遥感影像，精确监测和分析水体面积变化趋势'
                elif 'vegetation' in file.lower():
                    model_type = 'vegetation'
                    model_name = '植被分析模型'
                    description = '基于多波段数据分析植被覆盖度和健康状况，支持农业监测'
                elif 'yolo' in file.lower():
                    model_type = 'object'
                    model_name = 'YOLO目标检测模型'
                    description = '基于YOLO算法的目标检测模型，支持多种目标类型识别，适用于图像和视频'
                else:
                    model_name = f'通用遥感分析模型({file})'
                    description = '系统自动识别的遥感图像分析模型'

                cursor.execute(
                    """INSERT INTO models \
                    (name, type, description, path, example_image, result_example_image, is_system, is_shared, accuracy, usage_count, supports_video, created_at) \
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())""",
                    (model_name, model_type, description, model_path, example_image_path, result_example_image_path,
                     True, True, 0.0, 0, True)
                )
                print(f"添加模型: {model_name} (从文件: {file})")
        else:
            print("未找到现有模型文件，不初始化任何默认模型")

        conn.commit()
        conn.close()

        print("模型添加成功！")
        return True
    except Exception as e:
        print(f"添加模型失败: {e}")
        return False


def create_admin_user():
    try:
        # 创建数据库连接
        conn = pymysql.connect(
            host='localhost',
            user='root',
            password=PASSWORD,
            database='remote_sensing_db',
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )
        cursor = conn.cursor()

        # 直接插入管理员账户（如已存在则忽略重复）
        hashed_password = generate_password_hash('admin123')
        try:
            cursor.execute(
                """INSERT INTO users \
                   (username, email, password, organization, department, license_number, \
                    research_field, platforms, usage_purpose, name, phone, bio, is_admin, created_at) \
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                ('admin', 'admin@example.com', hashed_password, '系统管理', '管理部门', 'ADMIN-LICENSE',
                 '系统管理', json.dumps([]), '系统管理', '系统管理员', '13800138000',
                 '系统管理员账号，负责系统维护和用户支持', 1, datetime.datetime.now())
            )
            admin_id = cursor.lastrowid
        except Exception as e:
            print(f"插入管理员账户时可能已存在，忽略: {e}")
            # 查询admin用户id
            cursor.execute("SELECT id FROM users WHERE username = 'admin'")
            row = cursor.fetchone()
            admin_id = row['id'] if row else None

            # 确保admin用户有管理员权限
            if admin_id:
                cursor.execute("UPDATE users SET is_admin = 1 WHERE id = %s", (admin_id,))
                print(f"更新admin用户为管理员权限成功")

        # 为管理员添加一条欢迎通知（如有id）
        if admin_id:
            current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            cursor.execute(
                """INSERT INTO notifications \
                   (user_id, title, content, type, is_read, created_at)\
                   VALUES (%s, %s, %s, %s, %s, %s)""",
                (
                admin_id, '欢迎使用遥感图像识别系统', '感谢您使用本系统，这里是管理员账号，您可以管理系统资源和用户权限。',
                'info', 0, current_time)
            )

        conn.commit()
        conn.close()

        print("管理员账户已初始化！默认账号：admin，密码：admin123")
        return True
    except Exception as e:
        print(f"创建管理员账户失败: {e}")
        return False


def create_directories():
    try:
        # 创建必要的目录
        os.makedirs('uploads', exist_ok=True)
        os.makedirs('models/system', exist_ok=True)
        os.makedirs('models/custom', exist_ok=True)
        os.makedirs('static/results', exist_ok=True)  # 图像结果目录
        os.makedirs('static/videos', exist_ok=True)  # 视频结果目录

        print("目录创建成功！")
        return True
    except Exception as e:
        print(f"创建目录失败: {e}")
        return False


def update_database_schema():
    try:
        # 创建数据库连接
        conn = pymysql.connect(
            host='localhost',
            user='root',
            password=PASSWORD,
            database='remote_sensing_db',
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )
        cursor = conn.cursor()

        # 检查images表是否有file_type列
        cursor.execute("SHOW COLUMNS FROM images LIKE 'file_type'")
        if not cursor.fetchone():
            cursor.execute("ALTER TABLE images ADD COLUMN file_type VARCHAR(10) DEFAULT 'image'")
            print("images表添加file_type字段成功")

        # 检查images表是否有duration列
        cursor.execute("SHOW COLUMNS FROM images LIKE 'duration'")
        if not cursor.fetchone():
            cursor.execute("ALTER TABLE images ADD COLUMN duration FLOAT")
            print("images表添加duration字段成功")

        # 检查images表是否有frame_count列
        cursor.execute("SHOW COLUMNS FROM images LIKE 'frame_count'")
        if not cursor.fetchone():
            cursor.execute("ALTER TABLE images ADD COLUMN frame_count INT")
            print("images表添加frame_count字段成功")

        # 检查models表是否有supports_video列
        cursor.execute("SHOW COLUMNS FROM models LIKE 'supports_video'")
        if not cursor.fetchone():
            cursor.execute("ALTER TABLE models ADD COLUMN supports_video BOOLEAN DEFAULT TRUE")
            print("models表添加supports_video字段成功")

        # 检查models表是否有result_example_image列
        cursor.execute("SHOW COLUMNS FROM models LIKE 'result_example_image'")
        if not cursor.fetchone():
            cursor.execute("ALTER TABLE models ADD COLUMN result_example_image VARCHAR(255)")
            print("models表添加result_example_image字段成功")

        # 检查models表是否有parameters列
        cursor.execute("SHOW COLUMNS FROM models LIKE 'parameters'")
        if not cursor.fetchone():
            cursor.execute("ALTER TABLE models ADD COLUMN parameters TEXT")
            print("models表添加parameters字段成功")

        # 检查models表是否有instructions列
        cursor.execute("SHOW COLUMNS FROM models LIKE 'instructions'")
        if not cursor.fetchone():
            cursor.execute("ALTER TABLE models ADD COLUMN instructions TEXT")
            print("models表添加instructions字段成功")

        # 检查processing_history表是否有object_count列
        cursor.execute("SHOW COLUMNS FROM processing_history LIKE 'object_count'")
        if not cursor.fetchone():
            cursor.execute("ALTER TABLE processing_history ADD COLUMN object_count INT")
            print("processing_history表添加object_count字段成功")

        # 检查processing_history表是否有categories列
        cursor.execute("SHOW COLUMNS FROM processing_history LIKE 'categories'")
        if not cursor.fetchone():
            cursor.execute("ALTER TABLE processing_history ADD COLUMN categories JSON")
            print("processing_history表添加categories字段成功")

        # 检查processing_history表是否有file_type列
        cursor.execute("SHOW COLUMNS FROM processing_history LIKE 'file_type'")
        if not cursor.fetchone():
            cursor.execute("ALTER TABLE processing_history ADD COLUMN file_type VARCHAR(10) DEFAULT 'image'")
            print("processing_history表添加file_type字段成功")

        # 检查processing_history表是否有status列
        cursor.execute("SHOW COLUMNS FROM processing_history LIKE 'status'")
        if not cursor.fetchone():
            cursor.execute("ALTER TABLE processing_history ADD COLUMN status VARCHAR(20) DEFAULT 'completed'")
            print("processing_history表添加status字段成功")

        # 检查processing_history表是否有progress列
        cursor.execute("SHOW COLUMNS FROM processing_history LIKE 'progress'")
        if not cursor.fetchone():
            cursor.execute("ALTER TABLE processing_history ADD COLUMN progress INT DEFAULT 100")
            print("processing_history表添加progress字段成功")

        # 检查是否存在processing_jobs表
        cursor.execute("SHOW TABLES LIKE 'processing_jobs'")
        if not cursor.fetchone():
            cursor.execute("""
            CREATE TABLE processing_jobs (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT NOT NULL,
                history_id INT NOT NULL,
                status VARCHAR(20) NOT NULL,
                progress INT DEFAULT 0,
                message TEXT,
                start_time DATETIME,
                end_time DATETIME,
                created_at DATETIME NOT NULL,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                FOREIGN KEY (history_id) REFERENCES processing_history(id) ON DELETE CASCADE
            )
            """)
            print("创建processing_jobs表成功")

        conn.commit()
        conn.close()

        print("数据库架构更新成功！")
        return True
    except Exception as e:
        print(f"数据库架构更新失败: {e}")
        return False


if __name__ == '__main__':
    print("开始初始化遥感图像识别系统...")

    if create_directories() and initialize_database() and add_default_models() and create_admin_user() and update_database_schema():
        print("系统初始化完成！")
        print("系统现已支持图像和视频处理功能！")
    else:
        print("系统初始化失败，请检查错误信息并重试。")
        sys.exit(1)