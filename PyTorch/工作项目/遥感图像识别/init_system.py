import pymysql
import os
import sys
import json
from werkzeug.security import generate_password_hash
import datetime

PASSWORD = ''


# 数据库初始化主函数
def initialize_database():
    try:
        # 创建数据库连接（使用root用户进行初始设置）
        conn = pymysql.connect(
            host='localhost',
            user='root',
            password=PASSWORD
        )
        cursor = conn.cursor()

        # 创建项目专用数据库
        cursor.execute("CREATE DATABASE IF NOT EXISTS remote_sensing_db")

        # 切换到刚创建的数据库
        cursor.execute("USE remote_sensing_db")

        # 读取数据库架构SQL脚本
        with open('database_schema.sql', 'r', encoding='utf-8') as f:
            sql_script = f.read()

        # 执行SQL脚本中的每条语句
        for statement in sql_script.split(';'):
            if statement.strip():
                try:
                    cursor.execute(statement)
                except pymysql.err.InternalError as e:
                    # 忽略表已存在的错误，其他错误则打印提示
                    if "already exists" not in str(e):
                        print(f"执行SQL语句出错: {e}")
                        print(f"出错的SQL: {statement}")

        # 确保notifications表存在
        try:
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
                print("已创建notifications表")
        except Exception as e:
            print(f"创建notifications表出错: {e}")

        # 确保permission_requests表存在
        try:
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
                print("已创建permission_requests表")
        except Exception as e:
            print(f"创建permission_requests表出错: {e}")

        # 检查并添加users表中可能缺失的字段
        try:
            cursor.execute("DESCRIBE users")
            columns = [row[0] for row in cursor.fetchall()]

            if 'name' not in columns:
                cursor.execute("ALTER TABLE users ADD COLUMN name VARCHAR(100)")
                print("已添加users.name字段")

            if 'phone' not in columns:
                cursor.execute("ALTER TABLE users ADD COLUMN phone VARCHAR(20)")
                print("已添加users.phone字段")

            if 'bio' not in columns:
                cursor.execute("ALTER TABLE users ADD COLUMN bio TEXT")
                print("已添加users.bio字段")

            if 'is_admin' not in columns:
                cursor.execute("ALTER TABLE users ADD COLUMN is_admin BOOLEAN DEFAULT FALSE")
                print("已添加users.is_admin字段")
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

        # 创建数据库访问用户并授予权限
        try:
            cursor.execute("CREATE USER IF NOT EXISTS 'remote_sensing_user'@'localhost' IDENTIFIED BY 'your_password'")
            cursor.execute("GRANT ALL PRIVILEGES ON remote_sensing_db.* TO 'remote_sensing_user'@'localhost'")
            cursor.execute("FLUSH PRIVILEGES")
        except Exception as e:
            print(f"创建数据库用户出错: {e}")

        # 提交更改并关闭连接
        conn.commit()
        conn.close()

        print("数据库初始化成功！")
        return True
    except Exception as e:
        print(f"数据库初始化失败: {e}")
        return False


# 添加默认系统模型
def add_default_models():
    try:
        # 连接到项目数据库
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

        # 获取当前脚本所在目录
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # 确保系统模型目录存在
        model_dir = os.path.join(current_dir, 'models', 'system')
        os.makedirs(model_dir, exist_ok=True)

        # 扫描模型目录中的模型文件
        model_files = []
        for file in os.listdir(model_dir):
            if file.endswith(('.pt', '.pth', '.h5')):
                model_files.append(file)

        # 为每个模型文件创建数据库记录
        if model_files:
            print(f"发现{len(model_files)}个模型文件")
            example_image_path = os.path.join('static', 'images', 'default_input.jpg')
            result_example_image_path = os.path.join('static', 'images', 'default_result.jpg')
            for file in model_files:
                model_path = os.path.join(model_dir, file)

                # 根据文件名猜测模型类型和名称
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
                    model_type = 'general'
                    model_name = f'通用遥感分析模型({file})'
                    description = '系统自动识别的遥感图像分析模型'

                # 插入模型记录
                cursor.execute(
                    """INSERT INTO models 
                    (name, type, description, path, example_image, result_example_image, 
                    is_system, is_shared, accuracy, usage_count, supports_video, created_at) 
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())""",
                    (model_name, model_type, description, model_path, example_image_path,
                     result_example_image_path, True, True, 0.0, 0, True)
                )
                print(f"添加模型: {model_name} (文件: {file})")
        else:
            print("未找到模型文件，跳过添加默认模型步骤")

        # 提交更改并关闭连接
        conn.commit()
        conn.close()

        print("模型添加完成！")
        return True
    except Exception as e:
        print(f"添加模型失败: {e}")
        return False


# 创建管理员用户
def create_admin_user():
    try:
        # 连接到项目数据库
        conn = pymysql.connect(
            host='localhost',
            user='root',
            password=PASSWORD,
            database='remote_sensing_db',
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )
        cursor = conn.cursor()

        # 插入管理员账户（如果已存在则跳过）
        hashed_password = generate_password_hash('admin123')
        try:
            cursor.execute(
                """INSERT INTO users 
                   (username, email, password, organization, department, license_number, 
                    research_field, platforms, usage_purpose, name, phone, bio, is_admin, created_at) 
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                ('admin', 'admin@example.com', hashed_password, '系统管理', '管理部门', 'ADMIN-LICENSE',
                 '系统管理', json.dumps([]), '系统管理', '系统管理员', '13800138000',
                 '系统管理员账号，负责系统维护和用户支持', 1, datetime.datetime.now())
            )
            admin_id = cursor.lastrowid
        except Exception as e:
            print(f"插入管理员账户时可能已存在，忽略: {e}")
            # 查询admin用户ID
            cursor.execute("SELECT id FROM users WHERE username = 'admin'")
            row = cursor.fetchone()
            admin_id = row['id'] if row else None

            # 确保admin用户具有管理员权限
            if admin_id:
                cursor.execute("UPDATE users SET is_admin = 1 WHERE id = %s", (admin_id,))
                print(f"已更新admin用户为管理员权限")

        # 为管理员添加欢迎通知
        if admin_id:
            current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            cursor.execute(
                """INSERT INTO notifications 
                   (user_id, title, content, type, is_read, created_at) 
                   VALUES (%s, %s, %s, %s, %s, %s)""",
                (
                    admin_id, '欢迎使用遥感图像识别系统',
                    '感谢您使用本系统，这里是管理员账号，您可以管理系统资源和用户权限。',
                    'info', 0, current_time
                )
            )

        # 提交更改并关闭连接
        conn.commit()
        conn.close()

        print("管理员账户初始化完成！默认账号：admin，密码：admin123")
        return True
    except Exception as e:
        print(f"创建管理员账户失败: {e}")
        return False


# 创建必要的目录结构
def create_directories():
    try:
        # 创建上传文件目录
        os.makedirs('uploads', exist_ok=True)
        # 创建系统模型目录
        os.makedirs('models/system', exist_ok=True)
        # 创建自定义模型目录
        os.makedirs('models/custom', exist_ok=True)
        # 创建图像结果存储目录
        os.makedirs('static/results', exist_ok=True)
        # 创建视频结果存储目录
        os.makedirs('static/videos', exist_ok=True)

        print("目录结构创建完成！")
        return True
    except Exception as e:
        print(f"创建目录失败: {e}")
        return False


# 更新数据库架构（添加新字段或表）
def update_database_schema():
    try:
        # 连接到项目数据库
        conn = pymysql.connect(
            host='localhost',
            user='root',
            password=PASSWORD,
            database='remote_sensing_db',
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )
        cursor = conn.cursor()

        # 检查并添加images表的file_type字段
        cursor.execute("SHOW COLUMNS FROM images LIKE 'file_type'")
        if not cursor.fetchone():
            cursor.execute("ALTER TABLE images ADD COLUMN file_type VARCHAR(10) DEFAULT 'image'")
            print("已添加images.file_type字段")

        # 检查并添加images表的duration字段
        cursor.execute("SHOW COLUMNS FROM images LIKE 'duration'")
        if not cursor.fetchone():
            cursor.execute("ALTER TABLE images ADD COLUMN duration FLOAT")
            print("已添加images.duration字段")

        # 检查并添加images表的frame_count字段
        cursor.execute("SHOW COLUMNS FROM images LIKE 'frame_count'")
        if not cursor.fetchone():
            cursor.execute("ALTER TABLE images ADD COLUMN frame_count INT")
            print("已添加images.frame_count字段")

        # 检查并添加models表的supports_video字段
        cursor.execute("SHOW COLUMNS FROM models LIKE 'supports_video'")
        if not cursor.fetchone():
            cursor.execute("ALTER TABLE models ADD COLUMN supports_video BOOLEAN DEFAULT TRUE")
            print("已添加models.supports_video字段")

        # 检查并添加models表的result_example_image字段
        cursor.execute("SHOW COLUMNS FROM models LIKE 'result_example_image'")
        if not cursor.fetchone():
            cursor.execute("ALTER TABLE models ADD COLUMN result_example_image VARCHAR(255)")
            print("已添加models.result_example_image字段")

        # 检查并添加models表的parameters字段
        cursor.execute("SHOW COLUMNS FROM models LIKE 'parameters'")
        if not cursor.fetchone():
            cursor.execute("ALTER TABLE models ADD COLUMN parameters TEXT")
            print("已添加models.parameters字段")

        # 检查并添加models表的instructions字段
        cursor.execute("SHOW COLUMNS FROM models LIKE 'instructions'")
        if not cursor.fetchone():
            cursor.execute("ALTER TABLE models ADD COLUMN instructions TEXT")
            print("已添加models.instructions字段")

        # 检查并添加processing_history表的object_count字段
        cursor.execute("SHOW COLUMNS FROM processing_history LIKE 'object_count'")
        if not cursor.fetchone():
            cursor.execute("ALTER TABLE processing_history ADD COLUMN object_count INT")
            print("已添加processing_history.object_count字段")

        # 检查并添加processing_history表的categories字段
        cursor.execute("SHOW COLUMNS FROM processing_history LIKE 'categories'")
        if not cursor.fetchone():
            cursor.execute("ALTER TABLE processing_history ADD COLUMN categories JSON")
            print("已添加processing_history.categories字段")

        # 检查并添加processing_history表的file_type字段
        cursor.execute("SHOW COLUMNS FROM processing_history LIKE 'file_type'")
        if not cursor.fetchone():
            cursor.execute("ALTER TABLE processing_history ADD COLUMN file_type VARCHAR(10) DEFAULT 'image'")
            print("已添加processing_history.file_type字段")

        # 检查并添加processing_history表的status字段
        cursor.execute("SHOW COLUMNS FROM processing_history LIKE 'status'")
        if not cursor.fetchone():
            cursor.execute("ALTER TABLE processing_history ADD COLUMN status VARCHAR(20) DEFAULT 'completed'")
            print("已添加processing_history.status字段")

        # 检查并添加processing_history表的progress字段
        cursor.execute("SHOW COLUMNS FROM processing_history LIKE 'progress'")
        if not cursor.fetchone():
            cursor.execute("ALTER TABLE processing_history ADD COLUMN progress INT DEFAULT 100")
            print("已添加processing_history.progress字段")

        # 检查并创建processing_jobs表
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
            print("已创建processing_jobs表")

        # 提交更改并关闭连接
        conn.commit()
        conn.close()

        print("数据库架构更新完成！")
        return True
    except Exception as e:
        print(f"数据库架构更新失败: {e}")
        return False


# 主程序入口
if __name__ == '__main__':
    print("开始初始化遥感图像识别系统...")

    # 依次执行初始化步骤
    if (create_directories() and
            initialize_database() and
            add_default_models() and
            create_admin_user() and
            update_database_schema()):
        print("系统初始化成功！")
        print("系统现已支持图像和视频处理功能！")
    else:
        print("系统初始化失败，请根据提示信息检查并重试。")
        sys.exit(1)